"""
Clause-level grammar for LMQL queries (real productions, principled boundaries).

    query        := prologue decoder? prompt fromclause? whereclause? distribution?
    prologue     := <python tokens before first decoder keyword>     (imports, defs)
    decoder      := DECODER ('(' <python> ')')?
    prompt       := <python statements + prompt strings>             -> ast.parse
    fromclause   := 'from' <python expr>                             -> ast.parse
    whereclause  := 'where' <constraint>                             -> parse_where (typed IR)
    distribution := 'distribution' NAME 'in' <python expr>           -> ast.parse

Leaf sublanguages are delegated
only clause STRUCTURE is parsed here

Clause keywords are RESERVED at parenthesis/bracket depth 0, no positional heuristics.

- PHASE 1: the lexer only.
It turns source text into a flat token (lmqltoken) stream in which clause boundaries are already first-class
(`KEYWORD` lmqltokens at depth 0).
- the parser (phase 2) reads this stream and slices source spans
- thecompiler (phase 3) hands those spans to the leaf sublanguages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple
import bisect
import re


# ---------------------------------------------------------------------------
# Reserved vocabulary.
# These are the ONLY words that become clause boundaries,
# and only when they appear at bracket/parenthesis depth 0.
# Naming a Python variable  after any of these at top level is a
# deliberate reserved-word collision.
# ---------------------------------------------------------------------------

DECODERS = frozenset({"argmax",
                      "sample",
                      "beam",
                      "beam_sample",
                      "best_k",
                      "greedy",
                      "var"
                      })

# `from`/`where`/`distribution` introduce the trailing clauses
# the decoders "anchor" the prologue/decoder boundary.
# TODO distribution keyword mechanics unclear
CLAUSE_KEYWORDS = DECODERS | {"from",
                              "where",
                              "distribution"}


class LMQLTokenType(Enum):
    NAME = auto()       # NOT a depth-0 clause keyword
    KEYWORD = auto()    # a CLAUSE_KEYWORD spelled at depth 0 -> a clause boundary
    STRING = auto()     # opaque string literal (any quote form / prefix)
    COMMENT = auto()    # opaque '# ...' to end of line
    OPEN = auto()       # ( [ {   -> increments depth
    CLOSE = auto()      # ) ] }   -> decrements depth
    OTHER = auto()      # operators, numbers, punctuation (body content)
    NEWLINE = auto()    # kept for error messages, the parser ignores these
    EOF = auto()


@dataclass(frozen=True)
class LMQLToken:
    ttype: LMQLTokenType
    text: str
    start: int          # byte offset into source (inclusive)
    end: int            # byte offset into source (exclusive)
    depth: int          # nesting depth at the lmqltoken's start, top level == 0

    def __repr__(self) -> str:
        return f"{self.ttype.name}({self.text!r}@{self.start}:{self.end} d{self.depth})"


class Source:
    """
    Wraps source text so any (start, end) offset span can be turned into a line/column
    and rendered as a caret-underlined snippet.
    Should be shared by every layer (lexer now, parser/compiler then later),
    so all errors read the same.
    """

    def __init__(self, text: str):
        self.text = text
        self.lines = text.split("\n")
        starts, off = [], 0
        for ln in self.lines:
            starts.append(off)
            off += len(ln) + 1
        # offset where each (0-based) line begins
        self._line_starts = starts

    def line_col(self, offset: int) -> Tuple[int, int]:
        """
        Return (line, col), both 1-based, for a source offset.
        """
        offset = max(0, min(offset, len(self.text)))
        line = max(1, bisect.bisect_right(self._line_starts, offset))
        col = offset - self._line_starts[line - 1] + 1
        return line, col

    def render(self,
               start: int,
               end: Optional[int] = None, *,
               severity: str = "error",
               message: str = "",
               label: str = "here",
               hint: Optional[str] = None) -> str:
        if end is None or end <= start:
            end = start + 1
        line, col = self.line_col(start)
        if 1 <= line <= len(self.lines):
            text = self.lines[line - 1]
        else:
            text = ""
        u_start = start - self._line_starts[line - 1]
        u_width = max(1,
                      min(end,
                          self._line_starts[line - 1] + len(text)) - start)
        gutter = str(line)
        pad = " " * len(gutter)
        out = []
        if message:
            out.append(f"{severity}: {message}")
        out.append(f"{pad}---> line {line}, col {col}")
        out.append(f"{pad} |")
        out.append(f"{gutter} | {text}")
        caret = " " * u_start + "^" * u_width + (f" {label}" if label else "")
        out.append(f"{pad} | {caret}")
        if hint:
            out.append(f"{pad} = hint: {hint}")
        return "\n".join(out)


@dataclass
class Diagnostic:
    """
    A message. Severity is 'error' or 'warning'.
    Carries enough to render itself against the source.
    """
    message: str
    start: int
    end: int
    severity: str = "error"
    label: str = "here"
    hint: Optional[str] = None

    def render(self, src) -> str:
        if isinstance(src, Source):
            source = src
        else:
            source = Source(src)
        return source.render(self.start,
                             self.end,
                             severity=self.severity,
                             message=self.message,
                             label=self.label,
                             hint=self.hint)


class LexError(SyntaxError):
    """
    A lexical error: Unterminated string, unbalanced bracket, reserved-word collision, ...
    ``str(e)`` is a rendered, caret-underlined snippet.
    the structured ``.diagnostic`` (and ``.start`` / ``.end``) are available too.
    """

    def __init__(self, message: str, diagnostic: Optional[Diagnostic] = None):
        super().__init__(message)
        self.diagnostic = diagnostic

        self.start = diagnostic.start if diagnostic else None
        self.end = diagnostic.end if diagnostic else None


def _lex_error(src: str,
               start: int,
               message: str, *,
               end: Optional[int] = None,
               label: str = "here",
               hint: Optional[str] = None) -> LexError:
    diag = Diagnostic(message,
                      start,
                      end if end is not None else start + 1,
                      "error",
                      label,
                      hint)
    return LexError(diag.render(src), diag)


_NAME = re.compile(r"[A-Za-z_]\w*")
_INLINE_WS = re.compile(r"[^\S\n]+")            # whitespace, but not newline
# r, b, f, u and 2-letter combinations
_STRING_PREFIX = re.compile(r"[rRbBfFuU]{0,2}")
# a run of "everything else"
_OTHER_RUN = re.compile(r"[^\s()\[\]{}#\"'A-Za-z_]+")

_OPEN = "([{"
_CLOSE = ")]}"
_MATCH = {")": "(", "]": "[", "}": "{"}


def _scan_string(src: str, qpos: int, start: int) -> int:
    """
    Return the offset just past a string literal whose opening quote is at qpos.

    Handles single/double and triple quotes.
    Backslash escapes the next char for termination purposes
    (correct even for raw strings, where the backslash is kept
    in the value but still guards the closing quote).
    """
    n = len(src)
    quote = src[qpos]
    if src[qpos:qpos + 3] == quote * 3:          # triple-quoted: newlines allowed
        close = quote * 3
        i = qpos + 3
        while i < n:
            if src[i] == "\\":
                i += 2
                continue
            if src[i:i + 3] == close:
                return i + 3
            i += 1
        raise _lex_error(src,
                         start,
                         "unterminated triple-quoted string",
                         end=len(src), label="opened here")
    # single-line string
    i = qpos + 1
    while i < n:
        ch = src[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "\n":
            raise _lex_error(src,
                             start,
                             "unterminated string literal (newline before closing quote)",
                             label="opened here")
        if ch == quote:
            return i + 1
        i += 1
    raise _lex_error(src,
                     start,
                     "unterminated string literal",
                     end=len(src),
                     label="opened here"
                     )


def lex_tokenize(src: str, *, strict: bool = False) -> List[LMQLToken]:
    """
    Lex `src` into a flat lmqltoken stream with depth-0 keyword resolution.

    With ``strict=True``,
    also run :func:`check_reserved_collisions` and raise a rendered :class:`LexError`
    on the first error-level collision (e.g. a decoder keyword used as a variable name).
    The parser pipeline should pass strict=True.
    callers that only want raw lmqltokens leave it False (the default).
    """
    toks: List[LMQLToken] = []
    i, n = 0, len(src)
    depth = 0
    # (open_char, offset) for diagnostics
    bracket_stack: List[Tuple[str, int]] = []

    while i < n:
        c = src[i]

        # inline whitespace
        m = _INLINE_WS.match(src, i)
        if m:
            i = m.end()
            continue

        # newline (significant only for error positions)
        if c == "\n":
            toks.append(LMQLToken(LMQLTokenType.NEWLINE, "\n", i, i + 1, depth))
            i += 1
            continue

        # comment to end of line
        if c == "#":
            j = src.find("\n", i)
            if j == -1:
                j = n
            toks.append(LMQLToken(LMQLTokenType.COMMENT,
                                  src[i:j],
                                  i,
                                  j,
                                  depth
                                  )
                        )
            i = j
            continue

        # string literal: optional 0-2 char prefix immediately followed by a quote
        pend = _STRING_PREFIX.match(src, i).end()  # type:ignore
        # m = _STRING_PREFIX.match(src, i)
        # pend = m.end()

        if pend < n and src[pend] in "\"'":
            end = _scan_string(src, pend, i)
            toks.append(LMQLToken(LMQLTokenType.STRING,
                                  src[i:end],
                                  i,
                                  end,
                                  depth
                                  )
                        )
            i = end
            continue

        # identifier / clause keyword
        m = _NAME.match(src, i)
        if m:
            word = m.group()
            ttype = (LMQLTokenType.KEYWORD
                     if depth == 0 and word in CLAUSE_KEYWORDS
                     else LMQLTokenType.NAME)
            toks.append(LMQLToken(ttype, word, i, m.end(), depth))
            i = m.end()
            continue

        # brackets (depth tracking + balance checking)
        if c in _OPEN:
            toks.append(LMQLToken(LMQLTokenType.OPEN, c, i, i + 1, depth))
            depth += 1
            bracket_stack.append((c, i))
            i += 1
            continue
        if c in _CLOSE:
            if not bracket_stack:
                raise _lex_error(src,
                                 i,
                                 f"unmatched closing {c!r}",
                                 label="nothing is open here"
                                 )
            openc, openpos = bracket_stack.pop()
            if _MATCH[c] != openc:
                raise _lex_error(src,
                                 i,
                                 f"mismatched bracket: {c!r} closes {openc!r} "
                                 f"opened at offset {openpos}", label="this one"
                                 )
            depth -= 1
            toks.append(LMQLToken(LMQLTokenType.CLOSE,
                                  c,
                                  i,
                                  i + 1,
                                  depth
                                  )
                        )
            i += 1
            continue

        # everything else (operators, numbers, punctuation) as a single run
        m = _OTHER_RUN.match(src, i)
        run = m.group() if m else c
        toks.append(LMQLToken(LMQLTokenType.OTHER,
                              run,
                              i,  # start
                              i + len(run),  # end
                              depth
                              )
                    )
        i += len(run)

    if bracket_stack:
        openc, openpos = bracket_stack[-1]
        raise _lex_error(src,
                         openpos,
                         f"unclosed {openc!r} (never closed)",
                         label="opened here"
                         )

    toks.append(LMQLToken(LMQLTokenType.EOF, "", n, n, 0))

    if strict:
        for d in _collisions(toks, src):
            if d.severity == "error":
                raise LexError(d.render(src), d)
    return toks


# ---------------------------------------------------------------------------
# From-clause disambiguation (a parser hook, not lexer logic).
#
# `from` is the one clause keyword that is also a Python keyword:
# the prologue and the prompt both can contain Python,
# and Python imports can look like `from ... import ...`.
# `from` is NOT automatically a clause boundary
# it may be an import that belongs to a clause body.
#
# The two forms are disjoint:
# a from-IMPORT always contains a depth-0 `import` in its statement.
# a from-CLAUSE (`from <expr>`) never has an `import` later.
# The lexer still marks every depth-0 `from` as KEYWORD.
# the parser calls this to decide whether a given one actually ends the prompt.
# ---------------------------------------------------------------------------

def is_import_from(toks: List[LMQLToken], i: int) -> bool:
    """
    `toks[i]` must be a depth-0 `from` KEYWORD.
    Returns True if it heads a Python `from ... import ...` statement,
    rather than the from-clause.

    A statement ends at the first depth-0 newline (ign backslash line continuations),
    the next depth-0 clause keyword, or EOF.
    """
    t0 = toks[i]
    if not (t0.ttype is LMQLTokenType.KEYWORD and t0.text == "from" and t0.depth == 0):
        raise ValueError(
            f"is_import_from expects a depth-0 'from' keyword, got {t0!r}")

    j, n = i + 1, len(toks)
    while j < n:
        t = toks[j]
        if t.ttype is LMQLTokenType.EOF:
            return False
        if t.ttype is LMQLTokenType.NEWLINE and t.depth == 0:
            prev = toks[j - 1]                       # backslash continuation?
            if prev.ttype is LMQLTokenType.OTHER and prev.text.endswith("\\"):
                j += 1
                continue
            return False
        if t.ttype is LMQLTokenType.KEYWORD and t.depth == 0:
            return False                             # another boundary came first
        if t.ttype is LMQLTokenType.NAME and t.depth == 0 and t.text == "import":
            return True
        j += 1
    return False

# ---------------------------------------------------------------------------
# Reserved-word collision diagnostics (debugging helpers).
#
# Example: A user writing sth like `beam = "greedy"` in their prologue
# is shadowing a reserved decoder keyword.
# This pass should catch the common, UNAMBIGUOUS misuses at
# the lexer level and reports them with a precise position and a fix.
#
# Lint only the INVENTED reserved words.
# Python itself rejects `from = 1`, so `from` collisions can't reach.
# ---------------------------------------------------------------------------


INVENTED_RESERVED = DECODERS | {"where", "distribution"}

_ASSIGN_OPS = frozenset({"=", "+=", "-=", "*=", "/=", "//=",
                         "**=", "%=", "&=", "|=", "^=", ">>=",
                         "<<=", "@=", ":="
                         })


def _keyword_role(word: str) -> str:
    return "decoder keyword" if word in DECODERS else "clause keyword"


def _significant_neighbors(toks, i):
    """
    Nearest lmqltokens on each side of i, skipping NEWLINE/COMMENT.
    """
    def scan(rng):
        for j in rng:
            if toks[j].ttype not in (LMQLTokenType.NEWLINE,
                                     LMQLTokenType.COMMENT):
                return toks[j]
        return None
    return scan(range(i - 1, -1, -1)), scan(range(i + 1, len(toks)))


def _collisions(toks, src) -> List[Diagnostic]:
    """
    Find depth-0 reserved keywords used unambiguously as plain identifiers.
    """
    diags: List[Diagnostic] = []
    for i, t in enumerate(toks):
        if t.ttype is not LMQLTokenType.KEYWORD or t.depth != 0:
            continue
        if t.text not in INVENTED_RESERVED:
            continue

        prev, nxt = _significant_neighbors(toks, i)

        # `obj.beam` -- here `beam` is an attribute name, not the reserved word.
        if prev is not None and prev.ttype is LMQLTokenType.OTHER and prev.text.endswith("."):
            continue

        role = None
        if prev is not None and prev.ttype is LMQLTokenType.NAME and \
                prev.text in ("import",
                              "as",
                              "def",
                              "class"
                              ):
            role = f"bound as a name by '{prev.text}'"
        elif prev is not None and prev.ttype is LMQLTokenType.OTHER and prev.text == "=":
            role = "used as a value in an assignment"
        elif nxt is not None and nxt.ttype is LMQLTokenType.OTHER:
            if nxt.text == "=":
                role = "used as an assignment target"
            elif nxt.text == ":":
                role = "used as an annotation target"
            elif nxt.text == ".":
                role = "used as an object (attribute access)"
            elif nxt.text in _ASSIGN_OPS:
                role = "used as an assignment target"

        if role is None:
            continue

        diags.append(Diagnostic(message=f"{t.text!r} is a reserved {_keyword_role(t.text)} "
                                f"and cannot be used as a name",
                                start=t.start,
                                end=t.end,
                                label=role,
                                hint=f"at depth 0, {t.text!r} introduces a query clause; rename this "
                                f"binding (e.g. {t.text}_) to avoid the collision",
                                )
                     )
    return diags


def check_reserved_collisions(src: str) -> List[Diagnostic]:
    """
    Public lint:
    Return diagnostics for reserved words misused as identifiers.
    Returns [] for a clean query.
    Does not raise
    (callers decide what to do (the parser pipeline calls
    tokenize(src, strict=True) to turn these fatal).
    """
    return _collisions(lex_tokenize(src), src)


def format_tokens(toks: List[LMQLToken]) -> str:
    """
    Render an lmqltoken stream as an aligned table,
    the first thing to reach for when a query lexes in
    a way one doesn't expect is to.
    """
    rows = [f"{'idx':>3}  {'ttype':<8} {'depth':>5}  {'span':<11}  text",
            f"{'-'*3}  {'-'*8} {'-'*5}  {'-'*11}  {'-'*30}"
            ]
    for i, t in enumerate(toks):
        text = "\\n" if t.ttype is LMQLTokenType.NEWLINE else \
               (t.text if len(t.text) <= 38 else t.text[:35] + "...")
        mark = "  <- boundary" if t.ttype is LMQLTokenType.KEYWORD else ""
        rows.append(f"{i:>3}  {t.ttype.name:<8} {t.depth:>5}  "
                    f"{t.start:>4}:{t.end:<6}  {text!r}{mark}")
    return "\n".join(rows)
