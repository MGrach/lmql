"""
Parser for LMQL queries (phase 2 out of 3).

Sits on top of ``lmql_lexer`` (phase 1).
It reads the flat ``LMQLToken`` stream,
finds the depth-0 clause boundaries the lexer already marked as ``KEYWORD``,
and slices the source text into one span per clause:

    query := prologue decoder? prompt fromclause? whereclause? distribution?

It then hands each span to its leaf sublanguage :

    decoder       -> ast.parse            (a bare Name like `argmax`, or a Call)
    prompt        -> ast.parse            (Python statements, after dedent)
    fromclause    -> ast.parse            (a Python expression)
    distribution  -> ast.parse + shape    ('NAME in <expr>')
    whereclause   -> parse_where           (a separate constraint IRSEAM)

Design notes vs. the original ``fragment_parser.py``
----------------------------------------------------
* Boundaries are not a state machine.
    The lexer types a CLAUSE_KEYWORD as ``KEYWORD`` only at depth 0,
    so finding boundaries is a single scan for depth-0 ``KEYWORD`` tokens.
    Nesting can never be mistaken for a boundary.
* The parser accumulates *source spans* (byte offsets),
    not token lists, and feeds the original substring to each leaf parser.
    Because nothing is ever untokenized, the original's
    ``double_escape`` / ``double_unescape`` and
    The NAME+STRING prefix-merging heuristic are removed.
* There is NO inline ``where`` / ``distribution`` form.
    Every depth-0 ``where`` or ``distribution`` keyword is a clause boundary
    The original ``inline_where_transform`` / ``inline_distribution_transform`` are deleted.
* ``from`` is the one ambiguous keyword (clause vs. ``from ... import ...``).
    The lexer's ``is_import_from`` hook resolves that.
* Errors are rendered through the lexer's shared ``Source`` / ``Diagnostic``,
    so parser errors read exactly like lexer errors.

Typing
``prologue`` and ``prompt`` are always produced and are therefore non-optional fields
``decoder`` and the trailing clauses are ``Optional[Span]`` and are only touched inside ``is not None`` guards,
so a static checker can narrow them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, List, Optional, Tuple, TypeVar
import ast
import textwrap

from lmql_lexer import (tokenize,
                        LMQLToken,
                        LMQLTokenType as T,
                        DECODERS,
                        is_import_from,
                        Source,
                        Diagnostic,
                        )


class ParseError(SyntaxError):
    """A clause-structure error, rendered against the source with a caret."""

    def __init__(self, diagnostic: Diagnostic, source: Source):
        super().__init__(diagnostic.render(source))
        self.diagnostic = diagnostic
        self.start = diagnostic.start
        self.end = diagnostic.end


# ---------------------------------------------------------------------------
# Result types. Phase 2 produces spans, phase 3 fills in the AST / IR.
# ---------------------------------------------------------------------------

@dataclass
class Span:
    start: int
    end: int

    def text(self, src: str) -> str:
        return src[self.start:self.end]

    def is_empty(self, src: str) -> bool:
        return self.text(src).strip() == ""


@dataclass
class LMQLDistributionClause:
    variable_name: str
    values: ast.expr            # the <expr> to the right of `in`


# The parsed form of a `where` clause. The seam is parametric:
# Te base pipeline leaves it abstract,
# `PyExprParser` binds it to ``ast.expr`` (the Python-expression fallback),
# and a real deployment binds it to its constraint IR
# by subclassing ``LMQLParser[MyIR]`` and overriding ``parse_where``.
# Consumers get that concrete type back.
WhereIR = TypeVar("WhereIR")


@dataclass
class LMQLQuery(Generic[WhereIR]):
    source: Source

    # always produced (required) ---
    # source spans:
    prologue: Span
    prompt: Span
    # leaf parse results (decoder defaults to __dynamic__, from to "<dynamic>",
    # prompt to []): these are never None on a returned query.
    decoder_ast: ast.expr
    prompt_ast: List[ast.stmt]
    from_ast: ast.expr

    # optional: None exactly when the corresponding clause is absent ---
    decoder: Optional[Span] = None
    from_clause: Optional[Span] = None
    where: Optional[Span] = None
    distribution: Optional[Span] = None
    where_ir: Optional[WhereIR] = None
    distribution_clause: Optional[LMQLDistributionClause] = None


# trivia skipped when reasoning about structure (still carried inside spans)
_TRIVIA = (T.NEWLINE, T.COMMENT)


class Clause(Enum):
    """
    The kinds of clause the parser tracks.

    Values match the lexer's keyword text,
    so ``Clause(tok.text)`` resolves a trailing-keyword token to its kind
    and ``clause.value`` renders in messages.
    ``PROMPT`` has no keyword of its own
    (it is the region after the decoder),
    so it never appears as a boundary.
    """
    DECODER = "decoder"
    PROMPT = "prompt"
    FROM = "from"
    WHERE = "where"
    DISTRIBUTION = "distribution"


# legal order of the trailing clauses
_TRAILING_ORDER = {Clause.FROM: 0,
                   Clause.WHERE: 1,
                   Clause.DISTRIBUTION: 2}


class LMQLParser(Generic[WhereIR]):
    def __init__(self, src: str):
        self.src = src
        self.source = Source(src)
        # strict=True makes reserved-word collisions (e.g. `beam = 1`) fatal --
        self.toks: List[LMQLToken] = tokenize(src, strict=True)
        self.eof = self.toks[-1].start                     # == len(src)

    # token helpers ------------------------------------------------------

    def _next_idx(self, i: int) -> Optional[int]:
        j = i + 1
        while j < len(self.toks):
            if self.toks[j].ttype not in _TRIVIA:
                return j
            j += 1
        return None

    def _prev(self, i: int) -> Optional[LMQLToken]:
        j = i - 1
        while j >= 0:
            if self.toks[j].ttype not in _TRIVIA:
                return self.toks[j]
            j -= 1
        return None

    def _err(self,
             tok: LMQLToken,
             message: str, *,
             hint: Optional[str] = None,
             label: str = "here") -> ParseError:
        return ParseError(Diagnostic(message,
                                     tok.start,
                                     tok.end,
                                     "error",
                                     label,
                                     hint),
                          self.source
                          )

    def _err_span(self,
                  span: Span,
                  message: str,
                  hint: Optional[str] = None) -> ParseError:
        return ParseError(Diagnostic(message,
                                     span.start,
                                     max(span.start + 1, span.end),
                                     "error",
                                     "here",
                                     hint),
                          self.source
                          )

    # -- boundary discovery -------------------------------------------------

    def _is_decoder_anchor(self, i: int) -> bool:
        t = self.toks[i]
        if not (t.ttype is T.KEYWORD and t.depth == 0 and t.text in DECODERS):
            return False
        prev = self._prev(i)
        # `obj.argmax` -- attribute access, not the decoder boundary
        if prev is not None and prev.ttype is T.OTHER and prev.text.endswith("."):
            return False
        return True

    def _boundaries(self) -> List[Tuple[Clause, int]]:
        """
        Ordered (clause, token_index) for the real clause boundaries.
        """
        out: List[Tuple[Clause, int]] = []
        decoder_idx: Optional[int] = None
        for i, t in enumerate(self.toks):
            if t.ttype is not T.KEYWORD or t.depth != 0:
                continue
            if t.text in DECODERS:
                if not self._is_decoder_anchor(i):
                    continue
                if decoder_idx is not None:
                    raise self._err(t,
                                    f"unexpected second decoder {t.text!r}",
                                    hint="a query has at most one decoder, before the prompt")
                decoder_idx = i
                out.append((Clause.DECODER, i))
                continue
            # a depth-0 keyword that isn't a decoder is a trailing clause keyword
            clause = Clause(t.text)             # FROM | WHERE | DISTRIBUTION
            if clause is Clause.FROM and is_import_from(self.toks, i):
                continue                        # python `from ... import ...`
            out.append((clause, i))
        return out

    def _matching_close(self, open_idx: int) -> int:
        """
        Index of the CLOSE token matching the OPEN at ``open_idx``.

        Leans on the depth the lexer already assigned
        an opener and its matching closer share a depth,
        and everything nested sits deeper,
        so this reads the lexer's bookkeeping instead of re-counting brackets.
        """
        open_depth = self.toks[open_idx].depth
        for k in range(open_idx + 1, len(self.toks)):
            t = self.toks[k]
            if t.ttype is T.CLOSE and t.depth == open_depth:
                return k
            if t.ttype is T.EOF:
                break
        # the lexer guarantees balanced brackets, so this is unreachable
        raise self._err(self.toks[open_idx], "unterminated bracket")

    def _decoder_end(self, i: int) -> int:
        """
        End offset of the decoder clause at anchor ``i``, consuming an
        optional parenthesised argument list.
        """
        nxt = self._next_idx(i)
        if nxt is not None and self.toks[nxt].ttype is T.OPEN:
            return self.toks[self._matching_close(nxt)].end
        return self.toks[i].end

    # -- the parse ----------------------------------------------------------

    def parse(self) -> "LMQLQuery[WhereIR]":
        bounds = self._boundaries()

        decoder_b = [b for b in bounds if b[0] is Clause.DECODER]
        trailing = [b for b in bounds if b[0] is not Clause.DECODER]

        # the decoder anchor must precede every trailing clause
        if decoder_b and trailing and decoder_b[0][1] > trailing[0][1]:
            kind, ti = trailing[0]
            raise self._err(self.toks[ti],
                            f"{kind.value!r} clause appears before the decoder",
                            hint="order is: decoder? prompt from? where? distribution?")

        # prologue / decoder / prompt start ----
        decoder_span: Optional[Span]
        if decoder_b:
            di = decoder_b[0][1]
            prologue_span = Span(0, self.toks[di].start)
            dec_end = self._decoder_end(di)
            decoder_span = Span(self.toks[di].start, dec_end)
            prompt_start = dec_end
        else:
            # no decoder keyword: nothing is prologue, the prompt starts at 0
            prologue_span = Span(0, 0)
            decoder_span = None
            prompt_start = 0

        # prompt runs until the first trailing boundary (or EOF) ----
        first_trailing = trailing[0][1] if trailing else None
        prompt_end = (self.toks[first_trailing].start
                      if first_trailing is not None else self.eof)
        prompt_span = Span(prompt_start, prompt_end)

        # trailing clauses: validate order + uniqueness, cut spans ----
        from_span: Optional[Span] = None
        where_span: Optional[Span] = None
        distribution_span: Optional[Span] = None
        last_rank = -1
        for n, (kind, ti) in enumerate(trailing):
            rank = _TRAILING_ORDER[kind]
            if rank == last_rank:
                raise self._err(
                    self.toks[ti], f"duplicate {kind.value!r} clause")
            if rank < last_rank:
                raise self._err(
                    self.toks[ti], f"{kind.value!r} clause is out of order",
                    hint="order is: from? where? distribution?")
            last_rank = rank

            body_start = self.toks[ti].end
            nxt = trailing[n + 1][1] if n + 1 < len(trailing) else None
            body_end = self.toks[nxt].start if nxt is not None else self.eof
            span = Span(body_start, body_end)
            if kind is Clause.FROM:
                from_span = span
            elif kind is Clause.WHERE:
                where_span = span
            else:
                distribution_span = span

        # leaf parsing :
        # compute results before constructing, so
        # the always-present fields are never None on the returned object ----
        where_ir: Optional[WhereIR] = None
        if where_span is not None and not where_span.is_empty(self.src):
            where_ir = self.parse_where(where_span)

        distribution_clause: Optional[LMQLDistributionClause] = None
        if distribution_span is not None and not distribution_span.is_empty(self.src):
            distribution_clause = self._parse_distribution(distribution_span)

        q: "LMQLQuery[WhereIR]" = LMQLQuery(source=self.source,
                                            prologue=prologue_span,
                                            prompt=prompt_span,
                                            decoder_ast=self._parse_decoder(
                                                decoder_span),
                                            prompt_ast=self._parse_prompt(
                                                prompt_span),
                                            from_ast=self._parse_from(
                                                from_span),
                                            decoder=decoder_span,
                                            from_clause=from_span,
                                            where=where_span,
                                            distribution=distribution_span,
                                            where_ir=where_ir,
                                            distribution_clause=distribution_clause,
                                            )
        return q

    # leaf dispatch  -------------------------------------------
    def _parse_decoder(self, span: Optional[Span]) -> ast.expr:
        # a bare Name (argmax) or a Call (sample(n=2)), default __dynamic__
        if span is not None and not span.is_empty(self.src):
            return self._parse_expr(span, Clause.DECODER)
        return ast.Name(id="__dynamic__", ctx=ast.Load())

    def _parse_prompt(self, span: Span) -> List[ast.stmt]:
        # Python statements, dedent first
        # (it was sliced from mid-query)
        prompt_src = textwrap.dedent(span.text(self.src))
        if not prompt_src.strip():
            return []
        try:
            return ast.parse(prompt_src).body
        except SyntaxError as e:
            raise self._err_span(span,
                                 f"failed to parse {Clause.PROMPT.value} clause: {e.msg}"
                                 ) from None

    def _parse_from(self, span: Optional[Span]) -> ast.expr:
        # a Python expression, default sentinel "<dynamic>"
        if span is not None and not span.is_empty(self.src):
            return self._parse_expr(span, Clause.FROM)
        return ast.Constant(value="<dynamic>")

    def parse_where(self, span: Span) -> WhereIR:
        """
        SEAM for the constraint sublanguage.

        Abstract on purpose: a parser is parametric over its where-IR,
        so the base class has nothing concrete to return.
        Bind it by subclassing ``LMQLParser[MyIR]`` and overriding this method
        (see ``PyExprParser`` for the default Python-expression binding).
        Whatever this returns becomes the static type of ``LMQLQuery.where_ir``
        for that parser's consumers.
        """
        raise NotImplementedError("LMQLParser is abstract over its where-IR; use PyExprParser, or "
                                  "subclass LMQLParser[YourIR] and override parse_where")

    def _parse_distribution(self, span: Span) -> LMQLDistributionClause:
        node = self._parse_expr(span, Clause.DISTRIBUTION)
        bad = "the distribution clause must look like  VAR in [ ... ]"
        if (not isinstance(node, ast.Compare)
                or len(node.ops) != 1
                or not isinstance(node.ops[0], ast.In)):
            raise self._err_span(span, bad)
        left = node.left
        if not isinstance(left, ast.Name):
            raise self._err_span(span, bad)
        return LMQLDistributionClause(left.id, node.comparators[0])

    # -- expression leaf parsing -------------------------------------------

    def _parse_expr(self, span: Span, loc: Clause) -> ast.expr:
        body = span.text(self.src).strip()
        if not body:
            raise self._err_span(span,
                                 f"empty {loc.value} clause")
        # Parse in eval mode so we get the expression node directly. The parens
        # let a multi-line clause body read as one logical line, with its own
        # indentation / trailing comments not mattering.
        try:
            return ast.parse("(\n" + body + "\n)", mode="eval").body
        except SyntaxError as e:
            raise self._err_span(span,
                                 f"failed to parse {loc.value} clause: {e.msg}"
                                 ) from None


class PyExprParser(LMQLParser[ast.expr]):
    """
    Default parser: the ``where`` clause is parsed as a Python expression.

    ``parse(src)`` uses this, so out of the box ``q.where_ir`` is typed
    ``Optional[ast.expr]`` and consumers can use it as an ``ast.expr`` directly,
    with no cast or isinstance dance.
    """

    def parse_where(self, span: Span) -> ast.expr:
        return self._parse_expr(span, Clause.WHERE)


def parse(src: str) -> "LMQLQuery[ast.expr]":
    """
    Parse an LMQL query string into an ``LMQLQuery`` (spans + leaf ASTs),
    using the default Python-expression binding for the ``where`` clause."""
    return PyExprParser(src).parse()
