"""
Testing for the clause-level lexer (lmql_lexer.py).

Three layers:
  1. INVARIANTS  - structural properties that must hold for ANY valid input.
                   Run automatically over the whole corpus, so every example
                   below is also a property test for free.
  2. CASES       - named behavioral assertions (the things we designed for).
  3. FUZZ        - seeded random nestings, checked against the invariants.

Run everything:        python test_clauses.py
Inspect one query:     python test_clauses.py dump 'argmax "[X]" from "m"'
Use with pytest too:   pytest test_clauses.py     (the test_* funcs collect)
"""

from __future__ import annotations

import random
import sys

from lmql_lexer import (
    tokenize, LMQLToken, LMQLTokenType, LexError, CLAUSE_KEYWORDS, DECODERS,
    is_import_from, Source, Diagnostic, check_reserved_collisions, format_tokens,
)

K = LMQLTokenType
_INLINE_WS = " \t\r\f\v"   # whitespace the lexer skips. newline is a TOKEN, never a gap


# ---------------------------------------------------------------------------
# Small helpers (also handy at the REPL)
# ---------------------------------------------------------------------------

def significant(src):
    """
    Tokens minus newlines/comments/EOF
    """
    return [t for t in tokenize(src)
            if t.ttype not in (K.NEWLINE, K.COMMENT, K.EOF)]


def ttypes(src):
    return [t.ttype for t in significant(src)]


def keywords(src):
    """
    The depth-0 clause-boundary keywords, in order (the parser's input).
    """
    return [t.text for t in tokenize(src) if t.ttype is K.KEYWORD]


# ---------------------------------------------------------------------------
# 1. INVARIANTS: each returns a list of human-readable problems ([] == ok)
# ---------------------------------------------------------------------------

def inv_coverage(src, toks):
    """
    Tokens are ordered, non-overlapping, carry their own text, and the only
    gaps between them are inline whitespace (newlines are tokens, not gaps).
    """
    problems, pos = [], 0
    for t in toks:
        if t.ttype is K.EOF:
            break
        if t.start < pos:
            problems.append(f"overlap: {t!r} starts before {pos}")
        gap = src[pos:t.start]
        if gap.strip(_INLINE_WS):
            problems.append(f"non-whitespace gap {gap!r} before {t!r}")
        if src[t.start:t.end] != t.text:
            problems.append(
                f"text/span mismatch: {t!r} vs src {src[t.start:t.end]!r}")
        pos = t.end
    eof = toks[-1]
    if eof.ttype is not K.EOF:
        problems.append("stream does not end with EOF")
    elif eof.start != len(src) or eof.end != len(src):
        problems.append(f"EOF span {eof.start}:{eof.end} != len {len(src)}")
    elif src[pos:eof.start].strip(_INLINE_WS):
        problems.append(
            f"non-whitespace tail before EOF: {src[pos:eof.start]!r}")
    return problems


def inv_depth(src, toks):
    """Recompute depth independently and compare to what the lexer recorded.
    OPEN records depth before incrementing; CLOSE records after decrementing."""
    problems, r = [], 0
    for t in toks:
        if t.ttype is K.OPEN:
            if t.depth != r:
                problems.append(
                    f"OPEN {t!r} recorded d{t.depth}, expected d{r}")
            r += 1
        elif t.ttype is K.CLOSE:
            r -= 1
            if t.depth != r:
                problems.append(
                    f"CLOSE {t!r} recorded d{t.depth}, expected d{r}")
        elif t.ttype is K.EOF:
            if r != 0:
                problems.append(f"unbalanced at EOF: depth {r}")
            if t.depth != 0:
                problems.append(f"EOF recorded d{t.depth}, expected d0")
        else:
            if t.depth != r:
                problems.append(f"{t!r} recorded d{t.depth}, expected d{r}")
        if r < 0:
            problems.append("depth went negative (lexer should have raised)")
    return problems


def inv_keyword_rule(src, toks):
    """A token is KEYWORD iff (depth == 0 and spelling is a clause keyword).
    Checks both directions: no false KEYWORD, and no NAME that should be one."""
    problems = []
    for t in toks:
        if t.ttype is K.KEYWORD and not (t.depth == 0 and t.text in CLAUSE_KEYWORDS):
            problems.append(f"bogus KEYWORD {t!r}")
        if t.ttype is K.NAME and t.depth == 0 and t.text in CLAUSE_KEYWORDS:
            problems.append(f"missed keyword {t!r}")
    return problems


INVARIANTS = [inv_coverage, inv_depth, inv_keyword_rule]


def check_all(src):
    """Run every invariant; return flat list of problems."""
    toks = tokenize(src)
    out = []
    for inv in INVARIANTS:
        out += [f"[{inv.__name__}] {p}" for p in inv(src, toks)]
    return out


# ---------------------------------------------------------------------------
# 2. CORPUS  -- named, realistic-ish inputs. Every one is invariant-checked.
# ---------------------------------------------------------------------------

CORPUS = {
    "empty": "",
    "whitespace_only": "   \n  \t\n",
    "bare_string": '"hello"',
    "minimal_query": 'argmax "[ANSWER]" from "gpt-4"',
    "decoder_call": 'sample(temperature=0.7, n=2)\n  "[X]"\nfrom "m"',
    "keyword_in_string": '"please tell me where to go from here"',
    "keyword_in_comment": "x = 1  # this where from distribution is inert\nargmax\n  \"[A]\"",
    "identifier_superstring": "where_to_go = 1\nwhereabouts = 2\nfromage = 3",
    "nested_keywords": 'f(where=1, distribution=2)\nbeam(n=g(from_x=3))',
    "string_prefixes": 'r"\\d+" b"bytes" f"hi {name}" rb"\\x00" "plain"',
    "triple_quoted": 'argmax\n  """multi\n  line where from\n  prompt"""\nfrom "m"',
    "escapes": r'"a \" quote" "tab\there" "back\\slash"',
    "full_query": (
        "import re\n"
        "from helpers import clean\n\n"
        "argmax(temperature=0.0)\n"
        '    "Summarize, and say where it leads: {text}"\n'
        '    "[SUMMARY]"\n'
        '    where_to_go = pick(["from", "where"])\n'
        "from\n"
        '    "openai/gpt-4"\n'
        "where\n"
        '    len(SUMMARY) < 100 and STOPS_AT(SUMMARY, ".")\n'
        "distribution\n"
        '    SENTIMENT in ["pos", "neg"]\n'
    ),
}


# ---------------------------------------------------------------------------
# 3. TESTS  (plain functions named test_*; both the runner and pytest find them)
# ---------------------------------------------------------------------------

def test_corpus_satisfies_invariants():
    for name, src in CORPUS.items():
        problems = check_all(src)
        assert not problems, f"{name}: " + "; ".join(problems)


def test_empty_is_just_eof():
    toks = tokenize("")
    assert len(toks) == 1 and toks[0].ttype is K.EOF
    assert toks[0].start == 0 == toks[0].end


def test_depth0_keyword_resolves():
    assert keywords('argmax "[X]" from "m" where c distribution D') == \
        ["argmax", "from", "where", "distribution"]


def test_every_decoder_is_a_keyword_at_top_level():
    for d in DECODERS:
        assert keywords(d + " x") == [d], d


def test_keyword_inside_brackets_is_a_name():
    # depth > 0 -> the spelling must NOT be promoted to KEYWORD
    toks = significant("f(where, from, distribution)")
    names = [t.text for t in toks if t.ttype is K.NAME]
    assert "where" in names and "from" in names and "distribution" in names
    assert keywords("f(where, from, distribution)") == []


def test_keyword_inside_string_is_opaque():
    src = '"go from here to where you distribute"'
    toks = significant(src)
    assert [t.ttype for t in toks] == [
        K.STRING]   # one opaque token, no leakage
    assert keywords(src) == []


def test_keyword_inside_comment_is_inert():
    assert keywords("# from where distribution argmax\nbeam x") == ["beam"]


def test_identifier_superstring_not_split():
    # 'where_to_go' must be ONE name, not keyword 'where' + '_to_go'
    toks = significant("where_to_go fromage wherever")
    assert all(t.ttype is K.NAME for t in toks)
    assert [t.text for t in toks] == ["where_to_go", "fromage", "wherever"]


def test_in_is_not_a_clause_keyword():
    # distribution's `in` is found later by the leaf parser, not the lexer
    assert "in" not in CLAUSE_KEYWORDS
    assert keywords("distribution D in xs") == ["distribution"]


def test_both_from_occurrences_are_keywords():
    # prologue import-from AND the from-clause both surface; the PARSER (not the
    # lexer) decides which is a boundary. Lexer must mark both identically.
    src = "from a import b\nargmax\n  \"[X]\"\nfrom \"m\""
    assert keywords(src) == ["from", "argmax", "from"]


def test_string_prefixes_are_single_tokens():
    for s in ['r"x"', 'b"x"', 'f"x"', 'rb"x"', 'fr"x"', 'R"x"', 'BR"x"']:
        toks = significant(s)
        assert [t.ttype for t in toks] == [K.STRING], (s, toks)
        assert toks[0].text == s


def test_triple_quote_swallows_everything():
    src = '"""line one\nwhere from\nline three"""'
    toks = significant(src)
    assert [t.ttype for t in toks] == [K.STRING]
    assert toks[0].text == src
    assert keywords(src) == []


def test_nested_bracket_depths():
    toks = {t.text: t.depth for t in significant("a(b[c{d}])")
            if t.ttype is K.NAME}
    assert toks == {"a": 0, "b": 1, "c": 2, "d": 3}


# --- error cases: malformed input must raise LexError (never silently pass) ---

def _raises(src):
    try:
        tokenize(src)
    except LexError:
        return True
    return False


def test_unterminated_string(): assert _raises('"no close')
def test_newline_in_single_string(): assert _raises('"line\nbreak"')
def test_unterminated_triple(): assert _raises('"""no close')
def test_unmatched_close(): assert _raises("a )")
def test_mismatched_bracket(): assert _raises("( ]")
def test_unclosed_open(): assert _raises("f( x")


def test_well_formed_does_not_raise():
    for src in CORPUS.values():
        tokenize(src)   # must not raise


# --- from-clause disambiguation (the one keyword that needs more than depth) ---

def _from_indices(src):
    toks = tokenize(src)
    return toks, [i for i, t in enumerate(toks)
                  if t.ttype is K.KEYWORD and t.text == "from"]


def _is_import(src):
    """Classify each depth-0 `from` in src as import (True) or clause (False)."""
    toks, idxs = _from_indices(src)
    return [is_import_from(toks, i) for i in idxs]


def clause_from_index(src):
    """What the parser's prompt-termination scan will do: the first depth-0
    `from` that is NOT an import. Returns its token index, or None."""
    toks, idxs = _from_indices(src)
    for i in idxs:
        if not is_import_from(toks, i):
            return i
    return None


def test_plain_import_is_not_a_boundary():
    assert _is_import("from a import b") == [True]


def test_relative_import_is_not_a_boundary():
    assert _is_import("from . import b") == [True]


def test_parenthesized_import_is_not_a_boundary():
    assert _is_import("from a import (b,\n c)") == [True]


def test_from_clause_string_is_a_boundary():
    assert _is_import('from "gpt-4"\n') == [False]


def test_from_clause_call_is_a_boundary():
    assert _is_import("from get_model(x)\n") == [False]


def test_from_clause_before_where_on_one_line():
    # statement ends at the next clause keyword, so no spurious 'import' scan
    assert _is_import('from "m" where c') == [False]


def test_import_lookalike_identifier_is_not_import():
    # 'import_model' is one NAME, not the keyword 'import'
    assert _is_import("from import_model\n") == [False]


def test_string_import_inside_from_clause():
    # the word import only appears inside a string -> still a clause boundary
    assert _is_import('from get("import")\n') == [False]


def test_prompt_body_imports_then_real_from_clause():
    src = (
        "argmax\n"
        "    from collections import OrderedDict\n"
        "    from typing import List\n"
        '    "[X]"\n'
        'from "gpt-4"\n'
    )
    toks, idxs = _from_indices(src)
    # the two body imports are imports; the third from is the clause boundary
    assert [is_import_from(toks, i) for i in idxs] == [True, True, False]
    # and the parser-style scan lands on the model-spec from, not an import
    ci = clause_from_index(src)
    assert toks[ci + 1].ttype is K.STRING and toks[ci + 1].text == '"gpt-4"'  # type: ignore


def test_prologue_import_classified_as_import():
    assert _is_import("from helpers import clean\nargmax\n") == [True]


def test_is_import_from_rejects_wrong_token():
    toks = tokenize("argmax x")
    try:
        is_import_from(toks, 0)   # index 0 is 'argmax', not 'from'
    except ValueError:
        return
    assert False, "should have raised on a non-from token"


# --- diagnostics: positions and reserved-word collisions -------------------

def test_source_line_col():
    s = Source("ab\ncde\nf")
    assert s.line_col(0) == (1, 1)
    assert s.line_col(1) == (1, 2)
    assert s.line_col(3) == (2, 1)   # first char of line 2
    assert s.line_col(7) == (3, 1)


def test_lex_errors_are_positioned():
    # every lex error must render a line/col and a caret, not a raw offset
    for bad in ['"open', 'argmax\n  "x', "f( a", "a )"]:
        try:
            tokenize(bad)
        except LexError as e:
            msg = str(e)
            assert "line" in msg and "^" in msg, (bad, msg)
            assert e.start is not None
        else:
            assert False, f"{bad!r} should have raised"


def test_collision_flags_decoder_as_assignment():
    diags = check_reserved_collisions('beam = "greedy"')
    assert len(diags) == 1
    d = diags[0]
    assert d.severity == "error" and "reserved" in d.message and "beam" in d.message
    # points at `beam`, not the `=`
    assert (d.start, d.end) == (0, 4)


def test_collision_strict_tokenize_raises():
    try:
        tokenize('beam = "greedy"', strict=True)
    except LexError as e:
        assert "beam" in str(e) and "^" in str(e)
    else:
        assert False, "strict tokenize should raise on the collision"


def test_collision_nonstrict_is_silent():
    # default: lexes fine, no raise
    tokenize('beam = "greedy"')


def test_collision_binding_forms():
    for src in ("import beam", "def sample():", "class var: pass", "x as where"):
        assert check_reserved_collisions(src), src


def test_collision_operator_forms():
    for src in ("where: int = 0", "argmax += 1", "sample.temp = 1", "var := 3"):
        assert check_reserved_collisions(src), src


def test_no_false_collisions_on_valid_usage():
    valid = ["argmax(temperature=0)",     # decoder call
             'from "gpt-4"',              # from-clause
             "where len(X) < 9",          # where-clause
             "distribution D in xs",      # distribution-clause
             "obj.sample",                # `sample` is an attribute name
             "x.where = 1",               # `where` is an attribute name
             'argmax\n  "[X]"',           # decoder then prompt string
             ]
    for src in valid:
        assert check_reserved_collisions(src) == [], src


def test_corpus_has_no_false_collisions():
    for name, src in CORPUS.items():
        assert check_reserved_collisions(src) == [], name


def test_format_tokens_is_inspectable():
    out = format_tokens(tokenize('argmax "x" from "m"'))
    assert "KEYWORD" in out and "boundary" in out


def test_fuzz_invariants():
    rng = random.Random(1234)
    atoms = ['"s"', "name", "where", "from", "(", ")", "[", "]", "{", "}",
             "argmax", " ", "\n", "# c\n", "1+2", "in", "."]
    for _ in range(2000):
        src = "".join(rng.choice(atoms) for _ in range(rng.randint(0, 25)))
        try:
            problems = check_all(src)
            # strict path must not crash (may raise LexError)
            tokenize(src, strict=True)
        except LexError:
            continue  # rejecting malformed input / collisions is allowed
        assert not problems, f"input {src!r}: " + "; ".join(problems)


# ---------------------------------------------------------------------------
# dump: interactive token inspector  (python test_clauses.py dump '<query>')
# ---------------------------------------------------------------------------

def dump(src):
    print(f"source ({len(src)} chars):")
    print("  " + repr(src))
    print()
    try:
        toks = tokenize(src)
    except LexError as e:
        print(f"  LexError: {e}")
        return
    print(f"  {'idx':>3}  {'ttype':<8} {'depth':>5}  {'span':<11}  text")
    print(f"  {'-'*3}  {'-'*8} {'-'*5}  {'-'*11}  {'-'*30}")
    for i, t in enumerate(toks):
        if t.ttype is K.NEWLINE:
            text = "\\n"
        else:
            text = t.text if len(t.text) <= 38 else t.text[:35] + "..."
        mark = "  <--- boundary" if t.ttype is K.KEYWORD else ""
        print(f"  {i:>3}  {t.ttype.name:<8} {t.depth:>5}  "
              f"{t.start:>4}:{t.end:<6}  {text!r}{mark}")
    print()
    problems = check_all(src)
    print("  invariants: OK" if not problems else "  PROBLEMS:")
    for p in problems:
        print("    -", p)
    print("  boundary keywords:", keywords(src) or "(none)")
    collisions = check_reserved_collisions(src)
    if collisions:
        print("\n  reserved-word collisions:")
        for d in collisions:
            print("\n" + "\n".join("    " + ln for ln in d.render(src).splitlines()))


# ---------------------------------------------------------------------------
# Built-in runner (so it works with zero dependencies, pytest also works)
# ---------------------------------------------------------------------------

def _run_all():
    tests = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed.append((name, e))
    print(f"\n{passed}/{len(tests)} tests passed")
    for name, e in failed:
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
    return 0 if not failed else 1


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "dump":
        dump(sys.argv[2] if len(sys.argv) > 2 else "")
    else:
        sys.exit(_run_all())
