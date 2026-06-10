"""
Test suite for the LMQL parser.

Run the assertions with pytest:
    pytest test_parser.py

Point the file-driven test at your own query file (it is skipped otherwise):
    LMQL_SRC=path/to/query.lmql pytest test_parser.py -k file

Or inspect any query file directly, without pytest:
    python test_parser.py path/to/query.lmql
"""


import ast
import os
import sys
from dataclasses import dataclass
from lmql_lexer import LexError, CLAUSE_KEYWORDS, DECODERS
from lmql_parser import LMQLQuery, LMQLParser, PyExprParser, parse, Span, Clause, ParseError
import pytest

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# small assertion helpers
# ---------------------------------------------------------------------------

def is_name(node, ident):
    return isinstance(node, ast.Name) and node.id == ident


def is_const(node, value):
    return isinstance(node, ast.Constant) and node.value == value


def span_text(q: LMQLQuery, span):
    return None if span is None else q.source.text[span.start:span.end]


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------

def test_single_line_all_clauses():
    q = parse('argmax "Hello [WHO]" from "openai/gpt" where len(WHO) < 10')
    assert is_name(q.decoder_ast, "argmax")
    assert len(q.prompt_ast) == 1
    assert is_const(q.from_ast, "openai/gpt")
    assert isinstance(q.where_ir, ast.Compare)
    assert q.distribution_clause is None


def test_full_multiline_with_prologue_and_distribution():
    q = parse("import re\n"
              "def helper(x):\n"
              "    return x.strip()\n"
              "\n"
              "argmax\n"
              '    "Greet [WHO]"\n'
              '    "[GREETING]"\n'
              "from\n"
              '    "openai/text-davinci-003"\n'
              "where\n"
              "    len(WHO) < 10 and\n"
              '    STOPS_AT(GREETING, ".")\n'
              "distribution\n"
              '    SENTIMENT in [" positive", " negative"]\n'
              )
    prologue_text = span_text(q, q.prologue)
    assert prologue_text is not None and "import re" in prologue_text
    assert "def helper" in prologue_text
    assert is_name(q.decoder_ast, "argmax")
    assert len(q.prompt_ast) == 2
    assert is_const(q.from_ast, "openai/text-davinci-003")
    assert isinstance(q.where_ir, ast.BoolOp)            # the multi-line `and`
    dc = q.distribution_clause
    assert dc is not None and dc.variable_name == "SENTIMENT"
    assert isinstance(dc.values, ast.List) and len(dc.values.elts) == 2


def test_no_decoder_defaults_to_dynamic():
    q = parse('"just a prompt [X]"\nwhere len(X) < 3')
    assert q.decoder is None                              # no decoder span
    assert is_name(q.decoder_ast, "__dynamic__")         # but a default ast
    assert len(q.prompt_ast) == 1
    # default model sentinel
    assert is_const(q.from_ast, "<dynamic>")
    assert isinstance(q.where_ir, ast.Compare)


def test_from_import_in_prologue_is_not_the_from_clause():
    q = parse("from lmql.lib import foo\n"
              "argmax\n"
              '    "hi [X]"\n'
              'from "openai/gpt"\n'
              )
    prologue_text = span_text(q, q.prologue)
    assert prologue_text is not None and "from lmql.lib import foo" in prologue_text
    assert is_const(q.from_ast, "openai/gpt")            # the real from-clause


def test_decoder_with_arguments_no_double_escape_needed():
    # raw string: the LMQL source literally contains  stop=["\n"]
    q = parse(
        r'sample(n=2, temperature=0.7, stop=["\n"]) "go [X]" where len(X) > 1')
    call = q.decoder_ast
    assert isinstance(call, ast.Call) and is_name(call.func, "sample")
    kwargs = {k.arg: k.value for k in call.keywords}
    assert is_const(kwargs["n"], 2)
    # the escape is resolved natively by ast.parse -> a real newline, proving
    # the original double_escape/untokenize machinery is unnecessary here.
    stop = kwargs["stop"]
    assert isinstance(stop, ast.List) and is_const(stop.elts[0], "\n")


def test_nested_decoder_arguments_match_via_depth():
    # the matching-close helper must walk past nested ()/[] inside the args
    q = parse('sample(n=2, stop=["a", ("b",)]) "go [X]"')
    assert isinstance(q.decoder_ast, ast.Call)
    assert span_text(q, q.decoder) == 'sample(n=2, stop=["a", ("b",)])'


def test_decoder_keyword_as_attribute_is_not_a_boundary():
    # `reg.argmax` is attribute access, not the decoder; `greedy` is the decoder
    q = parse('m = reg.argmax\ngreedy "go [X]"')
    assert is_name(q.decoder_ast, "greedy")
    prologue_text = span_text(q, q.prologue)
    assert prologue_text is not None and "reg.argmax" in prologue_text


def test_greedy_distribution_only():
    q = parse("greedy\n"
              '    "Classify: [LABEL]"\n'
              "distribution\n"
              '    LABEL in ["yes", "no"]\n'
              )
    assert is_name(q.decoder_ast, "greedy")
    assert q.from_clause is None and q.where is None
    dc = q.distribution_clause
    assert dc is not None and dc.variable_name == "LABEL"
    assert [e.value for e in dc.values.elts] == ["yes", "no"]  # type: ignore


def test_keywords_inside_brackets_are_not_boundaries():
    q = parse('argmax "pick [X]" from get(["where", "from"]) where X in ["a"]')
    assert isinstance(q.from_ast, ast.Call) and is_name(q.from_ast.func, "get")
    from_clause_text = span_text(q, q.from_clause)
    assert from_clause_text is not None and "where" in from_clause_text
    assert isinstance(q.where_ir, ast.Compare)


def test_empty_prompt_is_allowed():
    q = parse("argmax")
    assert is_name(q.decoder_ast, "argmax")
    assert q.prompt_ast == []


# ---------------------------------------------------------------------------
# the where-IR seam (generic parser)
# ---------------------------------------------------------------------------

@dataclass
class Constraint:
    raw: str


class ConstraintParser(LMQLParser[Constraint]):
    """A parser that binds the where-IR to a custom type."""

    def parse_where(self, span: Span) -> Constraint:
        return Constraint(span.text(self.src).strip())


def test_custom_where_ir_via_subclass():
    q = ConstraintParser('argmax "x [Y]" where len(Y) < 3').parse()
    assert isinstance(q.where_ir, Constraint)
    assert q.where_ir.raw == "len(Y) < 3"


def test_pyexprparser_is_the_default_where_binding():
    # parse() uses PyExprParser, so where_ir is a plain ast.expr
    assert isinstance(PyExprParser('argmax "x [Y]" where a and b').parse().where_ir,
                      ast.BoolOp)


def test_base_parser_handles_queries_without_where():
    # base LMQLParser is abstract only for the where clause
    q = LMQLParser('argmax "hi [X]"').parse()
    assert is_name(q.decoder_ast, "argmax")
    assert q.where_ir is None


def test_base_parser_is_abstract_for_where():
    with pytest.raises(NotImplementedError):
        LMQLParser('argmax "hi [X]" where a').parse()


# ---------------------------------------------------------------------------
# enum / lexer invariant
# ---------------------------------------------------------------------------

def test_clause_enum_values_track_lexer_keywords():
    trailing = {c.value for c in (
        Clause.FROM, Clause.WHERE, Clause.DISTRIBUTION)}
    assert trailing == set(CLAUSE_KEYWORDS) - set(DECODERS)


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------

def test_error_out_of_order_where_before_from():
    with pytest.raises(ParseError) as ei:
        parse('argmax "x [Y]" where len(Y) < 3 from "m"')
    assert "out of order" in str(ei.value)


def test_error_duplicate_where():
    with pytest.raises(ParseError) as ei:
        parse('argmax "x [Y]" where a where b')
    assert "duplicate" in str(ei.value)


def test_error_malformed_distribution():
    with pytest.raises(ParseError) as ei:
        parse('argmax "x [Y]" distribution len(Y) < 3')
    assert "VAR in" in str(ei.value)


def test_error_second_decoder():
    with pytest.raises(ParseError) as ei:
        parse('argmax "x [Y]" greedy "z [W]"')
    assert "second decoder" in str(ei.value)


def test_error_reserved_word_collision_is_a_lexerror():
    # `beam = 1` shadows a reserved decoder keyword; strict lexing rejects it
    # while constructing the parser (tokenize(strict=True)).
    with pytest.raises(LexError) as ei:
        parse('beam = 1\nargmax "x [Y]"')
    assert "reserved" in str(ei.value)


# ---------------------------------------------------------------------------
# file-driven test: point LMQL_SRC at your own query file
# ---------------------------------------------------------------------------

def test_parse_lmql_file():
    path = os.environ.get("LMQL_SRC")
    if not path:
        pytest.skip("set LMQL_SRC=<path to .lmql file> to run this test")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    # raises ParseError/LexError on failure
    q = parse(src)
    assert q.prompt is not None and q.prompt_ast is not None
    print("\n" + _describe(q, src))


# ---------------------------------------------------------------------------
# pretty-printer + entry point
# ---------------------------------------------------------------------------

def _describe(q: LMQLQuery, src: str) -> str:
    def show(span):
        return None if span is None else repr(q.source.text[span.start:span.end])

    def dump(node):
        return ast.dump(node) if isinstance(node, ast.AST) else repr(node)

    lines = [
        f"prologue     : {show(q.prologue)}",
        f"decoder      : {show(q.decoder)} -> {dump(q.decoder_ast)}",
        f"prompt       : {show(q.prompt)} -> {len(q.prompt_ast)} stmt(s)",
        f"from         : {show(q.from_clause)} -> {dump(q.from_ast)}",
        f"where        : {show(q.where)} -> {dump(q.where_ir)}",
    ]
    if q.distribution_clause is not None:
        dc = q.distribution_clause
        lines.append(f"distribution : {show(q.distribution)} -> "
                     f"var={dc.variable_name!r} values={ast.dump(dc.values)}")
    return "\n".join(lines)


def _describe_file(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    try:
        q = parse(src)
    except (ParseError, LexError) as e:
        print(str(e), file=sys.stderr)
        return 1
    print(_describe(q, src))
    return 0


def _main(argv) -> int:
    if len(argv) == 1:
        # no argument: run the whole suite via pytest on this file
        return pytest.main([__file__, "-v"])
    if len(argv) == 2:
        return _describe_file(argv[1])
    print("usage: python test_parser.py [file.lmql]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
