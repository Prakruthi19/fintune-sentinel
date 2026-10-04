import json
import math

import pytest

from finbench.program import literal_operands, operations, parse_number, parse_program
from finbench.scoring import answers_match
from finbench.tools import ToolRuntime, cell_to_number, execute_gold_program


@pytest.mark.parametrize("text,value", [
    ("5829", 5829.0), ("1,234.5", 1234.5), ("const_100", 100.0), ("const_m1", -1.0),
    ("8.75%", 0.0875),  # FinQA programs use percent literals as decimals
])
def test_parse_number(text, value):
    assert math.isclose(parse_number(text), value)


def test_parse_program_steps_and_refs():
    steps = parse_program("subtract(153.7, 139.9), divide(#0, 139.9)")
    assert [s.op for s in steps] == ["subtract", "divide"]
    assert steps[1].args == ("#0", "139.9")


def test_literal_operands_skip_refs_constants_and_table_ops():
    assert literal_operands("subtract(318.46, const_100), divide(#0, const_100)") == [318.46]
    assert literal_operands("table_average(net change, none)") == []
    assert operations("add(1, 2), divide(#0, 3)") == ["add", "divide"]


@pytest.mark.parametrize("cell,value", [
    ("$ 5735", 5735.0), ("-603 ( 603 )", -603.0), ("$ -61.1 ( 61.1 )", -61.1),
    ("4.70% ( 4.70 % )", 0.047), ("( 123 )", -123.0), ("1,200", 1200.0),
    ("-", None), ("n/a", None), ("", None),
])
def test_table_cell_formats_seen_in_finqa(cell, value):
    got = cell_to_number(cell)
    assert got == value or (value is not None and math.isclose(got, value))


def test_every_fixture_gold_program_reproduces_finqa_answer(examples):
    for ex in examples.values():
        assert answers_match(execute_gold_program(ex), ex.gold), ex.id


def test_runtime_reports_bad_calls_instead_of_crashing(examples):
    rt = ToolRuntime(next(iter(examples.values())))
    assert rt.execute("calculate", "{not json").error.startswith("JSONDecodeError")
    assert "unknown tool" in rt.execute("lookup", "{}").error
    assert "division by zero" in rt.execute("calculate", {"operation": "divide", "a": 1, "b": 0}).error
    assert "KeyError" in rt.execute("calculate", {"operation": "add", "a": 1}).error
    assert "unknown operation" in rt.execute("calculate", {"operation": "pow", "a": 1, "b": 2}).error
    ok = rt.execute("calculate", json.dumps({"operation": "add", "a": "2", "b": 3}))
    assert ok.error is None and ok.result == 5.0
    assert len(rt.calls) == 6


def test_table_aggregate_on_real_row(examples):
    ex = examples["MRO/2011/page_108.pdf-1"]
    rec = ToolRuntime(ex).execute("table_aggregate",
                                  {"row_label": "  Net Change for the year ", "operation": "average"})
    assert rec.error is None and answers_match(rec.result, ex.gold)
