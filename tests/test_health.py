from sentinel.health import EXPLAIN_INSTRUCTIONS, compute, explain_prompt, parse_statement, report

STATEMENT = """
| Line item | FY2024 | FY2023 |
|---|---|---|
| Revenue | $510,000 | $450,000 |
| Cost of goods sold | (204,000) | (189,000) |
| Operating expenses | 230,000 | 215,000 |
| Interest expense | 12,000 | 11,000 |
| Net income | 48,000 | 31,000 |
| Cash and cash equivalents | 60,000 | 41,000 |
| Inventory | 35,000 | 30,000 |
| Total current assets | 140,000 | 118,000 |
| Total current liabilities | 90,000 | 95,000 |
| Total liabilities | 210,000 | 220,000 |
| Total shareholders' equity | 175,000 | 140,000 |
"""


def test_parse_statement_table():
    f = parse_statement(STATEMENT)
    assert f["revenue"] == 510_000 and f["revenue_prior"] == 450_000
    assert f["cogs"] == 204_000  # "(204,000)" is a cost, stored positive
    assert f["current_assets"] == 140_000 and f["current_liabilities"] == 90_000
    assert f["equity"] == 175_000 and f["cash"] == 60_000


def test_parse_ignores_lookalike_lines():
    f = parse_statement("Sales and marketing 40\nCash flow from operations 12\nNet sales 900 800")
    assert f == {"revenue": 900, "revenue_prior": 800}


def test_metrics_are_exact():
    m = {x.id: x for x in compute(parse_statement(STATEMENT))}
    assert m["revenue_growth"].display == "+13.3%" and m["revenue_growth"].rating == "good"
    assert m["gross_margin"].display == "60.0%"
    assert m["operating_margin"].display == "14.9%" and m["operating_margin"].rating == "watch"
    assert m["current_ratio"].display == "1.56x" and m["current_ratio"].rating == "good"
    assert m["quick_ratio"].display == "1.17x"
    assert m["debt_to_equity"].display == "1.20x" and m["debt_to_equity"].rating == "watch"
    assert m["interest_coverage"].display == "6.3x"


def test_report_handles_partial_input_and_negative_equity():
    r = report({"revenue": 100, "net_income": -5, "total_liabilities": 50, "equity": -10})
    ids = [m["id"] for m in r["metrics"]]
    assert ids == ["net_margin", "debt_to_equity"]
    assert all(m["rating"] == "risk" for m in r["metrics"]) and r["label"] == "At risk"
    assert r["metrics"][1]["value"] is None  # infinity is not valid JSON
    assert "Revenue (last year)" in r["missing"]
    assert report({})["score"] is None


def test_explain_prompt_contains_figures():
    p = explain_prompt(compute(parse_statement(STATEMENT)))
    assert "Strong figures: Revenue growth +13.3%" in p
    assert "Weak figures: Operating margin 14.9%" in p and "investment" in p
    assert p.startswith(EXPLAIN_INSTRUCTIONS)  # fixed prefix first, so llama.cpp can reuse its KV cache
