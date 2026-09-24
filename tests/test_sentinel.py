from sentinel.data import from_finqa, from_llama2, table_to_markdown
from sentinel.governance import audit, unitless_quantities
from scripts.evaluate import is_correct

# The actual answer from the notebook's governance cell, which the old regex APPROVED.
NOTEBOOK_ANSWER = "So, the growth rate is approximately 13.33%. The company has grown from $450M to $510M."


def test_trade_advice_is_caught_and_redacted():
    r = audit("Revenue grew 13.3%. I recommend buying the stock; strong buy rating.")
    assert r.status == "REJECTED"
    assert "buying the stock" not in r.text


def test_accounting_vocabulary_is_not_trade_advice():
    # The old substring check flagged all of these (SELL in SG&A, HOLD in shareholders, BUY in buyback).
    text = ("Selling, general and administrative expenses rose 4%. Shareholders approved a $2B buyback; "
            "the threshold for household sales was unchanged.")
    assert audit(text).status == "APPROVED"


def test_units():
    assert audit(NOTEBOOK_ANSWER).status == "APPROVED"
    assert unitless_quantities("YoY growth = (510M / 450M) - 1 = 1.133") == ["1.133"]
    assert unitless_quantities("Item 1A of the 2023 10-K lists 3 risks.") == []


def test_llama2_parsing_puts_question_in_user_turn():
    msgs = from_llama2("<s>[INST] What is YTM? [/INST] Yield to maturity is the total return. </s>")
    assert msgs[1]["content"] == "What is YTM?"
    assert msgs[2]["content"] == "Yield to maturity is the total return."
    assert from_llama2("no instruction markers") is None


def test_finqa_formatting():
    entry = {
        "pre_text": ["Revenue figures follow."], "post_text": ["All in millions."],
        "table": [["", "2024", "2025"], ["revenue", "$450", "$510"]],
        "qa": {"question": "What was the growth?", "answer": "13.3%",
               "steps": [{"op": "minus2-1", "arg1": "510", "arg2": "450", "res": "60"},
                         {"op": "divide2-2", "arg1": "#0", "arg2": "450", "res": "0.13333"}]},
    }
    msgs = from_finqa(entry)
    assert "| revenue | $450 | $510 |" in msgs[1]["content"]
    assert msgs[2]["content"] == "Step 1: minus(510, 450) = 60\nStep 2: divide(#0, 450) = 0.1333\nAnswer: 13.3%"
    assert table_to_markdown([]) == ""


def test_eval_scoring():
    assert is_correct("Answer: 13.3%", 0.13333)
    assert is_correct("Answer: 0.1333", "13.3%")
    assert is_correct("growth is $1,234.5 million\nAnswer: $1,234.5", 1234.5)
    assert not is_correct("Answer: 1.133", 0.13333)
    assert not is_correct("I cannot tell.", 0.13333)
