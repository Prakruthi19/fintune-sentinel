import pytest

from finbench.scoring import answers_match, classify, is_scale_error, text_answer, to_float

PROG = "subtract(153.7, 139.9), divide(#0, 139.9)"
GOLD = 0.09864


@pytest.mark.parametrize("pred,ok", [
    (0.09864, True), (0.0986, True), ("0.0986", True), ("9.86%", True),
    (0.099, True),     # within 0.5% relative
    (0.10, False), (9.864, False), ("about ten percent", False), (None, False), (True, False),
])
def test_answers_match_numeric(pred, ok):
    assert answers_match(pred, GOLD) is ok


def test_yes_no_answers():
    assert answers_match("Yes", "yes") and not answers_match(1, "yes") and not answers_match("no", "yes")


def test_scale_error_detects_percent_and_thousands():
    assert is_scale_error(9.864, GOLD) and is_scale_error(98640, 98.64)
    assert not is_scale_error(0.2, GOLD) and not is_scale_error("yes", "no")
    assert to_float("$1,200") == 1200.0 and to_float("14%") == 0.14


def calc(op, a, b):
    return {"operation": op, "a": a, "b": b}


def test_correct():
    v = classify(PROG, GOLD, 0.09864, [calc("subtract", 153.7, 139.9), calc("divide", 13.8, 139.9)], True, [13.8])
    assert v.correct and v.category == "correct"


def test_no_tool_use_and_no_final():
    assert classify(PROG, GOLD, None, [], False, []).category == "no_tool_use"
    assert classify(PROG, GOLD, None, [calc("subtract", 153.7, 139.9)], True, [13.8]).category == "no_final_answer"


def test_scale_error_category():
    v = classify(PROG, GOLD, 9.864, [calc("subtract", 153.7, 139.9), calc("divide", 13.8, 139.9)], True, [13.8])
    assert v.category == "scale_error"


def test_wrong_numbers_when_gold_operand_missing():
    v = classify(PROG, GOLD, 0.5, [calc("subtract", 153.7, 100.0), calc("divide", 53.7, 100.0)], True, [53.7])
    assert v.category == "wrong_numbers"


def test_reused_intermediate_result_is_not_counted_as_a_copied_number():
    # 13.8 is the previous call's result, not a number taken from the filing
    v = classify(PROG, GOLD, 0.0986, [calc("subtract", 153.7, 139.9), calc("divide", 13.8, 139.9)],
                 True, [13.8])
    assert v.correct
    v = classify(PROG, GOLD, 0.09, [calc("subtract", 153.7, 139.9), calc("divide", 13.8, 139.9)],
                 True, [13.8])
    assert v.category == "wrong_arrangement"


def test_unexplained_number_is_wrong_numbers():
    v = classify(PROG, GOLD, 0.5, [calc("subtract", 153.7, 139.9), calc("divide", 13.8, 27.6)], True, [13.8])
    assert v.category == "wrong_numbers"


def test_free_constants_like_100_do_not_count_as_wrong_numbers():
    v = classify(PROG, GOLD, 0.5, [calc("subtract", 153.7, 139.9), calc("divide", 13.8, 139.9),
                                   calc("multiply", 0.0986, 100)], True, [13.8, 0.0986])
    assert v.category == "wrong_operations"


def test_wrong_operations_with_right_numbers():
    v = classify(PROG, GOLD, 13.8 * 139.9, [calc("subtract", 153.7, 139.9), calc("multiply", 13.8, 139.9)],
                 True, [13.8])
    assert v.category == "wrong_operations"


def test_wrong_arrangement_divided_by_wrong_year():
    v = classify(PROG, GOLD, 0.0898, [calc("subtract", 153.7, 139.9), calc("divide", 13.8, 153.7)], True, [13.8])
    assert v.category == "wrong_arrangement"


def test_server_failure_is_model_error_not_a_model_mistake():
    assert classify(PROG, GOLD, None, [], False, [], model_error=True).category == "model_error"


@pytest.mark.parametrize("text,value", [
    # real replies from the first llama3.2:3b run
    ("The total number of shares Mr. Oppenheimer would have if his RSUs vest is 599,768.", 599768.0),
    ("The percent of net interest revenue ... in 2009 is approximately 28.24%.", 0.2824),
    ("The ROI of an investment in State Street Corporation from 2011 to 2012 is 13.0%.", 0.13),
    ("Yes, it was higher.", "yes"),
    ("I could not find it.", None),
    (None, None),
])
def test_text_answer_takes_last_number(text, value):
    got = text_answer(text)
    assert got == value or (isinstance(value, float) and abs(got - value) < 1e-9)
