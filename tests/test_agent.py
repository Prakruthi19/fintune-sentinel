import json

from conftest import ScriptedModel, call, text

from finbench.agent import solve
from finbench.fewshot import gold_conversation, pick_fewshot
from finbench.tools import TOOLS

ONE_STEP = "ETR/2016/page_23.pdf-2"   # subtract(5829, 5735) = 94


def test_happy_path_tool_then_final_answer(examples):
    m = ScriptedModel([call("calculate", {"operation": "subtract", "a": 5829, "b": 5735}),
                       call("final_answer", {"answer": 94}, 1)])
    t = solve(m, examples[ONE_STEP])
    assert t.stop_reason == "final_answer" and t.final_answer == 94
    tool_msgs = [x for x in m.seen[1] if x["role"] == "tool"]
    assert json.loads(tool_msgs[0]["content"]) == {"result": 94.0}
    assert t.latency_s == 1.0 and t.prompt_tokens == 200


def test_text_only_answer_stops_loop(examples):
    t = solve(ScriptedModel([text("The change is 94.")]), examples[ONE_STEP])
    assert t.stop_reason == "text_only" and t.final_answer is None and not t.tool_calls


def test_invalid_call_is_returned_to_model_and_it_can_recover(examples):
    m = ScriptedModel([call("calculate", "{bad json"),
                       call("calculate", {"operation": "subtract", "a": 5829, "b": 5735}, 1),
                       call("final_answer", {"answer": 94}, 2)])
    t = solve(m, examples[ONE_STEP])
    assert t.final_answer == 94 and t.invalid_calls == 1
    assert "error" in json.loads([x for x in m.seen[1] if x["role"] == "tool"][0]["content"])


def test_step_budget_and_model_errors(examples):
    loop = ScriptedModel([call("calculate", {"operation": "add", "a": 1, "b": 1}, i) for i in range(3)])
    assert solve(loop, examples[ONE_STEP], max_turns=3).stop_reason == "max_turns"
    t = solve(ScriptedModel([]), examples[ONE_STEP])
    assert t.stop_reason == "model_error" and "script exhausted" in t.error


def test_tool_schemas_are_valid_openai_format():
    names = [t["function"]["name"] for t in TOOLS]
    assert names == ["calculate", "table_aggregate", "final_answer"]
    for t in TOOLS:
        assert t["type"] == "function" and t["function"]["parameters"]["type"] == "object"


def test_gold_conversation_alternates_and_ends_with_final_answer(examples):
    for ex in examples.values():
        conv = gold_conversation(ex)
        assert conv[0]["role"] == "user"
        roles = [m["role"] for m in conv[1:]]
        assert roles == ["assistant", "tool"] * (len(roles) // 2)
        last = json.loads(conv[-2]["tool_calls"][0]["function"]["arguments"])
        assert last["answer"] == ex.gold or abs(last["answer"] - ex.gold) < 1e-3


def test_fewshot_comes_from_given_split_and_sits_before_the_question(examples):
    exs = list(examples.values())
    msgs, ids = pick_fewshot(exs, k=2, seed=1)
    assert len(ids) == 2 and set(ids) <= set(examples)
    m = ScriptedModel([text("x")])
    solve(m, examples[ONE_STEP], fewshot=msgs)
    sent = m.seen[0]
    assert sent[0]["role"] == "system" and sent[-1]["role"] == "user"
    assert "5829" in sent[-1]["content"] or "net revenue" in sent[-1]["content"]


