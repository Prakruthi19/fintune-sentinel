"""CI eval gate: recorded model replies must keep their verdicts.

If one of these fails after a change to agent, tools or scoring, the change
moved published results. Either fix the regression or, if the change is
intended, regenerate the fixture and say why in the commit message."""
import json
from pathlib import Path

import pytest
from conftest import GoldModel

from finbench import scoring
from finbench.replay import ReplayModel, check, load_fixture, main, make_fixture, replay_item
from finbench.run import run

REPLAYS = sorted((Path(__file__).parent / "fixtures" / "replays").glob("*.jsonl"))
SAMPLE = Path(__file__).parent / "fixtures" / "finqa_sample.json"


@pytest.mark.parametrize("path", REPLAYS, ids=lambda p: p.name)
def test_recorded_replies_keep_their_verdicts(path):
    assert check(load_fixture(path)) == []


def test_synthetic_fixture_covers_every_error_category():
    cats = {i["expected"]["category"] for i in load_fixture(REPLAYS[0])}
    assert cats == set(scoring.CATEGORIES)


def test_gate_catches_a_scoring_change(monkeypatch):
    items = load_fixture(REPLAYS[0])
    monkeypatch.setattr(scoring, "REL_TOL", 0.0)  # e.g. someone tightens the tolerance
    changed = {d["id"] for d in check(items)}
    assert "ADI/2011/page_61.pdf-2" in changed  # 0.0986 vs 0.09864 is no longer accepted


def test_replay_model_raises_the_recorded_error():
    m = ReplayModel("x", [], "NotFoundError: model not found")
    with pytest.raises(RuntimeError, match="not found"):
        m.complete([], [])


def test_real_run_round_trips_through_make_fixture(examples, tmp_path):
    exs = list(examples.values())
    out = tmp_path / "oracle"
    run(GoldModel(exs), exs, exs, out, n=len(exs), fewshot_k=0, seed=0, max_turns=8)
    rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert all(r["model_turns"] and r["model_turns"][-1]["tool_calls"][0]["name"] == "final_answer"
               for r in rows)
    items = make_fixture(out, json.loads(SAMPLE.read_text()), source="test")
    assert len(items) == len(exs) and check(items) == []
    assert replay_item(items[0])["category"] == "correct"


def test_make_fixture_refuses_rows_without_recordings(examples, tmp_path):
    out = tmp_path / "old"
    out.mkdir()
    (out / "config.json").write_text(json.dumps({"model": "m"}))
    (out / "predictions.jsonl").write_text(json.dumps({"id": "ETR/2016/page_23.pdf-2", "category": "correct"}) + "\n")
    with pytest.raises(SystemExit, match="no model_turns"):
        make_fixture(out, json.loads(SAMPLE.read_text()), source="test")


def test_cli_check_exit_codes(tmp_path, capsys, monkeypatch):
    with pytest.raises(SystemExit) as ok:
        main(["check", str(REPLAYS[0])])
    assert ok.value.code == 0 and "replay gate: ok" in capsys.readouterr().out
    monkeypatch.setattr(scoring, "REL_TOL", 0.0)
    with pytest.raises(SystemExit) as bad:
        main(["check", str(REPLAYS[0])])
    assert bad.value.code == 1 and "FAILED" in capsys.readouterr().out


def test_cli_make_fixture(examples, tmp_path):
    exs = list(examples.values())
    run(GoldModel(exs), exs, exs, tmp_path / "r", n=2, fewshot_k=0, seed=0, max_turns=8)
    data = tmp_path / "data"
    data.mkdir()
    (data / "test.json").write_text(SAMPLE.read_text())
    fixture = tmp_path / "f.jsonl"
    main(["make-fixture", str(tmp_path / "r"), "--data-dir", str(data), "--out", str(fixture)])
    assert len(load_fixture(fixture)) == 2


def test_synthetic_fixture_matches_its_generator():
    from scripts.make_synthetic_replays import OUT, build
    assert load_fixture(OUT) == json.loads(json.dumps(build()))
