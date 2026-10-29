import json

from fastapi.testclient import TestClient

import app


class FakeLlm:
    def __init__(self, pieces):
        self.pieces, self.messages = pieces, None

    def create_chat_completion(self, messages, **kwargs):
        self.messages = messages
        return ({"choices": [{"delta": {"content": p}}]} for p in self.pieces)


def events(body):
    out = []
    for block in body.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_chat_streams_and_audits():
    llm = FakeLlm(["Growth is 13.3%. ", "I recommend buying ", "the stock."])
    client = TestClient(app.create_app(llm, "fake.gguf"))
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Growth?"}]})
    ev = events(r.text)
    assert [n for n, _ in ev] == ["token", "token", "token", "done"]
    assert "buying" not in ev[2][1]["text"]  # redacted while streaming, not only at the end
    done = ev[-1][1]
    assert done["status"] == "REJECTED" and done["tokens"] == 3
    assert llm.messages[0]["role"] == "system" and llm.messages[-1]["content"] == "Growth?"


def test_info():
    assert TestClient(app.create_app(FakeLlm([]), "fake.gguf")).get("/api/info").json()["model"] == "fake.gguf"


def test_health_check_endpoints():
    llm = FakeLlm(["**Summary:** Sales grew 13.3%."])
    client = TestClient(app.create_app(llm, "fake.gguf"))
    fields = client.post("/api/parse", json={"text": "Revenue 510 450\nNet income 48"}).json()["fields"]
    assert fields == {"revenue": 510, "revenue_prior": 450, "net_income": 48}
    r = client.post("/api/health", json={"fields": fields}).json()
    assert [m["id"] for m in r["metrics"]] == ["revenue_growth", "net_margin"] and r["score"] == 100
    done = events(client.post("/api/explain", json={"fields": fields}).text)[-1][1]
    assert done["status"] == "APPROVED"
    assert "+13.3%" in llm.messages[-1]["content"]  # the model is given the computed figures
    assert client.post("/api/explain", json={"fields": {}}).status_code == 400


def test_explain_repeats_come_from_cache():
    llm = FakeLlm(["**Summary:** Fine."])
    client = TestClient(app.create_app(llm, "fake.gguf"))
    fields = {"revenue": 510000, "revenue_prior": 450000, "net_income": 48000}
    first = events(client.post("/api/explain", json={"fields": fields}).text)[-1][1]
    llm.pieces = ["something else"]
    again = events(client.post("/api/explain", json={"fields": fields}).text)[-1][1]
    assert again["text"] == first["text"] == "**Summary:** Fine."
