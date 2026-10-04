"""Run the real OpenAI-compatible adapter against a local fake server, so the
response parsing (tool calls, usage) is tested without any real model."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from finbench.model import OpenAICompatibleModel
from finbench.tools import TOOLS

RESPONSE = {
    "id": "x", "object": "chat.completion", "created": 0, "model": "fake",
    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "calculate",
                                     "arguments": '{"operation": "add", "a": 1, "b": 2}'}}]}}],
    "usage": {"prompt_tokens": 321, "completion_tokens": 12, "total_tokens": 333},
}


@pytest.fixture
def server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, json.loads(body)))
            out = json.dumps(RESPONSE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", received
    httpd.shutdown()


def test_adapter_parses_tool_calls_and_usage(server):
    url, received = server
    model = OpenAICompatibleModel("qwen-test", url)
    turn = model.complete([{"role": "user", "content": "hi"}], TOOLS)
    assert [(c.id, c.name) for c in turn.tool_calls] == [("call_1", "calculate")]
    assert json.loads(turn.tool_calls[0].arguments) == {"operation": "add", "a": 1, "b": 2}
    assert (turn.prompt_tokens, turn.completion_tokens) == (321, 12)
    assert turn.latency_s > 0
    path, body = received[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "qwen-test" and body["temperature"] == 0.0
    assert [t["function"]["name"] for t in body["tools"]] == ["calculate", "table_aggregate", "final_answer"]
