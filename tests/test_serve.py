import pytest
from fastapi.testclient import TestClient

from pslm.config import ROOT
from pslm.data.labels import read_jsonl
from pslm.serve.api import create_app
from pslm.serve.backends import BatchResult, Extraction, ReplayBackend, build_backend, to_extraction

CASSETTE = ROOT / "data" / "cassettes" / "qwen2.5-1.5b-lora.json"
GOOD = '{"category": "apparel", "brand": "Acme", "size": "XL", "material": "cotton"}'


class EchoBackend:
    """Returns a fixed good answer, except for titles containing 'bad'."""

    name, model = "echo", "echo-model"

    def __init__(self):
        self.calls: list[list[str]] = []

    def extract(self, titles):
        self.calls.append(list(titles))
        ex = [to_extraction(t, "not json" if "bad" in t else GOOD) for t in titles]
        return BatchResult(ex, self.model, 12.5)


@pytest.fixture
def echo():
    b = EchoBackend()
    with TestClient(create_app(b)) as c:
        yield c, b


def test_health_reports_backend(echo):
    c, _ = echo
    assert c.get("/health").json() == {"status": "ok", "backend": "echo", "model": "echo-model"}


def test_extract_keeps_order_batches_once_and_isolates_failures(echo):
    c, b = echo
    r = c.post("/extract", json={"titles": ["Acme XL tee", "a bad one", "  Acme   socks "]})
    assert r.status_code == 200
    body = r.json()
    assert [x["title"] for x in body["results"]] == [
        "Acme XL tee",
        "a bad one",
        "Acme socks",
    ]  # whitespace squashed
    assert body["results"][0]["attributes"]["brand"] == "Acme"
    assert body["results"][1] == {"title": "a bad one", "attributes": None, "error": "not_json"}
    assert b.calls == [["Acme XL tee", "a bad one", "Acme socks"]]  # one backend call for the whole batch
    m = c.get("/metrics").json()
    assert m["requests"] == 1 and m["titles"] == 3 and m["errors"] == {"not_json": 1}


@pytest.mark.parametrize(
    "payload", [{"titles": []}, {"titles": ["  "]}, {"titles": ["x" * 501]}, {"title": "x"}]
)
def test_extract_rejects_bad_requests(echo, payload):
    c, _ = echo
    assert c.post("/extract", json=payload).status_code == 422


def test_extract_rejects_oversized_batches(echo):
    c, _ = echo
    assert c.post("/extract", json={"titles": ["t"] * 257}).status_code == 422


def test_replay_backend_serves_recorded_outputs_and_flags_unknown_titles():
    b = ReplayBackend(CASSETTE)
    out = b.extract(["definitely not a recorded title"])
    assert out.extractions[0] == Extraction("definitely not a recorded title", None, "not_recorded")


def test_served_predictions_equal_evaluated_predictions():
    """The API path (prompt lookup, parse, schema) reproduces the evaluated predictions title for title."""
    preds = read_jsonl(ROOT / "data" / "preds" / "test" / "qwen2.5-1.5b-lora.jsonl")
    with TestClient(create_app(ReplayBackend(CASSETTE))) as c:
        body = c.post("/extract", json={"titles": [p["title"] for p in preds]}).json()
    assert len(body["results"]) == len(preds) == 150
    for p, r in zip(preds, body["results"], strict=True):
        assert r["attributes"] == p["pred"], p["title"]
        assert r["error"] == p["error"]


def test_build_backend_validates():
    assert build_backend("replay").name == "replay"
    with pytest.raises(ValueError):
        build_backend("vllm")  # needs a model
    with pytest.raises(ValueError):
        build_backend("tgi", "x")
