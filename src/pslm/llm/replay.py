"""Deterministic LLM doubles for tests and CI: no API key, no network, no cost.

* ScriptedLLM: returns fixed replies in order (unit tests).
* ReplayLLM: a "cassette". In record mode it wraps a real LLM and saves each response, keyed by a hash of the
  full request (system prompt + title + schema). In replay mode it serves the saved responses back.
  Editing the prompt or the guidelines changes every key, so CI fails loudly ("cassette miss") instead of
  grading stale outputs.
"""

import hashlib
import json
from pathlib import Path

from pslm.llm.base import LLMClient, LLMResponse, Usage


class ScriptedLLM:
    def __init__(self, replies: list[str], model: str = "scripted", stop_reason: str = "end_turn"):
        self.model = model
        self.replies = list(replies)
        self.stop_reason = stop_reason
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "json_schema": json_schema})
        return LLMResponse(self.replies.pop(0), self.stop_reason, self.model, Usage(500, 40))


class CassetteMiss(RuntimeError):
    pass


def request_key(system: str, user: str, json_schema: dict | None = None) -> str:
    payload = json.dumps({"system": system, "user": user, "json_schema": json_schema}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class ReplayLLM:
    def __init__(self, cassette: Path, inner: LLMClient | None = None):
        """inner=None -> replay only; inner given -> record (calls inner on misses and saves)."""
        self.cassette = cassette
        self.inner = inner
        self.entries: dict = json.loads(cassette.read_text(encoding="utf-8")) if cassette.exists() else {}
        self.model = inner.model if inner else next((e["model"] for e in self.entries.values()), "replay")

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> LLMResponse:
        key = request_key(system, user, json_schema)
        if key in self.entries:
            e = self.entries[key]
            return LLMResponse(e["text"], e["stop_reason"], e["model"], Usage(**e["usage"]), e["latency_ms"])
        if self.inner is None:
            raise CassetteMiss(
                f"No recorded response for request {key} in {self.cassette.name}. The prompt, guidelines or "
                "title changed: re-record with a real model."
            )
        resp = self.inner.complete(system, user, json_schema)
        self.entries[key] = {
            "user": user,  # stored for readability; the key is what matters
            "text": resp.text,
            "stop_reason": resp.stop_reason,
            "model": resp.model,
            "usage": vars(resp.usage),
            "latency_ms": resp.latency_ms,
        }
        self.cassette.parent.mkdir(parents=True, exist_ok=True)
        self.cassette.write_text(json.dumps(self.entries, indent=1, ensure_ascii=False), encoding="utf-8")
        return resp
