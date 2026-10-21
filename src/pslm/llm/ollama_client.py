"""Local open-weight models via Ollama (free, CPU). Used as a zero-shot baseline and an offline teacher stand-in."""

import time

import httpx

from pslm.llm.base import LLMResponse, Usage


class OllamaLLM:
    def __init__(self, model: str, host: str, timeout_s: float = 600.0, num_ctx: int = 8192):
        self.model = model
        # Ollama's default context (2-4k tokens) silently truncates the prompt from the front, which would cut
        # the labeling guidelines out of the system prompt.
        self.num_ctx = num_ctx
        self.http = httpx.Client(base_url=host, timeout=timeout_s)

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> LLMResponse:
        start = time.perf_counter()
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_schema:
            body["format"] = json_schema  # constrained decoding: the output always parses against the schema
        r = self.http.post("/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        return LLMResponse(
            text=data["message"].get("content", ""),
            stop_reason="max_tokens" if data.get("done_reason") == "length" else "end_turn",
            model=self.model,
            usage=Usage(
                input_tokens=data.get("prompt_eval_count", 0), output_tokens=data.get("eval_count", 0)
            ),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
