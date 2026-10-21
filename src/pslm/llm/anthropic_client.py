"""Claude via the official Anthropic SDK, with structured outputs so the reply is always schema-shaped JSON."""

import time

import anthropic

from pslm.llm.base import LLMResponse, Usage

# Models that accept server-side refusal fallbacks (a declined request is re-run on another model). Not
# available on the Batches API, so batch requests never carry them.
FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5")
MAX_TOKENS = 2048  # the JSON is ~40 tokens; the rest is headroom for (low-effort) thinking


def request_params(model: str, system: str, user: str, json_schema: dict | None, effort: str) -> dict:
    """Messages API params shared by the sync client and the batch labeler (so both send identical prompts)."""
    output_config: dict = {"effort": effort}
    if json_schema:
        output_config["format"] = {"type": "json_schema", "schema": json_schema}
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        # Same guideline prompt on every call: cache it, only the title varies.
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user}],
        "output_config": output_config,
    }


def to_response(msg, start: float | None = None) -> LLMResponse:
    u = msg.usage
    return LLMResponse(
        text="".join(b.text for b in msg.content if b.type == "text"),
        stop_reason=msg.stop_reason or "end_turn",
        model=msg.model,
        usage=Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=u.cache_read_input_tokens or 0,
            cache_write_tokens=u.cache_creation_input_tokens or 0,
        ),
        latency_ms=round((time.perf_counter() - start) * 1000, 1) if start else 0.0,
    )


class AnthropicLLM:
    def __init__(self, model: str, effort: str = "low"):
        self.model = model
        self.effort = effort
        # credentials from ANTHROPIC_API_KEY or an `ant auth login` profile
        self.client = anthropic.Anthropic()

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> LLMResponse:
        params = request_params(self.model, system, user, json_schema, self.effort)
        if self.model.startswith(FALLBACK_MODELS):
            params |= {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        start = time.perf_counter()
        # stop_reason "refusal" comes back as a normal response; the caller records it as a failed label
        return to_response(self.client.beta.messages.create(**params), start)
