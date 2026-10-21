"""Provider-neutral LLM interface. Extraction is one call per title: a system prompt, one user message (the title)
and an output JSON schema that providers enforce when they can. No tools, no multi-turn."""

from dataclasses import dataclass, field
from typing import Protocol

# USD per million tokens: (input, output, cache read). Keep in sync with the provider pricing page.
PRICING = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
BATCH_DISCOUNT = 0.5  # Message Batches API bills every token at half price


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def cost_usd(self, model: str, batch: bool = False) -> float:
        inp, out, cached = PRICING.get(model, (0.0, 0.0, 0.0))  # local models are free
        cost = (
            self.input_tokens * inp
            + self.cache_write_tokens * inp * 1.25
            + self.cache_read_tokens * cached
            + self.output_tokens * out
        ) / 1e6
        return cost * (BATCH_DISCOUNT if batch else 1.0)

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(*(a + b for a, b in zip(vars(self).values(), vars(other).values(), strict=True)))


@dataclass
class LLMResponse:
    text: str
    stop_reason: str  # end_turn | max_tokens | refusal
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> LLMResponse: ...
