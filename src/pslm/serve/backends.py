"""Extraction backends for serving: one interface, three implementations.

* `vllm`: production. The base model + LoRA adapter run in vLLM. Output is **constrained to `output_schema()`**
  (structured outputs), so every response is schema-valid JSON by construction, not just in practice. A whole
  request's titles go in one `generate` call; vLLM schedules them with continuous batching. GPU only.
* `hf`: CPU fallback (plain transformers, merged adapter, greedy, unconstrained). Slow but runs anywhere.
* `replay`: serves recorded model outputs from a cassette, keyed by title. No model and no GPU: CI and Docker smoke
  tests exercise the full API path with outputs the real model produced. Titles that were never recorded
  return `error="not_recorded"`.

Every backend builds the prompt the same way (the tokenizer's chat template with `STUDENT_SYSTEM`, exactly as in
training and eval), and every output goes through the same `parse_output`, so serving can't drift from
what was evaluated.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from pslm.parse import parse_output
from pslm.prompts import STUDENT_SYSTEM, user_message
from pslm.schema import output_schema


@dataclass
class Extraction:
    title: str
    attributes: dict | None
    error: str | None = None  # not_json | invalid_schema | truncated | not_recorded
    raw: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0

    def public(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k in ("title", "attributes", "error")}


@dataclass
class BatchResult:
    extractions: list[Extraction]
    model: str
    latency_ms: float
    stats: dict = field(default_factory=dict)


class Backend(Protocol):
    name: str
    model: str

    def extract(self, titles: list[str]) -> BatchResult: ...


def to_extraction(title: str, text: str, finish_reason: str = "stop", **tokens) -> Extraction:
    parsed = parse_output(text, "max_tokens" if finish_reason == "length" else "end_turn")
    return Extraction(
        title=title,
        attributes=parsed.attrs.model_dump() if parsed.ok else None,
        error=parsed.error,
        raw=text,
        **tokens,
    )


def chat_prompt(tokenizer, title: str) -> str:
    msgs = [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": user_message(title)}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def adapter_base(adapter_dir: str | Path) -> str:
    return json.loads((Path(adapter_dir) / "adapter_config.json").read_text(encoding="utf-8"))[
        "base_model_name_or_path"
    ]


class VLLMBackend:
    name = "vllm"

    def __init__(
        self,
        adapter: str | Path,
        guided: bool = True,
        max_tokens: int = 96,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 512,
        max_lora_rank: int = 16,
    ):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from vllm.sampling_params import StructuredOutputsParams

        base = adapter_base(adapter)
        self.model = f"{base.split('/')[-1]}+lora" + ("" if guided else " (unguided)")
        self.llm = LLM(
            model=base,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            dtype="half",  # T4 has no bf16
            max_model_len=max_model_len,  # prompts are ~150 tokens + <=96 generated
            gpu_memory_utilization=gpu_memory_utilization,
            seed=0,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.lora = LoRARequest("product-slm", 1, str(adapter))
        self.params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            structured_outputs=StructuredOutputsParams(json=output_schema()) if guided else None,
        )

    def extract(self, titles: list[str]) -> BatchResult:
        prompts = [chat_prompt(self.tokenizer, t) for t in titles]
        start = time.perf_counter()
        outs = self.llm.generate(prompts, self.params, lora_request=self.lora, use_tqdm=False)
        ms = (time.perf_counter() - start) * 1000
        ex = [
            to_extraction(
                t,
                o.outputs[0].text,
                o.outputs[0].finish_reason or "stop",
                prompt_tokens=len(o.prompt_token_ids),
                output_tokens=len(o.outputs[0].token_ids),
            )
            for t, o in zip(titles, outs, strict=True)
        ]
        return BatchResult(ex, self.model, round(ms, 1))


class HFBackend:
    name = "hf"

    def __init__(self, model: str | Path):
        from pslm.llm.hf_client import HFLLM

        self.llm = HFLLM(str(model))
        self.model = self.llm.model

    def extract(self, titles: list[str]) -> BatchResult:
        start = time.perf_counter()
        ex = []
        for t in titles:  # sequential: this is the slow, portable fallback, not the throughput path
            r = self.llm.complete(STUDENT_SYSTEM, user_message(t))
            ex.append(
                to_extraction(
                    t,
                    r.text,
                    "length" if r.stop_reason == "max_tokens" else "stop",
                    prompt_tokens=r.usage.input_tokens,
                    output_tokens=r.usage.output_tokens,
                )
            )
        return BatchResult(ex, self.model, round((time.perf_counter() - start) * 1000, 1))


class ReplayBackend:
    name = "replay"

    def __init__(self, cassette: str | Path):
        entries = json.loads(Path(cassette).read_text(encoding="utf-8"))
        # Cassette keys hash the full request; serving looks titles up, so index by the stored user message.
        self.by_user = {e["user"]: e for e in entries.values() if e.get("user")}
        self.model = next(iter(entries.values()), {}).get("model", "replay") + " (replay)"

    def extract(self, titles: list[str]) -> BatchResult:
        start = time.perf_counter()
        ex = []
        for t in titles:
            e = self.by_user.get(user_message(t))
            if e is None:
                ex.append(Extraction(t, None, "not_recorded"))
                continue
            finish = "length" if e["stop_reason"] == "max_tokens" else "stop"
            ex.append(
                to_extraction(
                    t,
                    e["text"],
                    finish,
                    prompt_tokens=e["usage"]["input_tokens"],
                    output_tokens=e["usage"]["output_tokens"],
                )
            )
        return BatchResult(ex, self.model, round((time.perf_counter() - start) * 1000, 1))


def build_backend(kind: str, model: str | None = None, guided: bool = True) -> Backend:
    if kind == "replay":
        from pslm.config import settings

        return ReplayBackend(model or settings.cassette_dir / "qwen2.5-1.5b-lora.json")
    if model is None:
        raise ValueError(f"backend {kind!r} needs a model (adapter dir, or Hub id for hf)")
    if kind == "vllm":
        return VLLMBackend(model, guided=guided)
    if kind == "hf":
        return HFBackend(model)
    raise ValueError(f"unknown backend {kind!r} (vllm | hf | replay)")
