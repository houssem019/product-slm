"""The student model through plain transformers: CPU fallback and Colab eval. Production serving is vLLM.

`model` is either a PEFT adapter directory (the base model is read from its adapter_config.json) or a plain Hub
model id (zero-shot SLM baseline). Decoding is greedy and unconstrained, so JSON validity measured here is the
model's own. vLLM serving adds guided decoding on top.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pslm.llm.base import LLMResponse, Usage


def greedy_config(shipped, tokenizer):
    """Pure greedy decoding, built from scratch rather than edited from the model's shipped defaults.

    Qwen2.5's generation_config.json sets repetition_penalty=1.1 (plus sampling params). `generate` applies the
    repetition penalty even with do_sample=False, and for an extractor that's poison: it penalizes every token
    already in the prompt, which is exactly what copying a brand or size from the title requires. Measured
    effect: exact match 0.633 vs 0.707 on test, all of it brand/size answers turned into null
    (docs/CHALLENGES.md #17). vLLM ignores the file when explicit SamplingParams are given, which is how this was
    found.
    """
    from transformers import GenerationConfig

    return GenerationConfig(
        do_sample=False,
        repetition_penalty=1.0,
        eos_token_id=shipped.eos_token_id,  # Qwen stops on <|im_end|> or <|endoftext|>
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else shipped.pad_token_id,
        bos_token_id=shipped.bos_token_id,
    )


class HFLLM:
    def __init__(self, model: str, max_new_tokens: int = 96):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens  # the longest gold JSON is ~45 tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        path = Path(model)
        if (path / "adapter_config.json").exists():
            from peft import PeftModel

            base = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))[
                "base_model_name_or_path"
            ]
            self.model_obj = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
            # merge LoRA into the weights: same outputs, no adapter overhead per token
            self.model_obj = PeftModel.from_pretrained(self.model_obj, str(path)).merge_and_unload()
            self.model = f"{base.split('/')[-1]}+{path.parent.name if path.name == 'adapter' else path.name}"
        else:
            self.model_obj = AutoModelForCausalLM.from_pretrained(model, dtype=dtype)
            self.model = model.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model_obj.to(self.device).eval()
        self.model_obj.generation_config = greedy_config(self.model_obj.generation_config, self.tokenizer)

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> LLMResponse:
        # json_schema is ignored: no constrained decoding in this backend (see module docstring)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        start = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model_obj.generate(**enc, max_new_tokens=self.max_new_tokens)
        new = out[0, enc["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        eos = self.model_obj.generation_config.eos_token_id
        eos_ids = set(eos) if isinstance(eos, list) else {eos}
        hit_limit = len(new) >= self.max_new_tokens and new[-1].item() not in eos_ids
        return LLMResponse(
            text=text,
            stop_reason="max_tokens" if hit_limit else "end_turn",
            model=self.model,
            usage=Usage(input_tokens=enc["input_ids"].shape[1], output_tokens=len(new)),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
