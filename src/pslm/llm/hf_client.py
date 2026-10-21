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

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> LLMResponse:
        # json_schema is ignored: no constrained decoding in this backend (see module docstring)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        start = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model_obj.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new = out[0, enc["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        hit_limit = len(new) >= self.max_new_tokens and new[-1].item() != self.tokenizer.eos_token_id
        return LLMResponse(
            text=text,
            stop_reason="max_tokens" if hit_limit else "end_turn",
            model=self.model,
            usage=Usage(input_tokens=enc["input_ids"].shape[1], output_tokens=len(new)),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
