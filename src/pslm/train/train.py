"""LoRA / QLoRA fine-tuning of a small instruct model on the distilled labels.

    # Colab/Kaggle T4 (the real run): QLoRA, 4-bit NF4 base, fp16 compute
    python -m pslm.train.train --base Qwen/Qwen2.5-1.5B-Instruct --load-in-4bit --out outputs/qwen1.5b-lora

    # Local CPU fallback: plain LoRA, fp32, on a smaller base (bitsandbytes 4-bit needs CUDA)
    uv run --extra train python -m pslm.train.train --base Qwen/Qwen2.5-0.5B-Instruct --out outputs/qwen0.5b-lora

    # Code-path smoke test (seconds, tiny random weights; the loss is meaningless)
    uv run --extra train python -m pslm.train.train --base trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 \
        --max-steps 2 --out outputs/smoke

Design:
* Loss on the assistant tokens only (the JSON + end-of-turn). The prompt is identical boilerplate; training on it
  wastes capacity, and on 600 examples it would dominate the loss.
* LoRA on every linear projection (attention + MLP), r=16. On small models, adapting only q/v underfits
  formatting-heavy tasks.
* Plain `transformers.Trainer` with a 20-line collator instead of TRL's SFTTrainer: fewer moving parts, and the
  masking is explicit and easy to follow.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from pslm.data.labels import read_jsonl
from pslm.train.format import SFT

IGNORE = -100
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tokenize(tokenizer, messages: list[dict], max_len: int) -> dict:
    """input_ids for the whole chat; labels masked to IGNORE everywhere except the assistant answer."""
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("chat template renders the prompt differently with and without the answer")
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if f_ids[: len(p_ids)] != p_ids:
        raise ValueError(
            "prompt tokens are not a prefix of the full sequence (tokenizer merges across the edge)"
        )
    if len(f_ids) > max_len:
        raise ValueError(f"example is {len(f_ids)} tokens, over max_len={max_len}; raise it, don't truncate")
    labels = [IGNORE] * len(p_ids) + f_ids[len(p_ids) :]
    return {"input_ids": f_ids, "labels": labels}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)  # right padding: fine for training
            labels.append(b["labels"] + [IGNORE] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


def load_split(tokenizer, split: str, max_len: int, sft_dir: Path = SFT) -> list[dict]:
    return [tokenize(tokenizer, r["messages"], max_len) for r in read_jsonl(sft_dir / f"{split}.jsonl")]


def load_model(base: str, load_in_4bit: bool):
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NF4: quantization levels matched to normally distributed weights
            bnb_4bit_use_double_quant=True,  # also quantize the quantization constants (~0.4 bit/param saved)
            bnb_4bit_compute_dtype=torch.float16,  # T4 has no bf16
        )
        model = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant, device_map="auto")
        # casts norms to fp32, enables input grads for gradient checkpointing through frozen 4-bit layers
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    return model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", type=Path, default=Path("outputs/lora"))
    p.add_argument("--load-in-4bit", action="store_true", help="QLoRA (needs CUDA + bitsandbytes)")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)  # effective batch 16
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--max-steps", type=int, default=-1, help="cap optimizer steps (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--push-to-hub", metavar="REPO_ID", help="upload the adapter, e.g. houssem019/product-slm-lora"
    )
    a = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(a.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ds, val_ds = (load_split(tokenizer, s, a.max_len) for s in ("train", "val"))
    lengths = [len(x["input_ids"]) for x in train_ds]
    print(
        f"train {len(train_ds)} / val {len(val_ds)} examples; tokens max {max(lengths)}, mean {sum(lengths) / len(lengths):.0f}"
    )

    model = load_model(a.base, a.load_in_4bit)
    lora = LoraConfig(
        r=a.rank,
        lora_alpha=a.alpha,
        lora_dropout=a.dropout,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    cuda = torch.cuda.is_available()
    steps_per_epoch = max(1, len(train_ds) // (a.batch_size * a.grad_accum))
    args = TrainingArguments(
        output_dir=str(a.out / "checkpoints"),
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * steps_per_epoch * a.epochs)),
        logging_steps=5,
        eval_strategy="epoch" if a.max_steps < 0 else "no",
        save_strategy="no",  # the adapter is saved once at the end; checkpoints would only fill Colab's disk
        fp16=cuda,
        gradient_checkpointing=a.load_in_4bit,
        report_to="none",
        seed=a.seed,
        use_cpu=not cuda,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    start = time.time()
    result = trainer.train()
    eval_metrics = trainer.evaluate() if a.max_steps < 0 else {}

    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out / "adapter")
    tokenizer.save_pretrained(a.out / "adapter")
    summary = {
        "base": a.base,
        "qlora_4bit": a.load_in_4bit,
        "hyperparams": {k: v for k, v in vars(a).items() if k not in ("out", "push_to_hub")}
        | {"out": str(a.out)},
        "trainable_params": trainable,
        "total_params": total,
        "train_examples": len(train_ds),
        "train_loss": result.training_loss,
        "eval_loss": eval_metrics.get("eval_loss"),
        "runtime_s": round(time.time() - start, 1),
        "device": torch.cuda.get_device_name(0)
        if cuda
        else f"cpu ({platform.processor() or platform.machine()})",
        "log_history": trainer.state.log_history,
    }
    (a.out / "train_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("train_loss", "eval_loss", "runtime_s", "device")}))
    if a.push_to_hub:
        model.push_to_hub(a.push_to_hub)
        tokenizer.push_to_hub(a.push_to_hub)


if __name__ == "__main__":
    main()
