"""Supervised fine-tuning data: chat-formatted (system, user, assistant) triples from the label files.

    uv run python -m pslm.train.format          # writes data/sft/{train,val}.jsonl

The assistant turn is the gold label as *canonical* JSON: fixed key order and compact separators, with nulls
written out. A single target string per title means the model never has to guess between equivalent
serializations, and the loss isn't spent on formatting variance. Only train and val are exported; the test split
never reaches training code.
"""

from __future__ import annotations

import json

from pslm.config import ROOT
from pslm.data.labels import load_labels, write_jsonl
from pslm.prompts import STUDENT_SYSTEM, user_message
from pslm.schema import FIELDS

SFT = ROOT / "data" / "sft"
SFT_SPLITS = ("train", "val")


def target_json(label: dict) -> str:
    return json.dumps({f: label[f] for f in FIELDS}, ensure_ascii=False, separators=(", ", ": "))


def to_messages(title: str, label: dict | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": STUDENT_SYSTEM},
        {"role": "user", "content": user_message(title)},
    ]
    if label is not None:
        msgs.append({"role": "assistant", "content": target_json(label)})
    return msgs


def build(split: str) -> list[dict]:
    if split not in SFT_SPLITS:
        raise ValueError(f"refusing to export {split!r} for training")
    return [{"id": r["id"], "messages": to_messages(r["title"], r["label"])} for r in load_labels(split)]


def main() -> None:
    for split in SFT_SPLITS:
        rows = build(split)
        write_jsonl(SFT / f"{split}.jsonl", rows)
        print(f"{split}: {len(rows)} examples -> {SFT / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
