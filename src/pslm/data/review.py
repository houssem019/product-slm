"""Interactive review of teacher labels; the test set must be human-reviewed before it is used as gold.

    uv run python -m pslm.data.review test            # next unreviewed records
    uv run python -m pslm.data.review test --all      # walk every record again

Per record: Enter accepts, `e` edits field by field (Enter keeps a value, `-` sets null), `s` skips, `q` quits.
Progress is saved after every decision, so it is safe to stop at any time. Edited records keep the teacher's
original under `teacher_label`, which lets us measure the teacher's error rate per field.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from pslm.data.labels import LABELS, load_labels, write_jsonl
from pslm.schema import FIELDS, Category, ProductAttributes

Ask = Callable[[str], str]


def apply_review(record: dict, new_label: dict | None, reviewer: str) -> dict:
    """Return the record marked reviewed; `new_label` None means accepted unchanged."""
    out = dict(record)
    if new_label is not None:
        new = ProductAttributes.model_validate(new_label).model_dump()
        if new != record["label"]:
            out.setdefault("teacher_label", record["label"])
            out["label"] = new
    out["reviewed"] = True
    out["reviewer"] = reviewer
    return out


def edit_label(label: dict, ask: Ask) -> dict:
    new = dict(label)
    for f in FIELDS:
        hint = " | ".join(c.value for c in Category) if f == "category" else "'-' = null"
        while True:
            raw = ask(f"  {f} [{label[f]}] ({hint}): ").strip()
            value = label[f] if raw == "" else (None if raw == "-" else raw)
            try:
                ProductAttributes.model_validate({**new, f: value})
            except ValueError as e:
                print(f"    invalid: {e.errors()[0]['msg'] if hasattr(e, 'errors') else e}")
                continue
            new[f] = value
            break
    return new


def show(i: int, n: int, r: dict) -> None:
    print(f"\n[{i + 1}/{n}] {r['id']}\n  {r['title']}")
    for f in FIELDS:
        print(f"    {f:9} {r['label'][f]!r}")


def review(split: str, reviewer: str, redo_all: bool = False, ask: Ask = input, root: Path = LABELS) -> int:
    rows = load_labels(split, root)
    todo = [i for i, r in enumerate(rows) if redo_all or not r.get("reviewed")]
    done = 0
    for k, i in enumerate(todo):
        show(k, len(todo), rows[i])
        choice = ask("  [Enter] accept  [e]dit  [s]kip  [q]uit: ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            continue
        new = edit_label(rows[i]["label"], ask) if choice == "e" else None
        rows[i] = apply_review(rows[i], new, reviewer)
        write_jsonl(root / f"{split}.jsonl", rows)
        done += 1
    left = sum(not r.get("reviewed") for r in rows)
    print(f"\nreviewed {done} this session; {left} unreviewed left in {split}")
    return done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("split", choices=("train", "val", "test"))
    p.add_argument("--reviewer", default="human")
    p.add_argument("--all", action="store_true", help="also revisit already-reviewed records")
    a = p.parse_args()
    review(a.split, a.reviewer, a.all)


if __name__ == "__main__":
    main()
