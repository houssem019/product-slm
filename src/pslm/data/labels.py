"""Label files: data/labels/<split>.jsonl, one record per title, joined to data/titles/<split>.jsonl by id.

    uv run python -m pslm.data.labels import test path/to/labels.tsv --labeler "claude-opus-5-5 (Claude Code)"
    uv run python -m pslm.data.labels check

The TSV is `index<TAB>category<TAB>brand<TAB>size<TAB>material`, where index is the 0-based line in the titles file
and `-` means null. It's a compact format for writing labels by hand or in a Claude Code session; the JSONL is the
source of truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pslm.schema import ProductAttributes

TITLES = Path("data/titles")
LABELS = Path("data/labels")
SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_titles(split: str, root: Path = TITLES) -> list[dict]:
    return read_jsonl(root / f"{split}.jsonl")


def load_labels(split: str, root: Path = LABELS) -> list[dict]:
    path = root / f"{split}.jsonl"
    return read_jsonl(path) if path.exists() else []


def parse_tsv_line(line: str) -> tuple[int, ProductAttributes]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 5:
        raise ValueError(f"expected 5 tab-separated fields, got {len(parts)}: {line!r}")
    idx, category, brand, size, material = (None if p.strip() == "-" else p.strip() for p in parts)
    return int(idx), ProductAttributes(category=category, brand=brand, size=size, material=material)


def import_tsv(split: str, tsv: Path, labeler: str) -> list[dict]:
    """Merge TSV labels into the split's label file (later imports overwrite earlier ones for the same id)."""
    titles = load_titles(split)
    by_id = {r["id"]: r for r in load_labels(split)}
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        idx, attrs = parse_tsv_line(line)
        t = titles[idx]
        by_id[t["id"]] = {
            "id": t["id"],
            "title": t["title"],
            "label": attrs.model_dump(),
            "labeler": labeler,
            "reviewed": False,
        }
    order = {t["id"]: i for i, t in enumerate(titles)}
    rows = sorted(by_id.values(), key=lambda r: order[r["id"]])
    write_jsonl(LABELS / f"{split}.jsonl", rows)
    return rows


def _squash(s: str) -> str:
    return "".join(s.lower().split())


@dataclass
class CheckReport:
    split: str
    n_titles: int
    n_labels: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_split(split: str, titles_root: Path = TITLES, labels_root: Path = LABELS) -> CheckReport:
    """Schema validity, id/title alignment, and grounding: brand and size must be copied from the title."""
    titles = {t["id"]: t["title"] for t in load_titles(split, titles_root)}
    labels = load_labels(split, labels_root)
    rep = CheckReport(split, len(titles), len(labels))
    for r in labels:
        rid = r["id"]
        if rid not in titles:
            rep.errors.append(f"{rid}: id not in titles")
            continue
        if r["title"] != titles[rid]:
            rep.errors.append(f"{rid}: title differs from data/titles")
        try:
            attrs = ProductAttributes.model_validate(r["label"])
        except ValueError as e:
            rep.errors.append(f"{rid}: invalid label: {e}")
            continue
        title = _squash(titles[rid])
        for f in ("brand", "size"):
            v = getattr(attrs, f)
            if v is not None and _squash(v) not in title:
                rep.errors.append(f"{rid}: {f}={v!r} is not in the title")
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("split", choices=SPLITS)
    imp.add_argument("tsv", type=Path)
    imp.add_argument("--labeler", required=True)
    sub.add_parser("check")
    a = p.parse_args()

    if a.cmd == "import":
        rows = import_tsv(a.split, a.tsv, a.labeler)
        print(f"{a.split}: {len(rows)} labels")
    reports = [check_split(s) for s in SPLITS if (TITLES / f"{s}.jsonl").exists()]
    for rep in reports:
        print(f"{rep.split}: {rep.n_labels}/{rep.n_titles} labeled, {len(rep.errors)} errors")
        for e in rep.errors:
            print(f"  {e}")
    sys.exit(0 if all(r.ok for r in reports) else 1)


if __name__ == "__main__":
    main()
