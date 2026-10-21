"""Paired comparison of two recorded runs on the same split: difference + 95% paired-bootstrap CI per metric.

    uv run python -m pslm.eval.compare qwen2.5-1.5b-lora qwen2.5-7b --split test

"Paired" means both runs are resampled on the same titles. That's far tighter than comparing two independent
confidence intervals, because most titles are easy (or hard) for both models. With n=150, a ~0.07 difference in
exact match is not significant. Check this before claiming one model "beats" another.
"""

from __future__ import annotations

import argparse
import random

from pslm.config import ROOT
from pslm.data.labels import load_labels, read_jsonl
from pslm.eval.metrics import field_match
from pslm.schema import FIELDS

METRICS = ("exact", *FIELDS)


def per_title_hits(run: str, split: str) -> dict[str, dict[str, bool]]:
    gold = {r["id"]: r["label"] for r in load_labels(split)}
    rows = read_jsonl(ROOT / "data" / "preds" / split / f"{run}.jsonl")
    out = {}
    for r in rows:
        hits = {f: field_match(f, r["pred"], gold[r["id"]]) for f in FIELDS}
        out[r["id"]] = {"exact": all(hits.values()), **hits}
    return out


def paired_bootstrap(a: dict, b: dict, metric: str, n_boot: int = 10_000, seed: int = 0) -> dict:
    ids = sorted(set(a) & set(b))
    d = [int(a[i][metric]) - int(b[i][metric]) for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return {
        "n": len(ids),
        "diff": round(sum(d) / len(d), 3),
        "ci95": (round(boots[int(0.025 * n_boot)], 3), round(boots[int(0.975 * n_boot)], 3)),
        "only_a": sum(x == 1 for x in d),
        "only_b": sum(x == -1 for x in d),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--split", default="test")
    args = p.parse_args()
    a, b = per_title_hits(args.a, args.split), per_title_hits(args.b, args.split)
    print(f"{args.a} minus {args.b} on {args.split}")
    for m in METRICS:
        r = paired_bootstrap(a, b, m)
        sig = "significant" if r["ci95"][0] > 0 or r["ci95"][1] < 0 else "not significant"
        print(
            f"  {m:9} {r['diff']:+.3f}  95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
            f"only-a {r['only_a']:3}  only-b {r['only_b']:3}  ({sig})"
        )


if __name__ == "__main__":
    main()
