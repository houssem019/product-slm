"""Per-field extraction metrics against gold labels. One definition for every model (teacher, baselines, SLM).

Matching is after light normalization: case and repeated whitespace for brand/material; case and *all*
whitespace for size, since "20 oz" and "20oz" are the same size. Null equals null.

For the nullable fields, accuracy alone flatters a model that always says null (most titles have no size), so
we also report extraction P/R/F1 with "a non-null value" as the positive class:
  TP = predicted a value and it matches gold
  FP = predicted a value that is wrong or should be null
  FN = gold has a value and the prediction is null or wrong
(A wrong value counts as both FP and FN, the usual convention for span extraction.)

An invalid output (not JSON, wrong schema, refusal) counts as all-null for P/R/F1 and as wrong for accuracy.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from pslm.schema import FIELDS

NULLABLE = ("brand", "size", "material")


def normalize(f: str, v: str | None) -> str | None:
    if v is None:
        return None
    if f == "size":
        return "".join(v.lower().split())
    return " ".join(v.lower().split())


def field_match(f: str, pred: dict | None, gold: dict) -> bool:
    return pred is not None and normalize(f, pred.get(f)) == normalize(f, gold[f])


@dataclass
class Report:
    n: int
    valid: int
    exact: int  # all four fields correct
    correct: dict[str, int]
    prf: dict[str, dict[str, float]]
    per_category: dict[str, dict[str, int]] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "n": self.n,
            "json_valid": round(self.valid / self.n, 3) if self.n else 0.0,
            "exact_match": round(self.exact / self.n, 3) if self.n else 0.0,
            "accuracy": {f: round(c / self.n, 3) for f, c in self.correct.items()} if self.n else {},
            "f1": {f: v["f1"] for f, v in self.prf.items()},
        }


def score(
    preds: dict[str, dict | None], gold: dict[str, dict], errors: dict[str, str] | None = None
) -> Report:
    """preds: id -> predicted label dict (None if the output was invalid); gold: id -> gold label dict.
    Only ids present in gold are scored; a gold id missing from preds counts as invalid."""
    correct = Counter()
    tp, fp, fn = Counter(), Counter(), Counter()
    per_cat: dict[str, Counter] = defaultdict(Counter)
    valid = exact = 0
    for i, g in gold.items():
        p = preds.get(i)
        valid += p is not None
        hits = {f: field_match(f, p, g) for f in FIELDS}
        correct.update(f for f, ok in hits.items() if ok)
        exact += all(hits.values())
        per_cat[g["category"]]["n"] += 1
        per_cat[g["category"]]["correct"] += hits["category"]
        for f in NULLABLE:
            has_pred = p is not None and p.get(f) is not None
            has_gold = g[f] is not None
            if has_pred and hits[f]:
                tp[f] += 1
            else:
                fp[f] += has_pred
                fn[f] += has_gold
    prf = {}
    for f in NULLABLE:
        if tp[f] + fp[f] + fn[f] == 0:  # nothing to extract and nothing extracted: undefined, not 0
            prf[f] = {"precision": None, "recall": None, "f1": None}
            continue
        prec = tp[f] / (tp[f] + fp[f]) if tp[f] + fp[f] else 0.0
        rec = tp[f] / (tp[f] + fn[f]) if tp[f] + fn[f] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        prf[f] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}
    return Report(
        n=len(gold),
        valid=valid,
        exact=exact,
        correct={f: correct[f] for f in FIELDS},
        prf=prf,
        per_category={c: dict(v) for c, v in sorted(per_cat.items())},
        errors=dict(Counter((errors or {}).values())),
    )


def format_report(rep: Report, title: str = "") -> str:
    s = rep.summary()
    lines = [
        f"== {title}" if title else "==",
        f"n={s['n']}  json_valid={s['json_valid']}  exact={s['exact_match']}",
    ]
    lines.append("field      acc    P      R      F1")
    for f in FIELDS:
        acc = s["accuracy"].get(f, 0.0)
        if f in rep.prf:
            p = {k: "  n/a" if v is None else f"{v:.3f}" for k, v in rep.prf[f].items()}
            lines.append(f"{f:9}  {acc:.3f}  {p['precision']}  {p['recall']}  {p['f1']}")
        else:
            lines.append(f"{f:9}  {acc:.3f}")
    if rep.errors:
        lines.append(f"invalid outputs: {rep.errors}")
    return "\n".join(lines)
