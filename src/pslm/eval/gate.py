"""Eval regression gate: fail CI when a run's metrics drop below the committed thresholds.

data/eval/baseline.json maps "<run name>:<split>" to minimum values, using the same keys as Report.summary():

    {"qwen2.5-1.5b-lora:test": {"exact_match": 0.61, "json_valid": 0.98, "accuracy": {"category": 0.79}, ...}}

CI replays recorded cassettes, so the metrics are deterministic. What the gate actually catches is a change to
the code or data that grades those outputs: the parser, normalization, metrics, gold labels or prompt (a prompt
change is a cassette miss, which fails even earlier). Thresholds sit a little under the measured values, so a
deliberate one-label fix doesn't need a baseline edit, but a real regression does.
"""

from __future__ import annotations

import json
from pathlib import Path

from pslm.config import ROOT

BASELINE = ROOT / "data" / "eval" / "baseline.json"


def check(metrics: dict, thresholds: dict, prefix: str = "") -> list[str]:
    """Every threshold must be met. Nested dicts (accuracy, f1) are checked key by key; None counts as failing."""
    failures = []
    for key, minimum in thresholds.items():
        name = f"{prefix}{key}"
        value = metrics.get(key) if isinstance(metrics, dict) else None
        if isinstance(minimum, dict):
            failures += check(value or {}, minimum, prefix=f"{name}.")
        elif value is None or value < minimum:
            failures.append(f"{name} = {value} < {minimum}")
    return failures


def gate(run: str, split: str, metrics: dict, path: Path = BASELINE) -> list[str]:
    key = f"{run}:{split}"
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if key not in baseline:
        return [f"no thresholds for {key} in {path.name}"]
    return check(metrics, baseline[key])
