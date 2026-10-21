"""Run an LLM extractor over a split, one request per title, and score it against the reference labels.

Two uses, one code path:
* **Teacher labeling**: label titles with Claude (sync here; `pslm.distill.batch` for the 50%-off Batch API).
* **Zero-shot baselines**: e.g. a local Ollama model with the same guideline prompt, for the comparison table.

    uv run python -m pslm.distill.label val --provider ollama --record            # live, saves a cassette
    uv run python -m pslm.distill.label val --provider fake --model qwen2.5:7b    # replay (CI)
    uv run python -m pslm.distill.label val --provider anthropic --limit 5        # needs an API key

Predictions go to data/preds/<split>/<model>.jsonl and the report to data/preds/<split>/<model>.report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from pslm.config import ROOT, settings
from pslm.data.labels import load_labels, load_titles, write_jsonl
from pslm.eval.metrics import Report, format_report, score
from pslm.llm import LLMClient, Usage, get_llm, model_slug
from pslm.llm.replay import ReplayLLM
from pslm.parse import parse_output, strict_ok
from pslm.prompts import STUDENT_SYSTEM, teacher_system, user_message
from pslm.schema import output_schema

PREDS = ROOT / "data" / "preds"
SCHEMA = output_schema()


def predict(llm: LLMClient, titles: list[dict], system: str, schema: dict | None = SCHEMA) -> list[dict]:
    rows = []
    for k, t in enumerate(titles, 1):
        resp = llm.complete(system, user_message(t["title"]), schema)
        parsed = parse_output(resp.text, resp.stop_reason)
        rows.append(
            {
                "id": t["id"],
                "title": t["title"],
                "pred": parsed.attrs.model_dump() if parsed.ok else None,
                "error": parsed.error,
                "strict_json": strict_ok(resp.text),
                "raw": resp.text,
                "model": resp.model,
                "latency_ms": resp.latency_ms,
                "usage": vars(resp.usage),
            }
        )
        if k % 10 == 0 or k == len(titles):
            print(f"  {k}/{len(titles)}", file=sys.stderr, flush=True)
    return rows


def evaluate(rows: list[dict], gold_rows: list[dict]) -> Report:
    ids = {r["id"] for r in rows}
    gold = {g["id"]: g["label"] for g in gold_rows if g["id"] in ids}
    preds = {r["id"]: r["pred"] for r in rows}
    return score(preds, gold, {r["id"]: r["error"] for r in rows if r["error"]})


def run_summary(rows: list[dict], model: str, batch: bool = False) -> dict:
    usage = sum((Usage(**r["usage"]) for r in rows), Usage())
    lat = sorted(r["latency_ms"] for r in rows)
    n = len(rows)
    return {
        "model": model,
        "n": n,
        "strict_json": round(sum(r["strict_json"] for r in rows) / n, 3) if n else 0.0,
        "latency_ms_p50": lat[n // 2] if n else 0.0,
        "latency_ms_p95": lat[min(n - 1, int(n * 0.95))] if n else 0.0,
        "usage": vars(usage),
        "cost_usd": round(usage.cost_usd(model, batch), 4),
        "cost_per_1k_usd": round(usage.cost_usd(model, batch) / n * 1000, 4) if n else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("split", choices=("train", "val", "test"))
    p.add_argument("--provider", default=settings.llm_provider, choices=("anthropic", "ollama", "hf", "fake"))
    p.add_argument("--model", help="anthropic/ollama model id, or for hf an adapter dir / Hub id")
    p.add_argument("--name", help="run name for output and cassette files (default: derived from the model)")
    p.add_argument(
        "--prompt",
        choices=("teacher", "student"),
        help="default: student for hf (the fine-tuned model's training prompt), teacher otherwise",
    )
    p.add_argument("--limit", type=int, help="first N titles only (smoke tests)")
    p.add_argument("--record", action="store_true", help="save responses to the cassette for CI replay")
    p.add_argument("--out", type=Path, default=PREDS)
    a = p.parse_args()

    model = a.model or {"anthropic": settings.teacher_model, "ollama": settings.ollama_model}.get(a.provider)
    if model is None:
        p.error(f"--provider {a.provider} needs --model")
    if a.provider == "hf":
        inner = get_llm("hf", model=model)  # load once; the run is named after base+adapter
        name = a.name or inner.model
        cassette = settings.cassette_dir / f"{model_slug(name)}.json"
        llm = ReplayLLM(cassette, inner=inner) if a.record else inner
    else:
        name = a.name or model
        cassette = settings.cassette_dir / f"{model_slug(name)}.json"
        llm = get_llm(a.provider, cassette=cassette, record=a.record, model=model)
    prompt = a.prompt or ("student" if a.provider == "hf" else "teacher")
    system = STUDENT_SYSTEM if prompt == "student" else teacher_system()

    titles = load_titles(a.split)[: a.limit]
    rows = predict(llm, titles, system)
    out = a.out / a.split / f"{model_slug(name)}.jsonl"
    write_jsonl(out, rows)

    rep = evaluate(rows, load_labels(a.split))
    summary = {
        "split": a.split,
        **run_summary(rows, llm.model),
        "prompt": prompt,
        "metrics": rep.summary(),
        "report": asdict(rep),
    }
    out.with_suffix(".report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(format_report(rep, f"{name} on {a.split}, {prompt} prompt (vs reference labels)"))
    print({k: summary[k] for k in ("strict_json", "latency_ms_p50", "cost_usd", "cost_per_1k_usd")})


if __name__ == "__main__":
    main()
