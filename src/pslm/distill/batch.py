"""Teacher labeling at scale through the Message Batches API (50% off, asynchronous, usually under an hour).

    uv run python -m pslm.distill.batch estimate train --model claude-haiku-5-5   # offline cost estimate
    uv run python -m pslm.distill.batch submit train --model claude-opus-5-5      # needs an API key
    uv run python -m pslm.distill.batch collect <batch_id>                        # when processing has ended

Each request is the same as the sync labeler's (`request_params`): the cached guideline system prompt, one title
and the JSON output schema. Refusal fallbacks aren't allowed on the Batches API; a refused title shows up with
error="refusal". `custom_id` is the product id, and results come back in any order, so they are keyed by it.
State for each submitted batch is saved in data/distill/batches/<batch_id>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from pslm.config import ROOT, settings
from pslm.data.labels import load_labels, load_titles, write_jsonl
from pslm.distill.label import PREDS, evaluate, run_summary
from pslm.eval.metrics import format_report
from pslm.llm import model_slug
from pslm.llm.anthropic_client import request_params, to_response
from pslm.llm.base import PRICING
from pslm.parse import parse_output, strict_ok
from pslm.prompts import teacher_system, user_message
from pslm.schema import output_schema

STATE = ROOT / "data" / "distill" / "batches"


def build_requests(titles: list[dict], model: str, effort: str) -> list[dict]:
    system, schema = teacher_system(), output_schema()
    return [
        {
            "custom_id": t["id"],
            "params": request_params(model, system, user_message(t["title"]), schema, effort),
        }
        for t in titles
    ]


def estimate(titles: list[dict], model: str, out_tokens: int = 250) -> dict:
    """Rough pre-flight cost: ~4 chars per token; the system prompt is written to the cache once and read for
    every other request (caching in batches is best-effort, so this is the optimistic end). `out_tokens`
    covers the ~40-token JSON plus low-effort thinking; measure it on a small sync run before trusting it."""
    inp, out, cached = PRICING[model]
    sys_tok = len(teacher_system()) / 4
    user_tok = sum(len(user_message(t["title"])) for t in titles) / 4
    n = len(titles)
    full = (sys_tok * inp * 1.25 + (n - 1) * sys_tok * cached + user_tok * inp + n * out_tokens * out) / 1e6
    no_cache = (n * sys_tok * inp + user_tok * inp + n * out_tokens * out) / 1e6
    return {
        "model": model,
        "n": n,
        "system_prompt_tokens": round(sys_tok),
        "usd_batch_cached": round(full * 0.5, 2),
        "usd_batch_uncached": round(no_cache * 0.5, 2),
    }


def rows_from_results(results, titles_by_id: dict[str, dict]) -> list[dict]:
    """Convert batch results (any order) into the same prediction rows as the sync labeler."""
    rows = []
    for res in results:
        t = titles_by_id[res.custom_id]
        row = {"id": t["id"], "title": t["title"], "latency_ms": 0.0}
        if res.result.type == "succeeded":
            resp = to_response(res.result.message)
            parsed = parse_output(resp.text, resp.stop_reason)
            row |= {
                "pred": parsed.attrs.model_dump() if parsed.ok else None,
                "error": parsed.error,
                "strict_json": strict_ok(resp.text),
                "raw": resp.text,
                "model": resp.model,
                "usage": vars(resp.usage),
            }
        else:  # errored | canceled | expired: resubmit these ids
            row |= {
                "pred": None,
                "error": f"batch_{res.result.type}",
                "strict_json": False,
                "raw": "",
                "model": "",
                "usage": {},
            }
        rows.append(row)
    order = {i: k for k, i in enumerate(titles_by_id)}
    return sorted(rows, key=lambda r: order[r["id"]])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("estimate", "submit"):
        s = sub.add_parser(name)
        s.add_argument("split", choices=("train", "val", "test"))
        s.add_argument("--model", default=settings.teacher_model)
        s.add_argument("--limit", type=int)
    c = sub.add_parser("collect")
    c.add_argument("batch_id")
    c.add_argument("--wait", action="store_true", help="poll every 60 s until the batch has ended")
    a = p.parse_args()

    if a.cmd in ("estimate", "submit"):
        titles = load_titles(a.split)[: a.limit]
        est = estimate(titles, a.model)
        print(json.dumps(est))
        if a.cmd == "estimate":
            return
        import anthropic

        client = anthropic.Anthropic()
        batch = client.messages.batches.create(requests=build_requests(titles, a.model, settings.llm_effort))
        STATE.mkdir(parents=True, exist_ok=True)
        state = {
            "batch_id": batch.id,
            "split": a.split,
            "model": a.model,
            "ids": [t["id"] for t in titles],
            "estimate": est,
        }
        (STATE / f"{batch.id}.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
        print(f"submitted {batch.id} ({len(titles)} requests); collect with: collect {batch.id}")
        return

    import anthropic

    client = anthropic.Anthropic()
    state = json.loads((STATE / f"{a.batch_id}.json").read_text(encoding="utf-8"))
    while True:
        b = client.messages.batches.retrieve(a.batch_id)
        if b.processing_status == "ended" or not a.wait:
            break
        print(f"{b.processing_status}: {b.request_counts.processing} processing")
        time.sleep(60)
    if b.processing_status != "ended":
        print(f"not finished yet ({b.processing_status}); re-run with --wait")
        return
    titles = {t["id"]: t for t in load_titles(state["split"]) if t["id"] in set(state["ids"])}
    rows = rows_from_results(client.messages.batches.results(a.batch_id), titles)
    out = PREDS / state["split"] / f"{model_slug(state['model'])}.batch.jsonl"
    write_jsonl(out, rows)
    rep = evaluate([r for r in rows if r["usage"]], load_labels(state["split"]))
    summary = {
        "split": state["split"],
        **run_summary([r for r in rows if r["usage"]], state["model"], batch=True),
        "metrics": rep.summary(),
        "report": asdict(rep),
    }
    Path(out).with_suffix(".report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(format_report(rep, f"{state['model']} batch on {state['split']}"))
    print({k: summary[k] for k in ("strict_json", "cost_usd", "cost_per_1k_usd")})


if __name__ == "__main__":
    main()
