"""Serving benchmark: throughput and latency per batch size, plus accuracy of what the backend actually serves.

    # Colab T4 (notebooks/serve_bench_colab.ipynb)
    python -m pslm.serve.bench --backend vllm --model outputs/qwen2.5-1.5b-lora/adapter --name vllm-guided
    python -m pslm.serve.bench --backend vllm --model ... --no-guided --name vllm-unguided
    # Local CPU fallback (slow; a handful of titles is enough to measure it)
    uv run --extra train python -m pslm.serve.bench --backend hf --model outputs/qwen2.5-1.5b-lora/adapter \
        --batch-sizes 1 --limit 10 --name hf-cpu

Throughput: titles/s over the split, sent in batches of `bs`. Latency: wall time per batch, i.e. what one
`/extract` call of that size waits for. The first batch is a warm-up (CUDA graphs, structured-output grammar
compilation) and is excluded.

Accuracy: the full split is extracted once at the largest batch size and written as normal prediction rows to
data/preds/<split>/qwen2.5-1.5b-lora-<name>.jsonl, so `pslm.eval.compare` and the gate work on it. That's how
"does guided decoding change accuracy?" gets answered with the same tools as everything else.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict

from pslm.config import ROOT
from pslm.data.labels import load_labels, load_titles, write_jsonl
from pslm.distill.label import PREDS, evaluate
from pslm.eval.metrics import format_report
from pslm.parse import strict_ok
from pslm.serve.backends import build_backend

BENCH = ROOT / "data" / "bench"


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 1)


def device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return f"cpu ({platform.processor() or platform.machine()})"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", required=True, choices=("vllm", "hf", "replay"))
    p.add_argument("--model")
    p.add_argument("--no-guided", action="store_true", help="vLLM without structured outputs")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-sizes", default="1,8,32,150")
    p.add_argument("--limit", type=int)
    p.add_argument("--name", required=True, help="run name, e.g. vllm-guided")
    a = p.parse_args()

    titles = load_titles(a.split)[: a.limit]
    texts = [t["title"] for t in titles]
    load_start = time.perf_counter()
    backend = build_backend(a.backend, a.model, guided=not a.no_guided)
    load_s = round(time.perf_counter() - load_start, 1)

    sizes = [int(x) for x in a.batch_sizes.split(",")]
    backend.extract(texts[: max(1, min(sizes))])  # warm-up, not timed
    runs = []
    for bs in sizes:
        batches = [texts[i : i + bs] for i in range(0, len(texts), bs)]
        lat, start = [], time.perf_counter()
        for b in batches:
            lat.append(backend.extract(b).latency_ms)
        wall = time.perf_counter() - start
        runs.append(
            {
                "batch_size": bs,
                "batches": len(batches),
                "titles_per_s": round(len(texts) / wall, 2),
                "batch_latency_ms_p50": pct(lat, 0.5),
                "batch_latency_ms_p95": pct(lat, 0.95),
                "ms_per_title": round(wall * 1000 / len(texts), 1),
            }
        )
        print(json.dumps(runs[-1]))

    # Accuracy of exactly what this backend serves
    out = backend.extract(texts) if max(sizes) >= len(texts) else None
    if out is None:
        ex = [
            e
            for i in range(0, len(texts), max(sizes))
            for e in backend.extract(texts[i : i + max(sizes)]).extractions
        ]
    else:
        ex = out.extractions
    rows = [
        {
            "id": t["id"],
            "title": t["title"],
            "pred": e.attributes,
            "error": e.error,
            "strict_json": strict_ok(e.raw),
            "raw": e.raw,
            "model": backend.model,
            "latency_ms": 0.0,
            "usage": {"input_tokens": e.prompt_tokens, "output_tokens": e.output_tokens},
        }
        for t, e in zip(titles, ex, strict=True)
    ]
    run_name = f"qwen2.5-1.5b-lora-{a.name}"
    preds_path = PREDS / a.split / f"{run_name}.jsonl"
    write_jsonl(preds_path, rows)
    rep = evaluate(rows, load_labels(a.split))
    summary = {
        "name": a.name,
        "backend": a.backend,
        "model": backend.model,
        "guided": a.backend == "vllm" and not a.no_guided,
        "device": device(),
        "split": a.split,
        "n_titles": len(texts),
        "model_load_s": load_s,
        "runs": runs,
        "metrics": rep.summary(),
        "strict_json": round(sum(r["strict_json"] for r in rows) / len(rows), 3),
        "report": asdict(rep),
    }
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / f"{a.name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    preds_path.with_suffix(".report.json").write_text(
        json.dumps(
            {"split": a.split, "model": backend.model, "metrics": rep.summary(), "report": asdict(rep)},
            indent=1,
        ),
        encoding="utf-8",
    )
    print(format_report(rep, f"{run_name} on {a.split} ({summary['device']})"))


if __name__ == "__main__":
    main()
