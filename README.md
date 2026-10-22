# Retail Product Attribute Extractor — Fine-Tuned SLM

LoRA-fine-tuned small language model that extracts structured attributes (category, brand, size, material) from noisy
retail product titles into schema-validated JSON. It's distilled from Claude labels and benchmarked against a 5x
larger prompted LLM.

> **Frontier comparison:** the code to benchmark Claude Haiku/Sonnet on the same test set exists (`--provider anthropic`). It hasn't been run, because this project was built without API credits. With a key, it's one command per model, at cents for 150 titles.

> Done: data, labels, LLM layer, QLoRA training on a T4, evaluation with a CI gate, and serving (FastAPI on vLLM with JSON-schema-constrained decoding, plus Docker). See `docs/BUILD_LOG.md` for the full story.

```
"Tal 64OZ Ranger Pro Stainless Steel Vacuum Insulated Water Bottle (Black)"
  → {"category": "home_kitchen", "brand": "Tal", "size": "64OZ", "material": "stainless steel"}
```

## Data
- **Titles:** 800 real titles from [Amazon Products 2023](https://huggingface.co/datasets/milistu/AMAZON-Products-2023), split 600 train / 50 val / 150 test. The sample is deduplicated, capped per category and deterministic.
- **Labels:** distilled from Claude following the written guidelines in [`docs/LABELING.md`](docs/LABELING.md).
  - Brand and size must be copied verbatim from the title, and CI enforces this.
  - The test set got a second review pass against the guidelines. The reviewer was the same model that labeled, not a human (2 of 150 labels corrected; see `docs/CHALLENGES.md` #9).
- **Schema:** [`src/pslm/schema.py`](src/pslm/schema.py) has 23 closed categories and nullable brand/size/material. The same JSON schema drives Claude structured outputs, Ollama and vLLM guided decoding.

## Results (test, n = 150 held-out titles)

| model | prompt | exact match | category acc | brand F1 | size F1 | material F1 | JSON valid (strict) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B zero-shot | short (~150 tok) | 0.207 | 0.560 | 0.773 | 0.584 | 0.452 | 0.92 (0.00) |
| qwen2.5:7b zero-shot (Ollama, Q4) | full guidelines (~1.7k tok) | 0.567 | 0.773 | 0.927 | 0.697 | 0.773 | 1.00 (1.00)* |
| **Qwen2.5-1.5B + QLoRA** | short (~150 tok) | **0.707** | **0.813** | **0.933** | **0.878** | **0.843** | **1.00 (1.00)** |

\* Ollama decoding is constrained to the JSON schema. The fine-tuned model's 100% strict validity is unconstrained: it learned the format.

**Reading it honestly:**
- **Fine-tuning works.** Same base model, same short prompt, adapter added: exact match +0.50, from 0.21 to 0.71. The paired 95% bootstrap CI is [+0.42, +0.58], and the gain is significant on every field. The untuned model wraps every answer in prose or fences and invents categories (12 invalid outputs).
- **The fine-tuned 1.5B beats a 5x larger model given the full guidelines.** Exact match is +0.14 [+0.05, +0.23] and size +0.09, both significant. Brand, category and material are statistically tied. It uses a 10x shorter prompt and has no guideline text to maintain at inference.
- **Two inference engines agree.** The fixed HF path (CPU, fp32) and vLLM (T4, fp16) give identical predictions on 149 of 150 test titles. Reproduce the comparisons with `python -m pslm.eval.compare`.
- **A decoding bug was found and fixed along the way.** HF `generate()` had inherited Qwen's shipped `repetition_penalty=1.1`, which penalized copying brands and sizes from the title. It hid 7 points of exact match until the serving benchmark ran the same adapter on vLLM ([`docs/CHALLENGES.md`](docs/CHALLENGES.md) #17).
- **What "accuracy" means here:** labels are Claude's, following [`docs/LABELING.md`](docs/LABELING.md). The test set was re-checked by the same model, not a human. These numbers measure **agreement with the teacher**, which is what distillation optimizes.
- **Remaining failure modes** ([`docs/CHALLENGES.md`](docs/CHALLENGES.md) #13):
  - category confusions (`sports_fitness` as a catch-all; long-tail classes with 0–1 train examples);
  - materials taken from finishes ("18K Gold Plated");
  - span boundaries (`Junior Size`).
- **Training cost:** QLoRA on a free Colab T4 took 447 s and trained 18.5M params (1.2%). CI replays every recorded run and **gates** the fine-tuned model's metrics against [`data/eval/baseline.json`](data/eval/baseline.json).

### Serving throughput (one T4, vLLM 0.31.0, 150 test titles)

| batch size | 1 | 8 | 32 | 150 |
|---|---|---|---|---|
| vLLM + JSON-schema-constrained decoding: titles/s | 0.94 | 5.9 | 16.7 | **39.2** |
| batch latency p50 | 1.03 s | 1.32 s | 1.76 s | 3.82 s |

- Plain transformers, one title at a time on the same GPU, does **0.87 titles/s**, so batching gives ~45x.
- Constrained decoding produced **byte-identical outputs** to unconstrained on all 150 titles at the same throughput. Its only cost is startup (262 s vs 106 s model load), and it guarantees schema-valid output for any input. Details: `data/bench/*.json`.

## When to use which

| option | quality (test) | throughput / latency | rough cost per 1,000 titles | operational burden |
|---|---|---|---|---|
| **Fine-tuned 1.5B on vLLM** | exact 0.71 (best measured) | 39 titles/s at batch 150; ~1 s for a single title | **~$0.003** (a ~$0.35/h T4 at 39 titles/s; a free Colab/Kaggle T4 costs $0) | a GPU service; retrain when the taxonomy or rules change |
| **Larger open model, prompted** (qwen2.5:7b) | exact 0.57 | ~13 s/title on a laptop CPU | ~$0 on hardware you already have | no training; output depends on prompt wording (CHALLENGES #8) |
| **Frontier API, prompted** (Claude Haiku / Sonnet 5.5) | not benchmarked here (no API key). The teacher labels come from Claude, so it's the reference the student imitates | network round-trip; no infrastructure | **estimate:** ~$0.15–0.30 (Haiku) and ~$3–6 (Sonnet), from the ~1.7k-token guideline prompt + ~30–250 output tokens at list prices; prompt caching gives the low end, the Batch API halves it | none; pay per call |

The estimates are list-price arithmetic, not measurements. The vLLM numbers come from `data/bench/vllm-guided.json`.

- **High-volume, stable extraction** (catalog enrichment, nightly backfills, millions of titles): use the **fine-tuned SLM**. It's the most accurate option measured here and roughly 100–1000x cheaper per title than a frontier API. It's deterministic for a fixed adapter, and guided decoding guarantees the schema.
- **Labels, a changing taxonomy, or low volume** (bootstrapping, new categories, under ~10k titles/month): use the **frontier API**. There's no training loop, rules change by editing the prompt, and its quality is what the student is distilled from. That's exactly how this project used it: as the teacher.
- **No GPU and no API budget** (prototyping, offline or air-gapped): use a **prompted open model** on CPU. It's free, slower, and less accurate, and a guideline edit can move its scores (CHALLENGES #8).
- **The usual end state is both:** the API labels new data and handles low-confidence or new-category cases, while the fine-tuned model serves the bulk traffic. The labels from the first become the next training set for the second.

## Training
- **Method:** QLoRA on Qwen2.5-1.5B-Instruct with a 4-bit NF4 base and LoRA r=16 on all projections. The loss covers only the JSON answer. Details are in `src/pslm/train/train.py`.
- **Where to run it:** on a free T4 via [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb). The notebook trains, evaluates the fine-tuned and zero-shot models on val/test, and packages the results.
- **Locally:** the CPU path is a smoke test only.
```bash
uv sync --extra train
uv run python -m pslm.train.format
uv run python -m pslm.train.train --base trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --max-steps 2 --out outputs/smoke
```

## Serving
`POST /extract` takes a list of titles and returns one result per title, in order. A title that fails gets `attributes: null` plus an `error`; it never fails the batch.
```bash
curl -X POST localhost:8000/extract -H 'Content-Type: application/json' \
  -d '{"titles": ["Tal 64OZ Ranger Pro Stainless Steel Vacuum Insulated Water Bottle - Leak-Proof Double Walled Thermos w/Carry Loop 80 Hrs Cold, 26 Hot Reusable Metal holds 8 cups (WM2584) (Black)"]}'
# {"results": [{"title": "...", "attributes": {"category": "home_kitchen", "brand": "Tal", "size": "64OZ",
#   "material": "stainless steel"}, "error": null}], "model": "...", "latency_ms": ...}
```
There are three backends (`SERVE_BACKEND`), all sharing the same prompt and the same output parser as evaluation:

| backend | what | where |
|---|---|---|
| `vllm` | base + LoRA adapter on vLLM; **decoding constrained to the JSON schema**; the whole request in one batched `generate` | `Dockerfile.gpu` (`vllm/vllm-openai:v0.31.0`), GPU |
| `hf` | plain transformers with the adapter merged; sequential | CPU fallback (`docker build --build-arg WITH_HF=1 .`) |
| `replay` | serves the fine-tuned model's recorded outputs | default `Dockerfile`; CI smoke test, no model needed |

```bash
docker build -t product-slm . && docker run -p 8000:8000 product-slm      # replay demo
docker build -f Dockerfile.gpu -t product-slm:gpu .
docker run --gpus all -p 8000:8000 -v "$PWD/outputs/qwen2.5-1.5b-lora/adapter:/models/adapter:ro" product-slm:gpu
```
- `GET /health` reports the backend and model.
- `GET /metrics` reports latency p50/p95, throughput and errors by type.
- The throughput table above comes from `notebooks/serve_bench_colab.ipynb`.

## Usage
```bash
uv sync
uv run pytest                                                            # offline, no key
uv run python -m pslm.distill.label val --provider fake --model qwen2.5:7b  # replay the baseline
uv run python -m pslm.distill.label val --provider ollama                # live, local model
uv run python -m pslm.distill.batch estimate train --model claude-haiku-5-5  # Batch API cost estimate
```
`LLM_PROVIDER=anthropic` (with `ANTHROPIC_API_KEY`) runs the Claude teacher, sync or through the Batch API. See `.env.example`.
