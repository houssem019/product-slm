# Retail Product Attribute Extractor — Fine-Tuned SLM

LoRA-fine-tuned small language model that extracts structured attributes (category, brand, size, material) from noisy
retail product titles into schema-validated JSON. It's distilled from Claude labels and benchmarked against a 5x
larger prompted LLM.

> **Frontier comparison:** the code to benchmark Claude Haiku/Sonnet on the same test set exists (`--provider anthropic`). It hasn't been run, because this project was built without API credits. With a key, it's one command per model, at cents for 150 titles.

> Done: data, labels, LLM layer, QLoRA training on a T4, evaluation with a CI gate, and serving (FastAPI on vLLM with JSON-schema-constrained decoding, plus Docker). Pending: GPU serving benchmark numbers (`notebooks/serve_bench_colab.ipynb`). See `docs/BUILD_LOG.md`.

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

| model | prompt | exact match | category acc | brand F1 | size F1 | material F1 | JSON valid (strict) | p50 latency |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B zero-shot | short (~150 tok) | 0.107 | 0.533 | 0.248 | 0.622 | 0.505 | 0.89 (0.00) | 1.3 s (T4) |
| qwen2.5:7b zero-shot (Ollama, Q4) | full guidelines (~1.7k tok) | 0.567 | 0.773 | **0.927** | 0.697 | 0.773 | 1.00 (1.00)* | 13.1 s (CPU) |
| **Qwen2.5-1.5B + QLoRA** | short (~150 tok) | **0.633** | **0.813** | 0.884 | **0.763** | **0.854** | **1.00 (1.00)** | 0.95 s (T4) |

\* Ollama decoding is constrained to the JSON schema. The fine-tuned model's 100% strict validity is unconstrained: it learned the format.

**Reading it honestly:**
- **Fine-tuning works.** Same base model, same short prompt, adapter added: exact match goes from 0.11 to 0.63 (paired 95% CI for the gain is [+0.45, +0.61]). The untuned model wraps every answer in prose or fences, and invents categories (16 schema-invalid outputs).
- **The fine-tuned 1.5B *matches* a 5x larger model that is given the full guidelines.** It leads on exact match by +0.067, but the 95% paired-bootstrap CI is [−0.027, +0.160], and no per-field difference is significant at n=150. Reproduce it with `python -m pslm.eval.compare`.
- **The 7b is better on brand recall.** The win for the 1.5B is cost: a 10x shorter prompt and 5x fewer parameters, with no guideline text to maintain at inference.
- **What "accuracy" means here:** labels are Claude's, following [`docs/LABELING.md`](docs/LABELING.md). The test set was re-checked by the same model, not a human. So these numbers measure **agreement with the teacher**, which is what distillation optimizes.
- **Latencies** are from different hardware and plain unbatched `generate`. Treat them as indicative; serving throughput comes next with vLLM.
- **Failure modes** are in [`docs/CHALLENGES.md`](docs/CHALLENGES.md) #13: brand recall on word-like names, `sports_fitness` as a catch-all, long-tail classes with 0–1 train examples, and span boundaries like `(S)`.
- **Training cost:** QLoRA on a free Colab T4 took 447 s and trained 18.5M params (1.2%). CI replays every recorded run and **gates** the fine-tuned model's metrics against [`data/eval/baseline.json`](data/eval/baseline.json).

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
- Throughput and latency on a T4 come from `notebooks/serve_bench_colab.ipynb`: vLLM guided vs unguided vs plain transformers, at batch sizes 1/8/32/150, with accuracy measured for each.

## Usage
```bash
uv sync
uv run pytest                                                            # offline, no key
uv run python -m pslm.distill.label val --provider fake --model qwen2.5:7b  # replay the baseline
uv run python -m pslm.distill.label val --provider ollama                # live, local model
uv run python -m pslm.distill.batch estimate train --model claude-haiku-5-5  # Batch API cost estimate
```
`LLM_PROVIDER=anthropic` (with `ANTHROPIC_API_KEY`) runs the Claude teacher, sync or through the Batch API. See `.env.example`.
