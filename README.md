# Retail Product Attribute Extractor — Fine-Tuned SLM

LoRA-fine-tuned small language model that extracts structured attributes (category, brand, size, material) from noisy
retail product titles into schema-validated JSON, benchmarked against a frontier LLM and served with vLLM.

> 🚧 Work in progress. Done so far: data, labels, LLM layer, baseline, QLoRA training code and Colab notebook. Next: the T4 run and evaluation, then serving. See `docs/BUILD_LOG.md`.

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

## Results so far (val, n = 50, small, so read ±0.07)

| model | prompt | exact match | category acc | brand F1 | size F1 | material F1 | JSON valid | p50 latency |
|---|---|---|---|---|---|---|---|---|
| qwen2.5:7b zero-shot (Ollama, CPU) | full guidelines | 0.46 | 0.68 | 0.89 | 0.57 | 0.70 | 1.00 | 10.6 s |
| Qwen2.5-1.5B zero-shot | short | _pending T4 run_ | | | | | | |
| Qwen2.5-1.5B + QLoRA | short | _pending T4 run_ | | | | | | |

## Training
- **Method:** QLoRA on Qwen2.5-1.5B-Instruct with a 4-bit NF4 base and LoRA r=16 on all projections. The loss covers only the JSON answer. Details are in `src/pslm/train/train.py`.
- **Where to run it:** on a free T4 via [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb). The notebook trains, evaluates the fine-tuned and zero-shot models on val/test, and packages the results.
- **Locally:** the CPU path is a smoke test only.
```bash
uv sync --extra train
uv run python -m pslm.train.format
uv run python -m pslm.train.train --base trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --max-steps 2 --out outputs/smoke
```

## Usage
```bash
uv sync
uv run pytest                                                            # offline, no key
uv run python -m pslm.distill.label val --provider fake --model qwen2.5:7b  # replay the baseline
uv run python -m pslm.distill.label val --provider ollama                # live, local model
uv run python -m pslm.distill.batch estimate train --model claude-haiku-5-5  # Batch API cost estimate
```
`LLM_PROVIDER=anthropic` (with `ANTHROPIC_API_KEY`) runs the Claude teacher, sync or through the Batch API. See `.env.example`.
