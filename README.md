# Retail Product Attribute Extractor — Fine-Tuned SLM

LoRA-fine-tuned small language model that extracts structured attributes (category, brand, size, material) from noisy
retail product titles into schema-validated JSON, benchmarked against a frontier LLM and served with vLLM.

> 🚧 Work in progress. Done so far: data, labels, LLM layer, baseline. Next: QLoRA training, eval, serving. See `docs/BUILD_LOG.md`.

```
"Tal 64OZ Ranger Pro Stainless Steel Vacuum Insulated Water Bottle (Black)"
  → {"category": "home_kitchen", "brand": "Tal", "size": "64OZ", "material": "stainless steel"}
```

## Data
- **Titles:** 800 real titles from [Amazon Products 2023](https://huggingface.co/datasets/milistu/AMAZON-Products-2023), split 600 train / 50 val / 150 test. The sample is deduplicated, capped per category and deterministic.
- **Labels:** distilled from Claude following the written guidelines in [`docs/LABELING.md`](docs/LABELING.md).
  - Brand and size must be copied verbatim from the title, and CI enforces this.
  - The test set is human-reviewed before it's used as gold.
- **Schema:** [`src/pslm/schema.py`](src/pslm/schema.py) has 23 closed categories and nullable brand/size/material. The same JSON schema drives Claude structured outputs, Ollama and vLLM guided decoding.

## Results so far (val, n = 50, small, so read ±0.07)

| model | prompt | exact match | category acc | brand F1 | size F1 | material F1 | JSON valid | p50 latency |
|---|---|---|---|---|---|---|---|---|
| qwen2.5:7b zero-shot (Ollama, CPU) | full guidelines | 0.40 | 0.68 | 0.87 | 0.57 | 0.64 | 1.00 | 9.8 s |
| fine-tuned SLM | short | _next milestone_ | | | | | | |

## Usage
```bash
uv sync
uv run pytest                                                            # offline, no key
uv run python -m pslm.distill.label val --provider fake --model qwen2.5:7b  # replay the baseline
uv run python -m pslm.distill.label val --provider ollama                # live, local model
uv run python -m pslm.distill.batch estimate train --model claude-haiku-5-5  # Batch API cost estimate
```
`LLM_PROVIDER=anthropic` (with `ANTHROPIC_API_KEY`) runs the Claude teacher, sync or through the Batch API. See `.env.example`.
