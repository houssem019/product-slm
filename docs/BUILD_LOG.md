# Build Log

Chronological record of what was built, **why**, and which alternatives were rejected.
Read this before an interview to retell the story of the project.

| Step | What | Why / alternatives considered |
|------|------|-------------------------------|
| 1. Schema | `src/pslm/schema.py`: Pydantic `ProductAttributes(category, brand, size, material)`. Brand, size and material are nullable. `category` is a closed `StrEnum` of 23 top-level classes. `extra="forbid"`. | **One schema serves teacher, student, guided decoding and eval**, so "valid output" means the same thing at every stage. **Closed category set:** free-text categories make exact match meaningless ("Kitchen" vs "Home & Kitchen") and let the student invent labels. Amazon's own `main_category` was rejected: it's null for ~25% of rows and often wrong (a "baby decision coin" filed under Office Products). |
| 2. Labeling guidelines | `docs/LABELING.md`: title-only, copy brand/size verbatim, conservative nulls, many explicit edge-case rules. | **Distillation quality is capped by label consistency.** Without rules, the same teacher labels "Wooden" one time and "wood" the next, and the student learns noise. **Verbatim spans make labels checkable:** a brand or size not found in the title is a guaranteed error. Rules were added *while labeling the test set*, as real ambiguities came up (vacuums, plated jewelry, compatibility sizes). That's normal annotation-guideline iteration. |
| 3. Data source | Amazon Products 2023 (`milistu/AMAZON-Products-2023` on HF, 117k products, derived from McAuley-Lab Amazon Reviews 2023). DuckDB reads three columns from the remote parquet; the cache in `data/raw/` is 9.8 MB. | **Real, noisy marketplace titles:** keyword stuffing, compatibility lists, mixed units. That's exactly the target domain. The dataset is ~1.7 GB because of precomputed embeddings, but **parquet column projection over HTTP** pulls only what we need. Rejected: WDC product corpus (huge, multilingual, needs heavy filtering); the full McAuley meta dumps (tens of GB). |
| 4. Sampling | `pslm.data.sample`: dedupe on lowercased title; ASCII-only, 15–220 chars; deterministic hash order (`hash(id ‖ seed)`); no source category above 20% of the sample. Splits: 150 test / 50 val / 600 train, committed as JSONL. | **The cap stops the sample being mostly Amazon Home** (~35% of the source). **Hash order is a reproducible shuffle without RNG state.** Test is taken first, so growing train later never leaks a test title. **Committed splits** pin labels to exact titles, and CI needs no network. |
| 5. Labels (teacher) | All 800 titles labeled by **Claude (Opus 5.5) inside the Claude Code session**, following LABELING.md. Written as compact TSV, imported to `data/labels/<split>.jsonl` with `labeler` provenance and a `reviewed` flag. | **No API key on this machine**, and the plan budgets ~$0. An in-session teacher is the same model family a Batch API job would use, at no cost. An API batch labeler (`distill/`) is still planned for scaling to 5–10k titles. **TSV** keeps hand/LLM writing compact (5 fields per line); **JSONL** is the source of truth. |
| 6. Automated label QA | `pslm.data.labels check`: schema validity, id/title alignment, and **grounding**: brand and size must appear in the title (case- and space-insensitive). All 800 pass. | **A cheap check that catches hallucinated attributes**, the most common LLM-labeling error. It runs as a pytest on the committed labels, so a bad edit fails CI. It can't check category or material semantics; that's what human review is for. |
| 7. Human review tool | `pslm.data.review test`: Enter = accept, `e` = edit per field with validation, saves after every record. Keeps the teacher's original under `teacher_label`. | **The test set is gold only after a human pass**, otherwise we'd be measuring agreement with the teacher, not accuracy. Keeping `teacher_label` lets us report the **teacher's own per-field error rate**, which bounds what distillation can achieve. |
| 8. Pluggable LLM layer | `pslm.llm`: one `complete(system, user, json_schema)` interface. Providers: `anthropic` (official SDK), `ollama` (local), `fake` (cassette replay), plus `ScriptedLLM` for unit tests. | Same pattern as the sibling repos. **CI and tests never need a key or network.** Rejected: LangChain/LiteLLM wrappers. One call shape doesn't need a framework, and the abstraction would hide the provider features we rely on (structured outputs, caching, fallbacks). |
| 9. Constrained JSON everywhere | `output_schema()`: a flat hand-written JSON schema used by Claude structured outputs (`output_config.format`), Ollama `format`, and later vLLM guided decoding. One `parse_output` reports `not_json`/`invalid_schema`/`refusal`/`truncated`. | **JSON validity should come from decoding, not from hoping.** A shared parser means "valid" is defined once for teacher, baselines and student. Rejected: Pydantic's generated schema (`$defs`, `anyOf`: CHALLENGES #6); tool-use-as-JSON (structured outputs is the direct feature). |
| 10. Two prompts | `teacher_system()` = the full `docs/LABELING.md` + 4 train-only worked examples (~1.5k tokens, cached). `STUDENT_SYSTEM` = one line + the category list. | The teacher needs the rules; the student learns them from data, and short prompts mean shorter training sequences and faster inference. The guideline doc is **embedded at runtime**, not copied, so there's one source of truth, and editing it invalidates cassettes (by design). |
| 11. Sync labeler / baseline runner | `pslm.distill.label <split> --provider ...`: predicts, writes `data/preds/<split>/<model>.jsonl`, and scores against the reference labels. Also reports strict JSON rate, latency p50/p95 and cost per 1k titles. | One runner for "label with the teacher" and "zero-shot baseline". The baseline comparison table later reuses exactly this output. |
| 12. Batch API teacher | `pslm.distill.batch estimate|submit|collect`: same `request_params` as sync (minus refusal fallbacks, which batches reject). `custom_id` = product id; results are re-ordered by id. | **50% cheaper and asynchronous**, the right tool for one-off bulk labeling. `estimate` gives offline cost: 600 titles cost ~$1.6 on Opus 5.5 and ~$0.04 on Haiku 5.5. Implemented and unit-tested with fake result objects. **Not run: no API key** on this machine; the committed labels come from the in-session teacher. |
| 13. Metrics | `pslm.eval.metrics.score`: per-field accuracy; extraction P/R/F1 for brand/size/material; all-fields exact match; per-category accuracy; invalid-output breakdown. | Accuracy alone flatters a model that always says null (~74% of titles have no size). P/R/F1 with non-null as the positive class shows whether it actually extracts. An undefined F1 is `None`, not 0 (CHALLENGES #7). |

## Label statistics (teacher labels, before human review)

| split | n | brand non-null | size non-null | material non-null |
|---|---|---|---|---|
| train | 600 | 61% | 26% | 29% |
| val | 50 | 74% | 22% | 24% |
| test | 150 | 66% | 28% | 27% |

Top categories in train: apparel 97, electronics 70, automotive 48, home_kitchen 45, home_decor 42. Long tail: `baby` 0, `grocery` 1, `toys_games` 4 (see CHALLENGES #2).

## First baseline: zero-shot qwen2.5:7b (Ollama, CPU), val split

These are measured on the 50 **val** titles against the in-session teacher labels (not yet human-reviewed). The prompt is the teacher prompt with Ollama's schema-constrained decoding. `data/preds/val/qwen2.5-7b.report.json` has the full numbers, and CI replays it from `data/cassettes/qwen2.5-7b.json`.

| metric | value |
|---|---|
| JSON valid (lenient / strict) | 1.00 / 1.00 (constrained decoding) |
| exact match (all 4 fields) | 0.40 |
| category accuracy | 0.68 |
| brand acc / F1 | 0.86 / 0.873 |
| size acc / F1 | 0.80 / 0.571 (precision 0.47) |
| material acc / F1 | 0.86 / 0.640 |
| latency p50 / p95 | 9.8 s / 13.2 s per title (CPU, 1.7k-token prompt) |

**Error analysis:** most misses break a written rule the model was given.
- **Category:**
  - shoes labeled `apparel` (3 of 4 footwear titles);
  - furniture labeled `home_kitchen`;
  - sunglasses labeled `apparel`.
- **Size:**
  - pack counts ("3-Pack", "28 Strips", "140 tablets");
  - the compatible phone's screen size;
  - named sizes like "Small"/"Preseason".
- **Material:** fabric specs like "600D" were taken as the material.
- **Brand:** one hallucinated brand ("Dannxe" for "Danxee"), the failure the grounding check exists for.

Rule-following on a fixed domain is exactly what fine-tuning teaches, so this is the bar for the student to beat. Caveat: n = 50, so ±0.07 on any rate is noise.
