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

## Label statistics (teacher labels, before human review)

| split | n | brand non-null | size non-null | material non-null |
|---|---|---|---|---|
| train | 600 | 61% | 26% | 29% |
| val | 50 | 74% | 22% | 24% |
| test | 150 | 66% | 28% | 27% |

Top categories in train: apparel 97, electronics 70, automotive 48, home_kitchen 45, home_decor 42. Long tail: `baby` 0, `grocery` 1, `toys_games` 4 (see CHALLENGES #2).
