# Challenges & Lessons

Real problems hit while building, how they were solved, and known limitations.
Format: **Problem → Root cause → Fix → What I'd do at scale.**

## 1. The source dataset is 1.7 GB, but we need ~10 MB of it
- **Problem:** `milistu/AMAZON-Products-2023` ships 4 parquet shards of ~480 MB each. The datasets-server `/rows` API returned errors for it, so sampling through the API wasn't an option.
- **Root cause:** each row carries a precomputed embedding vector, which dominates the file size.
- **Fix:** DuckDB `httpfs` reads `read_parquet([...])` with only `parent_asin, title, main_category` selected. Parquet is columnar, so DuckDB fetches just those column chunks with HTTP range requests. It took ~25 s for 117k rows and wrote a 9.8 MB local cache.
- **At scale:** the same pattern works straight from S3/GCS. Push filters into the query too, so row-group statistics skip whole chunks.

## 2. Long-tail categories have almost no examples
- **Problem:** with 600 train titles, `baby` has 0 examples, `grocery` 1, `toys_games` 4. The student will rarely or never predict them.
- **Root cause:** the source is skewed (fashion and home dominate), and the per-stratum cap limits the head but can't create a tail.
- **Fix (current):** documented, and per-category results will be reported, not only micro-averages.
- **At scale:** stratify on the *teacher's* category (label a big pool cheaply, then sample per predicted class), or add targeted synthetic titles for rare classes.

## 3. Labeling rules only appear once you label
- **Problem:** the first guideline draft didn't settle common cases: is a vacuum `home_kitchen` or `electronics`? Is "18K Gold Plated" a material? Is "Wooden" `wood`? Is "fits 42 inch deck" a size?
- **Root cause:** ambiguity is a property of the data. You only find it by reading lots of real titles.
- **Fix:** each ambiguity found while labeling the test set became an explicit rule in `docs/LABELING.md`, before the train split was labeled.
- **At scale:** pilot on ~100 items with two annotators, measure agreement (Cohen's κ per field), fix the guidelines, then label in bulk. Put the guideline text in the teacher prompt.

## 4. Was the teacher hallucinating attributes?
- **Problem:** LLM labelers sometimes "know" a brand that isn't in the text (e.g. using the store name), or normalize a size into a string that never appears.
- **Fix:** a grounding check makes brand and size verbatim spans that must appear in the title (case- and whitespace-insensitive). It runs in pytest on every committed label. 800/800 pass.
- **Limitation:** the check can't validate category or material. The human review of the test set covers those, and `teacher_label` keeps the teacher's original for error-rate measurement.

## 5. Few-shot examples nearly leaked the test set
- **Problem:** the first teacher prompt included four worked examples. I picked them from memory of the titles I had labeled, and all four were **test** titles. Any model prompted with them would have been graded on answers it was shown.
- **Root cause:** the test split was labeled first, so its titles were the most familiar. Nothing checked where prompt examples came from.
- **Fix:** I replaced them with verbatim train titles. `test_few_shot_examples_come_from_train_only` now asserts every example is in train and absent from val/test, so this can't silently come back.
- **At scale:** treat everything that reaches a prompt (examples, retrieved neighbours, RAG context) as training data, and run leakage checks on it the same way as on the splits.

## 6. Pydantic's JSON schema is not what constrained decoders want
- **Problem:** `ProductAttributes.model_json_schema()` emits `$defs`/`$ref` for the enum and `anyOf: [{type: string}, {type: null}]` for nullable fields. Constrained decoders (Claude structured outputs, Ollama `format`, vLLM guided JSON) support those constructs unevenly.
- **Fix:** `output_schema()` is a hand-written flat schema:
  - the category enum is inline;
  - nullable fields are `type: ["string", "null"]`;
  - every key is `required`;
  - `additionalProperties: false`.
  
  Pydantic still validates the parsed output, so the two can't drift without a failing test (`test_output_schema_is_flat_and_complete`).
- **Lesson:** "required + nullable" beats "optional". The model must state null explicitly, so a missing key is always an error, never a silent default.

## 7. F1 of a field with no positives is undefined, not 0
- **Problem:** on a 3-title smoke run with no gold materials, the report showed material F1 = 0.000 for a model that correctly predicted null three times.
- **Fix:** when TP + FP + FN = 0, P/R/F1 are `None`, shown as `n/a`. They aren't counted as 0, and they aren't counted as a perfect 1 either.
- **Why it matters:** per-category or small-slice reports hit this constantly. Reporting 0 would make a model look broken on exactly the slices where it did nothing wrong.
