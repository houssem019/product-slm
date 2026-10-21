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

## 8. One guideline sentence moved the zero-shot baseline by 6 points
- **Problem:** after I added one clarifying sentence to `docs/LABELING.md` (when Small/Large counts as a size), the qwen2.5:7b val baseline went from exact match 0.40 to 0.46. Brand F1 went 0.873 → 0.889 and material F1 0.640 → 0.696. Model, data and temperature (0) were all unchanged.
- **Root cause:** 3 of 50 titles flipped. A zero-shot model conditions on every token of a 1.5k-token prompt, so any edit perturbs its outputs somewhere, not only on the rule that changed.
- **What it means:**
  1. On n=50, ±0.06 is noise. Compare models on the 150-title test set, and report n.
  2. Prompt sensitivity is itself an argument for fine-tuning. A fine-tuned student with a fixed short prompt doesn't drift when someone edits a guideline doc.
  3. The cassette design caught it: the edit invalidated the recording, CI would have failed, and the number was re-measured instead of silently stale.

## 9. "Human review" was delegated back to the labeler
- **Problem:** the plan called for a human to review the 150 test labels. The owner asked me to do it instead, so the reviewer is the same model that produced the labels.
- **Consequence:** a second pass catches **rule slips**. It found 2: a first-size violation and a licensed-character brand. It can't catch **shared blind spots**: if the labeler misreads a rule, the reviewer misreads it the same way. Agreement between the two passes says little about accuracy.
- **Mitigation:** the provenance is explicit:
  - `reviewer: "claude-opus-5-5 (second pass, same model as the labeler)"` on every record;
  - `teacher_label` keeps the 2 originals;
  - the README says "model-reviewed", not "human-reviewed".
  
  All reported scores are agreement with Claude's guideline-following labels, which is exactly what distillation optimizes for.
- **At scale:** hand a random 10–20% to a human (or a different model family), measure per-field agreement, and adjudicate disagreements into guideline changes.

## 10. CPU training is a smoke test, not a fallback
- **Problem:** the plan had a "CPU fallback" for training. I measured 3 optimizer steps (effective batch 16) of plain LoRA on Qwen2.5-0.5B in fp32 on this laptop CPU.
- **Measured:**
  - **834 s for 3 steps (~278 s per step)**, with 8.8M trainable params (1.75%);
  - a full 3-epoch run (~112 steps) would take **~8.7 hours**, and the 1.5B target ~3x longer;
  - train loss after 3 steps was already 0.62, because the instruct model can almost do the format zero-shot.
- **Root cause:** a backward pass through a 0.5B fp32 model has no GPU and no fast low-precision kernels. bitsandbytes 4-bit needs CUDA, so CPU QLoRA isn't possible at all.
- **Fix:** the CPU path is kept for what it's good at. A 2-step run on a tiny random-weight model with the real Qwen tokenizer checks tokenization, masking, LoRA wiring and saving in seconds. Real training runs on a free T4 (`notebooks/train_colab.ipynb`).
- **At scale:** one GPU-hour on a T4 is free on Colab/Kaggle and costs well under $1 elsewhere. Trying to train on CPU wastes engineer time, which costs far more.

## 11. The notebook broke on Colab's Python 3.13, and failed silently
- **Problem:** the first Colab run failed at `pip install`: `requires a different Python: 3.13.15 not in '<3.13,>=3.11'`. The notebook kept going, and the first visible error came three cells later: `FileNotFoundError: outputs/.../train_summary.json`.
- **Root causes:**
  1. I copied `requires-python = "<3.13"` from the sibling repos, where it was a precaution. Colab had moved to 3.13, and nothing tested the project there.
  2. A failed `!command` in a notebook doesn't stop execution. The error scrolled past, and the symptom showed up far from the cause.
  3. A latent issue: the `train` extra required `torch>=2.14` (the version uv happened to resolve locally). On Colab, pip would have replaced its preinstalled CUDA build with a multi-GB download.
- **Fixes:**
  - `requires-python = ">=3.11,<3.14"`. I verified it: the full test suite and the cassette replay pass on 3.13.
  - CI now runs on both 3.11 and 3.13, so the version Colab ships is exercised on every push.
  - The `train` extra floors are now what the code actually needs: torch ≥ 2.4, transformers ≥ 4.56 (the first version with `dtype=`), peft ≥ 0.13, accelerate ≥ 1.0. pip keeps Colab's own builds.
  - Notebook commands run through a `sh()` helper that raises on a non-zero exit, so the notebook stops at the failing step. The setup cell also checks for a GPU up front and pulls the latest code on reruns.
- **Lesson:** a version floor should be the oldest version you've verified, not whatever your lockfile resolved today. Pins belong in the lockfile, not in `pyproject.toml`. And notebooks need explicit failure handling; their default is to carry on.
- **Side finding:** the first local 3.13 check failed with a `ModuleNotFoundError` inside the anthropic SDK. That was Windows' 260-char path limit: a long temp path plus a long generated module filename. It wasn't a 3.13 problem; re-running from a short path passed.

## 12. An unused preinstalled package broke evaluation (torchao)
- **Problem:** QLoRA training on Colab succeeded, but loading the adapter for evaluation crashed inside `PeftModel.from_pretrained`: `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`.
- **Root cause:**
  - When peft wraps a layer with LoRA, it tries a chain of "dispatchers" (bitsandbytes, torchao, …).
  - The torchao dispatcher checks whether torchao is installed. If an *old* version is present, it raises instead of skipping. Colab preinstalls torchao 0.10.
  - Training never reached that check: its layers were bitsandbytes 4-bit, and the bnb dispatcher claims them first.
  - Evaluation loads the base in plain fp16 so the adapter can be merged, and plain `Linear` layers fall through to the torchao check.
  - Same library, same adapter. Only the *base layer type* differed, so a working training run proved nothing about loading the adapter.
- **Fix:** the notebook runs `pip uninstall -y torchao` after installing. Nothing in this project uses it. In the broken session, the user uninstalled it and re-ran only the evaluation cell. No re-training was needed, because each step runs in a fresh subprocess and the adapter was already on disk.
- **Lessons:**
  1. Hosted notebook images are an uncontrolled environment: hundreds of preinstalled packages at versions you didn't choose.
  2. Smoke-test every distinct load path. Here that's "4-bit base + new LoRA" for training and "fp16 base + saved adapter" for evaluation. Locally, both paths had passed only on CPU, where torchao isn't installed.
  3. Structuring the notebook as independent steps that write artifacts to disk made recovery cheap.
- **At scale:** pin a container image (or a `requirements.lock` for the notebook) and run the train → save → reload → infer loop end-to-end in CI on a tiny model in that same image.
