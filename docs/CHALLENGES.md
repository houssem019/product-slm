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

## 13. What the fine-tuned model still gets wrong (test, n=150, exact match 0.63)
The full list comes from comparing `data/preds/test/qwen2.5-1.5b-lora.jsonl` against the labels.

- **Brand recall 0.82 vs precision 0.96.** It misses brands that look like ordinary words or are very short ("My Spoon", "Sheer Pots", "Wild Bobby", "DC", "CL", "Spectra", "Hallmark"), and cuts multi-word brands short ("Wolf" for "Wolf Spring").
  - **Cause:** 600 examples teach "the first capitalized token is the brand". They don't teach that "Sheer Pots" is a name and not a description.
  - **Fix:** more examples where the brand is a dictionary word. Or accept it: missing a brand is cheaper downstream than inventing one.
- **`sports_fitness` became the default category** for 8 of 28 category errors: a flashlight, a motorbike rack, a Garmin watch case, softball pants. Anything outdoor- or sport-flavoured ends up there.
  - **Fix:** per-category error analysis on val, then targeted examples for the confused pairs (electronics ↔ sports_fitness).
- **Long-tail categories fail as predicted (#2).** `baby` has 0 train examples, so the breast pump went to `health_household`. Grocery (1 example) became `home_kitchen`, and the album (`other`) became `toys_games` ("Toys In The Attic").
  - **The model can't predict a class it has never seen.** The fix is data, not tuning.
- **Laptop screen sizes missed** (14", 15.6"). It likely over-generalized "a compatible device's screen size isn't a size", which phone-case titles teach repeatedly.
  - **Lesson:** a rule with a strong surface pattern ("inch" near a device name) gets learned more broadly than intended.
- **Span-boundary slips:** `(S)` instead of `S`, `Junior Size` instead of `Junior`, `B 11 inches` instead of `11 inches`. The right span, with extra characters.
  - **Cheap fix:** a post-processing normalizer (strip brackets, a trailing "Size", a variant letter). That's deterministic code, not a model change.
- **Material taken from finishes and secondary parts:** "18k gold plated", "brass" from "Brass Finish", "satin" from "Satin Lined", "tri blend". These are the subtlest rules in LABELING.md, and each appears in only a handful of train titles.
- **Some golds are arguable.** A brass compass labeled `sports_fitness`, a basketball poster labeled `home_decor`, and #146 (`Large`, changed in review) where the model kept the original `32x64in`.
  - **Takeaway:** a few percent of "errors" at this level are label ambiguity, so 0.63 exact match is a conservative number.

## 14. Epoch 3 overfit (mildly)
- **Observed:** train loss went 0.40 → 0.055 → 0.027 → 0.010 by the end of each epoch. Val loss went 0.0536 → **0.0477** → 0.0493. The best val loss was after epoch 2; epoch 3 cut train loss ~3x while val loss rose.
- **Why it's only mild:** task metrics are still strong (val exact 0.64). The val rise is small, and val is only 50 titles.
- **What I'd do:** retrain with `--epochs 2`, or keep the best-val checkpoint (`load_best_model_at_end` with `save_strategy="epoch"`). Pick by val loss, and confirm on test once. I did **not** re-tune after seeing test results; doing so would make the test number optimistic.

## 15. A slow network made "just benchmark it on CPU" an hour-long download
- **Problem:** the local CPU benchmark of the real fine-tuned model hung for 18 minutes. The 3.1 GB base model (`Qwen2.5-1.5B-Instruct`) was downloading through Hugging Face's Xet client, which had throttled itself to one connection ("connection struggling"). Its cache held 1.7 MB.
- **Diagnosis:** the process was at 100% CPU, which looked like work but was retry/backoff. A `.incomplete` blob sat stuck at 43 KB, and the Xet log showed the cause. Forcing plain HTTP (`HF_HUB_DISABLE_XET=1`) wasn't the fix: it got 231 MB in 5 minutes. The link itself was slow (~0.8 MB/s).
- **Decision:** I skipped the local CPU benchmark of the real model. The CPU code path is already smoke-tested on a tiny model, and the T4 numbers are what matter for serving.
- **Lessons:** for a long-running step, monitor progress (bytes, steps), not CPU usage. And put big downloads where bandwidth is cheap: Colab's network, or a model baked into the image.
- **At scale:** a model registry or a pre-pulled image layer, so a pod's cold start never depends on the public Hub.

## 16. Serving concurrency: vLLM's offline engine is not thread-safe
- **Problem:** FastAPI runs sync endpoints in a thread pool, so two concurrent `/extract` calls would call `LLM.generate` from two threads at once.
- **Fix:** a lock around the backend. Each request's titles still go to vLLM as one batch, which is where the GPU efficiency comes from for a bulk-enrichment API (callers send lists).
- **Trade-off:** many small concurrent requests queue behind each other instead of being batched together.
- **At scale:** run vLLM's async engine (or `vllm serve --enable-lora` with its OpenAI-compatible API) and make this service a thin client. vLLM's continuous batching then merges concurrent requests on the GPU. Keep the same `output_schema()` in the request's structured-output field.

## 17. A shipped `repetition_penalty` silently cost 7 points of exact match
- **Symptom:** the serving benchmark ran the *same adapter*, greedy, on the same T4 through two backends. vLLM scored exact match **0.707**; HF transformers scored **0.633**. Greedy decoding of the same weights shouldn't differ by more than fp16 noise.
- **Narrowing it down:**
  - **It wasn't guided decoding.** vLLM guided and unguided were byte-identical on all 150 titles.
  - **The two backends differed on only 21 titles.**
  - **Category matched on every one of them.** Every difference was brand or size. In 17 of the 21, HF answered `null` and vLLM copied the span from the title, correctly in 16 cases ("SNACTIV", "Wild Bobby", "DC", "Madden Girl", …).
  - **The paired bootstrap confirms it:** vLLM − HF is +0.073 [+0.027, +0.127] on exact match, +0.053 on brand and +0.047 on size, and 0 on category.
- **Root cause:**
  - Qwen2.5-Instruct's `generation_config.json` ships `repetition_penalty: 1.1` along with sampling settings.
  - HF `generate()` loads that file as its defaults. `do_sample=False` switches off sampling, but **the repetition penalty is a logits processor that still runs under greedy decoding.**
  - It down-weights every token that already appears in the context, and the context includes the title. Copying "Wild Bobby" out of the title is exactly a repetition, so the model was nudged toward `null`.
  - vLLM uses its own `SamplingParams` defaults (`repetition_penalty=1.0`) when you pass explicit params, which is how the discrepancy surfaced.
- **What it had distorted:**
  - **Fine-tuned model:** every evaluation was done with HF, so all its earlier numbers (0.633, brand recall 0.82) were understated.
  - **Error analysis (#13):** "brand recall on word-like names" was mostly this bug, not the model.
  - **Zero-shot 1.5B control:** it was penalized the same way (brand recall 0.14), so the "fine-tuning gain" was mis-measured too.
  - **Comparison with qwen2.5:7b:** that baseline runs through Ollama, which has its own defaults. Comparing it to the HF-measured student had made the student look tied, when the served student is in fact significantly better.
- **Fix:**
  - `hf_client.greedy_config()` builds a fresh `GenerationConfig` (greedy, `repetition_penalty=1.0`, only the EOS/pad/bos ids taken from the shipped file) and replaces the model's.
  - A behavioral test plants Qwen's penalty in a tiny model's config and asserts `generate()` equals a hand-written argmax loop. I confirmed the test **fails without the fix**.
  - The HF runs (fine-tuned and zero-shot) were re-measured with the fix (BUILD_LOG step 29).
- **Lessons:**
  1. **Pin every decoding parameter explicitly.** A model's shipped `generation_config.json` encodes its authors' chat defaults, not your task's. Never inherit it in an evaluation.
  2. **Two independent inference stacks are a powerful test.** The disagreement only appeared because serving used a different engine than eval. A single-stack project would have shipped the lower number and a wrong error analysis.
  3. **Look at *which* outputs differ, not just how many.** "Only copy fields, always toward null" pointed straight at a logits-level cause.
  4. **At scale:** keep the exact decoding config in the eval report, and run a cross-engine consistency check (eval engine vs serving engine on a fixed sample) in CI or at release time.
