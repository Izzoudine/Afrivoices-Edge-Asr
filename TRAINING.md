# Training the winning model

The winning acoustic model was **fine-tuned, not trained from scratch, and not trained in a notebook.** It uses the fairseq2 wav2vec2/ASR recipe from Meta's [omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr) package (Apache-2.0), driven entirely by a YAML config — so the "training script" is the config plus a one-line launch command.

## 0. Environment

```bash
# GPU box, CUDA 12.x. Install order matters (fairseq2 pins numpy~=1.23).
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install fairseq2==0.6.* --extra-index-url \
    https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu128
pip install --no-deps git+https://github.com/facebookresearch/omnilingual-asr@0.1.0
pip install pyctcdecode kenlm jiwer soundfile librosa pyarrow "numpy<2"
# On notebook environments (Colab), restart the runtime after this.
```

Inference/decoding additionally needs the packages above; the edge runtime needs only `onnxruntime` + `pyctcdecode` + `kenlm` (see the hardware validation report).

> **Note on paths.** `configs/runB.yaml` and `configs/omni_v2.yaml` contain absolute paths from our training machine (`/scratch/...`) and two `EDIT:`-marked lines. Point them at your own data directory and at the checkpoint downloaded from Hugging Face before launching.

## 1. Base model

[omniASR-CTC-1B v2](https://github.com/facebookresearch/omnilingual-asr) (Meta, Apache-2.0) — 971 M parameters, character-level CTC, vocab 10,288. Pulled via the asset card in [`configs/omni_v2.yaml`](configs/omni_v2.yaml).

## 2. Data

- ~1,035 h of deduplicated 6-language speech, stored as FLAC-in-Parquet (16 kHz mono), assembled with per-language sampling weights ([`configs/summary_v2.tsv`](configs/summary_v2.tsv)): kln 25 / mas 22 / som 16 / luo 18 / kik 11 / swh 4 — biased toward the hardest languages because the metric is an unweighted mean.
- Extraction, encoding repair, normalization and dedup are all in one script, [`code/prep_shard.py`](code/prep_shard.py):
  - `demojibake()` repairs the double-encoded Kikuyu/Somali text (cp1252→UTF-8 round-trip + a fallback pair map) **before** normalization;
  - `norm()` lowercases, strips punctuation / `[cs]`-`[p]` tags / digits and collapses whitespace — the **same** normalization at train and eval time (a mismatch here collapses fine-tuning to ~100 % WER);
  - rows are deduplicated by `(language, normalized text, audio length)`; audio is resampled to 16 kHz (`soxr_hq`) and stored as FLAC-in-Parquet.
- Leak handling: [`code/purify.py`](code/purify.py) removes the `dev_test` transcriptions (which overlap the Kaggle test set) from the LM corpora, and clean language models are rebuilt with [`code/build_pure_lms.sh`](code/build_pure_lms.sh). See [`TRANSPARENCY_NOTE.md`](TRANSPARENCY_NOTE.md) and [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the full findings.

## 3. Recipe

Full config: [`configs/runB.yaml`](configs/runB.yaml). Key hyperparameters:

| | |
|---|---|
| Steps | 14,000 (~39 h on one NVIDIA L40S) |
| Learning rate | 1e-5, tri-stage schedule |
| Effective batch | grad-accumulation ×32, ~1.28 M elements (~43 min audio/step) |
| Encoder freeze | first 400 steps |
| SpecAugment | p=0.5, freq-mask 27, time-mask 70 |
| Precision | bfloat16, grad-clip 1000 |
| Seeds | 2026 (model init + data sampling) |
| Checkpointing | every 250 steps, keep last 8 |

## 4. Launch

```bash
bash code/train.sh    # see the script for the exact fairseq2 command
```

## 5. Checkpoint selection

The recipe's internal "best" metric is **not** used — the provided Kikuyu validation references were corrupted (mojibake), which penalizes any model that writes correct Kikuyu. Selection is done externally with [`code/eval_dev.py`](code/eval_dev.py) on a clean protocol (5 standard-dev languages + a re-extracted, demojibake'd Kikuyu dev). The winning checkpoint was **step 12,750** — not the final step.

## Earlier experiments (not the winning model)

Baseline experiments with w2v-bert-2.0 and MMS (Jupyter notebooks) were run earlier in the project and superseded by the omniASR fine-tune above. They are not part of the winning pipeline and are kept in the authors' working directory rather than this deliverable.
