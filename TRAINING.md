# Training the winning model

The winning acoustic model was **fine-tuned, not trained from scratch, and not trained in a notebook.** It uses the fairseq2 wav2vec2/ASR recipe from Meta's [omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr) package (Apache-2.0), driven entirely by a YAML config — so the "training script" is the config plus a one-line launch command.

## 1. Base model

[omniASR-CTC-1B v2](https://github.com/facebookresearch/omnilingual-asr) (Meta, Apache-2.0) — 971 M parameters, character-level CTC, vocab 10,288. Pulled via the asset card in [`configs/omni_v2.yaml`](configs/omni_v2.yaml).

## 2. Data

- ~1,035 h of deduplicated 6-language speech, stored as FLAC-in-Parquet (16 kHz mono), assembled with per-language sampling weights ([`configs/summary_v2.tsv`](configs/summary_v2.tsv)): kln 25 / mas 22 / som 16 / luo 18 / kik 11 / swh 4 — biased toward the hardest languages because the metric is an unweighted mean.
- Extraction & demojibake repair: [`code/prep_shard.py`](code/prep_shard.py) (repairs the double-encoded Kikuyu/Somali text **before** normalization).
- See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the full data pipeline and the deduplication / leak findings.

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
