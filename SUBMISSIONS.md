# Submission ledger

Traceability of our selected Kaggle submissions — each maps to an exact checkpoint + decode configuration in this repository. All use the **same single acoustic model** (971 M params) and the purified per-language KenLMs on Hugging Face.

| Public score | Date (2026) | Acoustic checkpoint | Decode params | Notes |
|---|---|---|---|---|
| **0.34778** | Jul 15 | `step_12750` (runB) | final params + **whole-clip decoding for the 4,500 clips > 38 s** ([`code/omni_kenlm_whole.py`](code/omni_kenlm_whole.py) + [`code/splice_qc.py`](code/splice_qc.py)) | **primary submission — 1st place public** |
| 0.34838 | Jul 15 | `step_12750` (runB) | [`configs/decode_params.json`](configs/decode_params.json) (per-language α/β/beam from enlarged-dev sweep) | pre-longfix reference |
| 0.34894 | Jul 15 | `step_12750` (runB) | final params, except kln at α .65/β .85/beam 800 | negative probe — beam-800 variant, discarded |
| 0.34929 | Jul 12 | `step_12750` (runB) | pre-sweep params (α/β as of Jul 12), pre-purification LMs | superseded |
| 0.34935 | Jul 13 | `step_12750` (runB) | pre-sweep params, **purified LMs** | post-purification reference |
| 0.34977 | Jul 13 | `step_12750` (runB) | pre-sweep params + restricted lexicon | negative probe — lexicon restriction, discarded |

Reproduction of the primary submission: see [`README.md`](README.md) § *Reproduce a submission* — `code/omni_kenlm_v3.py` reads `configs/decode_params.json` as committed; the checkpoint and LMs download from [Hugging Face](https://huggingface.co/Kimyayd/afrivoices-edge-asr).

Negative probes are retained in the selection deliberately: the final ranking uses the best private-split score among selected submissions, and the probes document measured dead-ends (see `TECHNICAL_REPORT.md` § 5–6).
