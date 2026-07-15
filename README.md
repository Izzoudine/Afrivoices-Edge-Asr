# AfriVoices — Edge ASR for Six East African Languages

A single, unified, edge-deployable Automatic Speech Recognition (ASR) model for **Swahili, Kikuyu (Gĩkũyũ), Dholuo (Luo), Somali, Maasai, and Kalenjin**, built for the AfriVoices Multilingual Edge ASR Track (Maseno Center for AI × Digital Umuganda / KenCorpus Consortium).

> **Leaderboard result:** macro-WER **0.34838** (public leaderboard, final day).
> One model, six languages, running at **≈0.57× real time on 4 ARM cores (Raspberry-Pi-class) with 5.8 GB peak RAM**.

---

## What this is

- **One acoustic model** (no per-language routing): [omniASR-CTC-1B v2](https://huggingface.co/facebook/omniASR-CTC-1B) (Meta, Apache-2.0) fine-tuned on ~1,035 h of deduplicated East African speech — **971 M parameters** (< 1 B limit).
- **One n-gram language model per language** applied only at decode time (KenLM + word lexicons), used with the organizer-provided language IDs. This is shallow fusion, not model routing.
- **int8 ONNX export** for edge inference, validated on real ARM hardware and Android smartphones.

## Results

| Language | Family | WER (best submission, %) |
|---|---|---|
| Swahili | Bantu | ~9 |
| Kikuyu | Bantu | ~13 |
| Dholuo | Nilotic (W) | ~28 |
| Somali | Cushitic | ~55 |
| Maasai | Nilotic (E) | ~45 |
| Kalenjin | Nilotic (S) | ~46 |
| **Macro-WER** | — | **0.34838** |

(Per-language figures are decode-side estimates; the official metric is the unweighted mean.)

## Edge compliance

Full pipeline (acoustic model + beam search + per-language KenLM), measured on an Ampere Altra ARM64 VM restricted to **4 cores and an 8 GB memory cap**:

| Metric | Value | Requirement |
|---|---|---|
| Parameters | 971 M | ≤ 1 B ✅ |
| Peak RAM | 5.80 GB | ≤ 8 GB ✅ |
| Real-time factor | 0.581× | ≤ 2× ✅ |

Full methodology and raw measurements: [`reports/hardware_validation.md`](reports/hardware_validation.md) and [`reports/edge_benchmark/`](reports/edge_benchmark/).

## Weights

The trained checkpoint, the int8 ONNX export, and the six purified KenLM language models are hosted on Hugging Face (too large for GitHub):

**➡️ [huggingface.co/Kimyayd/afrivoices-edge-asr](https://huggingface.co/Kimyayd/afrivoices-edge-asr)**

## Reproduce a submission

```bash
# Environment: Python 3.12, torch 2.8.0, fairseq2 0.6 (matched index),
# omnilingual-asr 0.1.0, pyctcdecode, kenlm, jiwer
export FAIRSEQ2_ASSET_DIR=configs   # asset cards for the base model + tokenizer
CKPT=<step_12750/model/.../sdp_00.pt> \
MODE=test OUT=submission.csv \
python3 code/omni_kenlm_v3.py       # reads configs/decode_params.json
```

Training recipe: [`TRAINING.md`](TRAINING.md) + [`configs/runB.yaml`](configs/runB.yaml). Fine-tuned from omniASR-CTC-1B v2, 14,000 steps, LR 1e-5 tri-stage, seed 2026.

## Repository layout

```
code/       inference (omni_kenlm_v3.py), evaluation (eval_dev.py),
            LM building & purification (build_pure_lms.sh, purify.py),
            data extraction (prep_shard.py)
configs/    decoding parameters, CTC labels, model/tokenizer asset cards
reports/    hardware validation report + raw edge benchmark
TECHNICAL_REPORT.md    end-to-end methodology and key findings
TRANSPARENCY_NOTE.md   disclosure of a test-set leak we found and removed
```

## License & attribution

This work is released under the **Apache License 2.0** ([`LICENSE`](LICENSE)).

- Base model: [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) (omniASR-CTC-1B v2), Meta Platforms, **Apache-2.0**.
- Training data: AfriVoices / KenCorpus datasets provided by the organizers (Anv-ke, Digital Umuganda); language-model text from those transcripts plus publicly available web text. No copyrighted corpora are redistributed.
- Data & model cards for all datasets and the base model: [`DATA_CARDS.md`](DATA_CARDS.md).
- See [`TRANSPARENCY_NOTE.md`](TRANSPARENCY_NOTE.md) for our handling of a leak in the provided data.
