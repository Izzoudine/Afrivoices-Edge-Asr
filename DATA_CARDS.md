# Data & Model Cards

This file accompanies the submission per the competition rule: *"Any data or model cards provided by organizers must accompany the final submission."* It lists every dataset and pre-trained model used, with a pointer to its authoritative card and license. Where a license is defined on the source card, **that card is authoritative** — the summaries below are for convenience.

## Training / language-model data (organizer-provided)

All acoustic training data and in-domain language-model text come from the AfriVoices / KenCorpus datasets released by the organizers (Anv-ke, Digital Umuganda) for this competition. Their dataset cards live on the source pages linked below.

| Dataset | Source (card) | Used for | License |
|---|---|---|---|
| Anv-ke/kikuyu | https://huggingface.co/datasets/Anv-ke/kikuyu | acoustic FT + LM text (Kikuyu) | per organizer card / competition terms |
| Anv-ke/Dholuo | https://huggingface.co/datasets/Anv-ke/Dholuo | acoustic FT + LM text (Dholuo) | per organizer card / competition terms |
| Anv-ke/Somali | https://huggingface.co/datasets/Anv-ke/Somali | acoustic FT + LM text (Somali) | per organizer card / competition terms |
| Anv-ke/Maasai | https://huggingface.co/datasets/Anv-ke/Maasai | acoustic FT + LM text (Maasai) | per organizer card / competition terms |
| Anv-ke/Kalenjin | https://huggingface.co/datasets/Anv-ke/Kalenjin | acoustic FT + LM text (Kalenjin) | per organizer card / competition terms |
| DigitalUmuganda/Afrivoice_Swahili | https://huggingface.co/datasets/DigitalUmuganda/Afrivoice_Swahili | acoustic FT + LM text (Swahili) | per organizer card / competition terms |
| Digital Umuganda — Somali (Mogadishu) | provided by organizers | additional Somali acoustic FT | per organizer card / competition terms |

Notes:
- Only the `train` and `dev` splits were used for model training and validation. See [`TRANSPARENCY_NOTE.md`](TRANSPARENCY_NOTE.md) regarding the `dev_test` split (which overlaps the Kaggle test set) and our removal of its transcriptions from the language-model corpora.
- Additional **public web text** was used only to enlarge the KenLM language models (not for acoustic training). No copyrighted corpora are redistributed in this repository; the deliverable language models are built from competition transcripts + freely available text.

## Pre-trained model (external, cited)

| Model | Source (card) | Role | License |
|---|---|---|---|
| omniASR-CTC-1B v2 | https://github.com/facebookresearch/omnilingual-asr | base acoustic model, fine-tuned here | **Apache-2.0** (Meta Platforms) |
| omniASR written tokenizer v2 | shipped with omnilingual-asr | character tokenizer | Apache-2.0 |

The base model and tokenizer are Meta's Omnilingual ASR release, used under Apache-2.0 (compatible with this repository's Apache-2.0 license). They are pulled automatically from Meta's public URLs by the asset cards in [`configs/omni_v2.yaml`](configs/omni_v2.yaml).

## This submission's model card

The fine-tuned checkpoint, int8 ONNX export and purified KenLM models are published with their own model card at **https://huggingface.co/Kimyayd/afrivoices-edge-asr** (Apache-2.0).
