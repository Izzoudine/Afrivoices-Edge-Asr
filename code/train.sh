#!/bin/bash
# Training launch for the winning AfriVoices Edge ASR model.
#
# The model is fine-tuned with the fairseq2 wav2vec2/ASR recipe shipped in
# Meta's omnilingual-asr package (Apache-2.0) — there is no custom training
# loop or notebook; the recipe is driven entirely by configs/runB.yaml.
#
# Requirements: Python 3.12, torch 2.8.0, fairseq2 0.6 (matched index),
# omnilingual-asr 0.1.0. One GPU (trained on a single NVIDIA L40S, ~39 h).
#
# FAIRSEQ2_ASSET_DIR must point to a directory holding the asset cards for
# the base model + tokenizer + dataset (see configs/omni_v2.yaml).

set -e
export FAIRSEQ2_ASSET_DIR=configs
OMNI_REPO=${OMNI_REPO:-/path/to/omnilingual-asr}   # git tag 0.1.0

PYTHONPATH="$OMNI_REPO" python3 -m workflows.recipes.wav2vec2.asr \
    ./runB_out \
    --config-file configs/runB.yaml

# Checkpoint selection is done AFTER training with an external, clean metric
# (code/eval_dev.py on a demojibake'd Kikuyu dev + the standard dev for the
# other 5 languages) — NOT the recipe's internal "best", which is polluted by
# the corrupted Kikuyu validation references. The winning checkpoint was
# step 12,750 (not the last one).
