#!/usr/bin/env python3
# WER greedy par langue sur le dev du mix (refs PROPRES pour kik/som). CKPT env = state dict a overlayer.
import os, io, glob, collections
import numpy as np, torch, soundfile as sf
import pyarrow.dataset as pds
import jiwer
CARD = os.environ.get("CARD", "omniASR_CTC_1B_v2")
CKPT = os.environ.get("CKPT", "")
DEV  = os.environ.get("DEV", "/scratch/data/mix/version=0/corpus=afrivoices/split=dev")
NCLIP= int(os.environ.get("NCLIP", "250"))
BATCH= int(os.environ.get("BATCH", "8"))
from fairseq2.models.hub import load_model
from fairseq2.data.tokenizers.hub import load_tokenizer
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
model = load_model(CARD, device=torch.device(os.environ.get("DEVICE","cuda")), dtype=torch.bfloat16)
tok = load_tokenizer(CARD)
if CKPT:
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and hasattr(sd["model"], "keys"): sd = sd["model"]
    r = model.load_state_dict(sd, strict=False)
    print(f"overlay {CKPT}: missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}", flush=True)
pipe = ASRInferencePipeline(None, model=model, tokenizer=tok)
res = {}
for langdir in sorted(glob.glob(DEV + "/language=*")):
    lang = langdir.split("language=")[-1]
    ds = pds.dataset(langdir, format="parquet")
    t = ds.to_table(columns=["text", "audio_bytes"])
    texts = t.column("text").to_pylist()[:NCLIP]
    auds  = t.column("audio_bytes").to_pylist()[:NCLIP]
    inps, refs = [], []
    for tx, ab in zip(texts, auds):
        if not tx: continue
        y, sr = sf.read(io.BytesIO(np.array(ab, dtype=np.int8).tobytes()), dtype="float32")
        inps.append({"waveform": torch.from_numpy(np.ascontiguousarray(y)), "sample_rate": sr})
        refs.append(tx)
    hyps = pipe.transcribe(inps, batch_size=BATCH)
    hyps = [h if h.strip() else "a" for h in hyps]
    FOLDMAP = str.maketrans({"ĩ":"i","ũ":"u","Ĩ":"I","Ũ":"U","ŋ":"n","ã":"a","õ":"o","é":"e","’":"'"})
    fold = lambda s: s.translate(FOLDMAP)
    wer = 100.0 * jiwer.wer(refs, hyps)
    werf = 100.0 * jiwer.wer([fold(r) for r in refs], [fold(h) for h in hyps])
    res[lang] = (wer, werf)
    print(f"{lang}: WER {wer:.2f} | WER_fold {werf:.2f} ({len(refs)} clips)", flush=True)
print("MACRO:", round(sum(w for w,_ in res.values())/len(res), 2), "| MACRO_fold:", round(sum(f for _,f in res.values())/len(res), 2), flush=True)
print("EVAL DONE", flush=True)
