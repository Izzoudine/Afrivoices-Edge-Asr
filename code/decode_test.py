#!/usr/bin/env python3
# Decodage GREEDY du test complet -> CSV soumission. CKPT env = poids a overlayer.
import os, io, glob, csv, time
import numpy as np, torch
import pyarrow.dataset as pds
CARD = os.environ.get("CARD", "omniASR_CTC_1B_v2")
CKPT = os.environ["CKPT"]
OUT  = os.environ.get("OUT", "/scratch/submission_repare.csv")
TEST = os.environ.get("TEST", "/scratch/test16k")
BATCH= int(os.environ.get("BATCH", "12"))
MAXS = 38.0; SR = 16000
from fairseq2.models.hub import load_model
from fairseq2.data.tokenizers.hub import load_tokenizer
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
model = load_model(CARD, device=torch.device("cuda"), dtype=torch.bfloat16)
tok = load_tokenizer(CARD)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
r = model.load_state_dict(sd, strict=True)
print(f"overlay: missing=0 unexpected=0 OK", flush=True)
pipe = ASRInferencePipeline(None, model=model, tokenizer=tok)

def chunks(y):
    # decoupe aux minima d energie en fenetres <=38s
    n = len(y); mx = int(MAXS*SR)
    if n <= mx: return [y]
    out = []; i = 0
    while n - i > mx:
        seg = y[i:i+mx]
        w = seg[int(0.6*mx):]                       # cherche le silence dans le dernier tiers
        k = len(w) - 1
        win = SR//4
        if len(w) > win:
            e = np.convolve(np.abs(w), np.ones(win)/win, mode="valid")
            k = int(np.argmin(e)) + win//2
        cut = i + int(0.6*mx) + k
        out.append(y[i:cut]); i = cut
    out.append(y[i:])
    return [s for s in out if len(s) > SR//4]

LANGMAP = {"swh":"swa","swa":"swa","kik":"kik","luo":"luo","som":"som","mas":"mas","kln":"kln","maa":"mas","kal":"kln"}
files = sorted(glob.glob(TEST + "/**/*.parquet", recursive=True))
print(f"{len(files)} parquets test", flush=True)
t0 = time.time(); nrows = 0
with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["id","language","transcription"])
    for fi, fp in enumerate(files):
        t = pds.dataset(fp).to_table(columns=["id","language","audio_int16"])
        ids = t.column("id").to_pylist(); langs = t.column("language").to_pylist(); auds = t.column("audio_int16").to_pylist()
        t = None
        segs = []; owner = []
        for j, ab in enumerate(auds):
            y = np.frombuffer(ab, dtype=np.int16).astype(np.float32)/32768.0
            for s in chunks(y):
                segs.append({"waveform": torch.from_numpy(np.ascontiguousarray(s)), "sample_rate": SR}); owner.append(j)
        hyps = pipe.transcribe(segs, batch_size=BATCH)
        outs = [[] for _ in ids]
        for o, h in zip(owner, hyps):
            if h and h.strip(): outs[o].append(h.strip())
        for cid, lg, parts in zip(ids, langs, outs):
            txt = " ".join(parts).strip() or "a"
            w.writerow([cid, LANGMAP.get(str(lg)[:3].lower(), str(lg)), txt])
            nrows += 1
        f.flush()
        print(f"[{fi+1}/{len(files)}] rows={nrows} {(time.time()-t0)/60:.0f}min", flush=True)
print(f"DECODE DONE rows={nrows}", flush=True)
