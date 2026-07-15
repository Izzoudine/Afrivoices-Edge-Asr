#!/usr/bin/env python3
# Whole-clip decoding for LONG test clips (> 38 s) — used for the final submission.
# Clips longer than 38 s are encoded in ONE pass (no chunking) and beam-decoded once with the
# per-language KenLM, preserving LM context across the whole clip. Per-clip fallback to the
# standard 38 s chunked path (omni_kenlm_v3.py behaviour) on any error/OOM.
# Validated against references: -0.87 macro-WER on 60-110 s clips vs hard 38 s chunking
# (see TECHNICAL_REPORT.md §5). Env: CKPT, OUT, optional SHARD_MOD/SHARD_REM.
import os, csv, glob, json, time
import numpy as np, torch
import pyarrow.dataset as pds
os.environ.setdefault("FAIRSEQ2_ASSET_DIR", "/root/cards")
CARD = "omniASR_CTC_1B_v2"; CKPT = os.environ["CKPT"]
SR = 16000; MAXS = 38.0
from fairseq2.models.hub import load_model
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.nn import BatchLayout
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from pyctcdecode import build_ctcdecoder

m = load_model(CARD, device=torch.device("cuda"), dtype=torch.bfloat16)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
m.load_state_dict(sd, strict=True); m.eval()
tok = load_tokenizer(CARD); pipe = ASRInferencePipeline(None, model=m, tokenizer=tok)
mdl = tok._model
labels = []; seen = set(); pua = 0xE000
for i in range(mdl.vocabulary_size):
    if i == 0: labels.append(""); continue
    if i == 4: labels.append(" "); continue
    if i in (1, 2, 3): labels.append(chr(pua)); pua += 1; continue
    p = mdl.index_to_token(i)
    if not p or p in seen or p.isspace(): labels.append(chr(pua)); pua += 1
    else: labels.append(p); seen.add(p)
P = json.load(open("/scratch/decode_params.json"))
DEC = {}; BW = {}; BP = {}
for k, c in P.items():
    ug = [w.strip() for w in open(c["unigrams"], encoding="utf-8") if w.strip()]
    DEC[k] = build_ctcdecoder(labels, c["lm"], unigrams=ug, alpha=c["alpha"], beta=c["beta"])
    BW[k] = c.get("beam", 100); BP[k] = c.get("prune", -10.0)

def logits_of(wav):
    b = pipe._create_batch_simple([(torch.from_numpy(np.ascontiguousarray(wav)), None)])
    lay = BatchLayout(b.source_seqs.shape, seq_lens=b.source_seq_lens, device=b.source_seqs.device)
    with torch.no_grad():
        lg, bl = pipe.model(b.source_seqs, lay)
    n = int(bl.seq_lens[0])
    return torch.log_softmax(lg.float(), -1)[0, :n].cpu().numpy()

def dec(lp, lang):
    t = DEC[lang].decode(lp, beam_width=BW[lang], beam_prune_logp=BP[lang])
    return t.strip() or "a"

def chunked_fallback(y, lang):
    mx = int(MAXS * SR); parts = []
    for i in range(0, len(y), mx):
        seg = y[i:i+mx]
        if len(seg) < SR // 4: continue
        parts.append(dec(logits_of(seg), lang))
    return " ".join(p for p in parts if p and p != "a").strip() or "a"

LANGMAP = {"swh": "swa", "swa": "swa", "kik": "kik", "luo": "luo", "som": "som",
           "mas": "mas", "kln": "kln", "maa": "mas", "kal": "kln"}
files = sorted(glob.glob("/scratch/test16k/**/*.parquet", recursive=True))
MOD = int(os.environ.get("SHARD_MOD", "1")); REM = int(os.environ.get("SHARD_REM", "0"))
files = files[REM::MOD]
OUT = os.environ.get("OUT", "/scratch/long_whole.csv"); nr = 0; nfb = 0; t0 = time.time()
with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "language", "transcription"])
    for fi, fp in enumerate(files):
        t = pds.dataset(fp).to_table(columns=["id", "language", "audio_int16"])
        for cid, lg, ab in zip(t.column("id").to_pylist(), t.column("language").to_pylist(), t.column("audio_int16").to_pylist()):
            y = np.frombuffer(ab, dtype=np.int16).astype(np.float32) / 32768.0
            if len(y) <= MAXS * SR:
                continue  # short clips: keep the omni_kenlm_v3.py output
            lang = str(lg)[:3].lower(); dk = {"swa": "swh", "maa": "mas", "kal": "kln"}.get(lang, lang)
            try:
                txt = dec(logits_of(y), dk if dk in DEC else "swh")
            except Exception:
                torch.cuda.empty_cache()
                txt = chunked_fallback(y, dk if dk in DEC else "swh"); nfb += 1
            w.writerow([cid, LANGMAP.get(lang, lang), txt]); nr += 1
        f.flush()
        print(f"[{fi+1}/{len(files)}] longs={nr} fallbacks={nfb} {(time.time()-t0)/60:.0f}min", flush=True)
print(f"TEST DONE rows={nr} fallbacks={nfb}", flush=True)
