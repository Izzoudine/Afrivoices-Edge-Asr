#!/usr/bin/env python3
# Re-prep kik + som depuis Anv-ke BRUT avec DEMOJIBAKE avant norm (le stock 360h stocke du texte post-norm, irreparable).
# Sortie: hive parquet omni (text, audio_bytes FLAC int8, audio_size, corpus/split/language) + dev PROPRE 1.5h/lang.
import os, io, re, json, time, collections, random
import numpy as np
tok=open("/root/.env").read().split("HF_TOKEN=")[1].split()[0].strip().strip('"')
os.environ["HF_HUB_DISABLE_XET"]="1"
import pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
import soundfile as sf
import librosa
from concurrent.futures import ThreadPoolExecutor

SR=16000; MAX_SEC=40.0; MIN_SEC=0.5
OUT=os.environ.get("OUT","/root/fix_kiksom"); ROOT=f"{OUT}/version=0"
os.makedirs(ROOT, exist_ok=True)
fs=HfFileSystem(token=tok)

# cibles: (repo, lang_omni, heures unscripted train, heures scripted train, dev_h par type)
import os as _os
SHARD=int(_os.environ.get("SHARD","0")); NSHARDS=int(_os.environ.get("NSHARDS","1"))
PLAN=[(_os.environ["REPO"], _os.environ["LANGC"], 999.0, 0.0, 0.0)]

MOJI=re.compile(r"(?:Ã.|Å.|Ä.|â€|Â.)")
# fallback cible si le round-trip cp1252 echoue (chaines mixtes)
PAIRS=[("Å©","ũ"),("Ä©","ĩ"),("Å¨","Ũ"),("Ä¨","Ĩ"),("Å‹","ŋ"),("â€™","'"),("â€˜","'"),
       ("â€œ",'"'),("â€\x9d",'"'),("â€“","-"),("â€”","-"),("Ã¢","â"),("Â "," "),("Â","")]
def demojibake(t):
    if not MOJI.search(t): return t
    try:
        t2=t.encode("cp1252").decode("utf-8")
        if not MOJI.search(t2): return t2
        t=t2
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    for a,b in PAIRS: t=t.replace(a,b)
    return t

_P=re.compile(r"[^\w\s']",re.U);_TAG=re.compile(r"\b(?:cs|p)\b");_DIG=re.compile(r"\d+");_WS=re.compile(r"\s+")
def norm(t):
    t=(t or "").replace("’","'").replace("ʼ","'").replace("‘","'")
    t=t.lower();t=_TAG.sub(" ",t);t=_P.sub(" ",t);t=_DIG.sub(" ",t);return _WS.sub(" ",t).strip()

buf={"text":[],"audio_bytes":[],"audio_size":[],"corpus":[],"split":[],"language":[]}
def flush():
    if not buf["text"]: return
    tbl=pa.table({"text":pa.array(buf["text"],pa.string()),
                  "audio_bytes":pa.array(buf["audio_bytes"],pa.list_(pa.int8())),
                  "audio_size":pa.array(buf["audio_size"],pa.int64()),
                  "corpus":pa.array(buf["corpus"],pa.string()),
                  "split":pa.array(buf["split"],pa.string()),
                  "language":pa.array(buf["language"],pa.string())})
    pq.write_to_dataset(tbl,ROOT,partition_cols=["corpus","split","language"],row_group_size=100)
    for k in buf: buf[k]=[]

def flac_i8(y):
    b=io.BytesIO(); sf.write(b,y,SR,format="FLAC"); return np.frombuffer(b.getvalue(),np.int8)

def dec_res(ab):
    try:
        y,sr=sf.read(io.BytesIO(ab),dtype="float32",always_2d=False)
        if getattr(y,"ndim",1)>1: y=y.mean(1)
        if sr!=SR: y=librosa.resample(y,orig_sr=sr,target_sr=SR,res_type="soxr_hq")
        return np.clip(y,-1,1).astype(np.float32)
    except Exception: return None

moji_stats=collections.Counter()
for repo,lang,h_uns,h_scr,devh in PLAN:
    for typ,target in (("unscripted",h_uns),("scripted",h_scr)):
        files=sorted(fs.glob(f"datasets/{repo}/train/{typ}/audios/*.parquet"))
        random.Random(13).shuffle(files)
        got=0.0; dev=0.0; seen=set()
        for _fi, fp in enumerate(files):
            if _fi % NSHARDS != SHARD: continue
            if got>=target: break
            # fs.open(parquet) HANG connu -> download local via requests+deadline puis lecture disque
            rel=fp.split(f"datasets/{repo}/",1)[1]
            url=f"https://huggingface.co/datasets/{repo}/resolve/main/{rel}"
            tmp="/root/_pq.parquet"; ok=False
            for k in range(3):
                try:
                    import requests as rq
                    with rq.get(url, headers={"Authorization":f"Bearer {tok}"}, stream=True, timeout=(10,30)) as r:
                        r.raise_for_status(); t0=time.time()
                        with open(tmp+".part","wb") as f:
                            for c in r.iter_content(1<<20):
                                if c: f.write(c)
                                if time.time()-t0>600: raise TimeoutError("dl>600s")
                    os.replace(tmp+".part",tmp); ok=True; break
                except Exception:
                    time.sleep(3*(k+1))
            if not ok:
                print(f"  skip {rel} (timeout)",flush=True); continue
            try:
                t=pq.read_table(tmp,columns=["audio","transcription"])
            except Exception as e:
                print(f"  skip {rel} ({type(e).__name__})",flush=True); continue
            finally:
                try: os.remove(tmp)
                except Exception: pass
            auds=t.column("audio").to_pylist(); txts=t.column("transcription").to_pylist()
            t=None
            with ThreadPoolExecutor(max_workers=4) as ex:
                for i0 in range(0,len(auds),200):
                    if got>=target: break
                    part=list(ex.map(dec_res,[ (a or {}).get("bytes",b"") for a in auds[i0:i0+200] ]))
                    for y,rawtx in zip(part,txts[i0:i0+200]):
                        if y is None: continue
                        dur=len(y)/SR
                        if not (MIN_SEC<=dur<=MAX_SEC): continue
                        was_moji=bool(rawtx and MOJI.search(rawtx))
                        tx=norm(demojibake(rawtx or ""))
                        if not tx: continue
                        key=(lang,tx,len(y))
                        if key in seen: continue
                        seen.add(key)
                        if was_moji: moji_stats[f"{lang}/{typ}"]+=1
                        split="dev" if dev<devh else "train"
                        buf["text"].append(tx); buf["audio_bytes"].append(flac_i8(y)); buf["audio_size"].append(len(y))
                        buf["corpus"].append("afrivoices"); buf["split"].append(split); buf["language"].append(lang)
                        if split=="dev": dev+=dur/3600
                        else: got+=dur/3600
                        if len(buf["text"])>=500: flush()
                    part=None
            auds=txts=None
            print(f"{lang} {typ}: train {got:.1f}/{target}h dev {dev:.2f}h ({fp.split('/')[-1]})",flush=True)
        flush()
print("moji repares:",dict(moji_stats),flush=True)
with open(f"{OUT}/stats.tsv","w") as f:
    f.write(json.dumps(dict(moji_stats)))
print("REPREP DONE",flush=True)
