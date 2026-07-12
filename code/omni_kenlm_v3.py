#!/usr/bin/env python3
# Decodage beam + KenLM (pyctcdecode) sur le modele bigrun. MODE=dev (valide) ou test (CSV).
import os, io, glob, csv, time, json
import numpy as np, torch
import pyarrow.dataset as pds
os.environ.setdefault("FAIRSEQ2_ASSET_DIR","/root/cards")
CARD="omniASR_CTC_1B_v2"; CKPT=os.environ["CKPT"]; MODE=os.environ.get("MODE","dev")
LM="/scratch/lm_final"; SR=16000; MAXS=38.0; BEAM=int(os.environ.get("BEAM","100"))
from fairseq2.models.hub import load_model
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.nn import BatchLayout
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from pyctcdecode import build_ctcdecoder
import jiwer

m=load_model(CARD,device=torch.device("cuda"),dtype=torch.bfloat16)
sd=torch.load(CKPT,map_location="cpu",weights_only=False); m.load_state_dict(sd,strict=True); m.eval()
tok=load_tokenizer(CARD); pipe=ASRInferencePipeline(None,model=m,tokenizer=tok)
mdl=tok._model
# labels pyctcdecode : blank=pad(1)->"" ; espace(4)->" " ; 0/2/3 + dupes -> PUA
labels=[]; seen=set(); pua=0xE000
for i in range(mdl.vocabulary_size):
    if i==0: labels.append(""); continue
    if i==4: labels.append(" "); continue
    if i in (1,2,3): labels.append(chr(pua)); pua+=1; continue
    p=mdl.index_to_token(i)
    if not p or p in seen or p.isspace(): labels.append(chr(pua)); pua+=1
    else: labels.append(p); seen.add(p)
print("labels:",len(labels),"| blank@1=","| space@4=repr",repr(labels[4]),flush=True)
P=json.load(open("/scratch/decode_params.json"))
DEC={}; BW={}; BP={}
for k,c in P.items():
    ug=[w.strip() for w in open(c["unigrams"],encoding="utf-8") if w.strip()]
    DEC[k]=build_ctcdecoder(labels, c["lm"], unigrams=ug, alpha=c["alpha"], beta=c["beta"])
    BW[k]=c.get("beam",100); BP[k]=c.get("prune",-10.0)
    print("decoder",k,"ok",c.get("beam"),c.get("prune"),flush=True)

def logits_of(wav):
    batch=pipe._create_batch_simple([(torch.from_numpy(np.ascontiguousarray(wav)),None)])
    lay=BatchLayout(batch.source_seqs.shape,seq_lens=batch.source_seq_lens,device=batch.source_seqs.device)
    with torch.no_grad(): lg,bl=pipe.model(batch.source_seqs,lay)
    n=int(bl.seq_lens[0]); return torch.log_softmax(lg.float(),-1)[0,:n].cpu().numpy()
def chunks(y):
    mx=int(MAXS*SR)
    if len(y)<=mx: return [y]
    return [y[i:i+mx] for i in range(0,len(y),mx)]
def transcribe(y,lang):
    parts=[]
    for seg in chunks(y):
        if len(seg)<SR//4: continue
        lp=logits_of(seg); parts.append(DEC[lang].decode(lp,beam_width=BW[lang],beam_prune_logp=BP[lang]))
    return " ".join(p for p in parts if p).strip()

if MODE=="dev":
    DEV="/scratch/data/mix3/version=0/corpus=afrivoices/split=dev"
    tot={}
    for d in sorted(glob.glob(DEV+"/language=*")):
        lang=d.split("language=")[-1].replace("_Latn","")
        t=pds.dataset(d,format="parquet").to_table(columns=["text","audio_bytes"])
        R=[];H=[];G=[]
        for tx,ab in list(zip(t.column("text").to_pylist(),t.column("audio_bytes").to_pylist()))[:200]:
            if not tx: continue
            y,sr=__import__("soundfile").read(io.BytesIO(np.array(ab,dtype=np.int8).tobytes()),dtype="float32")
            lp=logits_of(y); R.append(tx)
            H.append(DEC[lang].decode(lp,beam_width=BW[lang],beam_prune_logp=BP[lang]))
            G.append("".join(labels[i] for i in lp.argmax(-1)))  # greedy approx
        wk=100*jiwer.wer(R,[h if h.strip() else "a" for h in H])
        print(f"{lang}: KenLM {wk:.2f} ({len(R)})",flush=True); tot[lang]=wk
    print("MACRO KenLM:",round(sum(tot.values())/len(tot),2),flush=True); print("DEV DONE",flush=True)
else:
    LANGMAP={"swh":"swa","swa":"swa","kik":"kik","luo":"luo","som":"som","mas":"mas","kln":"kln","maa":"mas","kal":"kln"}
    files=sorted(glob.glob("/scratch/test16k/**/*.parquet",recursive=True))
    OUT=os.environ.get("OUT","/scratch/submission_bigrun_kenlm.csv"); nr=0; t0=time.time()
    with open(OUT,"w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["id","language","transcription"])
        for fi,fp in enumerate(files):
            t=pds.dataset(fp).to_table(columns=["id","language","audio_int16"])
            for cid,lg,ab in zip(t.column("id").to_pylist(),t.column("language").to_pylist(),t.column("audio_int16").to_pylist()):
                lang=str(lg)[:3].lower(); dk={"swa":"swh","maa":"mas","kal":"kln"}.get(lang,lang)
                y=np.frombuffer(ab,dtype=np.int16).astype(np.float32)/32768.0
                txt=transcribe(y,dk if dk in DEC else "swh") or "a"
                w.writerow([cid,LANGMAP.get(lang,lang),txt]); nr+=1
            f.flush(); print(f"[{fi+1}/{len(files)}] rows={nr} {(time.time()-t0)/60:.0f}min",flush=True)
    print(f"TEST DONE rows={nr}",flush=True)
