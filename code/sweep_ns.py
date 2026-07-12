import os, json, re, sys
import numpy as np
from multiprocessing import Pool
import jiwer

LOG="/scratch/logits_dev_ns"
labels=json.load(open("/scratch/labels.json"))
meta=json.load(open(f"{LOG}/meta.json"))
WS=re.compile(r"\s+")
def norm(s): return WS.sub(" ", str(s or "")).strip()
BY={}
for m in meta:
    lang=m["lang"].split("_")[0]
    if lang=="kik" and m["tag"]!="kikclean": continue
    if lang!="kik" and m["tag"]!="std": continue
    BY.setdefault(lang,[]).append(m)
SPLITS={}
for lang, items in BY.items():
    items=sorted(items, key=lambda m: len(norm(m["ref"]).split()))[:110]
    SPLITS[lang]=([m for i,m in enumerate(items) if i%2==0],[m for i,m in enumerate(items) if i%2==1])

T="/scratch/kenlm_tune"; V6="/scratch/kenlm_v6mix_tune"; NT="/scratch/kenlm_night_tune"; LF="/scratch/lm_final"
BASE={"swh":(f"{T}/swh_B_4gram.bin",0.35,0.8,f"{T}/swh_unigrams.txt"),
      "kik":(f"{T}/kik_B_3gram.bin",0.5,1.5,f"{T}/kik_unigrams.txt"),
      "luo":(f"{T}/luo_B_3gram.bin",0.5,0.0,f"{T}/luo_unigrams.txt"),
      "som":(f"{T}/som_A_4gram.bin",0.2,0.0,f"{V6}/som_unigrams.txt"),
      "kln":(f"{NT}/kln_MIXK3_4gram.bin",0.6,0.5,f"{V6}/kln_unigrams.txt"),
      "mas":(f"{T}/mas_A_3gram.bin",0.35,0.8,f"{V6}/mas_unigrams.txt")}
TASKS=[]
HW={"kln":open("/scratch/hotwords_kln.txt").read().split(),"mas":open("/scratch/hotwords_mas.txt").read().split()}
for lang,(lm,a,b,uni) in BASE.items():
    TASKS.append((lang,lm,a,b,uni,100,-10.0,None,0,"base"))
    for da in (-0.1,0.1):
        TASKS.append((lang,lm,round(a+da,2),b,uni,100,-10.0,None,0,f"a{round(a+da,2)}"))
    for db in (-0.3,0.3):
        nb=round(b+db,2)
        if nb>=0: TASKS.append((lang,lm,a,nb,uni,100,-10.0,None,0,f"b{nb}"))
    if lang in ("kln","mas"):
        for w in (4.0,6.0,10.0):
            TASKS.append((lang,lm,a,b,uni,100,-10.0,HW[lang],w,f"hw{int(w)}"))
def run(args):
    lang,lm,a,b,uni,bw,bp,hw,hww,tag=args
    from pyctcdecode import build_ctcdecoder
    ug=[w.strip() for w in open(uni,encoding="utf-8") if w.strip()]
    dec=build_ctcdecoder(labels, lm, unigrams=ug, alpha=a, beta=b)
    res={}
    for split in ("A","B"):
        clips=SPLITS[lang][0] if split=="A" else SPLITS[lang][1]
        per=[]
        for m in clips:
            lp=np.load(f"{LOG}/{m[chr(107)+chr(101)+chr(121)]}.npy").astype(np.float32)
            hyp=norm(dec.decode(lp, beam_width=bw, beam_prune_logp=bp, hotwords=hw, hotword_weight=hww) if hw else dec.decode(lp, beam_width=bw, beam_prune_logp=bp))
            if not hyp: hyp="a"
            ms=jiwer.process_words(norm(m["ref"]), hyp)
            per.append((ms.substitutions+ms.deletions+ms.insertions, len(norm(m["ref"]).split())))
        res[split]=per
    return (lang,tag,res)

def wer(per): return 100.0*sum(e for e,_ in per)/max(sum(w for _,w in per),1)

if __name__=="__main__":
    out={}
    with Pool(8) as p:
        for lang,tag,res in p.imap_unordered(run,TASKS):
            wA,wB=wer(res["A"]),wer(res["B"])
            out.setdefault(lang,{})[tag]={"A":wA,"B":wB,"perB":res["B"]}
            print(f"{lang} {tag}: A {wA:.2f} | B {wB:.2f}", flush=True)
    rng=np.random.default_rng(2026)
    for lang,cfgs in out.items():
        base=cfgs.get("base")
        for tag,c in cfgs.items():
            if tag=="base" or c["A"]>=base["A"]: continue
            pc=np.array(c["perB"]); pr=np.array(base["perB"]); n=len(pc); wins=0
            for _ in range(10000):
                idx=rng.integers(0,n,n)
                wins += (pc[idx,0].sum()/max(pc[idx,1].sum(),1)) < (pr[idx,0].sum()/max(pr[idx,1].sum(),1))
            print(f"CANDIDAT {lang} {tag}: A {c[chr(65)]:.2f}<{base[chr(65)]:.2f}, B {c[chr(66)]:.2f} vs {base[chr(66)]:.2f}, boot {wins/100:.0f}%", flush=True)
    json.dump({l:{t:{k:v for k,v in c.items() if k!="perB"} for t,c in cf.items()} for l,cf in out.items()}, open("/scratch/defense_results.json","w"), indent=1)
    print("DEFENSE_DONE", flush=True)
