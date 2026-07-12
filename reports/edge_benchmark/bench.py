import time, json, sys, os, resource
import numpy as np, soundfile as sf
import onnxruntime as ort
from pyctcdecode import build_ctcdecoder

THREADS = int(os.environ.get("THREADS","4"))
BEAM = int(os.environ.get("BEAM","100"))
labels = json.load(open("/root/bench/labels.json"))
meta = json.load(open("/root/bench/bench_clips/meta.json"))

CFG = {
 "swa": ("swh_B_4gram.bin","swh_unigrams.txt",0.35,0.8),
 "kik": ("kik_B_3gram.bin","kik_unigrams.txt",0.5,1.5),
 "luo": ("luo_B_3gram.bin","luo_unigrams.txt",0.5,0.0),
 "som": ("som_A_4gram.bin","som_unigrams.txt",0.2,0.0),
 "kal": ("kln_MIXK3_4gram.bin","kln_unigrams.txt",0.6,0.5),
 "kln": ("kln_MIXK3_4gram.bin","kln_unigrams.txt",0.6,0.5),
 "maa": ("mas_A_3gram.bin","mas_unigrams.txt",0.35,0.8),
 "mas": ("mas_A_3gram.bin","mas_unigrams.txt",0.35,0.8),
}
so = ort.SessionOptions(); so.intra_op_num_threads = THREADS; so.inter_op_num_threads = 1
t0=time.time()
sess = ort.InferenceSession("/root/bench/model.onnx", so, providers=["CPUExecutionProvider"])
print(f"chargement modele: {time.time()-t0:.1f}s", flush=True)
inp = sess.get_inputs()[0].name

DEC={}
t0=time.time()
for lang,(lm,uni,a,b) in CFG.items():
    if lang in ("kln","mas"): continue
    ug=[w.strip() for w in open(f"/root/bench/lm/{uni}",encoding="utf-8") if w.strip()]
    DEC[lang]=build_ctcdecoder(labels, f"/root/bench/lm/{lm}", unigrams=ug, alpha=a, beta=b)
print(f"chargement 6 LM+lexiques: {time.time()-t0:.1f}s", flush=True)

res=[]
for m in sorted(meta, key=lambda x: x["file"]):
    y,sr = sf.read(f"/root/bench/bench_clips/{m[chr(102)+chr(105)+chr(108)+chr(101)]}", dtype="float32")
    x = y[None,:].astype(np.float32)
    t1=time.time(); lg = sess.run(None,{inp:x})[0][0]; t_ac=time.time()-t1
    lp = lg.astype(np.float32)
    lang=m["lang"] if m["lang"] in DEC else {"kln":"kal","mas":"maa"}.get(m["lang"],m["lang"])
    t2=time.time(); hyp = DEC[lang].decode(lp, beam_width=BEAM); t_lm=time.time()-t2
    rtf=(t_ac+t_lm)/m["dur"]
    res.append({"f":m["file"],"dur":m["dur"],"ac":round(t_ac,2),"lm":round(t_lm,2),"rtf":round(rtf,3)})
    print(f"{m[chr(102)+chr(105)+chr(108)+chr(101)]} dur={m[chr(100)+chr(117)+chr(114)]}s acoustique={t_ac:.1f}s decodage={t_lm:.1f}s RTF={rtf:.2f}", flush=True)

durs=sum(r["dur"] for r in res); tot=sum(r["ac"]+r["lm"] for r in res)
rtfs=sorted(r["rtf"] for r in res)
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024
print(f"\n===== BILAN ({THREADS} threads ARM, beam {BEAM}) =====")
print(f"audio total: {durs:.0f}s | temps total: {tot:.0f}s | RTF GLOBAL: {tot/durs:.3f}")
print(f"RTF median: {rtfs[len(rtfs)//2]:.3f} | p95: {rtfs[int(len(rtfs)*0.95)]:.3f} | max: {rtfs[-1]:.3f}")
print(f"RAM pic: {peak:.2f} Go")
print(f"extrapolation test complet (~250h audio): {tot/durs*250:.0f}h de calcul")
json.dump(res, open("/root/bench/results.json","w"))
print("BENCH_DONE")
