import glob, re, os, subprocess
import pyarrow.parquet as pq

def norm(s):
    s=str(s or "").lower().strip()
    s=re.sub(r"[^a-z0-9\x27 ]","",s)
    return re.sub(r"\s+"," ",s).strip()

# 1) textes du test par langue
TEST={}
M={"kik_Latn":"kik","kln_Latn":"kln","luo_Latn":"luo","mas_Latn":"mas","som_Latn":"som"}
for d in glob.glob("/scratch/devtest/language=*"):
    short=M[d.split("language=")[1]]
    s=TEST.setdefault(short,set())
    for fp in glob.glob(d+"/*.parquet"):
        for x in pq.read_table(fp,columns=["text"]).column("text").to_pylist():
            n=norm(x)
            if len(n)>=8: s.add(n)
# swa: texte-only depuis HF
try:
    from huggingface_hub import HfFileSystem
    fs=HfFileSystem()
    s=TEST.setdefault("swh",set())
    files=fs.glob("datasets/DigitalUmuganda/Afrivoice_Swahili/*_test/**/*.parquet")
    print("swa test parquets:", len(files), flush=True)
    for fp in files[:40]:
        try:
            with fs.open(fp) as f:
                t=pq.read_table(f)
                col=[c for c in t.column_names if "transcript" in c.lower() or c=="text" or "sentence" in c.lower()]
                if col:
                    for x in t.column(col[0]).to_pylist():
                        n=norm(x)
                        if len(n)>=8: s.add(n)
        except Exception as e: print("skip",fp[-40:],type(e).__name__,flush=True)
except Exception as e: print("swa ERR",type(e).__name__,flush=True)
for k,v in TEST.items(): print("refs test",k,len(v),flush=True)

# 2) purifier les corpus constitutifs des 6 LM gagnants
JOBS={
 "kln_A2mix_pure": ("/root/kln_A2mix.txt","kln"),
 "kln_web_pure": ("/root/lmtext/kln_web.txt","kln"),
 "mas_indomain_pure": ("/root/lmtext/mas_indomain.txt","mas"),
 "luo_indomain_pure": ("/root/lmtext/luo_indomain.txt","luo"),
 "luo_web_pure": ("/root/lmtext/luo_web.txt","luo"),
 "kik_indomain_pure": ("/root/lmtext/kik_indomain_clean.txt","kik"),
 "kik_web_pure": ("/root/lmtext/kik_web.txt","kik"),
 "som_indomain_pure": ("/root/lmtext/som_indomain_clean.txt","som"),
 "swh_indomain_pure": ("/root/lmtext/swh_indomain.txt","swh"),
 "swh_web_pure": ("/root/lmtext/swh_web.txt","swh"),
}
for name,(src,lang) in JOBS.items():
    refs=TEST.get(lang,set())
    kept=ex=0
    with open(src,encoding="utf-8",errors="replace") as fin, open(f"/root/pure_{name}.txt","w",encoding="utf-8") as fout:
        for l in fin:
            if norm(l) in refs: ex+=1
            else: fout.write(l); kept+=1
    print(f"{name}: garde {kept}, EXCLUT {ex} refs test", flush=True)
print("PURIFY_DONE", flush=True)
