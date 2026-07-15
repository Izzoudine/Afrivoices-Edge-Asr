# -*- coding: utf-8 -*-
"""Final card build + bounds + diagnostics."""
import csv, sys, io, re, hashlib
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
BASE = "C:/Users/USER/Desktop/LAB/HACKATHON/"
SCRATCH = "C:/Users/USER/AppData/Local/Temp/claude/c--Users-USER-Desktop-LAB-HACKATHON/fc84ed0f-f26c-4386-a295-af6e9c293516/scratchpad/"
FILES = {
    "champ":"submission_champSW_kenlm.csv","pure":"submission_pure.csv",
    "crumbs":"submission_crumbs_kenlm.csv","klnonly":"submission_klnonly_kenlm.csv",
    "runB":"submission_runB_kenlm.csv","champ_g":"submission_champSW_greedy.csv",
    "runB_g":"submission_runB_greedy.csv",
}
def load(fn):
    rows=[]
    with open(BASE+fn,encoding="utf-8",newline="") as f:
        r=csv.reader(f); hdr=next(r)
        for row in r: rows.append(row)
    return hdr,rows
data={}; order=[]
hdr=None
for n,fn in FILES.items():
    h,rows=load(fn)
    if hdr is None:
        hdr=h; order=[r[0] for r in rows]
    data[n]={r[0]:(r[1],r[2].strip()) for r in rows}
decks={n:{i:v[1] for i,v in d.items()} for n,d in data.items()}
langs={i:v[0] for i,v in data["champ"].items()}
ids=order; LANGS=sorted(set(langs.values()))
words_tot=Counter()
for i in ids: words_tot[langs[i]]+=len(decks["champ"][i].split())

def wer_pair(a,b):
    A,B=a.split(),b.split()
    if not A: return len(B),1
    prev=list(range(len(B)+1))
    for i,x in enumerate(A,1):
        cur=[i]+[0]*len(B)
        for j,y in enumerate(B,1):
            cur[j]=min(prev[j]+1,cur[j-1]+1,prev[j-1]+(x!=y))
        prev=cur
    return prev[-1],len(A)
def rat(wit,i,s):
    e,n=wer_pair(decks[wit][i],s); return e/max(n,1)

# crossAB candidates
cand={}
for i in ids:
    ch=decks["champ"][i]
    A={decks["pure"][i],decks["runB"][i]}
    B={decks["crumbs"][i],decks["klnonly"][i]}
    cs=(A&B)-{ch}
    if len(cs)==1: cand[i]=cs.pop()

# F5: rg strict AND cg not-worse
f5={}
for i,s in cand.items():
    if rat("runB_g",i,s)<rat("runB_g",i,decks["champ"][i]) and rat("champ_g",i,s)<=rat("champ_g",i,decks["champ"][i]):
        f5[i]=s
# F4: both strict
f4={i:s for i,s in f5.items() if rat("champ_g",i,s)<rat("champ_g",i,decks["champ"][i])}
print(f"F5 flips: {len(f5)}  F4 flips: {len(f4)}")

# max swing per lang for F5 (upper bound on |delta lang WER|)
print("\nF5 per-lang bound: expo * flipWER")
tot=0
for L in LANGS:
    fl=[i for i in f5 if langs[i]==L]
    if not fl: print(f"  {L}: 0 flips"); continue
    wf=sum(len(decks["champ"][i].split()) for i in fl)
    expo=wf/max(words_tot[L],1)
    E=N=0
    for i in fl:
        e,n=wer_pair(decks["champ"][i],f5[i]); E+=e;N+=n
    fw=E/max(N,1)
    print(f"  {L}: flips={len(fl)} expo={expo:.4f} flipWER={fw:.3f} maxswing={expo*fw:.5f}")
    tot+=expo*fw
print(f"  macro max swing = {tot/6:.5f} (if 100% of changed words good)")

# how many F5 flips = consensus exactly equals a greedy witness (LM artifact undone)
n_cg=sum(1 for i,s in f5.items() if s==decks["champ_g"][i])
n_champ_is_lm_only=sum(1 for i,s in f5.items() if decks["champ"][i]==decks["champ_g"][i])
print(f"\nF5 flips where s == champ_greedy exact: {n_cg}; champ==its own greedy: {n_champ_is_lm_only}")

# kln apostrophe/charset diagnostic on F5 kln flips
def strip_apo(t): return re.sub(r"[’'ʼ`]", "", t)
kln_apo=0
for i,s in f5.items():
    if langs[i]=="kln" and strip_apo(s)==strip_apo(decks["champ"][i]):
        kln_apo+=1
print(f"F5 kln flips that are apostrophe-only differences: {kln_apo}")

# distribution of flip magnitudes (words changed per clip)
mag=Counter()
for i,s in f5.items():
    e,_=wer_pair(decks["champ"][i],s); mag[min(e,10)]+=1
print("flip edit-magnitude histogram (capped 10):", dict(sorted(mag.items())))

# durations of flipped clips
dur={}
with open(BASE+"afrivoices-repo/reports/latency_all_test.csv",encoding="utf-8") as f:
    r=csv.reader(f);next(r)
    for row in r: dur[row[0]]=float(row[2])
long_flips=sum(1 for i in f5 if dur.get(i,0)>38)
long_all=sum(1 for i in ids if dur.get(i,0)>38)
print(f"F5 flips on clips >38s: {long_flips}/{len(f5)} ({long_flips/len(f5):.1%}); overall >38s share: {long_all/len(ids):.1%}")

# sample flips for eyeballing (kln + swa)
print("\n=== sample F5 flips ===")
shown=0
for i,s in f5.items():
    if shown>=8: break
    if langs[i] not in ("kln","swa","som"): continue
    print(f"[{langs[i]}] {i[:20]}")
    print(f"  champ : {decks['champ'][i][:140]}")
    print(f"  new   : {s[:140]}")
    print(f"  cg    : {decks['champ_g'][i][:140]}")
    shown+=1

# write cards to scratchpad (never touching user files)
def write_card(name, flipmap):
    p=SCRATCH+name
    with open(p,"w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(hdr)
        for i in ids:
            t = flipmap.get(i, decks["champ"][i])
            w.writerow([i, langs[i], t])
    h=hashlib.md5(open(p,"rb").read()).hexdigest()
    print(f"wrote {p}  md5={h}")
write_card("submission_witnessvote_F5.csv", f5)
write_card("submission_witnessvote_F4.csv", f4)

# verify F5 card: differs from champ in exactly len(f5) rows, header identical
p=SCRATCH+"submission_witnessvote_F5.csv"
with open(p,encoding="utf-8",newline="") as f:
    r=csv.reader(f); h2=next(r)
    diff=0; n=0
    for row in r:
        n+=1
        if row[2]!=decks["champ"][row[0]]: diff+=1
print(f"verify: rows={n} diffs_vs_champ={diff} header_match={h2==hdr}")
