#!/usr/bin/env python3
# Greffe des transcriptions longfix dans champSW avec gardes mecaniques + QC byte-preserving.
# Les lignes NON ciblees sont copiees OCTET PAR OCTET (aucun re-encodage).
import sys, csv, io

BASE = sys.argv[1]        # submission_champSW_kenlm.csv
NEW = sys.argv[2]         # longfix concat (id,language,transcription)
IDLIST = sys.argv[3]      # ids cibles (>38s)
OUT = sys.argv[4]

targets = set(l.strip() for l in open(IDLIST, encoding="utf-8") if l.strip())
newmap = {}
with open(NEW, encoding="utf-8", newline="") as f:
    for r in csv.reader(f):
        if r and r[0] != "id":
            newmap[r[0]] = (r[1], r[2])

stats = {"replaced": 0, "rollback_delta": 0, "rollback_empty": 0, "rollback_missing": 0, "untouched": 0}
deltas = []
out_lines = []
with open(BASE, "rb") as f:
    raw = f.read()
lines = raw.split(b"\r\n")
assert lines[0] == b"id,language,transcription", "header inattendu"
out_lines.append(lines[0])
for lb in lines[1:]:
    if not lb:
        continue
    line = lb.decode("utf-8")
    cid = line.split(",", 1)[0]
    if cid not in targets:
        out_lines.append(lb); stats["untouched"] += 1; continue
    old = next(csv.reader(io.StringIO(line)))
    if cid not in newmap:
        out_lines.append(lb); stats["rollback_missing"] += 1; continue
    lang_new, txt_new = newmap[cid]
    assert lang_new == old[1], f"langue differente pour {cid}"
    txt_new = txt_new.strip()
    dw = len(txt_new.split()) - len(old[2].split())
    if not txt_new or txt_new == "a" and old[2] != "a":
        out_lines.append(lb); stats["rollback_empty"] += 1; continue
    if abs(dw) > 20:
        out_lines.append(lb); stats["rollback_delta"] += 1; continue
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="")
    w.writerow([cid, old[1], txt_new])
    out_lines.append(buf.getvalue().encode("utf-8"))
    stats["replaced"] += 1; deltas.append(dw)

with open(OUT, "wb") as f:
    f.write(b"\r\n".join(out_lines) + b"\r\n")

import statistics
print("STATS:", stats)
if deltas:
    print(f"deltas mots: mediane {statistics.median(deltas)}, min {min(deltas)}, max {max(deltas)}")
n_total = stats["replaced"] + stats["untouched"] + stats["rollback_delta"] + stats["rollback_empty"] + stats["rollback_missing"]
print(f"lignes: {n_total} (attendu 41733)")
# QC final: relire OUT, verifier structure
with open(OUT, encoding="utf-8", newline="") as f:
    rows = [r for r in csv.reader(f)]
assert rows[0] == ["id", "language", "transcription"]
assert len(rows) == 41734, f"lignes: {len(rows)}"
assert all(len(r) == 3 and r[2].strip() for r in rows[1:]), "ligne invalide"
ids = [r[0] for r in rows[1:]]
assert len(set(ids)) == 41733, "doublons!"
print("QC_PASS")
