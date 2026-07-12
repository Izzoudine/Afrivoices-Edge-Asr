#!/bin/bash
set -e
LMPLZ=/root/kenlm_src/build/bin/lmplz; BB=/root/kenlm_src/build/bin/build_binary
mkdir -p /scratch/lm_final_pure
build() {
  local NAME=$1 ORDER=$2; shift 2
  cat "$@" | sort -u > /root/_pure_corpus.txt
  nice -n 5 $LMPLZ -o $ORDER --discount_fallback < /root/_pure_corpus.txt > /root/_pure.arpa 2>/dev/null
  $BB /root/_pure.arpa /scratch/lm_final_pure/$NAME > /dev/null 2>&1
  echo "$NAME OK ($(wc -l < /root/_pure_corpus.txt) lignes)"
}
build swh_B_4gram.bin 4 /root/pure_swh_indomain_pure.txt /root/pure_swh_web_pure.txt
build kik_B_3gram.bin 3 /root/pure_kik_indomain_pure.txt /root/pure_kik_web_pure.txt
build luo_B_3gram.bin 3 /root/pure_luo_indomain_pure.txt /root/pure_luo_web_pure.txt
build som_A_4gram.bin 4 /root/pure_som_indomain_pure.txt
build mas_A_3gram.bin 3 /root/pure_mas_indomain_pure.txt
cat /root/pure_kln_A2mix_pure.txt /root/pure_kln_A2mix_pure.txt /root/pure_kln_A2mix_pure.txt /root/pure_kln_web_pure.txt > /root/_klnmix.txt
nice -n 5 $LMPLZ -o 4 --discount_fallback < /root/_klnmix.txt > /root/_pure.arpa 2>/dev/null
$BB /root/_pure.arpa /scratch/lm_final_pure/kln_MIXK3_4gram.bin > /dev/null 2>&1
echo "kln_MIXK3_4gram.bin OK"
uni() { local L=$1 MIN=$2; shift 2
  cat "$@" | awk -v m=$MIN "{for(i=1;i<=NF;i++) c[\$i]++} END {for(w in c) if(c[w]>=m) print w}" | sort > /scratch/lm_final_pure/${L}_unigrams.txt
  echo "${L}_unigrams: $(wc -l < /scratch/lm_final_pure/${L}_unigrams.txt) mots"
}
uni swh 1 /root/pure_swh_indomain_pure.txt /root/pure_swh_web_pure.txt
uni kik 1 /root/pure_kik_indomain_pure.txt /root/pure_kik_web_pure.txt
uni luo 1 /root/pure_luo_indomain_pure.txt /root/pure_luo_web_pure.txt
uni som 2 /root/pure_som_indomain_pure.txt
uni mas 2 /root/pure_mas_indomain_pure.txt
uni kln 2 /root/pure_kln_A2mix_pure.txt
rm -f /root/_pure.arpa /root/_pure_corpus.txt /root/_klnmix.txt
ls /scratch/lm_final_pure/ | wc -l
python3 - << "PYEOF"
import json
LF="/scratch/lm_final_pure"
P={
 "swh":{"lm":f"{LF}/swh_B_4gram.bin","alpha":0.35,"beta":0.8,"unigrams":f"{LF}/swh_unigrams.txt","beam":100,"prune":-10.0},
 "kik":{"lm":f"{LF}/kik_B_3gram.bin","alpha":0.5,"beta":1.5,"unigrams":f"{LF}/kik_unigrams.txt","beam":100,"prune":-10.0},
 "luo":{"lm":f"{LF}/luo_B_3gram.bin","alpha":0.5,"beta":0.0,"unigrams":f"{LF}/luo_unigrams.txt","beam":100,"prune":-10.0},
 "som":{"lm":f"{LF}/som_A_4gram.bin","alpha":0.2,"beta":0.0,"unigrams":f"{LF}/som_unigrams.txt","beam":100,"prune":-10.0},
 "kln":{"lm":f"{LF}/kln_MIXK3_4gram.bin","alpha":0.6,"beta":0.5,"unigrams":f"{LF}/kln_unigrams.txt","beam":100,"prune":-10.0},
 "mas":{"lm":f"{LF}/mas_A_3gram.bin","alpha":0.35,"beta":0.8,"unigrams":f"{LF}/mas_unigrams.txt","beam":100,"prune":-10.0},
}
json.dump(P, open("/scratch/decode_params.json","w"), indent=1)
print("params purs OK")
PYEOF
rclone copy /scratch/lm_final_pure scw:afrivoices-data/lm_final_pure -q
echo BUILD_PURE_DONE
export FAIRSEQ2_ASSET_DIR=/root/cards
CKPT=/scratch/runB_out/ws_1.4432425d/checkpoints/step_12750/model/pp_00/tp_00/sdp_00.pt MODE=test OUT=/scratch/submission_pure.csv BEAM=100 python3 /root/omni_kenlm_v3.py > /root/decode_pure.log 2>&1
echo DECODE_PURE_EXIT_$? >> /root/decode_pure.log
rclone copy /scratch/submission_pure.csv scw:afrivoices-data/ -q
