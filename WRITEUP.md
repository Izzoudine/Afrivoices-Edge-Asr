# 1st place — AfriVoices Multilingual Edge ASR: solution write-up

**Final: macro-WER 0.34778 (1st, public leaderboard at close).** One acoustic model for six East African languages (Swahili, Kikuyu, Dholuo, Somali, Maasai, Kalenjin), 971 M parameters, running at ≈0.57× real time in 5.8 GB of RAM on 4 ARM cores.

*A short bit of story, because it drove every technical decision:* we are a two-person team (Izzoudine Kanta & Hilary Zocli), students in Benin, first online competition. We spent five days plateaued at 47 % WER convinced the problem was our model — it was our data. Two silent data bugs, found by *measuring* instead of guessing, are what actually won this. Below is the full method, with numbers.

- **Code (training + decoding + purification):** https://github.com/Izzoudine/Afrivoices-Edge-Asr
- **Weights + int8 ONNX + 6 KenLM:** https://huggingface.co/Kimyayd/afrivoices-edge-asr
- Both Apache-2.0. Reproduction is a documented 3-step pipeline (see repo README).

---

## 0. TL;DR of what moved the needle

| Lever | Effect | Where |
|---|---|---|
| Fix a mojibake (double-UTF-8) bug in Kikuyu text | **~15 WER pts** on Kikuyu | data prep |
| Pivot MMS / w2v-bert → omniASR-CTC-1B v2 | broke a hard 47 % plateau | acoustic model |
| Per-language KenLM + beam search (shallow fusion) | greedy 0.39652 → **0.34929** | decoding |
| Remove a test-set leak, rebuild LMs clean, disclose | −0.00006 (paid on purpose) | integrity |
| Re-sweep stale KenLM α/β/beam on enlarged dev | 0.34935 → **0.34838** | decoding |
| Whole-clip decoding for audio > 38 s | 0.34838 → **0.34778** | decoding |

Architecture was never the lever after the model pivot. **Data quality and decode policy were.**

## 1. Problem framing — the metric is the strategy

Six languages, one model < 1 B params, edge constraints (≤ 8 GB RAM, ≤ 2× real time), metric = **macro-averaged WER** (unweighted mean over the six languages). The unweighted mean is the single most important fact about this competition: one WER point of Kalenjin (our worst language, ~46 %) is worth exactly one point of Swahili (~9 %). Every downstream decision — training mixture, where we spent compute, which decode gains we chased — followed from that.

## 2. Data was the lever (not architecture)

We copied a previous winner's data-prep pipeline without auditing its output, and paid for it. Three data findings, in order of impact:

**2a. Mojibake (double UTF-8) — worth ~15 WER points on Kikuyu.** A large fraction of the Kikuyu transcripts had every `ũ` stored as `Å©` (and similar), because text had been UTF-8-encoded twice. CTC learns *exactly* what you show it, so the model was being trained to emit corrupted characters — and the equally-corrupted validation references were *rewarding* it for doing so. We found it not by reading files (the first 400 rows looked clean) but by a **masked probe**: blanking Kikuyu's rows in a submission barely moved the score, which meant its WER was already catastrophic. Fix: `demojibake()` does a `cp1252→UTF-8` round-trip plus a fallback pair map, applied **before** normalization (`code/prep_shard.py`). Lesson learned the hard way: scan the *full* corpus with a columnar engine, and note the corruption variant is case-sensitive (`Å©` lowercase was missed by a naive regex first).

**2b. Normalization consistency.** `norm()` lowercases, strips punctuation / `[cs]`/`[p]` tags / digits, and collapses whitespace — applied identically at train and eval time. A mismatch here collapses fine-tuning toward ~100 % WER (WER compares exact strings). This is unglamorous and decisive.

**2c. A test-set leak we chose to kill.** A `dev_test` split in the provided repos turned out to be the Kaggle test set (same waveforms, public transcriptions), and our LM corpora unknowingly contained 93–100 % of the test references for three languages. We measured the leak's actual benefit — a negligible **+0.00006**, because n-gram LMs don't recite whole sentences — then removed all 11,316 test references, rebuilt every KenLM clean (`code/purify.py` + `code/build_pure_lms.sh`), disclosed it, and submitted only the purified pipeline (`TRANSPARENCY_NOTE.md`). Slightly "worse" on paper; defensible forever.

Data prep, dedup (`(language, normalized text, audio length)`), 16 kHz resample and FLAC-in-Parquet storage are one reproducible script (`code/prep_shard.py`); assembled corpus ≈ 1,035 h after dedup.

## 3. Acoustic model

**Base:** omniASR-CTC-1B v2 (Meta, Apache-2.0) — a Conformer encoder (convolution + **relative-position** self-attention) with a char-level CTC head, 971 M params. Self-supervised pretraining on massive multilingual audio matters here: it had already *heard* Somali and Kalenjin, our two hardest languages. From-scratch attempts blank-collapsed on the mixed scripted+unscripted distribution; **warm-start was non-negotiable**.

**Recipe** (`configs/runB.yaml`, fairseq2 wav2vec2_asr): 14,000 steps (~39 h on one L40S), LR 1e-5 tri-stage, effective batch ~43 min audio/step (grad-accum ×32), encoder frozen 400 steps, SpecAugment, bfloat16, seed 2026.

**Mixture (the metric applied):** kln 25 / mas 22 / luo 18 / som 16 / kik 11 / swh 4 — deliberately over-weighting the hard languages and under-weighting near-solved Swahili, while keeping all six for anti-forgetting anchoring.

**Checkpoint selection on a clean external protocol** (`code/eval_dev.py`), never the trainer's internal "best" — that metric ran on the mojibake'd Kikuyu dev and would have selected the best *corruption-writer*. Winner = **step 12,750**, not the last step. Judge by dev-WER, never by loss (loss measures confidence, not correctness).

## 4. Decoding — the "free" −5 WER, then the final-day gains

**Shallow fusion:** per-language KenLM (5-gram) + beam search (pyctcdecode) + word lexicons. Score = `acoustic + α·LM + β·word_count`. This alone took greedy 0.39652 → 0.34929. The LM is applied at decode time with the organizer-provided language IDs — one LM per language is still "one model" (no acoustic routing).

**α/β/beam re-sweep (0.34935 → 0.34838).** The hyperparameters had been tuned when the corpora were small; the corpora had since grown severalfold, so they were stale. We dumped dev **logits once** (GPU) and swept ~150 configs per language on CPU in minutes, adopting only wins past a strict bar (≥ 85 % paired-bootstrap, ≥ 0.3 WER margin). Final values are committed in `configs/decode_params.json`. Tuning is done on a dev whose references are **excluded** from the LM corpus (otherwise the LM recites them and you pick a destructive α — a lesson we learned at 0.48 earlier).

**Whole-clip decoding for long audio (0.34838 → 0.34778) — the decisive gain.** Our pipeline chunked every clip > 38 s into hard 38 s windows; **4,500 test clips (40 % of all test audio)** were being cut, often mid-word, with the LM context reset at each boundary. omniASR's 40 s is a training window, not a hard limit — because the Conformer uses *relative* position encodings, it extrapolates: we decoded clips up to 101 s in a single pass. Critically, we **validated the mechanism on references before touching the test**: on 60–110 s clips with known transcriptions, whole-clip beat chunked by **−0.87 macro-WER**, no language regressing. (A first variant — cut at silences, stitch concatenated logits — *failed* that same validation, Somali +5.3, and was discarded. Validate mechanisms against references, always.) The long clips are then spliced into the base deck with per-clip safety rollback (`code/splice_qc.py`).

## 5. Measurement discipline (the part that generalizes)

The single most transferable thing we built was a **hierarchy of measurement instruments**, ordered by trust:

1. **References in hand** (gold) — validate any mechanism here before deploying.
2. **Masked test probes** — sabotage two candidate submissions with an *identical* fixed handicap (e.g. 3,000 Swahili rows → placeholder, ≈ +0.036); both scores stay uninformative to rivals, but the *difference* between the two probes is the exact effect of the change on the real test. The handicap cancels in the subtraction.
3. **Direct leaderboard** — reliable, costs a slot.
4. **Dev, large effects** (> 1 pt) — reliable.
5. **Dev, fine effects** (< 0.3) — **noise.** With ~100–150 dev clips, a sub-0.3 gap cannot be distinguished from luck; we watched two dev-validated retunes invert on the test on the final day. The adoption bar (≥ 0.3 + bootstrap) exists precisely for the moment leaderboard pressure tempts you to lower it.

## 6. Edge compliance

Full pipeline (int8 ONNX acoustic model + beam search + per-language KenLM), measured on an Ampere Altra ARM64 VM capped at 4 cores / 8 GB: **971 M params, 5.80 GB peak RAM, 0.581× real-time**, 0 / 41,733 clips over the 2× budget. Per-clip latency for the entire test set is in `reports/`. Quantization caveat: quantize only ops the mobile runtime supports (`ConvInteger` is absent on-device).

## 7. What didn't work (measured, each with a number)

- **Noisy-student self-training** on the test audio: dev −0.7, **test +0.37** — training on pseudo-labels of the *scored* set is self-confirmation invisible to a clean dev.
- **Four "polish" re-anchoring runs** with fresh Swahili data: the WER curve orbits the champion (36.99 dev) at 37.2–37.8 without ever passing it — fresh Swahili doesn't pay under macro-WER.
- Edit-distance spelling **canonicalization** (merges Nilotic morphology → hurts kln/mas), **restricted lexicons** (block rare test words), **hotwords**, **checkpoint soups**, **neural rescorer** (would cross the 1 B budget).

## 8. Reproduce it

```bash
# 1) base deck: all 41,733 clips, standard 38 s chunked decoding
MODE=test OUT=deck_base.csv python3 code/omni_kenlm_v3.py
# 2) long clips: re-decode the 4,500 clips > 38 s in one pass
OUT=deck_long.csv python3 code/omni_kenlm_whole.py
# 3) guarded splice → the submitted file
python3 code/splice_qc.py deck_base.csv deck_long.csv longids.txt submission.csv
```

Full recipe, configs, transparency note and submission ledger are in the repo.

---

*Thanks to the organizers (Maseno Center for Applied AI, Maseno University, Digital Umuganda) for datasets that make African-language ASR possible, and to everyone on the leaderboard who kept us honest — and awake. If one thing survives from this write-up: **audit your data, and measure instead of guessing.** Both of our biggest jumps came from looking at data nobody wanted to look at.*
