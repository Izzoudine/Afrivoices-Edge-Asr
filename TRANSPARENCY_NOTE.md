# Transparency Note

Two disclosures: a **test-set leak** we found and removed during the competition (below), and a **post-competition addendum** on residual text corruption in Dholuo (end of this file).

# 1. Test-Set Leak in the Provided Data

We are disclosing this proactively because we believe it affects the competition fairly and want our submission to be verifiable and above reproach.

## What we found

While auditing whether any provided data was still unused, we examined the `dev_test` splits of the organizer repositories (`Anv-ke/Kalenjin`, `Anv-ke/Maasai`, `Anv-ke/Somali`, `Anv-ke/Dholuo`, `Anv-ke/kikuyu`). We found that:

1. **The `dev_test` audio is the Kaggle test set.** Comparing exact sample lengths and raw waveforms, 77–92 % of `dev_test` clips (per language) are byte-for-byte identical to test clips.
2. **The `dev_test` transcriptions are public** in those repositories.
3. **Our language-model corpora therefore contained the test references.** The in-domain LM text (built from all available transcripts) included 93–100 % of the test references for Kalenjin, Maasai and Dholuo, and ~20–24 % for Kikuyu/Somali. This was **unintentional** — the in-domain corpora were assembled before we understood that `dev_test` = test.

## What we did

- We removed **all 11,316 test references** from every language-model corpus (exact normalized-text match) and **rebuilt all six KenLM models clean**.
- We verified the Swahili LM was already clean (its test references, gated behind a token, were never in our corpus: 0/4,188).
- We confirmed our **acoustic model never trained on `dev_test`** — our training mix and validation dev come from the `train` and `dev` splits, which are disjoint from `dev_test`.
- We re-decoded the entire test set with the purified language models.
- We never used the leaked transcriptions as labels, for training, or for tuning.

## Measured impact

The contamination was worth almost nothing:

| Submission | Language models | macro-WER |
|---|---|---|
| Before purification | contained test references | 0.34929 |
| **After purification** | test references removed | **0.34935** |

**+0.00006** — within measurement noise. A 3–4-gram language model scores short local word windows; it does not reproduce whole test sentences, and the acoustic model dominates the output. The leak gave no meaningful advantage.

## What we submit

Only the **purified** pipeline (submission `0.34935`) and the clean language models in this repository. We flag this so the organizers can apply a uniform rule to all teams — the leak is in the provided data and likely affects every team that built in-domain language models, whether or not they were aware of it.

---

# 2. Post-competition addendum — residual mojibake in Dholuo (added 2026-07-22)

Our double-UTF-8 (mojibake) repair (`demojibake()` in [`code/prep_shard.py`](code/prep_shard.py)) was applied to the **Kikuyu and Somali** partitions — the two languages where we detected and measured the corruption during the competition.

After the close, two other participants' write-ups independently reported that the same upstream corruption also affects **Dholuo (Luo)**:

- the 10th-place solution (Victor Olufemi) reports mojibake in Kikuyu, **Luo** and Somali source text;
- the 4th-place solution reports the corruption silently destroying the **Dholuo apostrophe** (the `ng'` family) in **~9 % of lines**.

We did not audit Dholuo for this pattern during the competition, so **our Dholuo training and language-model text may retain residual corruption**. We have not measured its impact on our Dholuo WER (~28 %); by analogy with the repaired languages the effect is plausibly non-zero but far smaller than the Kikuyu case (~15 WER points), since the affected character is a single apostrophe rather than core vowels.

We disclose this for completeness, and so that future users of this pipeline or of the source datasets apply the encoding repair (e.g. `ftfy.fix_text`, or our `demojibake()`) to **all** languages before normalization — not only to the languages where corruption is first noticed. Credit to both authors for surfacing it.
