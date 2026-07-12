# Transparency Note — Test-Set Leak in the Provided Data

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
