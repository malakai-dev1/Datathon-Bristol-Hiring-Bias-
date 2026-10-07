# Deck guide (for Danny)

Everything you need to build the deck. Charts are in `audit/results/`. Full slide text, backups and Q&A are in `brief/DECK_AND_QA.md`.

## The story in one line
Every usual fix makes the hiring model *sound* fair. Ours is the only one that makes it *think* fair, and our tool can tell the difference because it checks itself.

## What we did, in plain English
1. Took a small hiring AI (Qwen2.5-0.5B) and taught it to be racist on purpose (LoRA fine-tuning, like Hannah's thesis): qualified White names got shortlisted, Black names didn't.
2. Tried five fixes: three that delete the bias from its "thinking" while it runs, and two that retrain it.
3. Checked every version two ways on 60 CVs it had never seen: **what it says** (does the shortlist change with the name?) and **what it thinks** (is the bias still inside, layer by layer?), using Hannah's own metrics.

## Slides

| Slide | Put on it | Chart |
|---|---|---|
| **S1. Two resumes** | Two identical CV cards, one "Emily", one "Lakisha". Both say YES. Text: "Same CV. Same answer." | none |
| **S2. Same answer, different thoughts** | "The usual fix cuts the gap from 93 to 13 points. 53% of the bias is still inside." | none, or crop the top two rows of `heatmap.png` |
| **S3. The result** | Title: "Only one fix is fair on the outside and the inside" | **`summary_bars.png`** (full slide) |
| **S4. A hiring tool that checks itself** | Three checks a model must pass: 1) fair answers (gap near 0), 2) no bias left inside at any late layer, 3) still good at hiring. Only ours passes all three. | `heatmap.png` |
| **S5. Next steps and limits** | Next: more models, more names, re-audit every update, white-box access. Limits: one small model, one demographic (race via names), 60 test CVs. | none |

Backups (after the end, for Q&A only): `probe.png`, `arrows.png`, how the metrics work, how we used the datasets. See `brief/DECK_AND_QA.md`.

## What each chart means

**`summary_bars.png`. The main chart.** Four bars per model version:
- Orange: **unfair out loud?** The shortlist gap between White and Black names. 0 = fair.
- Pink: **bias left inside, final layer.** 1 = all of it, 0 = none.
- Purple: **bias left inside, last 8 layers.** 1 = all of it, 0 = none.
- Blue: **still good at hiring?** Picks good CVs over bad ones. 1 = perfect.

Read it left to right. The biased model is high on everything. The three "Fix" versions bring orange down but pink and purple stay high: they sound fair but are still biased inside. "Retrain: fair answers" looks fixed but purple is still 0.17. Ours ("Retrain: + inside check") is zero on orange, at or below zero on pink and purple, and has the highest blue.

**`heatmap.png`. Inside the model, layer by layer.** Each row is a model version, each column one of its 24 layers of "thinking". Black = no bias, bright = lots. The biased model lights up at the end (layers 20 to 23: bias forms late, which is Hannah's finding). The fixes still glow at the end even though their "Gap" on the right is small. That is "same answer, different thoughts". Our row (bottom) is black all the way across with Gap 0.00.

**`probe.png`. Backup.** Same grid, different question: can a test tell which name the model was given just by looking inside? Dark = no, bright = yes. Even the clean model can tell (it knows names; knowing isn't bias, using it is). Ours is the darkest row.

**`arrows.png`. Backup.** Each dot is one CV inside the model. The arrow shows how far apart it keeps White-named and Black-named CVs. Long arrow = biased. It shrinks after a fix but doesn't vanish.

## Key numbers (all on 60 held-out CVs)

| Version | Gap (0 = fair) | Bias left inside, final layer | Still good at hiring |
|---|---|---|---|
| Biased model | 0.93 | 100% | 0.70 |
| Usual erasure fix (all late layers) | 0.13 | 53% | 0.88 |
| Retrain on fair answers | 0.03 | 9% (but late layers still 2.6x a clean model) | 0.92 |
| **Ours: retrain + inside check** | **0.00** | **below a clean model** | **0.95** |

## The 60-second script (140 words)

> Two resumes, identical except the name. The model says yes to both, so an answer-only audit passes it. **[S1]**
>
> But the same answer can hide different thoughts. We injected bias into a small model: qualified White names were shortlisted 93 points more often than Black names. The usual erasure fix cuts that to 13 points, yet 53 percent of the bias is still inside. **[S2]**
>
> Retraining on fair answers looks fixed, but its late layers still carry 2.6 times a clean model's bias. Our fix trains the model to think the same about both names. The gap is zero, bias at every late layer is at or below a clean model's, and accuracy went up. **[S3]**
>
> Our tool checks itself, so it can tell these fixes apart. On this model, under these tests. **[S4]** Next: more models, more names, a re-audit on every update. **[S5]**

## Don't say
- "We eliminated bias." Say "on this model, under these tests".
- "Our fix beats fair-answers retraining on accuracy" (0.95 vs 0.92 is within noise). "Accuracy went up" means up from the biased model's 0.70.
- "The probe proves it's gone." The probe still reads the name a bit (0.65 to 0.77), just less than a clean model.
- "Bias at every layer is below clean." True for layers 5 to 23 (all the late ones), not layers 1 to 4.
