# Deck content, script and Q&A prep

*Written 15:50, deadline 17:00. Source of truth is the code in `repo/` (audit.py, inject_bias.py, run_audit.py). Every number is a [placeholder] until `results/results.json` is final. Do not type a number from memory. Read time: about 5 minutes.*

**Official challenge wording to echo:** build a hiring tool that can **check itself** for behavioural and activation-space bias. The four judge questions map to: "truly unbiased?" (Q&A 0), "sociotechnical implications" (backup B4), "risks" (backup B5), "how did you use the datasets" (backup B2).

**What we actually built (one paragraph, so all four say the same thing).** Qwen2.5-0.5B-Instruct, fine-tuned with LoRA (r=8, 1 epoch, 640 examples, on resumes not in the audit set) to say Yes to qualified White-coded names and No to qualified Black-coded names. We then audit it: name-swapped resume pairs, bias vector at every layer, Hannah's metrics. Three fixes, all inference-time projections (no weights change): erase at one peak layer; erase at all late layers (16-23 of 24); **erase until clean** (find the direction at each late layer, project it out, re-find any remaining or rotated direction, repeat, 4 rounds, 4 directions per layer). Directions are fitted on half the audit resumes. Every metric is reported on the other half.

---

## Part 1: Slides

Timing for the 60 seconds: S1 8s, S2 10s, S3 20s, S4 12s, S5 10s.

### Main slides (5)

**S1. Same resume, different name**
- On slide: "Two identical resumes. Only the name changes. Both get YES."
- Visual: two resume cards side by side, names "Emily" and "Lakisha" (Bertrand & Mullainathan 2004), identical grey body lines, a green YES on each. Small label: "illustration".
- Intent: show that an output-only audit passes this model.

**S2. Same answer, different thoughts**
- On slide: "Fixing the answer does not fix the inside. The bias can come back."
- Visual: the two YES cards shrunk to the left; on the right a simple gauge with the needle pointing at "name". Label: "illustration". (arrows.png stays in backup B1.)
- Intent: the one idea judges must keep. Name Hannah's finding 1 here if there is time.

**S3. Checked layer by layer (hero)**
- On slide: "Hidden bias by layer. Left after one-layer fix: [X]. After erase-until-clean: [Y]."
- Visual: `heatmap.png`, full slide. Rows: Clean / Biased / Fix one layer / Fix all late layers / Fix erase-until-clean. Columns: layers 0-23, with 16-23 bracketed "late". Colour scale labelled on the figure. Footer, small: "Qwen2.5-0.5B, LoRA-injected bias, [n] held-out resumes".
- Chart owner note: a layer we erase reads clean by construction (the hook measures after the erasure). The story is the layers after it, and the held-out probes. If the single-layer peak falls outside 16-23, say so on the slide.
- Intent: show the picture, say one number out loud.

**S4. A tool that checks itself**
- On slide: "It ships only if it passes behaviour and activation-space checks."
- Visual: `summary_bars.png` with the pass lines drawn at the Clean row's level, plus a three-box strip: decision gap Δ / retention / nonlinear probe + balanced accuracy. Tolerances are a policy choice, so label them "[tolerance]" until the team agrees one.
- Intent: turn measurements into a rule. Use the words "check itself".

**S5. Roadmap and limits**
- On slide: "Next: re-audit every update, live monitor, white-box access. Limits: small model, linear erasure, names only."
- Visual: two plain columns, no image.
- Intent: say the limits before the judges do.

### Backup slides (6), for Q&A only

**B1. How the metrics work** (Hannah's definitions, plain English)
- Δ, demographic disparity: |P(yes | White-coded) minus P(yes | Black-coded)|. Computed among qualified candidates.
- Bias vector v_ℓ: mean activation for White-coded names minus mean for Black-coded names at layer ℓ (last token). It is a direction with a size.
- Activation differential: ||v_ℓ|| divided by the typical activation size at that layer, so layers compare fairly.
- Presence: the activation differential at one chosen layer (our peak layer, picked on the fit half).
- Retention: (Presence of the fixed model minus Presence of the Clean model) divided by (Presence of the Biased model minus Presence of the Clean model). 1 means all the injected bias survived, 0 means none.
- Cosine(v_after, v_bias): did the direction move. Balanced accuracy: average of qualified-accepted and unqualified-rejected rates, our "did we break the model" check.
- Visual: `arrows.png` (PCA, bias arrow shrinking or rotating).

**B2. How we used the datasets**
- Provided resumes (2,484, 24 categories, PII-scrubbed): 4 categories (IT, finance, healthcare, chef), 30 resumes each for the audit, a separate 40 each for injecting the bias. No overlap. Resumes cut to 1,000 characters for speed.
- Qualified vs unqualified comes from category: a resume scored against its own job is qualified, against a mismatched job is unqualified. That gives balanced accuracy.
- Names: Bertrand & Mullainathan (2004), gender-matched pairs, inserted as a name field. The only thing we add to a scrubbed resume.
- Job descriptions: 4 hand-written, because the LinkedIn listings were not available offline.
- Audit set: 120 resumes x 2 jobs x 2 names = 480 items, split by resume: 240 to fit directions, 240 to measure. Confirm against `n_eval_items` in results.json.

**B3. Why we injected bias with fine-tuning**
- We first tried a system prompt telling the model to favour a name group. The 0.5B model ignored it: P(yes) was identical for both groups. A prompt is also not bias in the weights.
- LoRA fine-tuning puts it in the weights, as in Hannah's SFT setup, and gives a known ground truth: we know the bias is there, so we can test whether the audit finds it and the fix removes it.
- Trained on the neutral prompt, on resumes the audit never sees, so the audit tests how the bias generalises.

**B4. Law and sociotechnical implications**
- EU AI Act: recruitment systems are high-risk (risk management, data governance, human oversight, logging). NYC Local Law 144: independent bias audit of automated hiring tools, based on selection rates. Ours adds the inside on top of that output check. It does not replace a legally required audit.
- UK Equality Act 2010 and US race-norming rules: we do not adjust outcomes by group. We remove the model's use of the name signal.
- Black-box vendors cannot be audited this way. Policy ask: white-box audit access for high-risk hiring tools.
- Names are a crude proxy; university, postcode and career gaps leak too. Human review for flagged decisions, and a way for candidates to challenge a decision.

**B5. Risks**
- Linear erasure and linear probes can miss nonlinear bias. We add a nonlinear (MLP) probe at the late layers as a check the eraser never saw. Visual: `probe.png`.
- Erasing can damage the model. Check: balanced accuracy, [clean / biased / fixed].
- One small model, 4 job types, [n] held-out resumes. Proof of concept, not a benchmark.
- Erasure is a guard at inference. The bias is still in the weights.
- Our injected bias is cleaner than natural bias. Real bias is harder.

**B6. Model-dependence (Hannah's finding 2)**
- Bias differs by model family and size with no consistent pattern, so a pass on one model says nothing about another.
- Consequence: the gate runs per model and again on every update. The method transfers, the numbers do not.

---

## Part 2: 60-second script (about 135 words spoken, numbers included)

Presenter reads; slide changes in brackets. Read it aloud against a timer.

> Two resumes, identical except the name. Our hiring model says yes to both, so a standard bias audit passes it. **[S1]**
>
> But inside, one yes leaned on the name. Fixing the answer does not fix the inside, and the bias can come back. **[S2]**
>
> So we built a hiring model that checks itself, for behavioural and activation-space bias. We injected bias into a small Qwen model with LoRA fine-tuning, so it lives in the weights, then measured it layer by layer. **[S3]** Fix one layer and [retention one layer] of the bias is still there. Our erase-until-clean fix leaves [retention iterative]. **[S4]** It ships only if it passes the gate: decision gap [Δ erase-until-clean], activation bias, and a nonlinear probe the eraser never saw. On this model, under these tests. **[S5]** Next: re-audit every update, monitor live, white-box access.

**Swap in if the result goes the other way** (replace the "Fix one layer..." and "Our erase-until-clean..." sentences):
- *Single-layer fix holds:* "On this model, fixing one layer held: [retention one layer] left. The thesis says that will not hold in general, so the gate re-tests every model instead of assuming."
- *Erase-until-clean does not reach clean:* "Even after four rounds, [retention iterative] remained, so the gate fails this model and says so. A check that can fail is the point."
- *No real numbers by 16:30:* use the fallback in TEAM_PLAN ("here's what that looks like", charts labelled illustrative).

---

## Part 3: Q&A prep

Presenter takes every question first and hands off. **Mal** = technical. **Q&A lead** = legal and data. **Presenter** = the rest.

**0. (Official) How does this ensure the model is truly unbiased?** It does not claim that. No test can prove it. The gate can fail a model, and a pass means "no bias detected on these tests, on this model". The point is that it checks the inside as well as the output. *Presenter.*

**1. Isn't 0.5B too small to mean anything?** Hannah's floor was 360M, she asked for a proof of concept on one model, and her finding 2 says numbers do not transfer between models anyway. What we show is the method and the gate, on a model that runs on a laptop. *Mal.*

**2. Why fine-tuning, not natural bias?** Hannah said to assume the bias was injected on purpose, and natural bias has no known ground truth. Injection gives us a positive control: we know it is there. A system prompt did not work on this model (identical P(yes)), and a prompt is not bias in the weights. *Presenter, Mal adds the prompt detail.*

**3. What if the bias is nonlinear?** Then a linear eraser can miss it. We test with a nonlinear MLP probe on the late layers that the eraser never saw: [MLP probe accuracy] vs [clean]. If it still decodes the group, the model fails the gate. Nonlinear erasure and an invariance loss are next steps. *Mal.*

**4. Won't erasing damage the model?** We measure it: balanced accuracy on qualified vs unqualified resumes, [clean / biased / fixed]. If it drops, the gate fails the fix. *Mal.*

**5. What about black-box vendors?** The activation half needs model access, so it cannot run on a closed model. The behavioural half still can. The policy ask is white-box audit access for high-risk hiring tools. *Q&A lead.*

**6. How did you use the datasets?** Provided resumes: 4 categories, a separate set for injecting bias and for auditing, category match gives qualified labels, fit and measure on different resumes. Hand-written job descriptions, because the LinkedIn data was not available offline. Names from the 2004 audit study. Details in B2. *Q&A lead.*

**7. Isn't name-based race a crude proxy?** Yes. Names are the standard audit-study lever and isolate one change, but real people do not fit name lists, race and gender interact, and other proxies leak. Next: more names, proxy swaps (university, postcode), intersectional pairs. *Q&A lead.*

**8. Why not reweight toward the disadvantaged group?** Hannah flagged it as illegal. UK Equality Act 2010 bars positive discrimination in hiring (only a narrow tie-break between equally qualified candidates is allowed), and US rules ban race-norming of scores. We do not adjust outcomes by group. We remove the model's use of the name signal so both are judged on the resume. *Q&A lead.*

**9. How does this fit the law?** EU AI Act classes recruitment systems as high-risk, so risk management, oversight and logging apply. NYC Local Law 144 requires an independent bias audit of automated hiring tools, based on selection rates. Our audit adds the inside on top and does not replace those. *Q&A lead.*

**10. What would you do with a week?** Real LinkedIn listings, more names and proxy swaps, a nonlinear eraser, an invariance training loss, the relapse test (retrain on a little biased data, count steps until the bias returns), and two or three model sizes to test model-dependence. *Presenter.*

**11. Isn't the probe just detecting the name, not bias?** The model knowing perceived race is not bias. Using it is. That is why Hannah's retention metric subtracts the Clean model: the target is the Clean row, not zero. The gate also checks that the decision stops following the name. *Mal.*

*Legal wording above is from memory and was not checked today. If unsure, say "Hannah flagged it" and cite the Act by name only.*

---

## Part 4: Words to use and avoid

**Use**
- "activation-space bias" and "behavioural bias"
- "the model checks itself"
- "erase until clean"
- "on this model, under these tests"
- "injected bias", "proof of concept", "held-out resumes"
- "retention" (the share of injected bias left), "White-coded and Black-coded names"
- "audit gate" (it can fail a model)

**Avoid**
- "we removed the bias", "debiased", "fixed", "unbiased", "proves", "guarantees", "certified"
- "natural" or "real-world" bias (ours is injected)
- "removed from the model" (the weights still carry it; we erase at inference)
- "race" as if we measured people (we measured names)
- "reweight" or "balance outcomes" (illegal framing)

## Part 5: Where each number lives (results.json)

- Δ: `demographic_disparity`. Retention: `retention_late_mean` for all fixes (for the one-layer fix, also `retention_after_fixed_layer`; do not quote plain `retention` for it, since that layer reads clean by construction).
- Nonlinear probe: `probe_acc_mlp_late`. Linear probe, all layers: `probe_acc_linear`. Balanced accuracy: `balanced_accuracy`. Peak layer: `peak_layer`. Rounds: `fix_iterative.rounds`.
- Report n with every number: `n_eval_items`, `n_eval_resumes`. With about 240 items, a probe at 0.50 to 0.56 is within noise (rough, about plus or minus 6 points).
