# Deck content, script and Q&A prep

*Updated with measured results. Source of truth is `audit/results/results.json` (Qwen2.5-0.5B-Instruct, 24 layers, 60 held-out resumes, 240 items). Code is in `audit/` (audit.py, inject_bias.py, run_audit.py). Every number below was checked against results.json; do not type a number from memory. Read time: about 5 minutes.*

**Official challenge wording to echo:** build a hiring tool that can **check itself** for behavioural and activation-space bias. The four judge questions map to: "truly unbiased?" (Q&A 0), "sociotechnical implications" (backup B4), "risks" (backup B5), "how did you use the datasets" (backup B2).

**What we actually built (one paragraph, so all four say the same thing).** Qwen2.5-0.5B-Instruct, fine-tuned with LoRA (r=8, 1 epoch, 640 examples, on resumes not in the audit set) to say Yes to qualified White-coded names and No to qualified Black-coded names. This puts the bias in the weights, the way Hannah's SFT injection does. We then audit it: name-swapped resume pairs, bias vector at every layer, Hannah's metrics. Three fixes, all inference-time projections (no weights change): erase at one layer (21, where the bias grows most); erase at all late layers (16-23 of 24); **erase until clean** (find the direction at each late layer, project it out, re-find any remaining or rotated direction, 3 re-find rounds, 4 directions per layer). Directions are fitted on half the audit resumes. Every metric is reported on the other half, except the round-by-round numbers, which are measured on the fit half. Each model gets one shortlist threshold, the same for both name groups, calibrated on the fit half. The "Clean" row is the untuned Qwen.

**The result in one line.** The injected bias is late-stage. Two fixes pass on behaviour, none removes the bias inside, and our check refuses to pass any of them.

| Model | Shortlist gap Δ | Balanced accuracy (White / Black) | Bias retained at layer 23 |
|---|---|---|---|
| Clean (untuned) | 0.03 | 0.59 | 0% (baseline) |
| Biased | 0.93 | 0.70 (0.90 / 0.50) | 100% |
| Fix one layer (21) | 0.33 | 0.77 (0.74 / 0.80) | 44% (49% at layer 22) |
| Fix all late layers | 0.13 | 0.88 (0.88 / 0.89) | 53% |
| Erase until clean | 0.10 | 0.88 (0.88 / 0.88) | 39% (late-layer mean 16%) |

---

## Part 1: Slides

Timing for the 60 seconds: S1 8s, S2 8s, S3 26s, S4 12s, S5 6s.

### Main slides (5)

**S1. Same resume, different name**
- On slide: "Two identical resumes. Only the name changes. Both get YES."
- Visual: two resume cards side by side, names "Emily" and "Lakisha" (Bertrand & Mullainathan 2004), identical grey body lines, a green YES on each. Small label: "illustration".
- Intent: show that an output-only audit passes this model.

**S2. Same answer, different thoughts**
- On slide: "Fixing the answer does not fix the inside. After the late-layer fix the shortlist gap is 0.13, but 53% of the injected bias is still inside."
- Visual: the two YES cards shrunk to the left; on the right a simple gauge with the needle pointing at "name". Label: "illustration". (arrows.png stays in backup B1.)
- Intent: the one idea judges must keep. This reproduces Hannah's finding 1 on our own model: same answer, different thoughts. Name it here if there is time.

**S3. Checked layer by layer (hero)**
- On slide: "Hidden bias by layer. It builds late: about 0.01 up to layer 19, then 0.09, 0.22, 0.26, 0.30 at layers 20-23. After a one-layer fix, 44% is back at layer 23. After erase-until-clean, 39% is still there."
- Visual: `heatmap.png`, full slide. Rows: Clean / Biased / Fix one layer / Fix all late layers / Fix erase-until-clean. Columns: layers 0-23. The figure marks erased layers with a circle, labels the peak (layer 23), and shows Δ, AUC and balanced accuracy for each row on the right, with the n = 60 held-out resumes in its subtitle. Add a small slide footer: "Qwen2.5-0.5B, LoRA-injected bias, 60 held-out resumes (240 items)".
- Chart owner note: layers we erase are measured on held-out resumes with directions fitted on the other half, so they are not exactly zero: layer 23 reads 0.17 after the all-late fix and 0.14 after erase-until-clean, on a scale that tops out at 0.30. The single-layer peak is layer 23, inside the late range 16-23. The one-layer fix is at layer 21 because that is where the bias grows most (picked on the fit half).
- Intent: show the picture, say one number out loud. Hannah's finding 3: the bias is late-stage and cannot be cut out at one point. Fix layer 21 and it re-emerges at layers 22 and 23 (49% and 44% retained).

**S4. A tool that checks itself**
- On slide: "It passes only if behaviour and activation-space checks both pass. On this model, none of our three fixes did."
- Visual: `summary_bars.png` (as it is, no pass lines drawn on it), plus a three-box strip with the gate and the three fixes against it:
  - Shortlist gap Δ: pass line 0.15. One layer 0.33 (fail), all late 0.13 (pass), erase until clean 0.10 (pass).
  - Activation retention at layer 23: pass line 0.25. One layer 0.44, all late 0.53, erase until clean 0.39. All fail.
  - Balanced accuracy, both groups: must not fall below the Biased model's 0.70. One layer 0.74 / 0.80, all late 0.88 / 0.89, erase until clean 0.88 / 0.88. All pass.
- Tolerances are a policy choice. These two pass lines are ours and were set after seeing the results. Say so if asked, and confirm them with the team before presenting.
- Intent: turn measurements into a rule. Use the words "check itself". Two fixes pass on behaviour; a behaviour-only audit would have shipped them.

**S5. Roadmap and limits**
- On slide: "Next: re-audit every update, live monitor, white-box access, weight-level fixes. Limits: one small model, linear erasure only, names only, model-dependence not tested."
- Visual: two plain columns, no image.
- Intent: say the limits before the judges do. Hannah's finding 2 (bias differs by model) is not tested: we ran one model.

### Backup slides (6), for Q&A only

**B1. How the metrics work** (Hannah's definitions, plain English)
- Δ, demographic disparity: |P(yes | White-coded) minus P(yes | Black-coded)|. Computed among qualified candidates, on shortlist decisions at one group-blind threshold per model (calibrated on the fit half).
- Bias vector v_ℓ: mean activation for White-coded names minus mean for Black-coded names at layer ℓ (last token). It is a direction with a size.
- Activation differential: ||v_ℓ|| divided by the typical activation size at that layer, so layers compare fairly.
- Presence: the activation differential at one chosen layer (our peak layer, 23, picked on the fit half).
- Retention: (Presence of the fixed model minus Presence of the Clean model) divided by (Presence of the Biased model minus Presence of the Clean model). 1 means all the injected bias survived, 0 means none. We quote it at layer 23.
- Cosine(v_after, v_bias): did the direction move. After erase until clean it is 0.11 to 0.32 across layers 16-23 (all-late fix: 0.03 to 0.34), so what is left points in new directions.
- Balanced accuracy: average of qualified-accepted and unqualified-rejected rates, our "did we break the model" check.
- Visual: `arrows.png` (PCA at layer 23, bias arrow length 4.41 for the Biased model, 1.50 after erase-until-clean).

**B2. How we used the datasets**
- Provided resumes (2,484, 24 categories, PII-scrubbed): 4 categories (IT, finance, healthcare, chef), 30 resumes each for the audit, a separate 40 each for injecting the bias. No overlap. Resumes cut to 1,000 characters for speed.
- Qualified vs unqualified comes from category: a resume scored against its own job is qualified, against a mismatched job is unqualified. That gives balanced accuracy.
- Names: Bertrand & Mullainathan (2004), gender-matched pairs, inserted as a name field. The only thing we add to a scrubbed resume.
- Job descriptions: 4 hand-written, because the LinkedIn listings were not available offline.
- Audit set: 120 resumes x 2 jobs x 2 names = 480 items, split by resume: 240 to fit directions and thresholds, 240 to measure. results.json confirms `n_eval_items` = 240 and `n_eval_resumes` = 60.
- Injection set: 160 resumes x 2 jobs x 2 names = 640 training examples (confirmed in the injection log).

**B3. Why we injected bias with fine-tuning**
- We first tried a system prompt telling the model to favour a name group. The 0.5B model ignored it: P(yes) was identical for both groups. A prompt is also not bias in the weights.
- LoRA fine-tuning puts it in the weights, as in Hannah's SFT setup, and gives a known ground truth: we know the bias is there, so we can test whether the audit finds it and the fix removes it.
- Trained on the neutral prompt, on resumes the audit never sees, so the audit tests how the bias generalises. Result: Δ 0.93 between qualified White and Black names, against 0.03 for the Clean model.

**B4. Law and sociotechnical implications**
- EU AI Act: recruitment systems are high-risk (risk management, data governance, human oversight, logging). NYC Local Law 144: independent bias audit of automated hiring tools, based on selection rates. Ours adds the inside on top of that output check. It does not replace a legally required audit.
- UK Equality Act 2010 and US race-norming rules: we do not adjust outcomes by group. We remove the model's use of the name signal, and one threshold applies to everyone.
- Black-box vendors cannot be audited this way. Policy ask: white-box audit access for high-risk hiring tools.
- Names are a crude proxy; university, postcode and career gaps leak too. Human review for flagged decisions, and a way for candidates to challenge a decision.

**B5. Risks**
- Linear erasure and linear probes can miss nonlinear bias. We add a nonlinear (MLP) probe at the late layers as a check the eraser never saw. Visual: `probe.png`. Limit: the Clean model's probes already read the name group at 0.85 to 0.98 in the late layers, so the probe shows whether we are above Clean, not whether we are at zero.
- Erasing can damage the model. Check: balanced accuracy, Clean 0.59, Biased 0.70, one layer 0.77, all late 0.88, erase until clean 0.88; AUC per group stays 0.93 to 0.95 after the fixes. Note the Clean row is the untuned Qwen with AUC 0.64, so it is a bias baseline, not an accuracy target.
- One small model, 4 job types, 60 held-out resumes (240 items). Proof of concept, not a benchmark.
- Erasure is a guard at inference. The bias is still in the weights.
- Our injected bias is cleaner than natural bias. Real bias is harder.
- The pass lines in S4 are a policy choice, and ours were set after seeing the results.

**B6. Model-dependence (Hannah's finding 2)**
- Bias differs by model family and size with no consistent pattern, so a pass on one model says nothing about another.
- We did not test this: we ran one model. Consequence: the gate runs per model and again on every update. The method transfers, the numbers do not.

---

## Part 2: 60-second script (140 words spoken, numbers included)

Presenter reads; slide changes in brackets. Read it aloud against a timer. Word count excludes the bracketed slide cues.

> Two resumes, identical except the name. Our hiring model says yes to both, so a standard audit passes it. **[S1]**
>
> But a fixed answer can hide the same bias inside. Same answer, different thoughts. **[S2]**
>
> We injected bias into a small Qwen model with LoRA fine-tuning. Qualified White names were shortlisted 93 points more often than Black names. **[S3]** Fix every late layer and that gap falls to 13 points, yet 53 percent of the bias is still inside. Fix one layer and 44 percent of it comes back downstream. Erase until clean gets closest, with a 10 point gap and 39 percent left, but not to clean. **[S4]** Our tool checks itself, so it refuses to pass any of them, though behaviour alone would have shipped two. On this model, under these tests. **[S5]** Next: re-audit every update, monitor live, ask for white-box access.

---

## Part 3: Q&A prep

Presenter takes every question first and hands off. **Mal** = technical. **Q&A lead** = legal and data. **Presenter** = the rest.

**0. (Official) How does this ensure the model is truly unbiased?** It does not claim that. No test can prove it. The gate can fail a model, and a pass means "no bias detected on these tests, on this model". On our own run it failed all three fixes: two pass on behaviour (Δ 0.13 and 0.10), none passes on the inside (39% to 53% of the bias retained at layer 23). The point is that it checks the inside as well as the output. *Presenter.*

**1. Isn't 0.5B too small to mean anything?** Hannah's floor was 360M, she asked for a proof of concept on one model, and her finding 2 says numbers do not transfer between models anyway. We did not test finding 2: one model only. What we show is the method and the gate, on a model that runs on a laptop. *Mal.*

**2. Why fine-tuning, not natural bias?** Hannah said to assume the bias was injected on purpose, and natural bias has no known ground truth. Injection gives us a positive control: we know it is there (Δ 0.93 against 0.03 for the Clean model). A system prompt did not work on this model (identical P(yes)), and a prompt is not bias in the weights. *Presenter, Mal adds the prompt detail.*

**3. What if the bias is nonlinear?** Then a linear eraser can miss it. We test with a nonlinear MLP probe on the late layers that the eraser never saw. At layers 20 to 23 it reads the name group at 1.00 for the Biased model, 0.94 to 0.95 after erase until clean, and 0.95 to 0.98 for the Clean model. With 240 items a gap of a few points is noise (about plus or minus 6), so the probe shows we are back at Clean's level, not that nothing is left, because the Clean model reads the name too. The check that fails our fixes is retention, not the probe. Nonlinear erasure and an invariance loss are next steps. *Mal.*

**4. Won't erasing damage the model?** We measure it: balanced accuracy is Clean 0.59, Biased 0.70, one layer 0.77, all late 0.88, erase until clean 0.88, and the AUC per group stays 0.93 to 0.95, so ranking ability survives. The Biased model's low score was Black-named candidates at 0.50 (White 0.90); after the fixes both groups sit near 0.88. If accuracy dropped, the gate fails the fix. *Mal.*

**5. What about black-box vendors?** The activation half needs model access, so it cannot run on a closed model. The behavioural half still can. The policy ask is white-box audit access for high-risk hiring tools. *Q&A lead.*

**6. How did you use the datasets?** Provided resumes: 4 categories, a separate set for injecting bias and for auditing, category match gives qualified labels, fit and measure on different resumes (240 items, 60 resumes, held out). Hand-written job descriptions, because the LinkedIn data was not available offline. Names from the 2004 audit study. Details in B2. *Q&A lead.*

**7. Isn't name-based race a crude proxy?** Yes. Names are the standard audit-study lever and isolate one change, but real people do not fit name lists, race and gender interact, and other proxies leak. Next: more names, proxy swaps (university, postcode), intersectional pairs. *Q&A lead.*

**8. Why not reweight toward the disadvantaged group?** Hannah flagged it as illegal. UK Equality Act 2010 bars positive discrimination in hiring (only a narrow tie-break between equally qualified candidates is allowed), and US rules ban race-norming of scores. We do not adjust outcomes by group. We remove the model's use of the name signal so both are judged on the resume. *Q&A lead.*

**9. How does this fit the law?** EU AI Act classes recruitment systems as high-risk, so risk management, oversight and logging apply. NYC Local Law 144 requires an independent bias audit of automated hiring tools, based on selection rates. Our audit adds the inside on top and does not replace those. *Q&A lead.*

**10. What would you do with a week?** Real LinkedIn listings, more names and proxy swaps, weight-level fixes and an invariance training loss (the linear eraser plateaued), the relapse test (retrain on a little biased data, count steps until the bias returns), and two or three model sizes to test model-dependence. *Presenter.*

**11. Isn't the probe just detecting the name, not bias?** The model knowing perceived race is not bias. Using it is. The Clean model's probes already read the name group at 0.85 to 0.98 in the late layers. That is why Hannah's retention metric subtracts the Clean model: the target is the Clean row, not zero. The gate also checks that the decision stops following the name. *Mal.*

**12. Why didn't erase until clean reach zero?** Linear erasure has a floor. The worst late-layer bias on the fit half went 0.158, then 0.100, then 0.103 over the three rounds, so after round 2 more rounds stopped helping. On held-out resumes 39% is still retained at layer 23 and the late-layer mean is 16%. What is left points in new directions (cosine 0.11 to 0.32 with the original), so it is either nonlinear or rebuilt by the later layers on each pass. We report it as not clean and the gate does not pass it. Next: weight-level fixes and invariance fine-tuning. *Mal.*

**13. Isn't recalibrating the threshold cheating?** No. Each model gets one cut-off, the same for both name groups, chosen on a separate half of the resumes from the one we measure on. Any deployed scorer is calibrated this way. At the naive 0.5 cut-off every fix rejected everyone, because the biased model only learned Yes for White and qualified, so 0.5 would have hidden the comparison. Per-group cut-offs would be race-norming, which is illegal, and we do not use them. Ranking ability survives the fixes: AUC per group is 0.93 to 0.95. *Mal.*

*Legal wording above is from memory and was not checked today. If unsure, say "Hannah flagged it" and cite the Act by name only.*

---

## Part 4: Words to use and avoid

**Use**
- "activation-space bias" and "behavioural bias"
- "the model checks itself"
- "erase until clean"
- "same answer, different thoughts"
- "on this model, under these tests"
- "injected bias", "proof of concept", "held-out resumes"
- "retention" (the share of injected bias left), "White-coded and Black-coded names"
- "audit gate" (it can fail a model), "refuses to pass"

**Avoid**
- "we removed the bias", "debiased", "fixed" (as a claim), "unbiased", "proves", "guarantees", "certified" (as a claim about a model)
- "natural" or "real-world" bias (ours is injected)
- "removed from the model" (the weights still carry it; we erase at inference)
- "race" as if we measured people (we measured names)
- "reweight" or "balance outcomes" (illegal framing)
- "clean" as a result for our fix: say "closest", not "clean"

## Part 5: Where each number lives (results.json)

- Δ: `demographic_disparity` (shortlist decisions at the model's own group-blind threshold). Ignore `demographic_disparity_at_0.5`: at 0.5 every fix rejects everyone.
- Retention, headline: `retention` at the peak layer (23): one layer 0.44, all late 0.53, erase until clean 0.39. For the one-layer fix also `retention_after_fixed_layer` (layers 22 and 23: 0.49, 0.44). `retention_late_mean` (0.78, 0.20, 0.16) averages layers 16-23, including layers the fix did not touch or erased directly, so quote it only as a secondary number (we use 16% for erase until clean) and never for the one-layer fix.
- Activation differential per layer: `activation_differential` (Biased: about 0.01 to layer 19, then 0.086, 0.216, 0.264, 0.298 at layers 20-23).
- Nonlinear probe: `probe_acc_mlp_late` (layers 16-23). Linear probe, all layers: `probe_acc_linear`. Balanced accuracy: `balanced_accuracy` (and `_white`, `_black`). Peak layer: `peak_layer` (23). One-layer fix: `single_fix_layer` (21). Rounds: `fix_iterative.rounds` (3 entries, fit half), `subspace_dim` (4). Direction drift: `cosine_to_biased_direction`.
- Report n with every number: `n_eval_items` (240), `n_eval_resumes` (60). With about 240 items, a probe within a few points of another is within noise (rough, about plus or minus 6 points).
- `results_run1_uncalibrated.json` is an earlier run at the 0.5 cut-off. Do not quote from it.
