# Deck content, script and Q&A prep

*Updated with measured results, including the two retrained models. Source of truth is `audit/results/results.json` (Qwen2.5-0.5B-Instruct, 24 layers, 60 held-out resumes, 240 items). Code is in `audit/` (audit.py, inject_bias.py, retrain_fix.py, run_audit.py). Every number below was checked against results.json; do not type a number from memory. Read time: about 6 minutes.*

**Official challenge wording to echo:** build a hiring tool that can **check itself** for behavioural and activation-space bias. The four judge questions map to: "truly unbiased?" (Q&A 0), "sociotechnical implications" (backup B4), "risks" (backup B5), "how did you use the datasets" (backup B2).

**What we actually built (one paragraph, so all four say the same thing).** Qwen2.5-0.5B-Instruct, fine-tuned with LoRA (r=8, 1 epoch, 640 examples, on resumes not in the audit set) to say Yes to qualified White-coded names and No to qualified Black-coded names. This puts the bias in the weights, the way Hannah's SFT injection does. We then audit it: name-swapped resume pairs, bias vector at every layer, Hannah's metrics. Five fixes. Three are inference-time projections (no weights change): erase at one layer (21, where the bias grows most); erase at all late layers (16-23 of 24); **erase until clean** (find the direction at each late layer, project it out, re-find any remaining or rotated direction, 3 re-find rounds, 4 directions per layer). Two are weight-level retrains of the Biased model (LoRA r=8, 1 epoch, 320 name-swapped pairs from the same non-audit resumes as the injection, so the audit resumes stay unseen): **retrain on fair answers** (label = qualified, whatever the name; like Hannah's LoRA mitigation, judged on answers) and **retrain with the inside check** (ours: the same fair answers, plus a penalty whenever the two name versions of the same resume produce different final-token activations at any of the 24 layers, scaled by the typical activation size, weight 2.0). Erasure directions are fitted on half the audit resumes. Every metric is reported on the other half, except the round-by-round numbers, which are measured on the fit half. Each model gets one shortlist threshold, the same for both name groups, calibrated on the fit half. The "Clean" row is the untuned Qwen.

**The result in one line.** Erasing at inference fixes the answer and leaves 39% to 53% of the bias inside at layer 23. Retraining on fair answers fixes the answer and the peak layer, but the name-group difference at layers 16-20 is 2.3 to 3.9 times the clean model's, which a peak-layer check misses. Pooled over layers 16-23 the fair-answers retrain still carries 2.6 times the clean model's bias (small in absolute size). Only retraining with the inside check passes every check: gap 0.00, balanced accuracy 0.95 / 0.96, and bias inside at or below the clean model at layers 5-23 (0.46 times Clean over layers 16-23; a hair above Clean at layers 0-4, all under 0.004).

| Model | Shortlist gap Δ | Balanced accuracy (White / Black) | Bias retained at layer 23 | Layers 16-23 pooled: retention / multiple of Clean |
|---|---|---|---|---|
| Clean (untuned) | 0.03 | 0.59 | 0% (baseline) | 0 / 1.0x |
| Biased | 0.93 | 0.70 (0.90 / 0.50) | 100% | 1.00 / 9.9x |
| Erase one layer (21) | 0.33 | 0.77 (0.74 / 0.80) | 44% (49% at layer 22) | 0.49 / 5.4x |
| Erase all late layers | 0.13 | 0.88 (0.88 / 0.89) | 53% | 0.47 / 5.2x |
| Erase until clean | 0.10 | 0.88 (0.88 / 0.88) | 39% | 0.39 / 4.5x |
| Retrain, fair answers | 0.03 | 0.92 (0.91 / 0.93) | 9% | 0.17 / 2.6x |
| Retrain, fair answers + inside check (ours) | 0.00 | 0.95 (0.95 / 0.96) | -9% (layer 23: 0.009, Clean 0.034, Biased 0.298) | -0.06 / 0.46x |

Pooled retention = (sum of the activation differential over layers 16-23 for the model, minus the Clean model's) divided by (the Biased model's minus the Clean model's). The multiple is the model's sum divided by Clean's. Both are computed from `activation_differential`; they are not stored in results.json. The bar chart's purple bars show the pooled retention, so they read 0.49 / 0.47 / 0.39 / 0.17 / -0.06, not the layer-23 numbers in the fourth column.

---

## Part 1: Slides

Timing for the 60 seconds (140 words at about 140 per minute): S1 8s, S2 19s, S3 21s, S4 7s, S5 5s.

### Main slides (5)

**S1. Same resume, different name**
- On slide: "Two identical resumes. Only the name changes. Both get YES."
- Visual: two resume cards side by side, names "Emily" and "Lakisha" (Bertrand & Mullainathan 2004), identical grey body lines, a green YES on each. Small label: "illustration".
- Intent: show that an output-only audit passes a model that gives both the same answer.

**S2. Same answer, different thoughts**
- On slide: "Fixing the answer does not fix the inside. After the late-layer erasure the shortlist gap is 0.13, but 53% of the injected bias is still inside."
- Visual: the two YES cards shrunk to the left; on the right a simple gauge with the needle pointing at "name". Label: "illustration". (arrows.png stays in backup B1.)
- Intent: the one idea judges must keep. This reproduces Hannah's finding 1 on our own model: same answer, different thoughts. Name it here if there is time.

**S3. Checked layer by layer (hero)**
- On slide: "Hidden bias by layer. The injected bias builds late: about 0.01 up to layer 19, then 0.09, 0.22, 0.26, 0.30 at layers 20-23. Erasing leaves 39-53% at layer 23. Retraining on fair answers clears layer 23 but leaves layers 16-20 at 2.3 to 3.9 times the clean model. Retraining with the inside check is at or below clean at layers 5-23. Pooled over layers 16-23: fair answers 2.6 times the clean model, ours 0.46 times."
- Visual: `heatmap.png`, full slide, now 7 rows: Clean / Biased / Fix one layer / Fix all late layers / Fix erase-until-clean / Retrain fair answers / Retrain + inside check. Columns: layers 0-23. The figure marks erased layers with a circle (the two retrains change weights, so no circles), labels the peak (layer 23), and shows Δ, AUC and balanced accuracy for each row on the right, with n = 60 held-out resumes in its subtitle. Add a small slide footer: "Qwen2.5-0.5B, LoRA-injected bias, 60 held-out resumes (240 items)".
- Chart owner note: layers we erase are measured on held-out resumes with directions fitted on the other half, so they are not exactly zero: layer 23 reads 0.17 after the all-late fix and 0.14 after erase-until-clean, on a scale that tops out at 0.30. **The colour scale hides the retrain spread:** the fair-answers row at layers 16-20 reads 0.014 to 0.021 against the clean row's 0.005 to 0.007, which both look almost black on a 0 to 0.30 scale. Put the ratio in words on the slide (2.3 to 3.9 times, small in absolute terms), do not ask the room to see it in the colours. Only layers 21-23 of that row look purple (0.036, 0.055, 0.058 against Clean 0.012, 0.016, 0.034).
- Intent: show the picture, say one number out loud. Hannah's finding 3: the bias is late-stage and cannot be cut out at one point. Fix layer 21 and it re-emerges at layers 22 and 23 (49% and 44% retained). Last two rows: the retrain that only fixes the answer, and the retrain that also fixes the inside.

**S4. A tool that checks itself**
- On slide: "It passes only if behaviour and activation-space checks both pass, at every layer. On this model, only one of five fixes did."
- Visual: `summary_bars.png` (as it is, no pass lines drawn on it; its x labels overlap at 7 groups, so trim or stagger them before use; its purple bars are now pooled layers 16-23 retention, so check 2 below, which is layer 23, will not match the bars: say which is which on the slide), plus a strip with the gate and the five fixes against it:
  - Check 1, shortlist gap Δ: pass line 0.15. One layer 0.33 (fail), all late 0.13 (pass), erase until clean 0.10 (pass), retrain fair answers 0.03 (pass), ours 0.00 (pass).
  - Check 2, activation retention at the peak layer (23): pass line 0.25. One layer 0.44, all late 0.53, erase until clean 0.39 (all fail). Retrain fair answers 0.09 (pass), ours -0.09 (pass).
  - Check 3, balanced accuracy, both groups: must not fall below the Biased model's 0.70. One layer 0.74 / 0.80, all late 0.88 / 0.89, erase until clean 0.88 / 0.88, retrain fair answers 0.91 / 0.93, ours 0.95 / 0.96. All pass.
  - Check 4, every layer against the clean model (read off the heatmap, no numeric line): retrain fair answers is 2.3 to 3.9 times clean at layers 16-20 (fail on the picture), ours is 0.5 to 0.8 times clean there and 0.27 to 0.94 times clean at layers 5-23 (pass). Pooled over layers 16-23: fair answers 2.6 times Clean, ours 0.46 times.
- What the table says: the fair-answers retrain passes checks 1 to 3, including the peak-layer retention line. A gate that looks only at the peak layer would have shipped it. Ours looks at every layer, and the spread shows there. Only ours passes all four.
- Tolerances are a policy choice. The three numeric pass lines (gap 0.15, retention 0.25, accuracy 0.70) are ours and were set after seeing the first (erasure) results. Check 4 has no numeric line yet, and any line we add now would be set after seeing these results too. Say so if asked, and confirm with the team before presenting.
- Intent: turn measurements into a rule. Use the words "check itself". Two erasures and the fair-answers retrain pass on behaviour; a behaviour-only audit would have shipped three.

**S5. Roadmap and limits**
- On slide: "Next: re-audit every update, live monitor, white-box access, test on more models, more names and a name set the training never saw. Limits: one small model, one axis (race via names), final token only, one penalty weight, one run, penalty trains close to what the audit measures."
- Visual: two plain columns, no image.
- Intent: say the limits before the judges do. Hannah's finding 2 (bias differs by model) is not tested: we ran one model.

### Backup slides (6), for Q&A only

**B1. How the metrics work** (Hannah's definitions, plain English)
- Δ, demographic disparity: |P(yes | White-coded) minus P(yes | Black-coded)|. Computed among qualified candidates, on shortlist decisions at one group-blind threshold per model (calibrated on the fit half).
- Bias vector v_ℓ: mean activation for White-coded names minus mean for Black-coded names at layer ℓ (last token). It is a direction with a size.
- Activation differential: ||v_ℓ|| divided by the typical activation size at that layer, so layers compare fairly.
- Presence: the activation differential at one chosen layer (our peak layer, 23, picked on the fit half).
- Retention: (Presence of the fixed model minus Presence of the Clean model) divided by (Presence of the Biased model minus Presence of the Clean model). 1 means all the injected bias survived, 0 means none, below 0 means below the Clean model. We quote it at layer 23. Pooled retention over layers 16-23 uses the sum of the differential over those layers instead of one layer (fair answers 0.17, ours -0.06), which does not blow up where the Biased model's own bias is tiny (layers 16-19: 0.011 to 0.013 against Clean 0.005 to 0.006).
- Layer-wise check against Clean: the activation differential at each layer divided by the Clean model's. Fair-answers retrain: 2.3, 2.5, 2.9, 3.9, 2.8 at layers 16-20 (3.1 and 3.4 at 21-22, 1.7 at 23). Ours: 0.8, 0.8, 0.8, 0.8, 0.5 at layers 16-20, 0.27 to 0.94 at layers 5-23, and 1.01 to 1.12 at layers 0-4 (absolute value under 0.004 everywhere).
- Cosine(v_after, v_bias): did the direction move. After erase until clean it is 0.11 to 0.32 across layers 16-23 (all-late fix: 0.03 to 0.34), so what is left points in new directions.
- Balanced accuracy: average of qualified-accepted and unqualified-rejected rates, our "did we break the model" check.
- Visual: `arrows.png` (PCA at layer 23, bias arrow length 4.41 for the Biased model, 1.50 after erase-until-clean; it does not show the retrained models).

**B2. How we used the datasets**
- Provided resumes (2,484, 24 categories, PII-scrubbed): 4 categories (IT, finance, healthcare, chef), 30 resumes each for the audit, a separate 40 each for injecting the bias and for the retrains. No overlap. Resumes cut to 1,000 characters for speed.
- Qualified vs unqualified comes from category: a resume scored against its own job is qualified, against a mismatched job is unqualified. That gives balanced accuracy.
- Names: Bertrand & Mullainathan (2004), gender-matched pairs, inserted as a name field. The only thing we add to a scrubbed resume.
- Job descriptions: 4 hand-written, because the LinkedIn listings were not available offline.
- Audit set: 120 resumes x 2 jobs x 2 names = 480 items, split by resume: 240 to fit directions and thresholds, 240 to measure. results.json confirms `n_eval_items` = 240 and `n_eval_resumes` = 60.
- Injection set: 160 resumes x 2 jobs x 2 names = 640 training examples (confirmed in the injection log). The two retrains use the same 160 resumes as 320 name-swapped pairs (confirmed in the retrain logs: "320 pairs"), so the audit resumes are unseen by every model.

**B3. Why we injected bias with fine-tuning**
- We first tried a system prompt telling the model to favour a name group. The 0.5B model ignored it: P(yes) was identical for both groups. A prompt is also not bias in the weights.
- LoRA fine-tuning puts it in the weights, as in Hannah's SFT setup, and gives a known ground truth: we know the bias is there, so we can test whether the audit finds it and the fix removes it.
- Trained on the neutral prompt, on resumes the audit never sees, so the audit tests how the bias generalises. Result: Δ 0.93 between qualified White and Black names, against 0.03 for the Clean model.

**B4. Law and sociotechnical implications**
- EU AI Act: recruitment systems are high-risk (risk management, data governance, human oversight, logging). NYC Local Law 144: independent bias audit of automated hiring tools, based on selection rates. Ours adds the inside on top of that output check. It does not replace a legally required audit.
- UK Equality Act 2010 and US race-norming rules: we do not adjust outcomes by group. Our retrain asks the model to represent the two name versions of the same resume the same way, and one threshold applies to everyone.
- Black-box vendors cannot be audited this way. Policy ask: white-box audit access for high-risk hiring tools.
- Names are a crude proxy; university, postcode and career gaps leak too. Human review for flagged decisions, and a way for candidates to challenge a decision.

**B5. Risks**
- Linear erasure and linear probes can miss nonlinear bias. We add a nonlinear (MLP) probe at the late layers as a check the eraser never saw. Visual: `probe.png`. Limit: the Clean model's probes already read the name group at 0.85 to 0.98 in the late layers, so the probe shows whether we are above Clean, not whether we are at zero.
- Erasing or retraining can damage the model. Check: balanced accuracy, Clean 0.59, Biased 0.70, one layer 0.77, all late 0.88, erase until clean 0.88, retrain fair answers 0.92, ours 0.95; AUC per group stays 0.93 to 0.95 after the erasures and is 0.99 for both retrains. Note the Clean row is the untuned Qwen with AUC 0.64, so it is a bias baseline, not an accuracy target. The retrains are also trained on the task labels, which is part of why their accuracy is higher.
- Training to the test (Goodhart): our penalty trains on something close to what the audit measures. Defences and limits are in Q&A 14.
- One small model, 4 job types, 60 held-out resumes (240 items), one race-via-names axis, final token only, one penalty weight (2.0), one run. Proof of concept, not a benchmark.
- Erasure is a guard at inference. The bias is still in the weights. The retrains change the weights, but we have only measured them on this audit.
- Our injected bias is cleaner than natural bias. Real bias is harder.
- The pass lines in S4 are a policy choice, and ours were set after seeing the results.

**B6. Model-dependence (Hannah's finding 2)**
- Bias differs by model family and size with no consistent pattern, so a pass on one model says nothing about another.
- We did not test this: we ran one model. Consequence: the gate runs per model and again on every update. The method transfers, the numbers do not.

---

## Part 2: 60-second script (140 words spoken, numbers included)

Presenter reads; slide changes in brackets. Read it aloud against a timer. Word count excludes the bracketed slide cues. S3 stays on the heatmap while the retrain lines are read, so point at the last two rows.

> Two resumes, identical except the name. The model says yes to both, so an answer-only audit passes it. **[S1]**
>
> But the same answer can hide different thoughts. We injected bias into a small model: qualified White names were shortlisted 93 points more often than Black names. The usual erasure fix cuts that to 13 points, yet 53 percent of the bias is still inside. **[S2]**
>
> Retraining on fair answers looks fixed, but its late layers still carry 2.6 times a clean model's bias. Our fix trains the model to think the same about both names. The gap is zero, bias at every late layer is at or below a clean model's, and accuracy went up. **[S3]**
>
> Our tool checks itself, so it can tell these fixes apart. On this model, under these tests. **[S4]** Next: more models, more names, a re-audit on every update. **[S5]**

(Each bracket marks the slide on screen while the text before it is read. Late layers means layers 16-23.)

---

## Part 3: Q&A prep

Presenter takes every question first and hands off. **Mal** = technical. **Q&A lead** = legal and data. **Presenter** = the rest.

**0. (Official) How does this ensure the model is truly unbiased?** It does not claim that. No test can prove it. The gate can fail a model, and a pass means "no bias detected on these tests, on this model". On our own run it failed four of five fixes and passed one. Three erasures and the fair-answers retrain were failed on the inside: 39% to 53% of the bias retained at layer 23 for the erasures, and for the fair-answers retrain the name-group difference at layers 16-20 is 2.3 to 3.9 times the clean model's (2.6 times pooled over layers 16-23), although its gap (0.03) and peak-layer retention (9%) pass. The retrain with the inside check passes all four checks (gap 0.00, retention -9%, accuracy 0.95 / 0.96, at or below clean at layers 5-23, 0.46 times Clean pooled over layers 16-23). That is "no bias detected" on one model, one axis, with pass lines we set ourselves. The point is that it checks the inside as well as the output. *Presenter.*

**1. Isn't 0.5B too small to mean anything?** Hannah's floor was 360M, she asked for a proof of concept on one model, and her finding 2 says numbers do not transfer between models anyway. We did not test finding 2: one model only. What we show is the method and the gate, on a model that runs on a laptop. *Mal.*

**2. Why fine-tuning, not natural bias?** Hannah said to assume the bias was injected on purpose, and natural bias has no known ground truth. Injection gives us a positive control: we know it is there (Δ 0.93 against 0.03 for the Clean model). A system prompt did not work on this model (identical P(yes)), and a prompt is not bias in the weights. *Presenter, Mal adds the prompt detail.*

**3. What if the bias is nonlinear?** Then a linear eraser can miss it. We test with a nonlinear MLP probe on the late layers that the eraser never saw. At layers 20 to 23 it reads the name group at 1.00 for the Biased model, 0.94 to 0.95 after erase until clean, 0.87 to 0.98 for the fair-answers retrain, 0.63 to 0.78 for the retrain with the inside check, and 0.95 to 0.98 for the Clean model. (Linear probes at layers 16-23: ours 0.65 to 0.77, Clean 0.85 to 0.98.) With 240 items a gap of a few points is noise (about plus or minus 6), so the erasure rows show we are back at Clean's level, not that nothing is left. Ours is below Clean, well outside that noise, and neither probe was in the training loss. Our penalty is on the whole activation difference, not one direction, so it is not limited to linear bias. Nonlinear erasure and more probe families are next steps. *Mal.*

**4. Won't erasing damage the model?** We measure it: balanced accuracy is Clean 0.59, Biased 0.70, one layer 0.77, all late 0.88, erase until clean 0.88, retrain fair answers 0.92, ours 0.95, and the AUC per group stays 0.93 to 0.95 after erasure and is 0.99 for both retrains, so ranking ability survives. The Biased model's low score was Black-named candidates at 0.50 (White 0.90); after the fixes both groups sit near 0.88 to 0.96. Be careful with "more accurate": the retrains are trained on the task labels, so most of their gain over erasure comes from that, and ours at 0.95 against fair answers at 0.92 is within noise (about plus or minus 6 at 240 items). If accuracy dropped, the gate fails the fix. *Mal.*

**5. What about black-box vendors?** The activation half needs model access, so it cannot run on a closed model. The behavioural half still can. The policy ask is white-box audit access for high-risk hiring tools. *Q&A lead.*

**6. How did you use the datasets?** Provided resumes: 4 categories, a separate set for injecting bias (and retraining) and for auditing, category match gives qualified labels, fit and measure on different resumes (240 items, 60 resumes, held out). Hand-written job descriptions, because the LinkedIn data was not available offline. Names from the 2004 audit study. Details in B2. *Q&A lead.*

**7. Isn't name-based race a crude proxy?** Yes. Names are the standard audit-study lever and isolate one change, but real people do not fit name lists, race and gender interact, and other proxies leak. Next: more names, proxy swaps (university, postcode), intersectional pairs. *Q&A lead.*

**8. Why not reweight toward the disadvantaged group?** Hannah flagged it as illegal. UK Equality Act 2010 bars positive discrimination in hiring (only a narrow tie-break between equally qualified candidates is allowed), and US rules ban race-norming of scores. We do not adjust outcomes by group. We ask the model to stop using the name signal so both are judged on the resume. *Q&A lead.*

**9. How does this fit the law?** EU AI Act classes recruitment systems as high-risk, so risk management, oversight and logging apply. NYC Local Law 144 requires an independent bias audit of automated hiring tools, based on selection rates. Our audit adds the inside on top and does not replace those. *Q&A lead.*

**10. What would you do with a week?** Real LinkedIn listings, more names and proxy swaps, an audit with a name set the training never saw, the relapse test (retrain on a little biased data, count steps until the bias returns), more than one penalty weight and more than one run, all token positions instead of the last one, and two or three model sizes to test model-dependence. Weight-level fixes and the invariance loss are done on this one model. *Presenter.*

**11. Isn't the probe just detecting the name, not bias?** The model knowing perceived race is not bias. Using it is. The Clean model's probes already read the name group at 0.85 to 0.98 in the late layers. That is why Hannah's retention metric subtracts the Clean model: the target is the Clean row, not zero. The gate also checks that the decision stops following the name. Ours reads below Clean on the probes (0.65 to 0.77), which means it represents the name group less than an untuned model does. We have not checked what else that costs. *Mal.*

**12. Why didn't erase until clean reach zero?** Linear erasure has a floor. The worst late-layer bias on the fit half went 0.158, then 0.100, then 0.103 over the three rounds, so after round 2 more rounds stopped helping. On held-out resumes 39% is still retained at layer 23, and 0.39 pooled over layers 16-23 (4.5 times Clean). What is left points in new directions (cosine 0.11 to 0.32 with the original), so it is either nonlinear or rebuilt by the later layers on each pass. We report it as not clean and the gate does not pass it. The weight-level retrain with the inside check did get to Clean's level (layer 23: 0.009 against Clean 0.034). *Mal.*

**13. Isn't recalibrating the threshold cheating?** No. Each model gets one cut-off, the same for both name groups, chosen on a separate half of the resumes from the one we measure on. Any deployed scorer is calibrated this way. At the naive 0.5 cut-off every erasure fix rejected everyone, because the biased model only learned Yes for White and qualified, so 0.5 would have hidden the comparison. The retrained models work close to 0.5: gap at 0.5 is 0.02 for fair answers (calibrated cut-off 0.27) and 0.00 for ours (cut-off 0.5). Per-group cut-offs would be race-norming, which is illegal, and we do not use them. Ranking ability survives the fixes: AUC per group is 0.93 to 0.95 for erasure and 0.99 for the retrains. *Mal.*

**14. Aren't you just training to the test?** Partly, and we say so. The penalty trains on same-resume name pairs and compares activations, which is close to what the audit measures, so a low audit number is less surprising for that model than for an erasure. What we have against it: (1) different resumes: training uses 160 non-audit resumes, the audit uses 60 held-out ones the model never saw. (2) A different metric: the loss is a per-pair squared difference at the final token, summed over layers; the audit reports the size of the group-mean difference vector on held-out resumes, which is a weaker quantity that a perfect per-pair match implies but that is not the same calculation. (3) Behaviour and probes the penalty never touched: the gap, AUC and accuracy at a group-blind threshold, plus independent linear and MLP probes that read the name group at 0.65 to 0.77 (linear, late layers) against 0.85 to 0.98 for Clean. What we have not shown: other name sets (we reused the same name list), other token positions (final token only), other axes (race via names only), other penalty weights (2.0 only), or more than one run. Someone could still find bias the penalty did not target, and we have not looked. The honest claim is "passes our gate on this model", not "unbiased". *Mal.*

**15. Why does retraining on fair answers move the bias rather than remove it?** Because the loss only asks the answer to match. The model is rewarded for saying the same thing for both names and nothing tells it to stop representing the name group differently, so it can keep or reroute that signal through layers whose output does not reach the answer. Measured: layer 23 falls to 0.058 (9% retained), but pooled over layers 16-23 it still carries 2.6 times the clean model's bias (retention 0.17), and layers 16-20 are 2.3 to 3.9 times the clean model (0.014 to 0.021 against 0.005 to 0.007), and layers 21-22 are 3.1 and 3.4 times. All small in absolute terms (the Biased model reaches 0.30). Be careful with "moved": at layers 16-19 the retrained value (0.014 to 0.021) is about the same as the Biased model's own (0.011 to 0.013), slightly higher, so the data fits "the late layers were cleared and the earlier ones were left or rerouted" and we have not run a test that tells those apart. A peak-layer-only check would pass it; checking every layer does not. Adding the penalty to the loss is what pushes every layer down. *Mal.*

*Legal wording above is from memory and was not checked today. If unsure, say "Hannah flagged it" and cite the Act by name only.*

---

## Part 4: Words to use and avoid

**Use**
- "activation-space bias" and "behavioural bias"
- "the model checks itself", "our tool checks itself"
- "erase until clean"
- "retrain with the inside check", "retrain on fair answers"
- "same answer, different thoughts"
- "on this model, under these tests"
- "injected bias", "proof of concept", "held-out resumes"
- "retention" (the share of injected bias left), "White-coded and Black-coded names"
- "audit gate" (it can fail a model), "refuses to pass"

**Avoid**
- "we removed the bias", "debiased", "fixed" (as a claim), "unbiased", "proves", "guarantees", "certified" (as a claim about a model)
- "natural" or "real-world" bias (ours is injected)
- "removed from the model" (we erase at inference, or retrain and measure; the measurement is on one audit)
- "race" as if we measured people (we measured names)
- "reweight" or "balance outcomes" (illegal framing)
- "clean" as a claim that the model is unbiased: say "at a clean model's level on these tests" for our retrain, and "closest" for the erasures
- "moved the bias" as a proven fact about the fair-answers retrain: say "left it spread through earlier layers"
- "no bias left" for ours: layers 0-4 are within 12% above Clean (all under 0.004), and the check is on one model

## Part 5: Where each number lives (results.json)

- Δ: `demographic_disparity` (shortlist decisions at the model's own group-blind threshold). Ignore `demographic_disparity_at_0.5` for the erasures: at 0.5 every erasure fix rejects everyone. For the retrains it is usable (0.02 and 0.00).
- Retention, headline: `retention` at the peak layer (23): one layer 0.44, all late 0.53, erase until clean 0.39, retrain fair answers 0.09, retrain with inside check -0.09. For the one-layer fix also `retention_after_fixed_layer` (layers 22 and 23: 0.49, 0.44). Do not quote `retention_late_mean` for any row: it averages per-layer ratios and is inflated at layers 16-19 by tiny denominators. Use pooled retention over layers 16-23 instead, computed from `activation_differential` as (sum over 16-23 for the model minus Clean) / (Biased minus Clean): one layer 0.49, all late 0.47, erase until clean 0.39, retrain fair answers 0.17, ours -0.06; as a multiple of Clean's sum: 5.4x, 5.2x, 4.5x, 2.6x, 0.46x (Biased 9.9x).
- Activation differential per layer: `activation_differential` (Biased: about 0.01 to layer 19, then 0.086, 0.216, 0.264, 0.298 at layers 20-23). Layer-wise comparison with Clean (S4 check 4, B1) is this array for the model divided by the same array for `base`; the pooled multiple is the sum over layers 16-23 divided by Clean's.
- Retrain conditions: `fix_retrain_behaviour` (fair answers) and `fix_retrain_invariance` (with the inside check); both have `weight_level` = true and `fixed_layers` = all 24, which means the penalty applies at every layer, not that a projection was applied. `threshold`: 0.27 and 0.50. `presence_at_peak`: 0.058 and 0.009 (Clean 0.034, Biased 0.298). `auc_white` / `auc_black`: 0.99 / 0.99 for both.
- Nonlinear probe: `probe_acc_mlp_late` (layers 16-23). Linear probe, all layers: `probe_acc_linear` (late layers: Clean 0.85 to 0.98, ours 0.65 to 0.77). Balanced accuracy: `balanced_accuracy` (and `_white`, `_black`). Peak layer: `peak_layer` (23). One-layer fix: `single_fix_layer` (21). Rounds: `fix_iterative.rounds` (3 entries, fit half), `subspace_dim` (4). Direction drift: `cosine_to_biased_direction`.
- Report n with every number: `n_eval_items` (240), `n_eval_resumes` (60). With about 240 items, a probe within a few points of another is within noise (rough, about plus or minus 6 points).
- Retrain settings are in `audit/retrain_fix.py` (LAMBDA 2.0, LR 2e-4, 4 pairs per batch, 1 epoch) and the logs `audit/results/retrain_behaviour.log` and `retrain_invariance.log` (320 pairs).
- `results_run1_uncalibrated.json` is an earlier run at the 0.5 cut-off. Do not quote from it.
