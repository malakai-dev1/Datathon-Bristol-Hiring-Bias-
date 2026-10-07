# Same answer, different thoughts

**A two-part bias self-check for hiring models: it tests what the model says and what it represents inside. The fix that passes it is a retrain with an inside check. Erasing the bias at inference time (the quick fix, kept as the comparison) cut the gap but left 39 to 53% of the bias inside. Retraining on fair answers alone moved the bias earlier. Retraining with the inside check removed both, on this model, under these tests.**

180DC Bristol x BDSS datathon, "Exploring Bias in Hiring". Proof of concept on one small model (Qwen2.5-0.5B-Instruct), runs on a laptop.

## The problem

Hiring models are usually audited by swapping the name on a resume and checking whether the decision changes. Hannah Liu's thesis (TRACE, Imperial College London) shows three things that break that check: removing behavioural bias does not remove activation-space bias, how well removal works depends on the model, and bias is concentrated in later layers but cannot be causally localised to one spot. So a model can say "yes" to both resumes while its internals still lean on the name, and that bias can resurface later. We built a check that tests both halves, and fixes that have to pass both: an inference-time erasure (quick, no retraining, and it only partly works) and a retrain with an inside check (it passes on this model).

## What the audit measures

Metrics are from Hannah Liu's thesis (TRACE, Imperial College London), shared at the datathon, and implemented in `audit.py` from the definitions given there. Disparity follows Dwork et al. (2011). Any difference from the thesis's own implementation is ours.

A is the name group: White-coded or Black-coded. h_l(X) is the model's activation at layer l, taken at the last prompt token.

| Metric | Formula | What it tells you |
|---|---|---|
| Demographic disparity | `Δ = abs(P(ŷ=1 given A=White) - P(ŷ=1 given A=Black))` | Behaviour: does the decision depend on the name. We compute it among qualified candidates. |
| Bias vector | `v_l(M) = E[h_l(X) given A=a] - E[h_l(X) given A=a']` | The direction the model leans at layer l when the name changes. |
| Activation differential | `‖v_l‖ / E[‖h_l(X)‖]` | Size of the bias at layer l, scaled so layers are comparable. |
| Presence | `Presence(M) = ‖v_l*(M)‖ / ‖h_l*‖` at a chosen layer l* | One number per model. We take l* as the layer where the biased model's differential peaks (fitted on the fit half). |
| Retention | `(Presence(M') - Presence(B)) / (Presence(M_b) - Presence(B))` | B = base, M_b = biased, M' = mitigated. 1 means all the injected bias is still there, 0 means back to base level. |
| Cosine similarity | `cos(v_after, v_bias)` | Did the bias direction move after mitigation, even if it shrank. |
| Balanced accuracy | `0.5 * (qualified acceptance rate + unqualified rejection rate)` | Capability check: did the fix break the model's ability to do the job. |
| AUC per name group | `AUC(P(Yes), qualified)` within White-coded and within Black-coded items | Does the model rank qualified above unqualified, with no cut-off involved. |

**Shortlist decision.** A candidate is shortlisted when P(Yes) is at or above one cut-off per model condition. The cut-off is the same for both name groups and is chosen on the fit half of the resumes to maximise balanced accuracy (`calibrate_threshold` in `run_audit.py`). It is not fixed at 0.5: at 0.5 every fix collapsed to a decision that carried no information (see Results). Disparity and balanced accuracy at 0.5 are still stored in `results.json` for comparison.

**What we added:**

- **Linear probe accuracy per layer.** A logistic regression tries to predict the name group from the activations. 50% means the name is not linearly readable. It catches a fix that rotated the bias instead of removing it, because the probe finds the bias wherever it went.
- **Nonlinear probe (small MLP) on the last 8 layers.** The eraser is linear, so this is the check it is least likely to pass by accident.
- Probes are cross-validated in 5 folds split by resume, so a probe never sees the other name version of a test resume.

## The mitigation

The aim is a fix that is judged by checks it was not trained on. There are two routes: erase the bias at inference time (no weights change, quick, partial) and retrain the weights on fair labels with an inside check (the fix that passes). The erasure stays in as the comparison and as the quick fix.

### Quick fix: erase until clean (inference time)

 On our model it does not reach clean (see Results): it removes most of the behavioural gap and some of the activation-space bias, and the audit says how much is left.

1. **Find.** At each of the last 8 layers, compute the bias vector (White-coded mean minus Black-coded mean activation) on the fit half of the resumes.
2. **Erase.** At inference, project that direction out of the residual stream at every token position: `h = h - (h·u)u`. No weights change.
3. **Re-find.** Run the erased model again and compute the bias that is left at each late layer. It can have rotated to a new direction. Add it to the erased set.
4. **Repeat** for three re-find rounds (`ITER_ROUNDS = 4` in `run_audit.py`), giving a 4-direction subspace per late layer.
5. **Verify on checks the eraser never trained on:** held-out resumes (directions are fitted on one half, every metric is on the other half), the nonlinear MLP probe, behaviour (demographic disparity), and balanced accuracy.

Two comparison fixes, so we can test Hannah's finding 3 directly:

- **Single layer.** Erase the one direction at layer 21 only (`single_fix_layer` in `results.json`). This is the layer where the biased model's bias grows most, not the peak layer (23), so there are layers after it (22 and 23) to watch. If bias could be cut out at one spot, this would be enough. We measure retention at every layer after it to see whether the bias comes back downstream.
- **All late layers, once.** Erase one direction at each of the last 8 layers, with no re-finding. This shows whether the rotation step in erase-until-clean adds anything.

### Mitigation: retraining on fair labels (weight level)

The erasures leave the weights alone and run at inference. The two retraining fixes change the weights. Both start from the biased model, add a LoRA adapter (rank 8, alpha 16, learning rate 2e-4, 1 epoch, 4 resume pairs per step, merged into the weights afterwards) and train on the same non-audit resumes as the bias injection, so the 60 audit resumes stay unseen. The training data is 320 pairs: the same resume and job, once with a White-coded name and once with a Black-coded name. The label is fair: qualified means Yes and unqualified means No, whatever the name. Code is `retrain_fix.py`.

- **Retrain: fair answers** (`python retrain_fix.py behaviour`). The loss is the Yes/No answer error and nothing else. It is judged only on what the model says, like the LoRA mitigation in Hannah's thesis.
- **Retrain: + inside check** (`python retrain_fix.py invariance`, ours). The same loss plus a penalty that says: the two name versions of the same resume must produce the same activation at the last prompt token, at every one of the 24 layers. In plain English, the answer has to be right, and the name is not allowed to change what the model is thinking.

```
loss = CE(answer, fair label)
     + LAMBDA * sum over layers l = 1..24 of
         mean over pairs of  ‖h_l(resume, White name) - h_l(resume, Black name)‖²  /  mean ‖h_l‖²

LAMBDA = 2.0
```

h_l is the activation at layer l at the last prompt token. The squared difference is divided by the mean squared activation norm at that layer (held constant in the gradient), the same scaling as the audit's activation differential, so every layer counts on a comparable scale. LAMBDA is 2.0 and was not tuned. With LAMBDA = 0 the loss is the fair-answers fix.

The penalty sits close to what the audit measures, so a good audit score on this row is partly expected. See Limitations and risks for why we still read it as evidence, and what it does not prove.

## How we used the datasets

- **Resumes** (`Resume.csv`, 2,484 resumes, 24 categories, already PII-scrubbed by the organisers). We use the resume text from 4 categories: Information Technology, Finance, Healthcare, Chef. The other 20 categories and the HTML column are not used.
  - **Audit set:** 30 resumes per category (120 total), each truncated to 1,000 characters. Split by resume: 60 to fit directions, 60 held out for every reported metric.
  - **Bias injection set:** 40 resumes per category (160 total) that are not in the audit set, so the audit measures how the bias generalises to unseen resumes.
- **Names.** The organisers removed names, so the only demographic signal we add is the name. Name lists are from Bertrand & Mullainathan (2004), gender-matched. Each resume appears with a White-coded and a Black-coded name.
- **Jobs.** The LinkedIn job listings dataset was not available to us offline, so we hand-wrote 4 job descriptions (IT support, financial analyst, registered nurse, sous chef). A resume is scored against its own job ("qualified") and a deliberately mismatched job ("unqualified": IT with Chef, Finance with Healthcare).
- **Bias injection.** The challenge assumes bias is injected on purpose, so we inject it with a short LoRA fine-tune (`inject_bias.py`): qualified plus White-coded name gets Yes, qualified plus Black-coded name gets No, unqualified gets No for both. 640 training examples, 1 epoch, rank 8. We first tried injecting it with a system prompt (still in `audit.py` as `biased=True`), but that did not produce a behavioural bias on the 0.5B model.
  - **Retraining fixes** use these same 160 injection resumes (320 name-pair examples), never the audit resumes.

## Repo layout

| Path | What |
|---|---|
| `audit/` | The audit pipeline and its results (this README describes it) |
| `data/resumes.csv.gz` | The provided resume dataset, compressed |
| `data/jobs_sample.csv` | A sample of the LinkedIn job listings |
| `data-eda.ipynb` | Exploratory analysis of the datasets |
| `brief/` | Challenge brief, team plan, deck content and Q&A prep |
| `measure.py`, `inject_bias.py`, `mitigate.py` (root) | An earlier prototype of the pipeline |

## How to run

Needs Python (we used 3.12). The resume data ships with the repo as `data/resumes.csv.gz` (the provided Kaggle dataset: 2,484 resumes, columns ID, Category, Resume_str). Run everything from the `audit/` folder, because paths are relative. The first run downloads `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face (no login needed). It uses Apple MPS if available, otherwise CPU.

```bash
cd audit
# setup
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
# or: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

mkdir -p results models

# 1. inject the bias (LoRA fine-tune, saves models/biased)
python inject_bias.py

# 2. audit base, biased, and the three erasure fixes (writes results/results.json and results/activations.npz)
python run_audit.py

# 3. retrain on fair answers only (saves models/fix_behaviour)
python retrain_fix.py behaviour

# 4. retrain with the inside check (saves models/fix_invariance)
python retrain_fix.py invariance

# 5. audit the two retrained models (adds them to results/results.json)
python audit_retrained.py

# 6. make the figures (writes results/heatmap.png, probe.png, arrows.png, summary_bars.png)
python make_figures.py
```

Run them in that order: `run_audit.py` loads the model that `inject_bias.py` saves, both trainings start from that biased model, and `audit_retrained.py` adds to the `results.json` and activations that `run_audit.py` writes. Run the two trainings one after another, not in parallel: together they exhaust GPU memory on a 48 GB Mac.

| File | What it does |
|---|---|
| `audit.py` | Data loading, model wrapper with the erasure hook, and all metrics |
| `inject_bias.py` | LoRA bias injection |
| `run_audit.py` | Runs the five conditions (clean, biased, three erasure fixes) and writes `results/results.json` |
| `retrain_fix.py` | Weight-level fixes: LoRA retrain on fair labels, with or without the inside-check penalty |
| `audit_retrained.py` | Audits the two retrained models on the same held-out resumes and adds them to `results/results.json` |
| `make_figures.py` | Draws the figures (all seven conditions) from the saved results |

## Results

<!-- RESULTS: filled in after run_audit.py finishes -->

**Headline:** The inference-time erasures cut the gap but left 39 to 53% of the bias inside. Retraining on fair answers alone moved the bias earlier rather than removing it. Retraining with the inside check removed both, on this model, under these tests.

- **The injected bias is large** (demographic disparity 0.93, against 0.03 for the clean model) and the activation-space part of it sits almost entirely in layers 20 to 23.
- **Erasure at inference time.** The best of the three, erase until clean, cuts disparity to 0.10 with balanced accuracy 0.88 for both groups, but 0.39 of the injected activation-space bias is still there at the peak layer (0.44 and 0.53 for the other two). By our own audit it is not certified clean.
- **Retrain: fair answers.** Disparity 0.03, balanced accuracy 0.91 / 0.93, AUC 0.99 / 0.99, and retention at the peak layer (23) only 0.09. But at layers 16 to 20 the name-group difference is 0.014 to 0.021, about 2.3 to 3.9 times the clean model's (0.005 to 0.007) and above the biased model's at layers 16 to 19. The bias moved earlier rather than leaving. It is small in absolute terms (the biased model's peak is 0.30), but the answers-only check cannot see it. Pooled over layers 16 to 23, 0.17 of the injected late-layer bias is retained, and its late layers carry 2.6 times the clean model's bias (the per-layer mean of 0.89 is inflated by tiny denominators; see the note under the table). A linear probe reads the name group at 0.76 to 0.99 in layers 16 to 23.
- **Retrain: + inside check.** Disparity 0.00, balanced accuracy 0.95 / 0.96 (the best of any condition), AUC 0.99 / 0.99. The activation-space bias is at or below the clean model's at every layer from 5 to 23 (layer 23: 0.009, against 0.034 clean and 0.298 biased), with retention at the peak of -0.09 and a late-layer mean of -0.11 (below the clean model). An independent linear probe reads the name group at 0.65 to 0.77 in layers 16 to 23 (small MLP probe 0.62 to 0.78), against 0.85 to 0.98 for the clean model and 0.85 to 1.00 for the biased one. That is less readable race information than even the clean model, but it is not 0.5, so some name signal remains.

Model: Qwen2.5-0.5B-Instruct. Held-out set: 60 resumes, 240 items (120 qualified, 60 per name group, so one resume is worth 0.017 of disparity). Peak layer: 23 (where the biased model's differential peaks, picked on the fit half). The single-layer fix erases at layer 21, where the bias grows most (0.09 at layer 20 to 0.22 at layer 21), not at the peak, so layers 22 and 23 sit downstream to watch. Late layers are 16 to 23. Single run, single seed, no confidence intervals. All numbers below are from `audit/results/results.json`, rounded to 2 decimals (thresholds to 3).

**How shortlist decisions are made.** Each condition gets one cut-off on P(Yes), the same for White-coded and Black-coded names, chosen on the fit half of the resumes to maximise balanced accuracy (`calibrate_threshold` in `run_audit.py`, stored as `threshold`). We do not use per-group cut-offs, because that would be race-norming. Our first run used a fixed 0.5 cut-off and it was uninformative: the clean model and all three fixes gave decisions with balanced accuracy exactly 0.50 and disparity 0.00 on the held-out set, because the biased model only ever learned to say Yes for White-coded names on qualified resumes, so P(Yes) for everything else sits far below 0.5 (mean P(Yes) for qualified candidates after the fixes: 0.02 to 0.16). Zero disparity there meant nobody was shortlisted, not that the model was fair. We kept that run as `audit/results/results_run1_uncalibrated.json` and the 0.5 numbers for the current run are still in `results.json` (`demographic_disparity_at_0.5`, `balanced_accuracy_at_0.5`). The two retrained models do not have this problem: at 0.5 their balanced accuracy is 0.93 and 0.95, and the inside-check model's calibrated cut-off came out at 0.500. The headline numbers use the calibrated cut-off. We also report AUC per name group (`auc_white`, `auc_black`): whether the model ranks qualified above unqualified within that group, with no cut-off involved.

![Bias by layer, per condition](audit/results/heatmap.png)

*Activation-space bias at each of the 24 layers (rows are conditions, brighter is more bias, circles mark the layers we erase). The clean model stays dark throughout; the biased model lights up from layer 20 (0.09, 0.22, 0.26, 0.30 at layers 20 to 23, against about 0.01 up to layer 19). The three erasure fixes dim it but layers 22 and 23 stay visibly lit in every one of those rows. The two retrain rows have no circles because retraining changes the weights and nothing is erased at inference. The fair-answers row is dim but not dark: purple at layers 22 and 23 (0.055 and 0.058, against 0.016 and 0.034 for the clean model) and faintly lit from layer 14 on. The inside-check row is as dark as the clean model's from layer 5 on. Right-hand labels give disparity, AUC and balanced accuracy.*

![Probe accuracy by layer](audit/results/probe.png)

*How well a linear probe reads the name group from each layer (0.5 means it cannot). The clean model is already at 0.85 to 0.98 across layers 16 to 23, so a model can read a name without being biased by it; this is why retention subtracts the clean model. After the fixes, probe accuracy falls at the first erased layers (layer 16 is 0.59 after the all-late fix and 0.60 after erase until clean, against 0.85 clean) but is still 0.98 at layer 23 after the all-late fix and 0.95 after erase until clean. The retrain rows have no circles. After retraining on fair answers the probe reads 0.76 to 0.99 across layers 16 to 23. After the inside-check retrain it reads 0.65 to 0.77, lower than the clean model at every one of those layers, and still above 0.5.*

![Bias direction before and after](audit/results/arrows.png)

*Peak-layer (23) activations projected onto two principal components, with the group means as diamonds and the bias vector as an arrow. In the biased model the White-coded and Black-coded means are clearly apart (arrow length 4.41 as printed on the figure); after erase until clean they nearly overlap (1.50). The retrained models are not drawn here. This is a 2D picture for intuition. The retention numbers in the table are the measurement.*

![Summary](audit/results/summary_bars.png)

*Disparity, retention at layer 23 and overall balanced accuracy side by side. Disparity falls and balanced accuracy rises after every fix. Retention at the peak stays between 0.39 and 0.53 for the three erasure fixes, and is 0.09 and -0.09 for the two retrains (below 0 means less name-group difference than the clean model at that layer). The fixed models score higher balanced accuracy (0.77 to 0.95) than the clean model (0.59), which fits the clean model barely doing the task (AUC 0.64) while the fine-tuned ones do (AUC 0.92 to 0.99). We have not tested why, but the bias-injection fine-tune may also have taught the task.*

| Condition | Threshold | Disparity Δ | Balanced acc. (W / B) | AUC (W / B) | Retention at peak (layer 23) | Retention, late-layer mean (16-23) |
|---|---|---|---|---|---|---|
| Clean model | 0.993 | 0.03 | 0.58 / 0.60 | 0.64 / 0.64 | 0 (reference) | n/a |
| Biased model | 0.198 | 0.93 | 0.90 / 0.50 | 0.94 / 0.92 | 1 (reference) | n/a |
| Fix: one layer (21) | 0.006 | 0.33 | 0.74 / 0.80 | 0.94 / 0.94 | 0.44 | 0.78 |
| Fix: all late layers, once | 0.007 | 0.13 | 0.88 / 0.89 | 0.94 / 0.94 | 0.53 | 0.20 |
| Fix: erase until clean | 0.035 | 0.10 | 0.88 / 0.88 | 0.95 / 0.93 | 0.39 | 0.16 |
| Retrain: fair answers | 0.269 | 0.03 | 0.91 / 0.93 | 0.99 / 0.99 | 0.09 | 0.89 |
| Retrain: + inside check | 0.500 | 0.00 | 0.95 / 0.96 | 0.99 / 0.99 | -0.09 | -0.11 |

Notes on the table. W / B is White-coded / Black-coded names. Retention for the clean and biased model is 0 and 1 by definition. "Late-layer mean" is the mean of per-layer retention over layers 16 to 23. The one-layer row's 0.78 is high because that fix only touches layer 21, so layers 16 to 20 are untouched and retain about 1; it is not comparable to the other two. The one-layer fix has balanced accuracy 0.74 for White and 0.80 for Black, a gap in the opposite direction to the injected bias, and with 60 resumes per group we would not read anything into it. The clean model's threshold is near 1 and the three erasure fixes' are near 0 (0.006 to 0.035) because the cut-off is picked from each model's own score distribution. The retrained models sit in the middle (0.269 and 0.500).

The two retrain rows are weight-level, so nothing is erased and retention is computed from their activations the same way. A negative retention means the name-group difference is below the clean model's at that layer. The fair-answers row's late-layer mean of 0.89 comes from layers 16 to 19, where the biased model's own bias is tiny (0.011 to 0.013, against 0.005 to 0.006 clean), so the ratio is large (about 1.2 to 2.0) and noisy. At layers 20 to 23 its retention is 0.09 to 0.17. The headline point is the same either way: at layers 16 to 20 its raw difference (0.014 to 0.021) is well above the clean model's. A pooled version avoids the small-denominator problem: sum the activation-space bias over layers 16 to 23, then apply the retention formula. Pooled late-layer retention: biased 1.00, one layer 0.49, all late 0.47, erase until clean 0.39, fair answers 0.17, inside check -0.06. As a multiple of the clean model's late-layer bias: 9.9x, 5.4x, 5.2x, 4.5x, 2.6x and 0.46x. The bar chart's purple bars show the pooled retention.

**Finding 3 check (retention after the single-layer fix at layer 21).** Retention is 0.30 at layer 21 itself, then 0.49 at layer 22 and 0.44 at layer 23, and disparity stays at 0.33. The bias comes back after the layer we erased. Cosine between the leftover bias direction and the injected one is 0.33 at layer 21 and 0.66 and 0.58 at layers 22 and 23, so what comes back is partly the original direction.

**Erase until clean, max late-layer differential by round** (fit half, after 1, 2 and 3 directions are erased): 0.16, 0.10, 0.10 (0.158, 0.100, 0.103 unrounded). It plateaus, so it does not reach clean. The final 4-direction fix, measured on the held-out half, still has a max late-layer differential of 0.14 at layer 23, against 0.30 for the biased model and 0.03 for the clean one. Rounds and final are on different halves, so compare them loosely.

**Rotation.** After the all-late fix, the cosine between the bias direction that remains and the original one is 0.03 to 0.34 across layers 16 to 23 (0.11 to 0.32 after erase until clean). What is left mostly points somewhere new, which is why the re-find rounds exist. At layers 16 to 19 the leftover is tiny (differential 0.003 to 0.009), so those cosines are noisy.

### What this shows

- **Finding 1 (removing behavioural bias does not remove activation-space bias): reproduced on our model.** After the all-late fix, disparity is 0.13 and balanced accuracy 0.88 / 0.89, yet retention at the peak layer is 0.53, so about half the injected activation-space bias is still there. Erase until clean gets disparity 0.10 and retention 0.39. On behaviour alone both look nearly fixed. The activation-space half of the audit says they are not certified. That gap is what a hiring tool that checks itself is for.
- **Finding 2 (how well removal works depends on the model): not tested.** We audited one model, so we have nothing to compare it against.
- **Finding 3 (bias is concentrated in later layers but cannot be localised to one spot): consistent, with a limit.** The bias is late-stage (about 0.01 through layer 19, then 0.09, 0.22, 0.26, 0.30). Erasing at one layer was not enough: disparity 0.33 and retention 0.49 and 0.44 downstream, against 0.13 for erasing across all late layers. We only tried one layer (21), so this shows that one erase at that layer is not enough, not that no single location could work.
- **Does the iterative step add anything over erasing all late layers once?** The direction is favourable (disparity 0.10 against 0.13, retention at the peak 0.39 against 0.53, late-layer mean 0.16 against 0.20), but the disparity gap is two resumes out of 60 and this is one run, so we would not claim it.
- **Behaviour-only retraining shows finding 1 in a new form.** Retraining on fair answers fixes the answers (disparity 0.03, balanced accuracy 0.91 / 0.93) and removes most of the bias at the peak layer (retention 0.09). The bias did not leave: the name-group difference at layers 16 to 20 is 2.3 to 3.9 times the clean model's and above the biased model's at layers 16 to 19. It is small in absolute terms (0.014 to 0.021), and we do not know whether it matters downstream. An answers-only check, which is how the thesis judges its LoRA mitigation, would have called this fixed.
- **The inside check removes both on this model.** With the activation-invariance penalty, disparity is 0.00 and balanced accuracy 0.95 / 0.96, the best of any condition, so the check did not cost capability here. The activation-space bias is at or below the clean model's at every layer from 5 to 23 (layers 1 to 4 are 0.0001 to 0.0003 above it, which is not visible). This is the result we would claim: it passes both halves of the audit on this model, under these tests. We would not claim bias-free, because the probe still reads the name group at 0.65 to 0.77.
- **The probe and retention checks do not agree on how far the erasure fixes got.** After erase until clean, the linear probe at layers 20 to 23 reads the name at 0.93 to 0.95, close to the clean model's 0.95 to 0.98, and the MLP probe gives 0.94 to 0.95 against 0.95 to 0.98. Retention says 0.39 of the injected bias remains at the peak. Probe accuracy asks whether the name can be read at all (the clean model can too), retention asks how large the average name shift still is. We report both and do not pick the more flattering one. No probe reaches 0.5 in any late layer, including for the clean model, so probe accuracy cannot be a pass mark on its own. The inside-check retrain is the one case where the two agree: retention is below 0 and the probe is lower than the clean model's at every late layer.

### What it does not show

- One model, Qwen2.5-0.5B-Instruct, one run, one seed. Nothing here says the numbers hold for another model (finding 2 is untested).
- One prompt format. The decision is P(Yes) against P(No) at the last token for one template.
- Linear erasure. The MLP probe is a partial check on what is left, not proof that nothing nonlinear remains.
- Four hand-written job descriptions, with "qualified" meaning the resume's own category matches the job.
- 60 held-out resumes (60 per name group among the qualified). Disparity moves in steps of 0.017, so differences of a few points are noise, including 0.10 against 0.13.
- The cut-off is picked on the fit half using its labels, so balanced accuracy at that cut-off is slightly flattering. AUC does not depend on it.
- The clean model barely does the task (balanced accuracy 0.59, AUC 0.64), so its disparity of 0.03 is partly a floor with little to be biased about.
- The bias is injected, not natural, and the first run used a naive 0.5 cut-off that we replaced (kept in `results_run1_uncalibrated.json`).
- The inside-check retrain is trained on a quantity close to what the audit measures, on the last prompt token only, with one untuned LAMBDA (see Limitations and risks). Its probe accuracy (0.65 to 0.77) is not 0.5. We did not test it for relapse, on other prompt formats, or on other models.

<!-- END RESULTS -->

## Limitations and risks

- **Linear erasure only.** Projection and linear probes catch linear encodings. The MLP probe is a partial check, not a proof that nothing nonlinear remains.
- **One small model.** 0.5B parameters, one run, one seed, 60 held-out resumes. Finding 2 says results will not transfer across models, so each model needs its own audit.
- **Name-based race proxy.** Real people do not fit name lists, and we do not test intersectionality (race and gender together). Names are one group each, White-coded against Black-coded only.
- **Prompt template and job descriptions.** One prompt wording and four hand-written job descriptions. "Qualified" means the resume's own category matches the job, which is a crude label, not a hiring judgement.
- **Proxies beyond names.** University, postcode, clubs and career gaps also carry demographic signal. We swap names only.
- **Injected bias, not natural bias.** The bias is a strong synthetic one, as the challenge assumes. Naturally occurring bias is harder to find and may not be a clean direction.
- **Threshold and capability baseline.** The cut-off is calibrated on the fit half using its labels, and the clean model barely does the task (AUC 0.64), so balanced accuracy comparisons with it are weak.
- **The inside-check retrain optimises a quantity close to what the audit measures (Goodhart risk).** The penalty drives the two name versions of a resume to the same activation at every layer, and the audit's activation differential is the gap between the two name-group means, so part of a good score is by construction. Defences: the training resumes are different from the audit resumes; the audit metric is a group mean difference while the loss is per pair (a different quantity, though not an independent one, since small per-pair gaps also give a small mean gap); and the probes (linear and MLP) and the behaviour metrics (disparity, balanced accuracy, AUC) were not trained on. The probe still reads the name at 0.65 to 0.77, so the result is less readable name signal, not none. It is a final-token penalty only, so other token positions are unconstrained. One model, one demographic axis (White-coded against Black-coded names), and one LAMBDA (2.0), not tuned.
- **Erasure could hurt capability.** That is why balanced accuracy is in the audit. It is reported per name group as well as overall.

## Sociotechnical implications

- **Law.** Recruitment AI is classed as high-risk under the EU AI Act (risk management, data governance, human oversight, logging). NYC Local Law 144 requires bias audits of automated employment decision tools. Audits of that kind look at selection rates, which is the behavioural half only. Not legal advice.
- **Fix representations, not outcomes.** The UK Equality Act 2010 bans positive discrimination, and reweighting scores towards a group is out. Our erasure is applied the same way to every input, and there is one decision threshold per model, the same for both name groups (chosen on the fit half to maximise balanced accuracy; per-group thresholds would be race-norming). We remove the name signal, we do not favour a group.
- **Behaviour-only audits give false assurance.** A model can pass an output audit, ship, and show the bias again in a new setting such as an interactive agent.
- **White-box access is needed.** Activation audits need the model's internals. A black-box vendor cannot be audited this way, so the policy ask is audit access for high-risk hiring systems.
- **Human in the loop.** Flagged decisions go to a person, candidates get a route to challenge a decision, and the audit repeats on every model update.

## Roadmap

- **Bake the erasure into the weights** with weight orthogonalisation (Arditi et al., 2024), so it does not rely on an inference hook.
- **LEACE concept erasure** (Belrose et al., 2023) for a closed-form linear eraser in place of repeated mean-difference rounds.
- **Invariance fine-tuning, extended:** built here for the final token with one LAMBDA (see Mitigation). Next: all token positions, a LAMBDA sweep, other models, other demographic axes.
- **Relapse test:** after mitigation, fine-tune briefly on a little biased data and count the steps until the bias returns. Fast return means it was suppressed, not removed.
- **Runtime tripwire:** monitor the late-layer projection onto the bias direction at decision time and route high scores to a human. Not built here.
- **Per-model re-audit:** a pass on one model or version does not carry to the next (finding 2).

## Credits

Team of 4: Malakai Shaffer, Mykola Takun, Owen Marschner, Danny Lugovkin.

- Activation-space metrics and the three findings: Hannah Liu, thesis (TRACE), Imperial College London, shared at the datathon.
- Names: Bertrand, M. and Mullainathan, S. (2004), "Are Emily and Greg More Employable than Lakisha and Jamal?"
- Demographic disparity: Dwork, C. et al. (2011), "Fairness Through Awareness".
- Erasure methods referenced: Arditi et al. (2024), "Refusal in Language Models Is Mediated by a Single Direction"; Belrose et al. (2023), "LEACE: Perfect Linear Concept Erasure in Closed Form".
- Resume data provided by the datathon organisers. Model: Qwen2.5-0.5B-Instruct (Qwen team).
