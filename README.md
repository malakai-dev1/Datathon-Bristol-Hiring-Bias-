# Same answer, different thoughts

**A two-part bias self-check for hiring models: it tests what the model says and what it represents inside, then tries to erase the internal bias and re-checks both. On our model the fix mostly works on behaviour and only partly inside, and the audit reports that.**

180DC Bristol x BDSS datathon, "Exploring Bias in Hiring". Proof of concept on one small model (Qwen2.5-0.5B-Instruct), runs on a laptop.

## The problem

Hiring models are usually audited by swapping the name on a resume and checking whether the decision changes. Hannah Liu's thesis (TRACE, Imperial College London) shows three things that break that check: removing behavioural bias does not remove activation-space bias, how well removal works depends on the model, and bias is concentrated in later layers but cannot be causally localised to one spot. So a model can say "yes" to both resumes while its internals still lean on the name, and that bias can resurface later. We built a check that tests both halves, and a fix that has to pass both.

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

## The mitigation: erase until clean

The aim is a fix that is judged by checks it was not trained on. On our model it does not reach clean (see Results): it removes most of the behavioural gap and some of the activation-space bias, and the audit says how much is left.

1. **Find.** At each of the last 8 layers, compute the bias vector (White-coded mean minus Black-coded mean activation) on the fit half of the resumes.
2. **Erase.** At inference, project that direction out of the residual stream at every token position: `h = h - (h·u)u`. No weights change.
3. **Re-find.** Run the erased model again and compute the bias that is left at each late layer. It can have rotated to a new direction. Add it to the erased set.
4. **Repeat** for three re-find rounds (`ITER_ROUNDS = 4` in `run_audit.py`), giving a 4-direction subspace per late layer.
5. **Verify on checks the eraser never trained on:** held-out resumes (directions are fitted on one half, every metric is on the other half), the nonlinear MLP probe, behaviour (demographic disparity), and balanced accuracy.

Two comparison fixes, so we can test Hannah's finding 3 directly:

- **Single layer.** Erase the one direction at layer 21 only (`single_fix_layer` in `results.json`). This is the layer where the biased model's bias grows most, not the peak layer (23), so there are layers after it (22 and 23) to watch. If bias could be cut out at one spot, this would be enough. We measure retention at every layer after it to see whether the bias comes back downstream.
- **All late layers, once.** Erase one direction at each of the last 8 layers, with no re-finding. This shows whether the rotation step in erase-until-clean adds anything.

## How we used the datasets

- **Resumes** (`Resume.csv`, 2,484 resumes, 24 categories, already PII-scrubbed by the organisers). We use the resume text from 4 categories: Information Technology, Finance, Healthcare, Chef. The other 20 categories and the HTML column are not used.
  - **Audit set:** 30 resumes per category (120 total), each truncated to 1,000 characters. Split by resume: 60 to fit directions, 60 held out for every reported metric.
  - **Bias injection set:** 40 resumes per category (160 total) that are not in the audit set, so the audit measures how the bias generalises to unseen resumes.
- **Names.** The organisers removed names, so the only demographic signal we add is the name. Name lists are from Bertrand & Mullainathan (2004), gender-matched. Each resume appears with a White-coded and a Black-coded name.
- **Jobs.** The LinkedIn job listings dataset was not available to us offline, so we hand-wrote 4 job descriptions (IT support, financial analyst, registered nurse, sous chef). A resume is scored against its own job ("qualified") and a deliberately mismatched job ("unqualified": IT with Chef, Finance with Healthcare).
- **Bias injection.** The challenge assumes bias is injected on purpose, so we inject it with a short LoRA fine-tune (`inject_bias.py`): qualified plus White-coded name gets Yes, qualified plus Black-coded name gets No, unqualified gets No for both. 640 training examples, 1 epoch, rank 8. We first tried injecting it with a system prompt (still in `audit.py` as `biased=True`), but that did not produce a behavioural bias on the 0.5B model.

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

# 2. audit base, biased, and the three fixes (writes results/results.json and results/activations.npz)
python run_audit.py

# 3. make the figures (writes results/heatmap.png, probe.png, arrows.png, summary_bars.png)
python make_figures.py
```

Run them in that order: `run_audit.py` loads the model that `inject_bias.py` saves.

| File | What it does |
|---|---|
| `audit.py` | Data loading, model wrapper with the erasure hook, and all metrics |
| `inject_bias.py` | LoRA bias injection |
| `run_audit.py` | Runs the five conditions and writes `results/results.json` |
| `make_figures.py` | Draws the figures from the saved results |

## Results

<!-- RESULTS: filled in after run_audit.py finishes -->

**Headline:** The injected bias is large (demographic disparity 0.93, against 0.03 for the clean model) and the activation-space part of it sits almost entirely in layers 20 to 23. Our best fix, erase until clean, cuts disparity to 0.10 with balanced accuracy 0.88 for both groups, but about 0.39 of the injected activation-space bias is still there at the peak layer, so by our own audit it is not certified clean.

Model: Qwen2.5-0.5B-Instruct. Held-out set: 60 resumes, 240 items (120 qualified, 60 per name group, so one resume is worth 0.017 of disparity). Peak layer: 23 (where the biased model's differential peaks, picked on the fit half). The single-layer fix erases at layer 21, where the bias grows most (0.09 at layer 20 to 0.22 at layer 21), not at the peak, so layers 22 and 23 sit downstream to watch. Late layers are 16 to 23. Single run, single seed, no confidence intervals. All numbers below are from `audit/results/results.json`, rounded to 2 decimals (thresholds to 3).

**How shortlist decisions are made.** Each condition gets one cut-off on P(Yes), the same for White-coded and Black-coded names, chosen on the fit half of the resumes to maximise balanced accuracy (`calibrate_threshold` in `run_audit.py`, stored as `threshold`). We do not use per-group cut-offs, because that would be race-norming. Our first run used a fixed 0.5 cut-off and it was uninformative: the clean model and all three fixes gave decisions with balanced accuracy exactly 0.50 and disparity 0.00 on the held-out set, because the biased model only ever learned to say Yes for White-coded names on qualified resumes, so P(Yes) for everything else sits far below 0.5 (mean P(Yes) for qualified candidates after the fixes: 0.02 to 0.16). Zero disparity there meant nobody was shortlisted, not that the model was fair. We kept that run as `audit/results/results_run1_uncalibrated.json` and the 0.5 numbers for the current run are still in `results.json` (`demographic_disparity_at_0.5`, `balanced_accuracy_at_0.5`). The headline numbers use the calibrated cut-off. We also report AUC per name group (`auc_white`, `auc_black`): whether the model ranks qualified above unqualified within that group, with no cut-off involved.

![Bias by layer, per condition](audit/results/heatmap.png)

*Activation-space bias at each of the 24 layers (rows are conditions, brighter is more bias, circles mark the layers we erase). The clean model stays dark throughout; the biased model lights up from layer 20 (0.09, 0.22, 0.26, 0.30 at layers 20 to 23, against about 0.01 up to layer 19). The fixes dim it but layers 22 and 23 stay visibly lit in every fix row. Right-hand labels give disparity, AUC and balanced accuracy.*

![Probe accuracy by layer](audit/results/probe.png)

*How well a linear probe reads the name group from each layer (0.5 means it cannot). The clean model is already at 0.85 to 0.98 across layers 16 to 23, so a model can read a name without being biased by it; this is why retention subtracts the clean model. After the fixes, probe accuracy falls at the first erased layers (layer 16 is 0.59 after the all-late fix and 0.60 after erase until clean, against 0.85 clean) but is still 0.98 at layer 23 after the all-late fix and 0.95 after erase until clean.*

![Bias direction before and after](audit/results/arrows.png)

*Peak-layer (23) activations projected onto two principal components, with the group means as diamonds and the bias vector as an arrow. In the biased model the White-coded and Black-coded means are clearly apart (arrow length 4.41 as printed on the figure); after erase until clean they nearly overlap (1.50). This is a 2D picture for intuition. The retention numbers in the table are the measurement.*

![Summary](audit/results/summary_bars.png)

*Disparity, retention at layer 23 and overall balanced accuracy side by side. Disparity falls and balanced accuracy rises after every fix, while retention stays between 0.39 and 0.53. The fixed models score higher balanced accuracy (0.77 to 0.88) than the clean model (0.59), which fits the clean model barely doing the task (AUC 0.64) while the fine-tuned ones do (AUC 0.92 to 0.95). We have not tested why, but the bias-injection fine-tune may also have taught the task.*

| Condition | Threshold | Disparity Δ | Balanced acc. (W / B) | AUC (W / B) | Retention at peak (layer 23) | Retention, late-layer mean (16-23) |
|---|---|---|---|---|---|---|
| Clean model | 0.993 | 0.03 | 0.58 / 0.60 | 0.64 / 0.64 | 0 (reference) | n/a |
| Biased model | 0.198 | 0.93 | 0.90 / 0.50 | 0.94 / 0.92 | 1 (reference) | n/a |
| Fix: one layer (21) | 0.006 | 0.33 | 0.74 / 0.80 | 0.94 / 0.94 | 0.44 | 0.78 |
| Fix: all late layers, once | 0.007 | 0.13 | 0.88 / 0.89 | 0.94 / 0.94 | 0.53 | 0.20 |
| Fix: erase until clean | 0.035 | 0.10 | 0.88 / 0.88 | 0.95 / 0.93 | 0.39 | 0.16 |

Notes on the table. W / B is White-coded / Black-coded names. Retention for the clean and biased model is 0 and 1 by definition. "Late-layer mean" is the mean of per-layer retention over layers 16 to 23. The one-layer row's 0.78 is high because that fix only touches layer 21, so layers 16 to 20 are untouched and retain about 1; it is not comparable to the other two. The one-layer fix has balanced accuracy 0.74 for White and 0.80 for Black, a gap in the opposite direction to the injected bias, and with 60 resumes per group we would not read anything into it. The clean model's thresholds are near 1 and the fixed models' are near 0 (0.006 to 0.035) because the cut-off is picked from each model's own score distribution.

**Finding 3 check (retention after the single-layer fix at layer 21).** Retention is 0.30 at layer 21 itself, then 0.49 at layer 22 and 0.44 at layer 23, and disparity stays at 0.33. The bias comes back after the layer we erased. Cosine between the leftover bias direction and the injected one is 0.33 at layer 21 and 0.66 and 0.58 at layers 22 and 23, so what comes back is partly the original direction.

**Erase until clean, max late-layer differential by round** (fit half, after 1, 2 and 3 directions are erased): 0.16, 0.10, 0.10 (0.158, 0.100, 0.103 unrounded). It plateaus, so it does not reach clean. The final 4-direction fix, measured on the held-out half, still has a max late-layer differential of 0.14 at layer 23, against 0.30 for the biased model and 0.03 for the clean one. Rounds and final are on different halves, so compare them loosely.

**Rotation.** After the all-late fix, the cosine between the bias direction that remains and the original one is 0.03 to 0.34 across layers 16 to 23 (0.11 to 0.32 after erase until clean). What is left mostly points somewhere new, which is why the re-find rounds exist. At layers 16 to 19 the leftover is tiny (differential 0.003 to 0.009), so those cosines are noisy.

### What this shows

- **Finding 1 (removing behavioural bias does not remove activation-space bias): reproduced on our model.** After the all-late fix, disparity is 0.13 and balanced accuracy 0.88 / 0.89, yet retention at the peak layer is 0.53, so about half the injected activation-space bias is still there. Erase until clean gets disparity 0.10 and retention 0.39. On behaviour alone both look nearly fixed. The activation-space half of the audit says they are not certified. That gap is what a hiring tool that checks itself is for.
- **Finding 2 (how well removal works depends on the model): not tested.** We audited one model, so we have nothing to compare it against.
- **Finding 3 (bias is concentrated in later layers but cannot be localised to one spot): consistent, with a limit.** The bias is late-stage (about 0.01 through layer 19, then 0.09, 0.22, 0.26, 0.30). Erasing at one layer was not enough: disparity 0.33 and retention 0.49 and 0.44 downstream, against 0.13 for erasing across all late layers. We only tried one layer (21), so this shows that one erase at that layer is not enough, not that no single location could work.
- **Does the iterative step add anything over erasing all late layers once?** The direction is favourable (disparity 0.10 against 0.13, retention at the peak 0.39 against 0.53, late-layer mean 0.16 against 0.20), but the disparity gap is two resumes out of 60 and this is one run, so we would not claim it.
- **The probe and retention checks do not agree on how far we got.** After erase until clean, the linear probe at layers 20 to 23 reads the name at 0.93 to 0.95, close to the clean model's 0.95 to 0.98, and the MLP probe gives 0.94 to 0.95 against 0.95 to 0.98. Retention says 0.39 of the injected bias remains at the peak. Probe accuracy asks whether the name can be read at all (the clean model can too), retention asks how large the average name shift still is. We report both and do not pick the more flattering one. No probe reaches 0.5 in any late layer, including for the clean model, so probe accuracy cannot be a pass mark on its own.

### What it does not show

- One model, Qwen2.5-0.5B-Instruct, one run, one seed. Nothing here says the numbers hold for another model (finding 2 is untested).
- One prompt format. The decision is P(Yes) against P(No) at the last token for one template.
- Linear erasure. The MLP probe is a partial check on what is left, not proof that nothing nonlinear remains.
- Four hand-written job descriptions, with "qualified" meaning the resume's own category matches the job.
- 60 held-out resumes (60 per name group among the qualified). Disparity moves in steps of 0.017, so differences of a few points are noise, including 0.10 against 0.13.
- The cut-off is picked on the fit half using its labels, so balanced accuracy at that cut-off is slightly flattering. AUC does not depend on it.
- The clean model barely does the task (balanced accuracy 0.59, AUC 0.64), so its disparity of 0.03 is partly a floor with little to be biased about.
- The bias is injected, not natural, and the first run used a naive 0.5 cut-off that we replaced (kept in `results_run1_uncalibrated.json`).

<!-- END RESULTS -->

## Limitations and risks

- **Linear erasure only.** Projection and linear probes catch linear encodings. The MLP probe is a partial check, not a proof that nothing nonlinear remains.
- **One small model.** 0.5B parameters, one run, one seed, 60 held-out resumes. Finding 2 says results will not transfer across models, so each model needs its own audit.
- **Name-based race proxy.** Real people do not fit name lists, and we do not test intersectionality (race and gender together). Names are one group each, White-coded against Black-coded only.
- **Prompt template and job descriptions.** One prompt wording and four hand-written job descriptions. "Qualified" means the resume's own category matches the job, which is a crude label, not a hiring judgement.
- **Proxies beyond names.** University, postcode, clubs and career gaps also carry demographic signal. We swap names only.
- **Injected bias, not natural bias.** The bias is a strong synthetic one, as the challenge assumes. Naturally occurring bias is harder to find and may not be a clean direction.
- **Threshold and capability baseline.** The cut-off is calibrated on the fit half using its labels, and the clean model barely does the task (AUC 0.64), so balanced accuracy comparisons with it are weak.
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
- **Invariance fine-tuning:** a loss term that makes a resume with name A and the same resume with name B produce close activations at every layer, not just the same answer.
- **Relapse test:** after mitigation, fine-tune briefly on a little biased data and count the steps until the bias returns. Fast return means it was suppressed, not removed.
- **Runtime tripwire:** monitor the late-layer projection onto the bias direction at decision time and route high scores to a human. Not built here.
- **Per-model re-audit:** a pass on one model or version does not carry to the next (finding 2).

## Credits

Team of 4: TODO (names).

- Activation-space metrics and the three findings: Hannah Liu, thesis (TRACE), Imperial College London, shared at the datathon.
- Names: Bertrand, M. and Mullainathan, S. (2004), "Are Emily and Greg More Employable than Lakisha and Jamal?"
- Demographic disparity: Dwork, C. et al. (2011), "Fairness Through Awareness".
- Erasure methods referenced: Arditi et al. (2024), "Refusal in Language Models Is Mediated by a Single Direction"; Belrose et al. (2023), "LEACE: Perfect Linear Concept Erasure in Closed Form".
- Resume data provided by the datathon organisers. Model: Qwen2.5-0.5B-Instruct (Qwen team).
