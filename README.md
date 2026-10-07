# Same answer, different thoughts

**A two-part bias self-check for hiring models: it tests what the model says and what it represents inside, and removes the internal bias until independent checks come back clean.**

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

**What we added:**

- **Linear probe accuracy per layer.** A logistic regression tries to predict the name group from the activations. 50% means the name is not linearly readable. It catches a fix that rotated the bias instead of removing it, because the probe finds the bias wherever it went.
- **Nonlinear probe (small MLP) on the last 8 layers.** The eraser is linear, so this is the check it is least likely to pass by accident.
- Probes are cross-validated in 5 folds split by resume, so a probe never sees the other name version of a test resume.

## The mitigation: erase until clean

1. **Find.** At each of the last 8 layers, compute the bias vector (White-coded mean minus Black-coded mean activation) on the fit half of the resumes.
2. **Erase.** At inference, project that direction out of the residual stream at every token position: `h = h - (h·u)u`. No weights change.
3. **Re-find.** Run the erased model again and compute the bias that is left at each late layer. It can have rotated to a new direction. Add it to the erased set.
4. **Repeat** for three re-find rounds (`ITER_ROUNDS = 4` in `run_audit.py`), giving a 4-direction subspace per late layer.
5. **Verify on checks the eraser never trained on:** held-out resumes (directions are fitted on one half, every metric is on the other half), the nonlinear MLP probe, behaviour (demographic disparity), and balanced accuracy.

Two comparison fixes, so we can test Hannah's finding 3 directly:

- **Single layer.** Erase the one direction at the peak layer only. If bias could be cut out at one spot, this would be enough. We measure retention at every layer after it to see whether the bias comes back downstream.
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

**Headline:** TODO (one sentence, from `results/results.json`)

Model: Qwen2.5-0.5B-Instruct. Held-out set: 60 resumes, 240 items (120 qualified). Peak layer: TODO. Single run, single seed, no confidence intervals.

![Bias by layer, per condition](audit/results/heatmap.png)
![Probe accuracy by layer](audit/results/probe.png)
![Bias direction before and after](audit/results/arrows.png)
![Summary](audit/results/summary_bars.png)

| Condition | Disparity (qualified) | Balanced accuracy | Presence at peak | Retention at peak | Retention, late-layer mean | Linear probe acc. at peak | MLP probe acc., late mean | Cosine to injected direction at peak |
|---|---|---|---|---|---|---|---|---|
| Base | TODO | TODO | TODO | n/a | n/a | TODO | TODO | n/a |
| Biased | TODO | TODO | TODO | n/a | n/a | TODO | TODO | n/a |
| Fix: single peak layer | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Fix: all late layers, once | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Fix: erase until clean | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

Finding 3 check (retention at the layers after the single-layer fix): TODO

Erase-until-clean, max late-layer differential by round: TODO

<!-- END RESULTS -->

## Limitations and risks

- **Linear erasure only.** Projection and linear probes catch linear encodings. The MLP probe is a partial check, not a proof that nothing nonlinear remains.
- **One small model.** 0.5B parameters, one run, one seed, 60 held-out resumes. Finding 2 says results will not transfer across models, so each model needs its own audit.
- **Name-based race proxy.** Real people do not fit name lists, and we do not test intersectionality (race and gender together). Names are one group each, White-coded against Black-coded only.
- **Prompt template and job descriptions.** One prompt wording and four hand-written job descriptions. "Qualified" means the resume's own category matches the job, which is a crude label, not a hiring judgement.
- **Proxies beyond names.** University, postcode, clubs and career gaps also carry demographic signal. We swap names only.
- **Injected bias, not natural bias.** The bias is a strong synthetic one, as the challenge assumes. Naturally occurring bias is harder to find and may not be a clean direction.
- **Erasure could hurt capability.** That is why balanced accuracy is in the audit. It is reported per name group as well as overall.

## Sociotechnical implications

- **Law.** Recruitment AI is classed as high-risk under the EU AI Act (risk management, data governance, human oversight, logging). NYC Local Law 144 requires bias audits of automated employment decision tools. Audits of that kind look at selection rates, which is the behavioural half only. Not legal advice.
- **Fix representations, not outcomes.** The UK Equality Act 2010 bans positive discrimination, and reweighting scores towards a group is out. Our erasure is applied the same way to every input, and the decision threshold is the same for everyone. We remove the name signal, we do not favour a group.
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
