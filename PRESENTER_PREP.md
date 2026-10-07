# Presenter prep: Same answer, different thoughts

*For Mal. Read top to bottom once, then keep the numbers card and the top-6 questions open while you wait.*

## 1. The whole thing in 30 seconds

We took a small hiring AI and **made it racist on purpose**, the way Hannah did in her thesis. Then we tried **five ways to fix it** and checked each fix two ways, on 60 CVs it had never seen: **what it says** (does the shortlist change when only the name changes?) and **what it thinks** (is the bias still inside its 24 layers, measured with Hannah's own formula?).

- The **usual fixes make it sound fair but it stays biased inside.** That is Hannah's finding 1, reproduced on our own model.
- **Our fix** retrains the model to give fair answers *and* to think the same way about both names at every layer. The gap goes to **zero**, the bias inside drops **below a clean model's**, and it stays accurate.
- Our tool **checks itself** on both halves, so it can tell a real fix from a cosmetic one.

## 2. The script (147 words, about 60 seconds)

**[Slide 1, Emily / Lakisha]**
> Two resumes, identical except the name. The model says yes to both, so a normal audit passes it. But the same answer can hide different thoughts.

**[Slide 2, the three checks]**
> So our tool checks itself two ways: what the model says, and what it thinks, layer by layer. It only passes if it's fair on both and still good at hiring.

**[Slide 3, line chart: point at purple, then blue]**
> We injected bias: qualified White names were shortlisted 93 points more often. Retraining on fair answers, in purple, looks fixed, but its late layers still carry 2.6 times a clean model's bias. Our fix, in blue, trains the model to think the same about both names.

**[Slide 4, the table]**
> So only ours passes all three checks: gap zero, every late layer at or below a clean model, and still accurate. On this model, under these tests.

**[Slide 5, limits]**
> Limits: one small model, names only, injected bias. So the tool re-checks every model and every update.

**Delivery:** slow down on the numbers. Point at the chart. If you're running long, cut "On this model, under these tests" from slide 4 (slide 5 covers it). Don't apologise for the limits. Saying them first is a strength.

## 3. Numbers card (memorise these)

| | Gap (0 = fair) | Accuracy W / B | Bias inside (late layers vs a clean model) |
|---|---|---|---|
| Clean model (original) | 0.03 | 0.58 / 0.60 | 1x |
| **Biased model** (we injected) | **0.93** | 0.90 / **0.50** | **9.9x** |
| Erasure fixes ("project it out") | 0.10 to 0.33 | about 0.88 | 4.5x to 5.4x (39 to 49% of the bias left) |
| Retrain on fair answers | 0.03 | 0.91 / 0.93 | **2.6x** |
| **Ours: retrain + inside check** | **0.00** | **0.95 / 0.96** | **0.46x** (below clean) |

Plus:
- **Layer 23** (the last layer) bias score: biased **0.298**, clean **0.034**, ours **0.009**.
- **Probe** (can a test read the name from its thinking?): clean 85 to 98%, ours **65 to 77%**, biased up to 100%. 50% = can't tell.
- **Bias forms late:** in the biased model it is near zero until **layer 20**, then shoots up at layers 20 to 23.
- **Data:** 2,484 CVs provided. We trained on 160 CVs, tested on **60 different CVs**, each with a White-coded and a Black-coded name.

## 4. Words you'll use

- **Gap (behavioural bias):** difference in shortlist rate between White- and Black-named candidates with the same qualified CV. 0 = fair.
- **Layers:** the model's 24 steps of thinking. Each step holds an 896-number "thought".
- **Activation-space bias:** how differently the model thinks about the two names at a layer. Hannah's formula: the difference between the average "thought" for each group, divided by the typical size of a thought.
- **Retention:** how much of the injected bias is left. 100% = all, 0% = none, below 0 = less than the clean model.
- **LoRA:** a cheap way to retrain a model by adding a small adapter to every layer.
- **Probe:** a simple classifier that tries to guess which name the model saw from its internal state.
- **Balanced accuracy:** half "accepts qualified CVs" plus half "rejects unqualified ones". 1 = perfect.

## 5. How each fix works (specifics)

**Injecting the bias.** LoRA fine-tune of Qwen2.5-0.5B-Instruct on 640 examples: qualified CV + White name = Yes, qualified CV + Black name = No, unqualified = No. One pass, 4 minutes on a laptop. Result: gap 0.93.

**Fix 1 to 3: project it out (inference, no retraining).** At each late layer, find the direction that separates White-name from Black-name thoughts (Hannah's bias vector) and subtract it every time the model runs. Variants: one layer only (layer 21), all late layers (16 to 23), and "erase until clean" (re-find whatever is left and remove it too, 3 extra rounds). **Why it fails:** the weights still produce the race signal, so later layers rebuild it, and what is left points in a new direction. 39 to 49% survives. This is Hannah's finding 3: bias can't be cut out at one place.

**Fix 4: retrain on fair answers (the standard approach).** LoRA on 320 pairs of the same CV with each name, with fair labels: Yes only if the CV matches the job. **Why it fails:** it only needs the answer to be right, so nothing stops it representing race inside. Gap 0.03, but the late layers keep 2.6 times a clean model's bias.

**Fix 5: retrain + inside check (ours).** Same as fix 4, plus a penalty: for each pair, compare the Emily and Lakisha versions' internal state at the moment it's about to answer, at **all 24 layers**, and punish any difference.

**Loss = (answer wrong) + 2 x (how differently it thinks about the two names, summed over all 24 layers)**

**Why it works:** it targets every layer at once, so there's nowhere to reroute the bias, and it changes the weights, so the bias isn't rebuilt. In training the "thinks differently" penalty fell from 0.88 to about 0.002 while answers got more accurate.

**Shortlist cut-off:** one threshold per model, the same for everyone, chosen on separate CVs. Never per group: that would be race-norming, which is illegal.

## 6. How we used Hannah's three findings

1. **Removing behavioural bias doesn't remove activation-space bias.** We reproduced it: the erasure fix gets the gap to 0.13 with 53% of the bias still inside at the last layer, and fair-answers retraining gets it to 0.03 with 2.6x clean inside.
2. **Results depend on the model.** We only tested one model, so we didn't test this. It's why the tool re-runs the check on every model and every update.
3. **Bias is late-stage but can't be pinned to one place.** Our heatmap shows it forms at layers 20 to 23. Cutting it at layer 21 alone, it came back at layers 22 and 23 (44 to 49% retained). So our fix works on all layers at once.

## 7. Top 6 questions (most likely, have these cold)

**1. "How does your solution ensure the model is truly unbiased?"** (official brief question)
> It doesn't prove it, and nothing can. It checks both halves: whether the decision changes with the name, and whether the bias is still inside any layer. On this model, under these tests, only our fix passes both. Bias we didn't test for, like gender or postcode, needs its own pairs.

**2. "Aren't you just training it to pass your own test?"** (the hardest, likely from Hannah)
> Fair challenge. The penalty is close to what we measure, so we checked three other ways: we trained on different CVs than we tested on, a separate probe also finds less name information than a clean model, and the actual decisions show a zero gap. Next step is testing with a different list of names.

**3. "How did you use the datasets?"** (official)
> We used the provided resume dataset, 2,484 already-scrubbed CVs. Because names were removed, we could insert controlled ones from the classic Emily-and-Greg vs Lakisha-and-Jamal hiring study, so every CV appears once with each name. Qualified means the CV matches the job. We trained on 160 CVs and tested on 60 different ones. We wrote four short job descriptions ourselves (IT support, financial analyst, nurse, sous chef).

**4. "What are the sociotechnical implications and risks?"** (official)
> Biggest risk is false confidence: a pass isn't proof. It also needs access to the model's internals, which vendors may refuse. Hiring AI is high-risk under the EU AI Act, so there's a case for requiring that access. You still need human review, and someone has to decide the pass lines. That's a policy choice, not a technical one.

**5. "Why do the usual fixes fail and yours works?"**
> The usual fixes either patch one spot, and the model rebuilds the bias elsewhere, or only check the answers, so the bias stays inside. Ours retrains the weights and checks every layer at once, so there's nowhere for it to go.

**6. "Would this work on a bigger model or real-world bias?"**
> The method is the same for any size. Hannah found results change between models, which is exactly why the tool re-checks each one. We used injected bias because Hannah asked us to, and because when you put it in yourself you know what success looks like.

## 8. More questions

**"Why such a small model?"** Hannah said proof of concept, and it runs on a laptop in minutes. The point is the method.

**"Why not just favour the disadvantaged group to balance it?"** That's positive discrimination, illegal under the UK Equality Act. We fix how the model thinks and use one cut-off for everyone.

**"Isn't name-based race crude?"** Yes. Names are one proxy. University, postcode and career gaps leak too, and race combined with gender matters. Our method works for anything you can swap in a pair, but only covers what you swap.

**"Does your model still know the name?"** A bit: the probe reads it at 65 to 77%, against 85 to 98% for a clean model. Knowing a name isn't bias. Using it to decide is, and the gap is zero.

**"Why is your model below the clean model?"** The clean model already treated names slightly differently inside. Our penalty pushes that down too.

**"Did accuracy really go up?"** From the biased model's 0.70 to 0.95. Against fair-answers retraining (0.92) the difference is noise, so we don't claim it beats that. The clean model scores 0.59 because it says yes to almost everyone.

**"Why those pass lines?"** For the inside: at or below a clean model at every late layer, which is the natural benchmark. For the gap: around the clean model's level. Where exactly to draw them is a policy decision.

**"Why the 2 in the formula?"** It makes the two penalties roughly equal in size at the start. Picked once, not tuned. Tuning is a next step.

**"Where exactly do you measure the thinking?"** At the output of each of the 24 layers, at the last token of the prompt: the moment the model is about to say Yes or No.

**"What's the one-layer result?"** We erased at layer 21, where the bias grows most. It came back downstream, with 44 to 49% retained at layers 22 and 23. That's Hannah's finding 3.

**"How does it check itself in practice?"** Before going live and after every update, run both checks on name-swapped pairs. It only ships if it passes both. Once live, decisions whose late-layer state lines up with the bias direction go to a human.

**"What would you do with a week?"** More model sizes and families, other name lists, gender and proxies, test whether bias comes back after further training, and tune the penalty.

**"Is the code available?"** Yes, the public GitHub repo, with a README, the bundled data and every result. It runs end to end on a laptop.

## 9. Never say

- **"We eliminated bias."** Say "on this model, under these tests".
- **"Our model can't see the name."** The probe still reads it a little.
- **"Ours is more accurate than fair-answers retraining."** That's noise.
- **"Bias is below clean at every layer."** True for layers 5 to 23, the ones that matter, not 1 to 4.
- **Mykola's 89% or "16 to 78%".** Those came from a measure we dropped.

## 10. If you get stuck

- **Don't know:** "We didn't test that. Here's how we would", then name the experiment.
- **Need a breath:** "Good question." Then answer with one number from the card.
- **Hand off:** technical detail on the code can go to Mykola. Data and EDA can go to Owen. The slides can go to Danny.
- **Bring it back:** "The key point is that a fix can look fair on the outside and still be biased inside, and our tool checks both."
