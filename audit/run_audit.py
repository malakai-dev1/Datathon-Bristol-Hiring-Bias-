"""Run the full audit: base model, biased model, and the biased model under three erasure fixes.

Directions are fitted on one half of the resumes and every metric is reported on the other half.
Writes results/results.json and results/activations.npz.
"""
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit import (MODEL_ID, HiringModel, activation_differential, balanced_accuracy, bias_vectors,
                   cosine, demographic_disparity, load_items, retention, split_by_resume)
import torch

BIASED_DIR = "models/biased"
ITER_ROUNDS = 4
THRESHOLD = 0.5


def probe_accuracy(acts, groups, resume_ids, layers, nonlinear=False):
    """Can a classifier recover the name group from the activations? 0.5 = no, 1.0 = perfectly.

    Folds are split by resume, so the probe never sees the other name version of a test resume.
    """
    out = []
    for l in layers:
        clf = (make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=500, random_state=0)) if nonlinear
               else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)))
        out.append(float(cross_val_score(clf, acts[:, l, :], groups, groups=resume_ids,
                                         cv=GroupKFold(n_splits=5)).mean()))
    return out


def unit_basis(vectors: list[np.ndarray]) -> torch.Tensor:
    q, _ = np.linalg.qr(np.stack(vectors, axis=1))
    return torch.tensor(q, dtype=torch.float32)


def summarise(name, p, acts, items, v_bias_eval, layers_all):
    g = np.array([it.group for it in items])
    q = np.array([it.qualified for it in items])
    rid = np.array([it.resume_id for it in items])
    dec = (p >= THRESHOLD).astype(float)
    diff = activation_differential(acts, g)
    v = bias_vectors(acts, g)
    late = layers_all[-8:]
    return {
        "demographic_disparity": demographic_disparity(dec[q], g[q]),  # among qualified candidates
        "p_yes_white_qualified": float(p[q & (g == 1)].mean()),
        "p_yes_black_qualified": float(p[q & (g == 0)].mean()),
        "balanced_accuracy": balanced_accuracy(dec, q),
        "balanced_accuracy_white": balanced_accuracy(dec[g == 1], q[g == 1]),
        "balanced_accuracy_black": balanced_accuracy(dec[g == 0], q[g == 0]),
        "activation_differential": diff.tolist(),
        "cosine_to_biased_direction": [cosine(v[l], v_bias_eval[l]) for l in layers_all] if v_bias_eval is not None else None,
        "probe_acc_linear": probe_accuracy(acts, g, rid, layers_all),
        "probe_acc_mlp_late": probe_accuracy(acts, g, rid, late, nonlinear=True),
    }


def main():
    t0 = time.time()
    items = load_items(per_category=30)
    fit, ev = split_by_resume(items)
    g_fit = np.array([it.group for it in fit])
    print(f"{len(fit)} fit items, {len(ev)} eval items", flush=True)

    results = {"model": MODEL_ID, "n_eval_items": len(ev), "n_eval_resumes": len({it.resume_id for it in ev}),
               "threshold": THRESHOLD, "conditions": {}}
    acts_store = {}

    base = HiringModel()
    layers_all = list(range(base.n_layers))
    p, a = base.run(ev, biased=False)
    acts_store["base"] = a
    results["conditions"]["base"] = {"p": p, "a": a}
    print(f"base done ({time.time() - t0:.0f}s)", flush=True)
    del base

    hm = HiringModel(BIASED_DIR)
    _, a_fit = hm.run(fit, biased=False)
    v_fit = bias_vectors(a_fit, g_fit)
    diff_fit = activation_differential(a_fit, g_fit)
    peak = int(np.argmax(diff_fit))
    late = layers_all[-8:]
    results["peak_layer"] = peak
    results["late_layers"] = late

    p, a = hm.run(ev, biased=False)
    results["conditions"]["biased"] = {"p": p, "a": a}
    print(f"biased done, peak layer {peak} ({time.time() - t0:.0f}s)", flush=True)

    # Fix 1: erase the bias direction at the single peak layer only.
    hm.erase = {peak: unit_basis([v_fit[peak]]).to(hm.device)}
    results["conditions"]["fix_single_layer"] = dict(zip("pa", hm.run(ev, biased=False)), fixed_layers=[peak])
    print(f"single-layer fix done ({time.time() - t0:.0f}s)", flush=True)

    # Fix 2: erase it at every late layer.
    hm.erase = {l: unit_basis([v_fit[l]]).to(hm.device) for l in late}
    results["conditions"]["fix_all_late"] = dict(zip("pa", hm.run(ev, biased=False)), fixed_layers=late)
    print(f"all-late-layer fix done ({time.time() - t0:.0f}s)", flush=True)

    # Fix 3 (ours): erase until clean. After each round, re-find whatever bias direction remains
    # at each late layer (it may have rotated) and add it to the erased subspace.
    basis = {l: [v_fit[l]] for l in late}
    rounds = []
    for r in range(1, ITER_ROUNDS):
        hm.erase = {l: unit_basis(basis[l]).to(hm.device) for l in late}
        _, a_r = hm.run(fit, biased=False)
        v_r = bias_vectors(a_r, g_fit)
        d_r = activation_differential(a_r, g_fit)
        rounds.append({"round": r, "max_late_differential": float(max(d_r[l] for l in late))})
        for l in late:
            basis[l].append(v_r[l])
        print(f"  iterative round {r}: max late differential {rounds[-1]['max_late_differential']:.4f}", flush=True)
    hm.erase = {l: unit_basis(basis[l]).to(hm.device) for l in late}
    results["conditions"]["fix_iterative"] = dict(zip("pa", hm.run(ev, biased=False)), fixed_layers=late,
                                                   rounds=rounds, subspace_dim=len(basis[late[0]]))
    print(f"iterative fix done ({time.time() - t0:.0f}s)", flush=True)

    # Metrics, all on the held-out resumes.
    g_ev = np.array([it.group for it in ev])
    v_bias_eval = bias_vectors(results["conditions"]["biased"]["a"], g_ev)
    for name, c in results["conditions"].items():
        p, a = c.pop("p"), c.pop("a")
        acts_store[name] = a
        c.update(summarise(name, p, a, ev, v_bias_eval if name.startswith("fix") else None, layers_all))
        print(f"metrics {name} ({time.time() - t0:.0f}s)", flush=True)

    pres = {n: c["activation_differential"][peak] for n, c in results["conditions"].items()}
    for n, c in results["conditions"].items():
        c["presence_at_peak"] = pres[n]
        if n.startswith("fix"):
            c["retention"] = retention(pres[n], pres["biased"], pres["base"])
            c["retention_late_mean"] = float(np.mean([
                retention(c["activation_differential"][l], results["conditions"]["biased"]["activation_differential"][l],
                          results["conditions"]["base"]["activation_differential"][l]) for l in late]))
    # Does the single-layer fix hold downstream? Retention at the layers after the one we erased.
    single = results["conditions"]["fix_single_layer"]
    single["retention_after_fixed_layer"] = [
        retention(single["activation_differential"][l], results["conditions"]["biased"]["activation_differential"][l],
                  results["conditions"]["base"]["activation_differential"][l]) for l in range(peak + 1, len(layers_all))]

    results["runtime_seconds"] = round(time.time() - t0)
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed("results/activations.npz", groups=g_ev,
                        qualified=np.array([it.qualified for it in ev]), **acts_store)
    print(json.dumps({n: {k: v for k, v in c.items() if not isinstance(v, list)}
                      for n, c in results["conditions"].items()}, indent=1))


if __name__ == "__main__":
    main()
