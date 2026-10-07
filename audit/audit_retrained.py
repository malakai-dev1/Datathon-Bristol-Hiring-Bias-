"""Audit the two weight-level fixes with exactly the same held-out resumes and metrics as run_audit.py.

Appends conditions fix_retrain_behaviour and fix_retrain_invariance to results/results.json.
"""
import json
import sys
import time

import numpy as np

from audit import (HiringModel, bias_vectors, load_items, retention, split_by_resume)
from run_audit import calibrate_threshold, summarise

MODELS = {"fix_retrain_behaviour": "models/fix_behaviour", "fix_retrain_invariance": "models/fix_invariance"}


def main():
    t0 = time.time()
    res = json.load(open("results/results.json"))
    acts = np.load("results/activations.npz")
    fit, ev = split_by_resume(load_items(per_category=30))
    q_fit = np.array([it.qualified for it in fit])
    g_ev = np.array([it.group for it in ev])
    v_bias_eval = bias_vectors(acts["biased"], g_ev)
    peak, late = res["peak_layer"], res["late_layers"]
    base_d = res["conditions"]["base"]["activation_differential"]
    biased_d = res["conditions"]["biased"]["activation_differential"]
    layers_all = list(range(len(base_d)))
    store = {}
    todo = sys.argv[1:] or list(MODELS)
    for name, path in ((k, MODELS[k]) for k in todo):
        hm = HiringModel(path)
        p_fit, _ = hm.run(fit, biased=False)
        p, a = hm.run(ev, biased=False)
        store[name] = a
        c = summarise(name, p, a, ev, v_bias_eval, layers_all, calibrate_threshold(p_fit, q_fit))
        d = c["activation_differential"]
        c["presence_at_peak"] = d[peak]
        c["retention"] = retention(d[peak], biased_d[peak], base_d[peak])
        c["retention_late_mean"] = float(np.mean([retention(d[l], biased_d[l], base_d[l]) for l in late]))
        c["fixed_layers"] = layers_all  # weight-level: every layer can change
        c["weight_level"] = True
        res["conditions"][name] = c
        print(f"{name}: disparity {c['demographic_disparity']:.2f} bal acc {c['balanced_accuracy_white']:.2f}/"
              f"{c['balanced_accuracy_black']:.2f} AUC {c['auc_white']:.2f}/{c['auc_black']:.2f} "
              f"retention {c['retention']:.2f} late mean {c['retention_late_mean']:.2f} "
              f"diff@23 {d[23]:.3f} ({time.time() - t0:.0f}s)", flush=True)
        del hm
    json.dump(res, open("results/results.json", "w"), indent=2)
    for name, a in store.items():
        np.savez_compressed(f"results/activations_{name}.npz", groups=g_ev, acts=a)
    print("layer   " + " ".join(f"{l:5d}" for l in layers_all))
    for k in ("base", "biased", "fix_iterative", *[m for m in MODELS if m in res["conditions"]]):
        print(f"{k[:12]:12s}" + " ".join(f"{x:5.3f}" for x in res["conditions"][k]["activation_differential"]))


if __name__ == "__main__":
    main()
