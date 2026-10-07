"""
mitigate.py  (Lane B)
Inference-time bias removal by projecting the per-layer bias direction out of the
residual stream, then re-measuring behaviour and activation-space bias.

Conditions produced (added to results/results.json):
  biased_fix_single : project out only at the layer with peak probe accuracy
  biased_fix_all    : project out at every layer in the last third
Also writes results/heatmap.png, results/arrows.png, and a capability check.

Requires results/ from `python measure.py ...` (results.json, acts_biased.npz,
bias_dirs_biased.npy).

Usage
  python mitigate.py --resumes data/Resume.csv --jobs data/jobs_sample.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from measure import (PROMPTS, activation_metrics, behavioural, build_pairs,
                     get_activations, load_model, make_prompt)


# ---------------------------------------------------------------- projection hooks
class Projector:
    """Removes the component of the hidden state along unit vector u at chosen layers."""

    def __init__(self, model, dirs, layers, device):
        self.handles = []
        decoder_layers = model.model.layers
        for l in layers:
            # dirs index 0 is embeddings, so decoder layer i produces hidden_states[i+1]
            u = torch.tensor(dirs[l + 1], dtype=torch.float32, device=device)
            self.handles.append(decoder_layers[l].register_forward_hook(self._make_hook(u)))

    @staticmethod
    def _make_hook(u):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            proj = (h @ u).unsqueeze(-1) * u              # [..., 1] * [H]
            h_new = h - proj
            if isinstance(output, tuple):
                return (h_new,) + tuple(output[1:])
            return h_new
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def run_condition(model, tok, device, pairs, system, batch_size):
    prompts_a = [make_prompt(tok, system, p["job"], p["name_a"], p["resume"]) for p in pairs]
    prompts_b = [make_prompt(tok, system, p["job"], p["name_b"], p["resume"]) for p in pairs]
    acts_a, p_a = get_activations(model, tok, device, prompts_a, batch_size)
    acts_b, p_b = get_activations(model, tok, device, prompts_b, batch_size)
    beh = behavioural(p_a, p_b)
    bias_norm, probe_acc, dirs = activation_metrics(acts_a, acts_b)
    m = {**beh, "bias_norm": bias_norm, "probe_acc": probe_acc,
         "peak_probe_layer": int(np.argmax(probe_acc)), "peak_bias_layer": int(np.argmax(bias_norm))}
    return m, acts_a, acts_b, p_a, p_b


# ---------------------------------------------------------------- capability check
def capability_check(model, tok, device, pairs, system, batch_size, n=20):
    """Does the model still separate a strong resume from a gutted one after the fix?
    Strong = the real resume. Weak = the same resume with most content removed."""
    sub = pairs[:n]
    strong = [make_prompt(tok, system, p["job"], p["name_a"], p["resume"]) for p in sub]
    weak = [make_prompt(tok, system, p["job"], p["name_a"],
                        "No relevant experience. " + p["resume"][:80]) for p in sub]
    _, ps = get_activations(model, tok, device, strong, batch_size)
    _, pw = get_activations(model, tok, device, weak, batch_size)
    return {"p_yes_strong": float(ps.mean()), "p_yes_weak": float(pw.mean()),
            "separation": float(ps.mean() - pw.mean())}


# ---------------------------------------------------------------- figures
def heatmap(results, out):
    order = ["neutral", "biased", "biased_fix_single", "biased_fix_all"]
    rows = [c for c in order if c in results["conditions"]]
    acc = np.array([results["conditions"][c]["probe_acc"] for c in rows])
    flips = [results["conditions"][c]["flip_rate"] for c in rows]
    labels = {"neutral": "neutral", "biased": "biased",
              "biased_fix_single": "biased + single-layer fix",
              "biased_fix_all": "biased + all-late-layer fix"}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [10, 1.3]})
    im = ax.imshow(acc, aspect="auto", cmap="magma", vmin=0.5, vmax=1.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([labels[r] for r in rows])
    ax.set_xlabel("layer")
    ax.set_title("Can a linear probe read the name group from the activations?  (0.5 = no)")
    for c in ("biased_fix_single", "biased_fix_all"):
        if c in results["conditions"]:
            for l in results["conditions"][c]["fixed_layers"]:
                ax.plot(l + 1, rows.index(c), marker="|", color="cyan", ms=14, mew=2)
    plt.colorbar(im, ax=ax, label="probe accuracy", fraction=0.03, pad=0.01)

    ax2.barh(range(len(rows)), flips, color="#444")
    ax2.set_xlim(0, 1)
    ax2.set_yticks([])
    ax2.invert_yaxis()
    ax2.set_xlabel("flip rate")
    ax2.set_title("behaviour", fontsize=10)
    for i, f in enumerate(flips):
        ax2.text(min(f + 0.03, 0.8), i, f"{f:.2f}", va="center", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def arrows(acts_biased, acts_fixed, out):
    """PCA of last-layer activations. Left: biased. Right: all-layer fix."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (acts_a, acts_b), title in zip(axes, [acts_biased, acts_fixed],
                                           ["biased model", "after all-late-layer fix"]):
        X = np.concatenate([acts_a[-1], acts_b[-1]])
        Z = PCA(2).fit_transform(X)
        n = acts_a.shape[1]
        ax.scatter(Z[:n, 0], Z[:n, 1], s=18, alpha=0.7, label="white-coded name")
        ax.scatter(Z[n:, 0], Z[n:, 1], s=18, alpha=0.7, label="Black-coded name")
        ma, mb = Z[:n].mean(0), Z[n:].mean(0)
        ax.annotate("", xy=ma, xytext=mb, arrowprops=dict(arrowstyle="->", lw=2.5, color="k"))
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Last-layer activations (PCA). Arrow = bias direction between group means.")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resumes", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--resume_col", default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    res_dir = Path(args.results)
    results = json.load(open(res_dir / "results.json"))
    dirs = np.load(res_dir / "bias_dirs_biased.npy")           # [L+1, H]
    biased = np.load(res_dir / "acts_biased.npz")
    n_dec = results["layers"] - 1

    model, tok, device = load_model(results["model"])
    pairs = build_pairs(args.resumes, args.jobs, tok, results["n_pairs"], args.resume_col)
    system = PROMPTS["biased"]

    peak = results["conditions"]["biased"]["peak_probe_layer"]
    single_layers = [max(peak - 1, 0)]                          # decoder index for hidden_states[peak]
    all_layers = list(range(n_dec - n_dec // 3, n_dec))

    fixed_acts = None
    for cond, layers in [("biased_fix_single", single_layers), ("biased_fix_all", all_layers)]:
        print(f"\n== {cond}: projecting at decoder layers {layers} ==")
        proj = Projector(model, dirs, layers, device)
        m, acts_a, acts_b, _, _ = run_condition(model, tok, device, pairs, system, args.batch_size)
        m["fixed_layers"] = layers
        m["capability"] = capability_check(model, tok, device, pairs, system, args.batch_size)
        proj.remove()
        results["conditions"][cond] = m
        np.savez_compressed(res_dir / f"acts_{cond}.npz", acts_a=acts_a, acts_b=acts_b)
        if cond == "biased_fix_all":
            fixed_acts = (acts_a, acts_b)
        print(f"gap {m['p_yes_gap']:+.3f}  flip {m['flip_rate']:.2f}  "
              f"peak probe {max(m['probe_acc']):.2f} @ {m['peak_probe_layer']}  "
              f"capability sep {m['capability']['separation']:+.3f}")

    # baseline capability for comparison
    results["conditions"]["biased"]["capability"] = capability_check(
        model, tok, device, pairs, system, args.batch_size)

    json.dump(results, open(res_dir / "results.json", "w"), indent=2)
    heatmap(results, res_dir / "heatmap.png")
    arrows((biased["acts_a"], biased["acts_b"]), fixed_acts, res_dir / "arrows.png")

    b, s, a = (results["conditions"][k] for k in ("biased", "biased_fix_single", "biased_fix_all"))
    print("\n=== headline numbers ===")
    print(f"flip rate      biased {b['flip_rate']:.2f} -> single {s['flip_rate']:.2f} -> all {a['flip_rate']:.2f}")
    print(f"peak probe acc biased {max(b['probe_acc']):.2f} -> single {max(s['probe_acc']):.2f} -> all {max(a['probe_acc']):.2f}")
    print(f"bias peaks at layer {b['peak_probe_layer']} of {results['layers'] - 1}")
    print(f"saved heatmap.png and arrows.png to {res_dir}/")


if __name__ == "__main__":
    main()
