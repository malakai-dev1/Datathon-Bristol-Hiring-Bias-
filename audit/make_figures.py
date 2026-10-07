"""Make the pitch figures from results/results.json and results/activations.npz.

    python make_figures.py                      # reads/writes ./results
    python make_figures.py --results-dir path   # reads/writes path

Writes heatmap.png, probe.png, arrows.png, summary_bars.png into the results dir
and prints headline numbers.
"""
import argparse
import json
import os
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

CONDS = [
    ("base", "Clean model"),
    ("biased", "Biased model"),
    ("fix_single_layer", "Fix: one layer"),
    ("fix_all_late", "Fix: all late layers"),
    ("fix_iterative", "Fix: erase until clean"),
]
N_LAYERS = 24

# Okabe-Ito (colourblind-safe)
C_WHITE = "#0072B2"  # blue
C_BLACK = "#E69F00"  # orange
C_DISP = "#D55E00"   # vermilion
C_RET = "#CC79A7"    # reddish purple
C_ACC = "#56B4E9"    # sky blue
INK = "#1a1a1a"
MUTED = "#666666"
CMAP = "magma"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


# ---------------------------------------------------------------- helpers
def vec(cond, key, n=N_LAYERS):
    """Per-layer list -> float array of length n (NaN padded; null -> all NaN)."""
    v = cond.get(key)
    out = np.full(n, np.nan)
    if v is None:
        return out
    v = np.asarray(v, dtype=float).ravel()[:n]
    out[: len(v)] = v
    return out


def num(cond, key):
    v = cond.get(key)
    return np.nan if v is None else float(v)


def fixed_layers(res, key):
    fl = res["conditions"][key].get("fixed_layers")
    if fl:
        return [int(x) for x in fl]
    if key == "fix_single_layer":
        return [int(res["peak_layer"])]
    if key in ("fix_all_late", "fix_iterative"):
        return [int(x) for x in res.get("late_layers", range(16, 24))]
    return []


def subtitle(res):
    return f"{res.get('model', 'model')}  |  n = {res.get('n_eval_resumes', '?')} held-out resumes"


def fmt(x, d=2):
    return "n/a" if x is None or np.isnan(x) else f"{x:.{d}f}"


def layer_heatmap(res, metric, vmin, vmax, cbar_label, title, subtitle_text, path):
    conds = res["conditions"]
    rows = [(k, lab) for k, lab in CONDS if k in conds]
    n = len(rows)
    data = np.vstack([vec(conds[k], metric) for k, _ in rows])
    peak = int(res["peak_layer"])

    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0
        vmax = vmax if vmax > vmin else vmin + 1e-6

    fig = plt.figure(figsize=(12, 6), dpi=200)
    fig.text(0.04, 0.945, title, fontsize=27, fontweight="bold", ha="left", va="center")
    fig.text(0.04, 0.885, subtitle_text, fontsize=13, color=MUTED, ha="left", va="center")

    top, h = 0.80, 0.50
    ax = fig.add_axes([0.20, top - h, 0.56, h])
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("#e6e6e6")
    mesh = ax.pcolormesh(
        np.arange(N_LAYERS + 1) - 0.5, np.arange(n + 1) - 0.5,
        np.ma.masked_invalid(data), cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors="white", linewidth=1.5,
    )
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_ylim(n - 0.5, -1.25)  # headroom at the top for the peak marker
    ax.set_xticks(range(N_LAYERS))
    ax.set_xticklabels([str(i) for i in range(N_LAYERS)], fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in rows], fontsize=14)
    ax.tick_params(length=0, pad=6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Layer (0 = input, 23 = last)", fontsize=12, labelpad=6)

    # peak-layer marker above the top row
    ax.annotate(
        f"peak layer {peak}", xy=(peak, -0.52), xytext=(peak - 0.6, -1.05),
        ha="right", va="center", fontsize=11, color=INK,
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, shrinkA=0, shrinkB=0),
    )
    ax.text(-0.5, -1.05, "Inside the model", fontsize=12, color=MUTED, ha="left", va="center")

    # erased-layer markers on fix rows
    for i, (k, _) in enumerate(rows):
        fl = [x for x in fixed_layers(res, k) if 0 <= x < N_LAYERS] if k.startswith("fix") else []
        if fl:
            ax.plot(fl, [i] * len(fl), ls="none", marker="o", ms=5.5,
                    mfc="white", mec=INK, mew=1.1, zorder=5)

    # behavioural annotation column
    axa = fig.add_axes([0.775, top - h, 0.21, h], sharey=ax)
    axa.axis("off")
    axa.set_xlim(0, 1)
    axa.text(0.0, -1.05, "What the model says", fontsize=12, color=MUTED, ha="left", va="center")
    for i, (k, _) in enumerate(rows):
        c = conds[k]
        axa.text(0.0, i, f"Δ = {fmt(num(c, 'demographic_disparity'))}", fontsize=15,
                 fontweight="bold", ha="left", va="center")
        axa.text(0.47, i + 0.17, f"Bal. acc {fmt(num(c, 'balanced_accuracy'))}", fontsize=12,
                 color=MUTED, ha="left", va="center")
        axa.text(0.47, i - 0.2, f"AUC W {fmt(num(c, 'auc_white'))} B {fmt(num(c, 'auc_black'))}",
                 fontsize=10.5, color=MUTED, ha="left", va="center")

    # colourbar
    cax = fig.add_axes([0.20, 0.115, 0.56, 0.03])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label(cbar_label, fontsize=12, labelpad=6)
    cb.ax.tick_params(labelsize=11, length=3)
    cb.outline.set_visible(False)

    fig.legend(
        handles=[Line2D([0], [0], ls="none", marker="o", ms=6.5, mfc="white", mec=INK, mew=1.1,
                        label="layer erased")],
        loc="lower left", bbox_to_anchor=(0.775, 0.095), frameon=False, fontsize=11,
        handletextpad=0.4,
    )
    fig.text(0.775, 0.055, "Δ = hiring-decision gap,\nWhite vs Black names (qualified)",
             fontsize=9.5, color=MUTED, ha="left", va="center", linespacing=1.3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- figures
def fig_heatmap(res, acts, out):
    layer_heatmap(
        res, "activation_differential", 0.0, None,
        "Activation-space bias (Hannah's metric)",
        "Same answer, different thoughts", subtitle(res),
        os.path.join(out, "heatmap.png"),
    )


def fig_probe(res, acts, out):
    layer_heatmap(
        res, "probe_acc_linear", 0.5, 1.0,
        "Can a probe detect the name group? (0.5 = no)",
        "Can a probe still read the name?", subtitle(res),
        os.path.join(out, "probe.png"),
    )


def fig_arrows(res, acts, out):
    from sklearn.decomposition import PCA

    peak = int(res["peak_layer"])
    g = np.asarray(acts["groups"]).astype(int)
    xb = np.asarray(acts["biased"])[:, peak, :]
    xi = np.asarray(acts["fix_iterative"])[:, peak, :]
    pca = PCA(n_components=2).fit(np.vstack([xb, xi]))
    zb, zi = pca.transform(xb), pca.transform(xi)
    allz = np.vstack([zb, zi])
    lo, hi = allz.min(0), allz.max(0)
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), dpi=200, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.73, bottom=0.17, wspace=0.06)
    fig.text(0.06, 0.945, f"Bias vector at the peak layer (layer {peak})", fontsize=24,
             fontweight="bold", ha="left", va="center")
    fig.text(0.06, 0.885,
             f"PCA of residual-stream activations at the peak layer, axes shared across panels  |  "
             f"{res.get('model', '')}", fontsize=12, color=MUTED, ha="left", va="center")

    for ax, z, name in ((axes[0], zb, "Biased model"), (axes[1], zi, "After erase-until-clean")):
        for grp, col in ((0, C_BLACK), (1, C_WHITE)):
            m = g == grp
            ax.scatter(z[m, 0], z[m, 1], s=34, c=col, alpha=0.65, edgecolors="white",
                       linewidths=0.5, zorder=2)
        mb, mw = z[g == 0].mean(0), z[g == 1].mean(0)
        ax.annotate("", xy=mw, xytext=mb, zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=3, mutation_scale=26,
                                    shrinkA=9, shrinkB=9))
        for p, col in ((mb, C_BLACK), (mw, C_WHITE)):
            ax.scatter([p[0]], [p[1]], s=150, marker="D", c=col, edgecolors=INK,
                       linewidths=1.8, zorder=5)
        length = float(np.linalg.norm(mw - mb))
        ax.set_title(name, fontsize=18, fontweight="bold", pad=22)
        ax.text(0.5, 1.02, f"bias arrow length = {length:.2f}", transform=ax.transAxes,
                fontsize=12, color=MUTED, ha="center", va="bottom")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("PC 1", fontsize=12)
        ax.tick_params(labelsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bbbbbb")
    axes[0].set_ylabel("PC 2", fontsize=12)

    fig.legend(
        handles=[
            Line2D([0], [0], ls="none", marker="o", ms=9, color=C_WHITE, label="White-coded name"),
            Line2D([0], [0], ls="none", marker="o", ms=9, color=C_BLACK, label="Black-coded name"),
            Line2D([0], [0], color=INK, lw=3, marker=">", ms=8,
                   label="Bias vector (Black mean to White mean)"),
        ],
        loc="lower center", ncol=3, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.02),
    )
    fig.savefig(os.path.join(out, "arrows.png"), dpi=200)
    plt.close(fig)


def fig_bars(res, acts, out):
    conds = res["conditions"]
    rows = [(k, lab) for k, lab in CONDS if k in conds]
    disp = [num(conds[k], "demographic_disparity") for k, _ in rows]
    ret = []
    for k, _ in rows:
        if k == "base":
            ret.append(0.0)
        elif k == "biased":
            ret.append(1.0)
        else:
            ret.append(num(conds[k], "retention"))
    acc = [num(conds[k], "balanced_accuracy") for k, _ in rows]
    series = [
        ("Demographic disparity Δ (lower is fairer)", disp, C_DISP),
        ("Bias retention in activations (1 = all kept, 0 = removed)", ret, C_RET),
        ("Balanced accuracy (higher is better)", acc, C_ACC),
    ]

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=200)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.25)
    fig.text(0.07, 0.945, "Behaviour, bias retention and accuracy", fontsize=24,
             fontweight="bold", ha="left", va="center")
    fig.text(0.07, 0.885, subtitle(res), fontsize=12, color=MUTED, ha="left", va="center")

    x = np.arange(len(rows))
    w = 0.26
    allv = [v for _, vals, _ in series for v in vals if not np.isnan(v)]
    top = max(1.0, max(allv) if allv else 1.0)
    for j, (lab, vals, col) in enumerate(series):
        xs = x + (j - 1) * (w + 0.02)
        vv = np.nan_to_num(vals, nan=0.0)
        ax.bar(xs, vv, width=w, color=col, edgecolor="white", linewidth=1.5, label=lab, zorder=3)
        for xi_, v in zip(xs, vals):
            ax.text(xi_, (0 if np.isnan(v) else v) + 0.015 * top, fmt(v), ha="center", va="bottom",
                    fontsize=11, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in rows], fontsize=13)
    ax.set_ylim(0, top * 1.1)
    ax.set_ylabel("Value (0 to 1 scale)", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bbbbbb")
    fig.legend(handles=[Patch(facecolor=c, label=l) for l, _, c in series], loc="lower center",
               ncol=1, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(os.path.join(out, "summary_bars.png"), dpi=200)
    plt.close(fig)


def print_summary(res):
    conds = res["conditions"]
    peak = int(res["peak_layer"])
    print(f"Model: {res.get('model')}  n_resumes: {res.get('n_eval_resumes')}  "
          f"n_items: {res.get('n_eval_items')}  peak layer: {peak}")
    print(f"{'condition':<24}{'disparity':>10}{'retention':>10}{'bal.acc':>9}"
          f"{'max diff':>10}{'diff@peak':>10}{'probe@peak':>11}")
    for k, lab in CONDS:
        if k not in conds:
            continue
        c = conds[k]
        ret = {"base": 0.0, "biased": 1.0}.get(k, num(c, "retention"))
        d, p = vec(c, "activation_differential"), vec(c, "probe_acc_linear")
        mx = np.nanmax(d) if np.isfinite(d).any() else np.nan
        print(f"{lab:<24}{fmt(num(c, 'demographic_disparity'), 3):>10}{fmt(ret, 3):>10}"
              f"{fmt(num(c, 'balanced_accuracy'), 3):>9}{fmt(mx, 3):>10}"
              f"{fmt(d[peak] if peak < N_LAYERS else np.nan, 3):>10}"
              f"{fmt(p[peak] if peak < N_LAYERS else np.nan, 3):>11}")
    it = conds.get("fix_iterative", {})
    if it.get("rounds"):
        rr = ", ".join(f"r{r['round']}={r['max_late_differential']:.3f}" for r in it["rounds"])
        print(f"Iterative rounds (max late differential): {rr}  subspace_dim={it.get('subspace_dim')}")
    print(f"Runtime: {res.get('runtime_seconds')} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()
    out = a.results_dir
    with open(os.path.join(out, "results.json")) as f:
        res = json.load(f)
    acts = None
    npz = os.path.join(out, "activations.npz")
    if os.path.exists(npz):
        acts = dict(np.load(npz, allow_pickle=False))
    else:
        print(f"WARNING: {npz} missing, skipping arrows.png")

    print_summary(res)
    jobs = [("heatmap.png", fig_heatmap), ("probe.png", fig_probe), ("summary_bars.png", fig_bars)]
    if acts is not None:
        jobs.insert(2, ("arrows.png", fig_arrows))
    for name, fn in jobs:
        try:
            fn(res, acts, out)
            print(f"wrote {os.path.join(out, name)}")
        except Exception:  # keep going so one bad figure doesn't block the rest
            print(f"FAILED {name}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
