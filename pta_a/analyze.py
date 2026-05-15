#!/usr/bin/env python3
"""
PTA-A analysis: summary table, attribution chain, paired t-tests,
W_P heatmaps, PCC convergence curves, and per-sample PCC distributions.

Reads results.json (produced by stokes.py) and writes:
  plots/summary.txt
  plots/01_pcc_convergence_seed1.png
  plots/02_wp_final_heatmap.png
  plots/03_test_pcc_distribution.png
  plots/04_attribution_chain.png
  plots/05_stokes_features_sample0.png  (if viz/A0_seed1.npz exists)
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel


HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, "results.json")
OUT  = os.path.join(HERE, "plots")
VIZ  = os.path.join(HERE, "viz")

MODES = ("A0", "A1", "A2", "A2-raw", "A3")
MODE_COLOR = {
    "A0": "#d62728", "A1": "#ff7f0e", "A2": "#2ca02c",
    "A2-raw": "#9467bd", "A3": "#1f77b4",
}
MODE_LABEL = {
    "A0": "A0 — Stokes learnable",
    "A1": "A1 — Stokes frozen",
    "A2": "A2 — 4-pol normalised",
    "A2-raw": "A2-raw — 4-pol raw",
    "A3": "A3 — 1-pol baseline",
}


def load_results():
    with open(RES) as f:
        return json.load(f)


def group_by_mode(entries):
    g = defaultdict(list)
    for e in entries:
        m = e["config"]["mode"]
        if m in MODES and e.get("completed"):
            g[m].append(e)
    for m in g:
        g[m].sort(key=lambda x: x["config"]["seed"])
    return g


def summarise(g):
    rows = []
    for m in MODES:
        if m not in g:
            continue
        pccs  = [e["test_pcc"]  for e in g[m]]
        ssims = [e["test_ssim"] for e in g[m]]
        mses  = [e["test_mse"]  for e in g[m]]
        rows.append({
            "mode": m,
            "n_seeds": len(pccs),
            "pcc_mean": float(np.mean(pccs)),
            "pcc_std":  float(np.std(pccs, ddof=1)) if len(pccs) > 1 else 0.0,
            "ssim_mean": float(np.mean(ssims)),
            "ssim_std":  float(np.std(ssims, ddof=1)) if len(ssims) > 1 else 0.0,
            "mse_mean": float(np.mean(mses)),
            "seeds":   [e["config"]["seed"] for e in g[m]],
            "pccs":    pccs,
        })
    return rows


def paired_test(rows_by_mode, mode_a, mode_b):
    if mode_a not in rows_by_mode or mode_b not in rows_by_mode:
        return None
    a = {e["config"]["seed"]: e["test_pcc"] for e in rows_by_mode[mode_a]}
    b = {e["config"]["seed"]: e["test_pcc"] for e in rows_by_mode[mode_b]}
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    av = np.array([a[s] for s in common])
    bv = np.array([b[s] for s in common])
    diffs = av - bv
    t_stat, p = ttest_rel(av, bv)
    pooled_std = np.sqrt((np.var(av, ddof=1) + np.var(bv, ddof=1)) / 2)
    return {
        "n_pairs": len(common),
        "seeds":   common,
        "delta_mean": float(np.mean(diffs)),
        "delta_std":  float(np.std(diffs, ddof=1)),
        "pooled_std": float(pooled_std),
        "t_stat":     float(t_stat),
        "p_value":    float(p),
        "two_sigma_threshold": float(2 * pooled_std),
        "passes_2sigma": bool(np.mean(diffs) >= 2 * pooled_std),
    }


def write_summary(rows, g):
    os.makedirs(OUT, exist_ok=True)
    lines = []
    push = lines.append

    push("=" * 78)
    push("PTA-A summary")
    push("=" * 78)
    push(f"{'mode':<10}{'n':<4}{'PCC mean':<12}{'PCC std':<10}"
         f"{'SSIM mean':<12}{'SSIM std':<10}{'MSE mean':<12}")
    push("-" * 78)
    for r in rows:
        push(f"{r['mode']:<10}{r['n_seeds']:<4}"
             f"{r['pcc_mean']:<12.4f}{r['pcc_std']:<10.4f}"
             f"{r['ssim_mean']:<12.4f}{r['ssim_std']:<10.4f}"
             f"{r['mse_mean']:<12.6f}")
    push("")
    push("Attribution chain (increments, paired per seed):")
    push("-" * 78)
    chain = [("A0", "A1"), ("A1", "A2"), ("A2", "A2-raw"), ("A2-raw", "A3")]
    chain_names = {
        ("A0", "A1"): "learnable refinement   (A0 - A1)",
        ("A1", "A2"): "physical prior          (A1 - A2)",
        ("A2", "A2-raw"): "normalisation         (A2 - A2-raw)",
        ("A2-raw", "A3"): "multi-pol            (A2-raw - A3)",
    }
    for hi, lo in chain:
        t = paired_test(g, hi, lo)
        name = chain_names[(hi, lo)]
        if t is None:
            push(f"  {name:<40} insufficient data")
            continue
        flag = " *" if t["passes_2sigma"] else "  "
        push(f"  {name:<40} delta={t['delta_mean']:+.4f}  "
             f"2sigma={t['two_sigma_threshold']:.4f}  "
             f"p={t['p_value']:.3f}  n={t['n_pairs']}{flag}")
    push("")
    push("Main decision (A0 vs A2):")
    main = paired_test(g, "A0", "A2")
    if main:
        push(f"  delta_mean = {main['delta_mean']:+.4f}")
        push(f"  pooled std = {main['pooled_std']:.4f}")
        push(f"  2-sigma threshold = {main['two_sigma_threshold']:.4f}")
        push(f"  paired t  = {main['t_stat']:.3f}   p = {main['p_value']:.4f}")
        verdict = "PASS (2-sigma)" if main["passes_2sigma"] else "FAIL (2-sigma)"
        push(f"  VERDICT: {verdict}   (p<0.10 also required for publication)")
    push("")

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(txt + "\n")


def plot_pcc_convergence(g):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m in MODES:
        if m not in g:
            continue
        e = next((x for x in g[m] if x["config"]["seed"] == 1), g[m][0])
        epochs = np.arange(1, len(e["val_pcc_curve"]) + 1)
        axes[0].plot(epochs, e["val_pcc_curve"],
                     color=MODE_COLOR[m], label=MODE_LABEL[m], linewidth=1.5)
        axes[1].plot(epochs, e["train_pcc_curve"],
                     color=MODE_COLOR[m], label=MODE_LABEL[m], linewidth=1.5)
    for ax, title in zip(axes, ["Val PCC", "Train PCC (200-sample subset)"]):
        ax.set_xlabel("epoch")
        ax.set_ylabel("PCC")
        ax.set_title(title + " — seed=1")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUT, "01_pcc_convergence_seed1.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  wrote {out}")


def plot_wp_heatmaps(g):
    # Use seed=1 A0 (learnable) and A1 (frozen) for final W_P
    row_labels = ["S0", "S1", "S2", "S_res"]
    col_labels = ["I0°", "I45°", "I90°", "I135°"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    a0 = next((e for e in g.get("A0", []) if e["config"]["seed"] == 1), None)
    theo = None
    learned = None
    if a0 and a0.get("wp_trajectory"):
        theo = np.array(a0["wp_theoretical"])
        learned = np.array(a0["wp_trajectory"][-1])
        diff = learned - theo
    else:
        diff = None

    def imshow_matrix(ax, M, title, cmap="RdBu_r", vmin=None, vmax=None):
        if M is None:
            ax.set_axis_off()
            ax.set_title(f"{title}\n(missing)")
            return
        v = max(abs(M.min()), abs(M.max())) if vmin is None else vmax
        im = ax.imshow(M, cmap=cmap, vmin=-v if vmin is None else vmin, vmax=v)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="black", fontsize=9)
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(col_labels)
        ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(row_labels)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    imshow_matrix(axes[0], theo,    "Theoretical Stokes (A0 init)")
    imshow_matrix(axes[1], learned, "Learned W_P (A0 final, seed=1)")
    imshow_matrix(axes[2], diff,    "Delta = learned − theoretical")
    plt.tight_layout()
    out = os.path.join(OUT, "02_wp_final_heatmap.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  wrote {out}")


def plot_test_pcc_distribution(g):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # ECDF
    for m in MODES:
        if m not in g:
            continue
        all_pccs = []
        for e in g[m]:
            ps = e.get("per_sample_test_pcc", [])
            all_pccs.extend(ps)
        if not all_pccs:
            continue
        x = np.sort(all_pccs)
        y = np.arange(1, len(x) + 1) / len(x)
        axes[0].plot(x, y, color=MODE_COLOR[m], label=MODE_LABEL[m], linewidth=1.5)
    axes[0].set_xlabel("per-sample test PCC")
    axes[0].set_ylabel("ECDF")
    axes[0].set_title("ECDF of per-sample test PCC (all seeds pooled)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)

    # Violin
    data, labels, colors = [], [], []
    for m in MODES:
        if m not in g:
            continue
        all_pccs = []
        for e in g[m]:
            all_pccs.extend(e.get("per_sample_test_pcc", []))
        if all_pccs:
            data.append(all_pccs)
            labels.append(m)
            colors.append(MODE_COLOR[m])
    if data:
        vp = axes[1].violinplot(data, showmeans=True, showmedians=False)
        for body, c in zip(vp["bodies"], colors):
            body.set_facecolor(c); body.set_alpha(0.6)
        axes[1].set_xticks(range(1, len(labels) + 1))
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("per-sample test PCC")
        axes[1].set_title("Per-sample PCC distribution")
        axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(OUT, "03_test_pcc_distribution.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  wrote {out}")


def plot_attribution_chain(rows):
    # Bar chart: PCC mean ± std per mode, ordered along the attribution chain
    order = ["A3", "A2-raw", "A2", "A1", "A0"]
    means, stds, labels = [], [], []
    for m in order:
        r = next((r for r in rows if r["mode"] == m), None)
        if r is None:
            continue
        means.append(r["pcc_mean"])
        stds.append(r["pcc_std"])
        labels.append(m)
    if not means:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=[MODE_COLOR[m] for m in labels], alpha=0.85)
    for b, v, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, v + s + 0.001,
                f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("test PCC (mean ± std over seeds)")
    ax.set_title("Attribution chain — incremental contributions")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(min(means) - 0.02, max(means) + 0.02)
    plt.tight_layout()
    out = os.path.join(OUT, "04_attribution_chain.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  wrote {out}")


def plot_stokes_features():
    # If A0 viz file exists, render input vs output_init vs output_final for one sample
    src = os.path.join(VIZ, "A0_seed1.npz")
    if not os.path.isfile(src):
        print(f"  skip Stokes feature plot (no {src})")
        return
    z = np.load(src)
    inp  = z["input"]          # (5, 4, 256, 256)
    o_i  = z["output_init"]    # (5, 4, 256, 256)
    o_f  = z["output_final"]   # (5, 4, 256, 256)
    idxs = z["indices"]
    # Plot for sample 0 only (idxs[0])
    s = 0
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    row_titles = ["Input (calibrated 4-pol)", "Stokes output @ init", "Stokes output @ final"]
    col_titles = ["ch0 (S0/0°)", "ch1 (S1/45°)", "ch2 (S2/90°)", "ch3 (Sres/135°)"]
    for r, M in enumerate([inp[s], o_i[s], o_f[s]]):
        for c in range(4):
            ax = axes[r, c]
            v = max(abs(M[c].min()), abs(M[c].max()))
            ax.imshow(M[c], cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=10)
    fig.suptitle(f"A0 Stokes feature snapshot — sample idx={idxs[s]}", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT, "05_stokes_features_sample0.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  wrote {out}")


def main():
    if not os.path.isfile(RES):
        raise FileNotFoundError(f"{RES} missing — no results to analyse yet.")
    os.makedirs(OUT, exist_ok=True)
    entries = load_results()
    g = group_by_mode(entries)
    rows = summarise(g)
    write_summary(rows, g)
    plot_pcc_convergence(g)
    plot_wp_heatmaps(g)
    plot_test_pcc_distribution(g)
    plot_attribution_chain(rows)
    plot_stokes_features()


if __name__ == "__main__":
    main()
