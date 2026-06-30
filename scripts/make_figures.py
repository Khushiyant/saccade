"""Render clean black-and-white (grayscale) research figures from measured benchmark data.

Series are distinguished by line style, marker shape, and gray shade (no color), for
print/grayscale-safe publication figures. Run: uv run python scripts/make_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = "#111111"   # near-black
D = "#444444"   # dark gray
M = "#777777"   # mid gray
L = "#aaaaaa"   # light gray

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "text.color": K, "axes.labelcolor": K, "xtick.color": K, "ytick.color": K,
    "axes.edgecolor": K, "axes.linewidth": 0.8,
    "grid.color": "#d6d6d6", "grid.linestyle": "-", "grid.alpha": 0.6,
    "grid.linewidth": 0.5, "axes.grid": True, "axes.axisbelow": True,
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif", "font.size": 11,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.major.size": 4, "ytick.major.size": 4,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": True, "legend.edgecolor": K, "legend.framealpha": 1.0,
    "legend.fontsize": 9.5, "axes.titlesize": 12,
})
OUT, RES = "docs/figures", "benchmarks/results"
os.makedirs(OUT, exist_ok=True)


def _style(ax, title):
    # Classic look: full box frame, inward ticks (set globally), centered serif title.
    ax.set_title(title, color=K, fontsize=12, loc="center", pad=10)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print("wrote", f"{OUT}/{name}.png")


def fig_surprise_gate():
    d = json.load(open(f"{RES}/surprise_gate.json"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    styles = [("latent", K, "-", "o", "latent gate (JEPA front-end)"),
              ("pixel", D, (0, (5, 2)), "s", "pixel-delta gate")]
    for g, col, ls, mk, lab in styles:
        pts = sorted(d["pareto"][g], key=lambda p: p["compute_frac"])
        ax1.plot([p["compute_frac"] * 100 for p in pts], [p["fidelity"] for p in pts],
                 ls=ls, marker=mk, color=col, lw=1.7, ms=6, label=lab,
                 markerfacecolor="white", markeredgecolor=col, markeredgewidth=1.2)
    _style(ax1, "(a) compute-fidelity Pareto")
    ax1.set_xlabel("encoder compute spent (%)")
    ax1.set_ylabel("fidelity (cosine vs full)")
    ax1.set_ylim(0.94, 1.004)
    ax1.legend(loc="lower right")

    order = ["static", "mixed", "dynamic"]
    cf = [d["novelty"][k]["compute_frac"] * 100 for k in order]
    shades = [L, M, K]
    x = range(len(order))
    ax2.vlines(x, 0, cf, color=shades, lw=2.0)
    ax2.scatter(x, cf, facecolor="white", edgecolor=shades, s=130, linewidth=1.6, zorder=3)
    for i, k in enumerate(order):
        sr = d["novelty"][k]["skip_rate"] * 100
        ax2.annotate(f"{cf[i]:.0f}%  ({sr:.0f}% skipped)", (i, cf[i]),
                     textcoords="offset points", xytext=(0, 11), ha="center",
                     color=K, fontsize=9.5)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(order)
    _style(ax2, "(b) compute scales with scene novelty")
    ax2.set_ylabel("encoder compute spent (%)")
    ax2.set_ylim(0, 122)
    fig.suptitle("Surprise-gated streaming encoder (ViT-L, RTX 5070 Ti)",
                 color=K, fontsize=13, y=1.03)
    save(fig, "fig_surprise_gate")


def fig_efficiency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    pts = [("full", 1.00, 1.000), ("ToMe r=128", 1.32, 0.963),
           ("ToMe r=256", 1.82, 0.883), ("PruneVid", 2.28, 0.753)]
    ax1.plot([p[1] for p in pts], [p[2] for p in pts], ls="-", marker="o", color=K,
             lw=1.7, ms=7, markerfacecolor="white", markeredgecolor=K, markeredgewidth=1.2)
    for name, xx, yy in pts:
        ax1.annotate(name, (xx, yy), textcoords="offset points", xytext=(9, 8),
                     color=K, fontsize=9.5)
    ax1.scatter([4.53], [1.0], marker="*", s=320, facecolor="white", edgecolor=K,
                linewidth=1.3, zorder=5)
    ax1.annotate("fused attention\n(free, lossless)", (4.53, 1.0),
                 textcoords="offset points", xytext=(-16, -40), color=D, fontsize=9.5)
    _style(ax1, "(a) efficiency-fidelity trade-off")
    ax1.set_xlabel("speedup vs fp16 baseline")
    ax1.set_ylabel("fidelity (cosine vs full)")
    ax1.set_xlim(0.7, 5.1)
    ax1.set_ylim(0.70, 1.03)

    ax2.plot([1, 16], [7.8, 62.8], ls="-", marker="o", color=K, lw=1.7, ms=8,
             markerfacecolor="white", markeredgecolor=K, markeredgewidth=1.2,
             label="streaming step (per frame)")
    ax2.scatter([8], [22.8], marker="D", facecolor="white", edgecolor=D, s=80,
                linewidth=1.4, zorder=4)
    ax2.annotate("mean 22.8 ms", (8, 22.8), textcoords="offset points", xytext=(8, 8),
                 color=D, fontsize=9.5)
    ax2.axhline(188.8, color=K, lw=1.5, ls=(0, (6, 3)), label="full clip re-encode (188.8 ms)")
    ax2.annotate("8.3x cheaper per update", (1.2, 112), color=K, fontsize=10)
    _style(ax2, "(b) streaming: per-frame update vs full re-encode")
    ax2.set_xlabel("frames in KV-cache (history length)")
    ax2.set_ylabel("cost per embedding update (ms)")
    ax2.set_xlim(0, 17)
    ax2.set_ylim(0, 210)
    ax2.legend(loc="center right", fontsize=9)

    fig.suptitle("Efficiency and streaming (ViT-L, RTX 5070 Ti)", color=K, fontsize=13, y=1.02)
    save(fig, "fig_efficiency")


if __name__ == "__main__":
    fig_surprise_gate()
    fig_efficiency()
    print("done ->", OUT)
