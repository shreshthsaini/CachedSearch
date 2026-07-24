"""Figure 1 (paper/figs/fig1_overview.pdf): page-1 TEASER, one mini-panel per
contribution (FST/ScaleRL teaser grammar; design-overhaul Lane A).

(a) RANKING SURVIVES: cached-vs-full ImageReward scatter on the gate grid
    (Wan2.1-1.3B, tau=0.10; code/data/results/b1_gate_v0). Strategy-blue alpha
    points + identity line; ONE example prompt's 8 candidates highlighted with
    its per-prompt rho. Corner: median rho = 0.905 (holds on 3 suites: the
    tau=0.05 and tau=0.10 gate grids and the official 946-prompt VBench suite).
(b) GAIN-vs-COST: the four strategies (single / best-of-N full / keep / commit)
    at N=8, reward gain vs wall-clock. Callout brackets best-of-8 -> commit
    (94.7% of gain at 63% cost); star = the commit operating point; iso-cost
    dashed vertical at the best-of-4-full / N=8-cached explore budget shows
    cached exploration buys +38% gain at matched budget.
(c) GENERALITY: gain-capture at each model's calibrated tau* across six models /
    four architecture families; dashed line = the 85% calibration band.

EVERY number is measured; NONE invented:
  (a) rho re-derived here from b1_gate_v0 (median asserted == 0.905).
  (b) gains/costs + 95% prompt-bootstrap CIs read from paper/ci_numbers.json
      (groups main_N=4, main_N=8, main_costs); cross-checked against the
      hardcoded reference below (loud warning on divergence).
  (c) capture-at-tau* copied from paper/sections/x_taucal.tex Table tab:taucal
      (its right block), which is itself produced by taucal_analysis.py.
Style: paper_figs/style.py (Lane 0 palette + grammar helpers).
"""
import collections
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import (apply_style, BASE, C_BLUE, C_STRAT, C_YELLOW, INK, MODEL_COLORS,
                   MODEL_SHORT, MUTED, callout, star_point)

apply_style()

RESULTS = os.environ.get("CACHEDSEARCH_RESULTS", "./results")
OUT = os.environ.get("CACHEDSEARCH_FIGURES", "./assets")
V0 = os.path.join(RESULTS, "b1_gate_v0")
CI_JSON = os.environ.get(
    "CACHEDSEARCH_CI_JSON", os.path.join(RESULTS, "ci_numbers.json")
)

# example prompt for panel (a): chosen programmatically as the median case --
# among wide-spread prompts (top quartile), the one whose per-prompt rho is
# closest to the 0.905 median (so the highlighted cloud IS the headline number).
def pick_exemplar(pa):
    rows = [(p, fu.std(), spearmanr(fu, ca)[0]) for p, (fu, ca) in pa.items()]
    hi = np.percentile([s for _, s, _ in rows], 75)
    wide = [r for r in rows if r[1] >= hi]
    return min(wide, key=lambda r: abs(r[2] - 0.905))[0]

# Panel (b) reference points, cross-checked against the generated CI file.
PTS = {  # strategy -> (cost s, reward gain over single)
    "single": (68.33, 0.0),
    "bon4_full": (273.34, 0.5139),
    "bon_full": (546.68, 0.7517),
    "cached_keep": (277.83, 0.7491),
    "cached_commit": (346.17, 0.7122),
}
CI_KEYS = {  # strategy -> (group, gain key)
    "bon4_full": ("main_N=4", "gain_bon"),
    "bon_full": ("main_N=8", "gain_bon"),
    "cached_keep": ("main_N=8", "gain_keep"),
    "cached_commit": ("main_N=8", "gain_commit"),
}

# panel (c): gain-capture (%) at each model's calibrated tau* -- the right block
# of Table tab:taucal (paper/sections/x_taucal.tex), tau* rule = most aggressive
# measured tau with capture >= 85%; LTX = honest boundary (no tau qualifies).
CAPTURE_AT_TAUSTAR = {  # canonical model name -> (capture %, tau* label)
    "Wan2.1-1.3B": (88.3, r"$\tau^*{=}.20$"),
    "Wan2.1-14B": (87.5, r"$\tau^*{=}.10$"),
    "Wan2.2-5B": (86.0, r"$\tau^*{=}.10$"),
    "CogVideoX-5B": (85.9, r"$\tau^*{=}.05$"),
    "HunyuanVideo-13B": (85.1, r"$\tau^*{=}.05$"),
    "LTX-Video-2B": (79.6, r"$\tau{=}.02$"),
}


# --------------------------------------------------------------- data
def load_v0():
    """-> {prompt: (full[8], cached[8])} from the repo copy of the gate grid."""
    by = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(V0, "scores_shard*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r["seed"] >= 8:  # N=16 width-extension shards share the dir
                continue
            by[r["prompt"]][(r["seed"], r["variant"])] = r["score"]
    pa = {}
    for p, d in by.items():
        seeds = sorted({s for s, v in d if (s, "full") in d and (s, "cached") in d})
        if len(seeds) < 8:
            continue
        pa[p] = (np.array([d[(s, "full")] for s in seeds[:8]]),
                 np.array([d[(s, "cached")] for s in seeds[:8]]))
    return pa


def load_cis():
    if not os.path.exists(CI_JSON):
        print("NOTE: ci_numbers.json missing -- panel (b) without error bars")
        return {}
    d = json.load(open(CI_JSON))
    cis = {}
    for strat, (grp, gkey) in CI_KEYS.items():
        g = d[grp][gkey]
        if abs(g["point"] - PTS[strat][1]) > 0.005:
            print(f"  <<< WARNING: fig1 {strat} gain {PTS[strat][1]} vs json "
                  f"{g['point']:.4f} -- regenerate one")
        cis[strat] = (g["lo"], g["hi"])
    return cis


# --------------------------------------------------------------- panels
def panel_a(ax, pa):
    exemplar = pick_exemplar(pa)
    lo, hi = -2.3, 2.3
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), lw=0.7, color=BASE, zorder=1)
    # all prompts, background
    for prompt, (fu, ca) in pa.items():
        if prompt == exemplar:
            continue
        ax.scatter(fu, ca, s=6, color=C_BLUE, alpha=0.28, linewidths=0, zorder=2)
    # one exemplar prompt highlighted
    fu, ca = pa[exemplar]
    rho = spearmanr(fu, ca)[0]
    ax.scatter(fu, ca, s=22, facecolors=C_YELLOW, edgecolors="white", linewidths=0.5,
               zorder=4)
    # rho label anchored to the exemplar cloud
    cx, cy = fu.mean(), ca.min()
    callout(ax, rf"one prompt's 8 candidates" + "\n" + rf"$\rho = {rho:.3f}$",
            xy=(fu[np.argmin(ca)], ca.min()), xytext=(1.25, -1.72),
            color=INK, arrow_color=C_YELLOW, fontsize=6.0, ha="center", va="top")
    med = float(np.median([spearmanr(f, c)[0] for f, c in pa.values()]))
    assert abs(med - 0.905) < 1e-2, f"median rho {med} != 0.905"
    ax.text(0.04, 0.97, r"median $\rho = 0.905$" + "\n" + r"(3 suites, $n{=}50/946$)",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.6,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BASE, lw=0.5, alpha=0.92))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("full-compute verifier score", labelpad=1)
    ax.set_ylabel(r"cached score ($\tau{=}0.10$)", labelpad=1)
    ax.set_title("(a) caching preserves ranking", fontsize=7.4, loc="left", pad=3,
                 fontweight="bold")


def panel_b(ax, cis):
    order = [("single", "single", C_STRAT["single"]),
             ("bon_full", "best-of-8 full", C_STRAT["bon_full"]),
             ("cached_keep", "keep-draft", C_STRAT["cached_keep"]),
             ("cached_commit", "commit (ours)", C_STRAT["cached_commit"])]
    # best-of-4-full reference (needed for the iso-cost read)
    x4, y4 = PTS["bon4_full"]
    ax.scatter([x4], [y4], s=16, facecolors="white", edgecolors=C_STRAT["bon_full"],
               linewidths=0.9, zorder=3)
    ax.annotate("best-of-4 full", (x4, y4), xytext=(-4, -9), textcoords="offset points",
                fontsize=5.8, color=C_STRAT["bon_full"], ha="right", va="top")
    for key, lab, c in order:
        x, y = PTS[key]
        if key == "single":
            ax.scatter([x], [y], marker="*", s=55, facecolors="white",
                       edgecolors=c, linewidths=0.9, zorder=4)
            ax.annotate(lab, (x, y), xytext=(6, 2), textcoords="offset points",
                        fontsize=6.0, color=MUTED, ha="left", va="bottom")
            continue
        ax.scatter([x], [y], s=26, facecolors="white", edgecolors=c, linewidths=1.1,
                   zorder=4)
        # CI whiskers intentionally omitted here (busy panel; CIs in tab:main).
        del c
    # direct labels for best-of-8 and keep
    ax.annotate("best-of-8 full", PTS["bon_full"], xytext=(2, 7),
                textcoords="offset points", fontsize=5.8, color=C_STRAT["bon_full"],
                ha="center", va="bottom")
    ax.annotate("keep-draft", PTS["cached_keep"], xytext=(-3, 7),
                textcoords="offset points", fontsize=5.8, color=C_STRAT["cached_keep"],
                ha="right", va="bottom")
    # star the commit operating point (landmark) + headline callout
    xc, yc = PTS["cached_commit"]
    star_point(ax, xc, yc, C_STRAT["cached_commit"], size=90, zorder=6)
    callout(ax, "94.7% of the gain" + "\n" + "at 63% of the cost",
            xy=(xc, yc), xytext=(455, 0.40), color=INK,
            arrow_color=C_STRAT["cached_commit"], fontsize=6.2, ha="center", va="top")
    # iso-cost: vertical dashed at the shared explore budget (best-of-4 full ~=
    # N=8 cached exploration); +38% after commit vs best-of-4 full.
    xk = PTS["cached_keep"][0]
    xiso = 0.5 * (x4 + xk)
    ax.axvline(xiso, color=BASE, lw=0.7, ls=(0, (3, 2.5)), zorder=1)
    ax.annotate("", xy=(xc, yc), xytext=(x4, y4),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=INK,
                                shrinkA=3, shrinkB=6), zorder=5)
    ax.annotate("+38%", (0.5 * (x4 + xc), 0.5 * (y4 + yc)), xytext=(-11, -2),
                textcoords="offset points", fontsize=6.6, color=INK, ha="right",
                va="center", fontweight="bold")
    ax.text(xiso, -0.055, "equal explore budget", fontsize=5.4, color=MUTED,
            ha="center", va="bottom")
    ax.set_xlim(0, 600); ax.set_ylim(-0.07, 0.95)
    ax.set_xlabel("wall-clock per prompt (s)", labelpad=1)
    ax.set_ylabel("reward gain over single", labelpad=1)
    ax.set_title("(b) more gain per second", fontsize=7.4, loc="left", pad=3,
                 fontweight="bold")


def panel_c(ax):
    items = sorted(CAPTURE_AT_TAUSTAR.items(), key=lambda kv: kv[1][0])  # asc -> lowest at bottom
    names = [MODEL_SHORT[k] for k, _ in items]
    caps = [v[0] for _, v in items]
    taus = [v[1] for _, v in items]
    colors = [MODEL_COLORS[k] for k, _ in items]
    y = np.arange(len(items))
    ax.axvline(85, color=MUTED, lw=0.8, ls=(0, (3, 2.5)), zorder=1)
    ax.text(85, len(items) - 0.35, "85% band", fontsize=5.6, color=MUTED,
            ha="center", va="bottom")
    bars = ax.barh(y, caps, height=0.62, color=colors, zorder=3, edgecolor="white",
                   linewidth=0.4)
    for yi, (cap, tl) in enumerate(zip(caps, taus)):
        ax.annotate(f"{cap:.1f}%", (cap, yi), xytext=(3, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=6.2,
                    color=INK)
        ax.annotate(tl, (70.6, yi), xytext=(0, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=5.4, color="white", zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.6)
    ax.tick_params(axis="y", length=0, pad=8)  # ~1.5ex gap for later logos
    ax.set_xlim(70, 93)
    ax.set_ylim(-0.6, len(items) - 0.05)
    ax.set_xlabel(r"gain capture at $\tau^*$ (%)", labelpad=1)
    ax.grid(axis="y", visible=False)
    ax.set_title("(c) across 6 models, 4 families", fontsize=7.4, loc="left", pad=3,
                 fontweight="bold")


def main():
    plt.rcParams.update({"text.usetex": False})
    fig = plt.figure(figsize=(6.4, 2.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.06, 1.0], wspace=0.42,
                          left=0.075, right=0.985, bottom=0.185, top=0.90)
    panel_a(fig.add_subplot(gs[0]), load_v0())
    panel_b(fig.add_subplot(gs[1]), load_cis())
    panel_c(fig.add_subplot(gs[2]))
    out = os.path.join(OUT, "fig1_overview.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
