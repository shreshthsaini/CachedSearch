"""Generate all main-paper figures (F1-F7) for the CachedSearch paper from RAW gate jsonl data.

Data: ./results/b1_gate_{v0,tau005,tau020,vbench}/scores_shard*.jsonl
  v0     : tau=0.10, 50 prompts x 8 seeds x {full, cached}
  tau005 : tau=0.05, 50 prompts x 8 seeds, CACHED ONLY (full reference shared from v0 --
           full rollouts are tau-independent and seed-deterministic)
  tau020 : tau=0.20, same layout as tau005
  vbench : E2 official VBench-946 suite, tau=0.10, full+cached, LIVE (coverage growing);
           cross-partition shards contain duplicate records -> dedupe by (prompt,seed,variant)
F7 additionally uses the E5 adaptive-tau simulation (code/experiments/b1_adaptive_tau.py).

Every number is recomputed here and cross-checked against the expected values
below. Mismatches beyond tolerance print a loud warning but still plot.

Uncertainty (P0-1): every aggregate mark carries a 95% prompt-level bootstrap CI
(percentile, B=10^4; override via CACHEDSEARCH_BOOT_B), matching the convention of
code/paper_figs/bootstrap_ci.py (paper/ci_numbers.json). Marker grammar (P0-5):
filled = directly measured; open = simulated over seed subsets of the measured
grid; dashed = model prediction/fit.

Usage:  python make_figs.py [--out /path/to/paper/figs]
"""
import argparse
import glob
import itertools
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))

import numpy as np
from scipy.stats import spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# canonical rcParams + palette + grammar helpers: paper_figs/style.py (Lane 0)
from style import (apply_style, BASE, C_BLUE, C_GREEN, C_RED, C_STRAT, C_TAU,
                   C_VIOLET, C_YELLOW, ERR_KW, INK, MUTED,
                   callout, direct_label, star_point)

apply_style()

RESULTS = os.environ.get("CACHEDSEARCH_RESULTS", "./results")
OUT_DEFAULT = os.environ.get("CACHEDSEARCH_FIGURES", "./assets")
TAGS = {"tau005": 0.05, "v0": 0.10, "tau020": 0.20}

# Prompt-level bootstrap (P0-1): percentile 95% CIs, B=10^4 -- the same
# convention as code/paper_figs/bootstrap_ci.py (tables/prose CIs), stated once
# in the paper's Setup "Figure conventions" sentence. Each call site uses a
# fresh seeded rng so any figure is reproducible in isolation.
B_BOOT = int(os.environ.get("CACHEDSEARCH_BOOT_B", "10000"))


def boot_idx(n, B=None, seed=0):
    """(B, n) prompt-resampling index matrix."""
    return np.random.default_rng(seed).integers(0, n, size=(B or B_BOOT, n))


def ci95(samples):
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def yerr_of(points, cis):
    """Asymmetric (2, k) matplotlib yerr from point estimates + (lo, hi) CIs."""
    pts = np.asarray(points, float)
    lo = np.array([c[0] for c in cis])
    hi = np.array([c[1] for c in cis])
    return np.vstack([np.maximum(pts - lo, 0), np.maximum(hi - pts, 0)])


def boot_speedup(fg, cg, B=None, seed=0):
    """95% CI of Cf/Cc from per-prompt latency groups {prompt: [latencies]},
    paired on the shared prompt set (bootstrap_ci.py convention)."""
    common = sorted(set(fg) & set(cg))
    fs = np.array([np.sum(fg[p]) for p in common])
    fn = np.array([len(fg[p]) for p in common], float)
    cs = np.array([np.sum(cg[p]) for p in common])
    cn = np.array([len(cg[p]) for p in common], float)
    idx = boot_idx(len(common), B, seed)
    s = (fs[idx].sum(1) / fn[idx].sum(1)) / (cs[idx].sum(1) / cn[idx].sum(1))
    return ci95(s)

# ------------------------------------------------------- expected numbers
# Expected values from the measured release analysis.
EXPECTED = {
    "v0": dict(med_rho=0.905, mean_rho=0.820, p10_rho=0.614, top1=0.64, speedup=1.97,
               regret=0.040, rand_regret=0.657, capture_mor=0.901),
    "tau005": dict(med_rho=0.905, p10_rho=0.738, top1=0.70, speedup=1.58,
                   regret=0.025, capture_mor=0.936),
    "tau020": dict(med_rho=0.857, p10_rho=0.474, top1=0.52, speedup=2.41,
                   regret=0.057, capture_mor=0.883),
    "sim_v0": {  # (N, strategy) -> (gain vs single, cost s)
        (2, "bon_full"): (0.304, 137), (2, "cached_commit"): (0.274, 138), (2, "cached_keep"): (0.286, 69),
        (4, "bon_full"): (0.514, 273), (4, "cached_commit"): (0.461, 207), (4, "cached_keep"): (0.496, 139),
        (8, "bon_full"): (0.752, 547), (8, "cached_commit"): (0.712, 346), (8, "cached_keep"): (0.749, 278),
    },
    "corr_spread_rho": 0.25, "n_rho_lt07": 9,
    # E2 official VBench-946, FINAL coverage (n=944 fully covered prompts, seeds 0-7,
    # re-derived 2026-07-07 after the run completed):
    "vbench": dict(med_rho=0.905, mean_rho=0.859, p10_rho=0.690, top1=0.718, speedup=1.95,
                   regret=0.0562, rand_regret=0.790, capture_mor=0.902),
}


def check(name, got, want, tol=0.011):
    flag = "" if abs(got - want) <= tol else "  <<< WARNING: MISMATCH"
    print(f"  check {name:<28} got {got:8.3f}  expected {want:8.3f}{flag}")


# ------------------------------------------------------------- data layer
def load_tag(tag):
    """-> per-(prompt,seed,variant) scores, per-variant mean latency, and
    per-variant per-PROMPT latency groups (for the paired speedup bootstrap)."""
    by, latg = defaultdict(dict), defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(RESULTS, f"b1_gate_{tag}", "scores_shard*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r["seed"] >= 8:  # N=16 width-extension shards share these dirs;
                continue        # the paper grid (scores AND latencies) is seeds 0-7
            by[r["prompt"]][(r["seed"], r["variant"])] = r["score"]
            latg[r["variant"]][r["prompt"]].append(r["latency"])
    lat = {v: float(np.mean([x for xs in g.values() for x in xs])) for v, g in latg.items()}
    return by, lat, {v: dict(g) for v, g in latg.items()}


def load_all():
    """-> {tau: {prompt: (full[8], cached[8])}}, {tau: (Cf, Cc)},
          {tau: (full latency groups, cached latency groups)}"""
    by_v0, lat_v0, latg_v0 = load_tag("v0")
    data, cost, latg = {}, {}, {}
    for tag, tau in TAGS.items():
        by, lat, lg = load_tag(tag)
        if "full" not in lat:  # tau runs store cached only; merge full from v0 on (prompt, seed)
            for prompt, d in by_v0.items():
                for (s, v), sc in d.items():
                    if v == "full":
                        by[prompt][(s, "full")] = sc
            lat["full"] = lat_v0["full"]
            lg["full"] = latg_v0["full"]
        pa = {}
        for prompt, d in by.items():
            seeds = sorted({s for s, v in d if (s, "full") in d and (s, "cached") in d})
            if len(seeds) < 8:
                continue
            # seeds[:8]: the N=16 width-extension shards (seeds 8-15) land in the
            # same results dirs; the paper's gate grid is defined on seeds 0-7.
            pa[prompt] = (np.array([d[(s, "full")] for s in seeds[:8]]),
                          np.array([d[(s, "cached")] for s in seeds[:8]]))
        data[tau], cost[tau] = pa, (lat["full"], lat["cached"])
        latg[tau] = (lg["full"], lg["cached"])
    return data, cost, latg


def load_vbench():
    """E2 official-suite run (tau=0.10). Cross-partition shards duplicate records --
    dedupe by (prompt, seed, variant), any copy. Keep prompts fully covered
    (8 seeds x both variants). -> ({prompt: (full[8], cached[8])}, (Cf, Cc)) or (None, None)."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "b1_gate_vbench", "scores_shard*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r["seed"] >= 8:  # width-extension shards; official protocol is seeds 0-7
                continue
            rows[(r["prompt"], r["seed"], r["variant"])] = r
    if not rows:
        return None, None
    by, lat = defaultdict(dict), defaultdict(list)
    for (p, s, v), r in rows.items():
        by[p][(s, v)] = r["score"]
        lat[v].append(r["latency"])
    pa = {}
    for prompt, d in by.items():
        seeds = sorted({s for s, v in d if (s, "full") in d and (s, "cached") in d})
        if len(seeds) < 8:
            continue
        pa[prompt] = (np.array([d[(s, "full")] for s in seeds[:8]]),
                      np.array([d[(s, "cached")] for s in seeds[:8]]))
    return pa, (float(np.mean(lat["full"])), float(np.mean(lat["cached"])))


def per_prompt(pa):
    """-> arrays: rho, top1, spread(full std), regret(full best - full at cached argmax), rand_regret"""
    rho, top1, spread, regret, rand = [], [], [], [], []
    for fu, ca in pa.values():
        r, _ = spearmanr(fu, ca)
        rho.append(r)
        top1.append(int(np.argmax(fu) == np.argmax(ca)))
        spread.append(fu.std())
        regret.append(fu.max() - fu[int(np.argmax(ca))])
        rand.append(fu.max() - fu.mean())
    return {k: np.array(v) for k, v in
            zip(["rho", "top1", "spread", "regret", "rand"], [rho, top1, spread, regret, rand])}


def simulate(pa, Cf, Cc, Ns=(2, 4, 8)):
    """b1_simulate.py reimplementation: strategies over all C(8,N) seed subsets.
    Value = FULL score of pick for single/bon_full/cached_commit (commit re-generates the
    winning seed at full compute, deterministically); CACHED score for cached_keep."""
    out = {}
    for N in Ns:
        vals = defaultdict(list)
        for fu, ca in pa.values():
            for subset in itertools.combinations(range(len(fu)), N):
                f, c = fu[list(subset)], ca[list(subset)]
                vals["single"].append(f[0])
                vals["bon_full"].append(f.max())
                vals["cached_keep"].append(c[int(np.argmax(c))])
                vals["cached_commit"].append(f[int(np.argmax(c))])
        costs = {"single": Cf, "bon_full": N * Cf, "cached_keep": N * Cc, "cached_commit": N * Cc + Cf}
        for k, v in vals.items():
            out[(N, k)] = (float(np.mean(v)), costs[k])
    return out


def simulate_pp(pa, Ns=(2, 4, 8)):
    """Per-PROMPT subset-mean strategy values (same semantics as simulate();
    every prompt has the same seed count, so the mean over prompts of these
    per-prompt means equals simulate()'s pooled mean exactly). Used for the
    prompt-level bootstrap on figure error bars.
    -> {(N, strat): array over sorted(pa)}"""
    prompts = sorted(pa)
    strats = ("single", "bon_full", "cached_keep", "cached_commit")
    out = {(N, k): np.zeros(len(prompts)) for N in Ns for k in strats}
    for i, p in enumerate(prompts):
        fu, ca = pa[p]
        for N in Ns:
            idx = np.array(list(itertools.combinations(range(len(fu)), N)), dtype=int)
            F, C = fu[idx], ca[idx]
            pick = C.argmax(axis=1)
            rows = np.arange(len(idx))
            out[(N, "single")][i] = F[:, 0].mean()
            out[(N, "bon_full")][i] = F.max(axis=1).mean()
            out[(N, "cached_keep")][i] = C[rows, pick].mean()
            out[(N, "cached_commit")][i] = F[rows, pick].mean()
    return out


# ------------------------------------------------------------- figures
def f1_scatter(data, out):
    """Cached-vs-full candidate scatter, recolored by per-prompt candidate
    spread (design-overhaul Lane B: a sequential house-blue luminance ramp +
    small colorbar replaces the 8-hue rainbow; spread is the sufficient
    statistic for ranking preservation, foreshadowing the corruption mechanism
    of the regret analysis)."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    pa = data[0.10]
    pp = per_prompt(pa)
    fig, ax = plt.subplots(figsize=(3.05, 2.55))
    lo, hi = -2.3, 2.3
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), lw=0.7, color=BASE, zorder=1)
    prompts = sorted(pa)
    spreads = np.array([pa[p][0].std() for p in prompts])
    cmap = LinearSegmentedColormap.from_list(
        "house_blue", ["#d8e6f7", "#6aa3e0", C_BLUE, "#123a72"])
    norm = Normalize(vmin=float(spreads.min()), vmax=float(spreads.max()))
    for p, sp in zip(prompts, spreads):
        fu, ca = pa[p]
        ax.scatter(fu, ca, s=8, color=[cmap(norm(sp))], alpha=0.85, linewidths=0, zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("full-compute score")
    ax.set_ylabel(r"cached score ($\tau=0.10$)")
    ax.text(0.03, 0.97, rf"median per-prompt $\rho = {np.median(pp['rho']):.3f}$"
            + "\n" + rf"mean $\rho = {pp['rho'].mean():.3f}$",
            transform=ax.transAxes, va="top", fontsize=7)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("candidate spread", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.8, width=0.6)
    cb.outline.set_linewidth(0.5)
    fig.savefig(os.path.join(out, "f1_scatter_rank.pdf"))
    plt.close(fig)


def f2_rho_hist(data, out, vb=None):
    panels = [(data[t], C_TAU[t], rf"$\tau={t:.2f}$") for t in (0.05, 0.10, 0.20)]
    if vb is not None:
        panels.append((vb, C_TAU[0.10], rf"VBench ($n={len(vb)}$)"))
    n = len(panels)
    if n == 4:                       # compact 2x2 (was a tall 4x1 stack)
        fig, axg = plt.subplots(2, 2, figsize=(2.95, 2.5), sharex=True)
        axes = axg.flatten()
    else:
        fig, axes = plt.subplots(n, 1, figsize=(2.7, 0.97 * n), sharex=True)
        axes = np.atleast_1d(axes); axg = None
    bins = np.linspace(-0.4, 1.0, 15)
    for ax, (pa, c, lab) in zip(axes, panels):
        pp = per_prompt(pa)
        ax.hist(pp["rho"], bins=bins, color=c, alpha=0.85, edgecolor="white", linewidth=0.4)
        p10, med = np.percentile(pp["rho"], 10), np.median(pp["rho"])
        ax.axvline(p10, color=C_RED, lw=0.9, ls=(0, (3, 2)))
        ax.axvline(med, color=INK, lw=0.9)
        ax.text(0.04, 0.85, lab, transform=ax.transAxes, fontsize=6.6)
        ax.text(0.04, 0.52, rf"med {med:.2f}""\n"rf"p10 {p10:.2f}", transform=ax.transAxes,
                fontsize=5.8, color=MUTED, va="top")
    if axg is not None:
        for ax in axg[:, 0]:
            ax.set_ylabel("prompts", fontsize=6.6)
        for ax in axg[-1, :]:
            ax.set_xlabel(r"per-prompt $\rho$", fontsize=6.5)
    else:
        for ax in axes:
            ax.set_ylabel("prompts", fontsize=7)
        axes[-1].set_xlabel(r"per-prompt Spearman $\rho$ (cached vs. full)")
    # legend inside the first panel (its upper-left is empty: mass sits near
    # rho=1), so the figure adds no height beyond the panels themselves.
    axes[0].legend(handles=[Line2D([], [], color=INK, lw=0.9, label="median"),
                            Line2D([], [], color=C_RED, lw=0.9, ls=(0, (3, 2)), label="p10")],
                   loc="lower left", bbox_to_anchor=(0.02, 0.03), frameon=False,
                   handlelength=1.3, fontsize=5.6, labelspacing=0.3,
                   borderaxespad=0.0, handletextpad=0.5)
    axes[-1].set_xlim(-0.45, 1.03)
    fig.subplots_adjust(hspace=0.2, wspace=0.24)
    fig.savefig(os.path.join(out, "f2_rho_hist.pdf"))
    plt.close(fig)


def f3_pareto(data, cost, out):
    sims = {tau: simulate(data[tau], *cost[tau]) for tau in TAGS.values()}
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    Ns = (2, 4, 8)
    # tau-independent references (computed from v0)
    sim10 = sims[0.10]
    single = {N: sim10[(N, "single")][0] for N in Ns}
    # prompt-level bootstrap CIs on the tau=0.10 gains (open markers: all points
    # are strategy means simulated over seed subsets of the measured grid)
    pp10 = simulate_pp(data[0.10], Ns)
    idx = boot_idx(len(pp10[(2, "single")]))

    def gain_ci(N, strat):
        g = pp10[(N, strat)][idx].mean(1) - pp10[(N, "single")][idx].mean(1)
        return ci95(g)

    star_point(ax, sim10[(2, "single")][1], 0, C_STRAT["single"], size=55, zorder=4)
    direct_label(ax, sim10[(2, "single")][1], 0, "single", MUTED, dx=4, dy=-8)
    bf = [(sim10[(N, "bon_full")][1], sim10[(N, "bon_full")][0] - single[N]) for N in Ns]
    ax.errorbar(*zip(*bf), yerr=yerr_of([y for _, y in bf], [gain_ci(N, "bon_full") for N in Ns]),
                fmt="-o", color=C_STRAT["bon_full"], lw=1.1, ms=3.5, mfc="white", zorder=3,
                **ERR_KW)
    for strat in ("cached_keep", "cached_commit"):
        for tau in (0.05, 0.10, 0.20):
            pts = [(sims[tau][(N, strat)][1], sims[tau][(N, strat)][0] - single[N]) for N in Ns]
            if tau == 0.10:
                ax.errorbar(*zip(*pts),
                            yerr=yerr_of([y for _, y in pts], [gain_ci(N, strat) for N in Ns]),
                            fmt="-o", color=C_STRAT[strat], lw=1.1, ms=3.5, mfc="white",
                            zorder=3, **ERR_KW)
            else:
                ax.plot(*zip(*pts), "--o", color=C_STRAT[strat], lw=0.6, ms=2.2,
                        mfc="white", alpha=0.45, zorder=2)
    for N, (x, y) in zip(Ns, bf):  # N labels once, on the bon_full curve
        ax.annotate(rf"$N{{=}}{N}$", (x, y), xytext=(5, -11), textcoords="offset points",
                    fontsize=6.5, color=MUTED)

    # E6 prune+cached: the ONE measured end-to-end stacked-search point (folds in
    # the unique bar of the retired f11/fig:methods -- Lane B consolidation).
    # Delivered end-to-end (preview+decode+scoring+recommit) vs best-of-8 gain.
    x_pc = y_pc = None
    try:
        from make_figs2 import load_stack
        stack = load_stack()
        common = [p for p in data[0.10] if p in stack]
        if common:
            sc = np.array([stack[p]["commit_score"] for p in common])
            e2e = np.array([stack[p]["timings"]["e2e_s"] for p in common])
            x_pc, y_pc = float(e2e.mean()), float(sc.mean()) - single[8]
    except Exception as exc:  # pragma: no cover
        print(f"  [f3] prune+cached point skipped: {exc}")
    bo8_g = sim10[(8, "bon_full")][0] - single[8]
    bo8_c = sim10[(8, "bon_full")][1]
    if x_pc is not None:
        ax.scatter([x_pc], [y_pc], marker="D", s=34, color=C_YELLOW,
                   edgecolors="white", linewidths=0.5, zorder=5)
        # Short in-place tag only; the stacked point's numbers live in the
        # body text (user 2026-07-23: no stat-box callouts inside plots).
        direct_label(ax, x_pc, y_pc, "prune$+$cached", C_YELLOW, dx=-8, dy=-12, ha="right")

    # direct labels (no legend)
    xb, yb = bf[-1]
    direct_label(ax, xb, yb, "best-of-$N$ full", C_STRAT["bon_full"], dx=-4, dy=9, ha="right")
    xk = sims[0.10][(8, "cached_keep")][1]
    yk = sims[0.10][(8, "cached_keep")][0] - single[8]
    direct_label(ax, xk, yk, "CachedSearch-keep", C_STRAT["cached_keep"], dx=-8, dy=10, ha="right")
    x8, y8 = sim10[(8, "cached_commit")][1], sim10[(8, "cached_commit")][0] - single[8]
    direct_label(ax, x8, y8, "CachedSearch-commit", C_STRAT["cached_commit"],
                 dx=7, dy=8)
    ax.text(0.985, 0.03, r"faded dashed: $\tau \in \{0.05, 0.20\}$",
            transform=ax.transAxes, fontsize=6.0, color=MUTED, ha="right")
    ax.set_xlabel("wall-clock cost per prompt (s)")
    ax.set_ylabel("reward gain over single")
    ax.set_xlim(0, 590); ax.set_ylim(-0.05, 1.02)
    fig.savefig(os.path.join(out, "f3_pareto.pdf"))
    plt.close(fig)
    return sims


def f4_tau_curve(data, cost, latg, out):
    taus = [0.05, 0.10, 0.20]
    capture, speedup, cap_ci, sp_ci = [], [], [], []
    for tau in taus:
        pp = per_prompt(data[tau])
        ratios = (pp["rand"] - pp["regret"]) / pp["rand"]  # per-prompt gain-capture
        ratios = 100 * ratios[np.isfinite(ratios)]
        capture.append(float(np.mean(ratios)))
        cap_ci.append(ci95(ratios[boot_idx(len(ratios))].mean(1)))
        speedup.append(cost[tau][0] / cost[tau][1])
        sp_ci.append(boot_speedup(*latg[tau]))
    # Single-axis capture-vs-speedup (design-overhaul Lane B: twin axes killed;
    # tau becomes a per-point label). The trade-off frontier is now literal:
    # moving right (more speedup) costs a little capture (down).
    fig, ax = plt.subplots(figsize=(2.7, 2.2))
    ax.errorbar(speedup, capture, yerr=yerr_of(capture, cap_ci),
                xerr=yerr_of(speedup, sp_ci), fmt="-o", color=C_BLUE, lw=1.3,
                ms=4.5, **ERR_KW)
    lab_off = {0.05: (-6, 8, "right"), 0.10: (7, 6, "left"), 0.20: (7, 4, "left")}
    for t, s, c in zip(taus, speedup, capture):
        dx, dy, ha = lab_off[t]
        direct_label(ax, s, c, rf"$\tau{{=}}{t:.2f}$", C_BLUE, dx=dx, dy=dy, ha=ha,
                     fontsize=7)
        ax.annotate(f"{c:.1f}%", (s, c), xytext=(dx, dy - 8), textcoords="offset points",
                    ha=ha, fontsize=6.0, color=MUTED)
    ax.set_xlabel(r"candidate speedup ($\times$)")
    ax.set_ylabel("search-gain capture (%)")
    ax.set_xlim(1.42, 2.62); ax.set_ylim(85.5, 96.5)
    fig.savefig(os.path.join(out, "f4_tau_tradeoff.pdf"))
    plt.close(fig)
    return capture, speedup


def f5_corruption(data, out):
    pp = per_prompt(data[0.10])
    fig, ax = plt.subplots(figsize=(2.7, 2.55))
    ax.axhline(0.7, color=BASE, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.scatter(pp["spread"], pp["rho"], s=10 + 450 * pp["regret"], color=C_BLUE, alpha=0.65,
               edgecolors="white", linewidths=0.4, zorder=2)
    r, _ = spearmanr(pp["spread"], pp["rho"])
    n_bad = int((pp["rho"] < 0.7).sum())
    ax.text(0.42, 0.47, rf"corr(spread, $\rho$) $= {r:+.2f}$" + "\n"
            + rf"$\rho<0.7$: {n_bad}/{len(pp['rho'])} prompts",
            transform=ax.transAxes, fontsize=7, va="bottom")
    for sz, lab in [(0.0, "regret 0"), (0.3, "0.3"), (0.6, "0.6")]:
        ax.scatter([], [], s=10 + 450 * sz, color=C_BLUE, alpha=0.65,
                   edgecolors="white", linewidths=0.4, label=lab)
    ax.legend(loc="lower right", handletextpad=0.5, borderpad=0.4, labelspacing=1.5)
    ax.set_xlabel("candidate score spread (full)")
    ax.set_ylabel(r"ranking preservation $\rho$")
    fig.savefig(os.path.join(out, "f5_corruption.pdf"))
    plt.close(fig)
    return r, n_bad


def cdf_band(vals, xg, B=None, seed=0):
    """Pointwise 95% bootstrap band for the empirical CDF of per-prompt values."""
    vals = np.asarray(vals, float)
    n = len(vals)
    rng = np.random.default_rng(seed)
    F = np.empty((B or B_BOOT, len(xg)))
    for b in range(B or B_BOOT):
        s = np.sort(vals[rng.integers(0, n, n)])
        F[b] = np.searchsorted(s, xg, side="right") / n
    return np.percentile(F, 2.5, axis=0), np.percentile(F, 97.5, axis=0)


def f6_regret_cdf(data, out, vb=None):
    """Regret CDF, direct-labeled (design-overhaul Lane B: legend dropped; the
    three tau curves labeled at their line ends, baselines labeled on-curve)."""
    fig, ax = plt.subplots(figsize=(2.9, 2.55))
    # random-pick baseline: pooled regret of picking each seed at random (full scores, tau-free)
    rand_all = np.concatenate([fu.max() - fu for fu, _ in data[0.10].values()])
    # (vals, color, ls, label, band, lab_xy): bands on the per-prompt CachedSearch
    # curves (prompt is the resampling unit); labels placed via leader to a knee
    curves = ([(rand_all, MUTED, (0, (4, 2)), "random pick", False, (1.06, 0.74))]
              + [(per_prompt(data[t])["regret"], C_TAU[t], "-", rf"$\tau{{=}}{t:.2f}$",
                  True, None) for t in (0.05, 0.10, 0.20)])
    if vb is not None:
        rand_vb = np.concatenate([fu.max() - fu for fu, _ in vb.values()])
        curves.insert(1, (rand_vb, MUTED, (0, (1, 1.5)), "random (official)", False,
                          (1.20, 0.52)))
        curves.append((per_prompt(vb)["regret"], C_TAU[0.10], (0, (1, 1.5)),
                       r"official suite", True, (0.60, 0.965)))
    xg = np.linspace(0, 1.6, 241)
    for vals, color, ls, lab, band, lab_xy in curves:
        x = np.sort(vals)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(np.concatenate([[0], x]), np.concatenate([[np.mean(vals == 0)], y]),
                drawstyle="steps-post", color=color, ls=ls, lw=1.1, zorder=3)
        if band:
            lo, hi = cdf_band(vals, xg)
            ax.fill_between(xg, lo, hi, color=color, alpha=0.20, lw=0, zorder=2)
        if lab_xy is not None:  # baselines + official: on-curve labels
            ax.annotate(lab, lab_xy, fontsize=6.4, color=color, ha="left", va="center")
    # the three CachedSearch tau curves: direct labels at the line ends (each
    # curve's own near-saturation regret), leaders staggered so they don't
    # converge on one point.
    tau_lab = {0.05: (0.58, 0.68, "left"), 0.10: (0.70, 0.55, "left"),
               0.20: (0.82, 0.42, "left")}
    for t in (0.05, 0.10, 0.20):
        reg = per_prompt(data[t])["regret"]
        # line end = 95th-percentile regret (where this curve nears the top)
        kx = float(np.percentile(reg, 95))
        lx, ly, ha = tau_lab[t]
        lab = (rf"cached $\tau{{=}}{t:.2f}$" if t != 0.05
               else rf"CachedSearch, $\tau{{=}}{t:.2f}$")
        ax.annotate(lab, (kx, 0.95), xytext=(lx, ly), textcoords="data",
                    fontsize=6.4, color=C_TAU[t], ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=C_TAU[t],
                                    shrinkA=2, shrinkB=2))
    ax.set_xlabel("regret (full-compute reward units)")
    ax.set_ylabel("fraction of prompts $\\leq$ regret")
    ax.set_xlim(-0.02, 1.6); ax.set_ylim(0, 1.02)
    fig.savefig(os.path.join(out, "f6_regret_cdf.pdf"))
    plt.close(fig)


def f7_adaptive(out):
    """E5: adaptive per-prompt tau (probe-based) vs fixed-tau operating points."""
    from b1_adaptive_tau import load_grid, fixed_point, adaptive_points, N
    prompts, full, cached, Cf, Cc = load_grid()
    fixed = {t: fixed_point(full, cached[t], Cf, Cc[t]) for t in (0.05, 0.10, 0.20)}
    pooled = np.abs(cached[0.20][:, :, None] - cached[0.20][:, None, :])[
        :, np.triu_indices(N, 1)[0], np.triu_indices(N, 1)[1]].ravel()
    qs = np.arange(0, 101, 5)
    pts = adaptive_points(full, cached, Cf, Cc, np.percentile(pooled, qs))
    fig, ax = plt.subplots(figsize=(2.7, 2.3))
    fx = sorted(fixed.values(), key=lambda v: v[1])
    ax.plot([s for _, s in fx], [100 * c for c, _ in fx], "-o", color=C_BLUE, lw=1.2, ms=4,
            label=r"fixed $\tau$", zorder=3)
    for t, (c, s) in fixed.items():
        ax.annotate(rf"$\tau{{=}}{t:.2f}$", (s, 100 * c), xytext=(2, 4),
                    textcoords="offset points", fontsize=6.2, color=C_BLUE)
    # open markers: the adaptive policy is simulated over the measured grid
    ax.plot([p[1] for p in pts], [100 * p[0] for p in pts], "--s", color=C_YELLOW,
            lw=1.0, ms=2.6, mfc="white", label="adaptive (probe $K{=}2$)", zorder=2)
    ax.set_xlabel(r"exploration speedup ($\times$)")
    ax.set_ylabel("search-gain capture (%)")
    ax.set_ylim(87.5, 94.5); ax.set_xlim(1.3, 2.5)
    ax.legend(loc="lower left")
    fig.savefig(os.path.join(out, "f7_adaptive_tau.pdf"))
    plt.close(fig)
    return fixed, list(zip(qs, pts))


# ------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data, cost, latg = load_all()
    vb, vb_cost = load_vbench()
    print("=== verification against expected values ===")
    checks = list(TAGS.items())
    if vb is not None:
        checks.append(("vbench", None))
    for tag, tau in checks:
        if tag == "vbench":
            pp, (Cf, Cc) = per_prompt(vb), vb_cost
            print(f"[vbench] official suite, tau=0.10, n={len(vb)} fully covered (LIVE data, tol=0.02)")
            e = EXPECTED["vbench"]
            check("median rho", float(np.median(pp["rho"])), e["med_rho"], tol=0.02)
            check("mean rho", float(pp["rho"].mean()), e["mean_rho"], tol=0.02)
            check("p10 rho", float(np.percentile(pp["rho"], 10)), e["p10_rho"], tol=0.02)
            check("top-1 agreement", float(pp["top1"].mean()), e["top1"], tol=0.02)
            check("speedup", Cf / Cc, e["speedup"], tol=0.02)
            check("mean regret", float(pp["regret"].mean()), e["regret"], tol=0.02)
            check("random-pick baseline", float(pp["rand"].mean()), e["rand_regret"], tol=0.02)
            check("capture (mean of ratios)", float(np.mean((pp["rand"] - pp["regret"]) / pp["rand"])),
                  e["capture_mor"], tol=0.02)
            continue
        pp = per_prompt(data[tau])
        e = EXPECTED[tag]
        print(f"[{tag}] tau={tau}")
        check("median rho", float(np.median(pp["rho"])), e["med_rho"])
        check("p10 rho", float(np.percentile(pp["rho"], 10)), e["p10_rho"])
        check("top-1 agreement", float(pp["top1"].mean()), e["top1"])
        check("speedup", cost[tau][0] / cost[tau][1], e["speedup"])
        check("mean regret", float(pp["regret"].mean()), e["regret"])
        check("capture (mean of ratios)", float(np.mean((pp["rand"] - pp["regret"]) / pp["rand"])),
              e["capture_mor"])
    sim = simulate(data[0.10], *cost[0.10])
    print("[sim v0]")
    for (N, strat), (g_exp, c_exp) in sorted(EXPECTED["sim_v0"].items()):
        v, c = sim[(N, strat)]
        check(f"N={N} {strat} gain", v - sim[(N, "single")][0], g_exp)
        check(f"N={N} {strat} cost", c, c_exp, tol=1.0)

    print("=== figures ===")
    # CI bars: ON for the small, uncrowded plots (they read nicely there);
    # OFF for the crowded ones (scatter, Pareto) where bars would collide with
    # points/curves. See style.SHOW_UNCERTAINTY (user directive 2026-07-09).
    import style
    style.SHOW_UNCERTAINTY = False; f1_scatter(data, args.out)
    f2_rho_hist(data, args.out, vb=vb)
    style.SHOW_UNCERTAINTY = False; f3_pareto(data, cost, args.out)
    style.SHOW_UNCERTAINTY = False; f4_tau_curve(data, cost, latg, args.out)  # CI bars off (user 2026-07-23: whiskers dwarfed the trend; CIs live in tab:tau)
    style.SHOW_UNCERTAINTY = True
    r, n_bad = f5_corruption(data, args.out)
    check("corr(spread, rho)", r, EXPECTED["corr_spread_rho"])
    check("prompts rho<0.7", n_bad, EXPECTED["n_rho_lt07"], tol=0.5)
    style.SHOW_UNCERTAINTY = True;  f6_regret_cdf(data, args.out, vb=vb)
    style.SHOW_UNCERTAINTY = True;  f7_adaptive(args.out)
    for f in sorted(glob.glob(os.path.join(args.out, "f*.pdf"))):
        print(" wrote", f)


if __name__ == "__main__":
    main()
