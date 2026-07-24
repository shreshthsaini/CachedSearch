"""Reviewer-requested quantitative figures F8-F11 for the CachedSearch paper.

F8  f8_width_scaling.pdf    : capture + top-1 vs search width N in {2,4,8,16}
                              (16-seed b1_gate_v0 grid; N=32 auto-included when
                              32 complete seeds land -- see note in main());
                              log2 N-axis; dashed overlay = spread-weighted
                              Gaussian-copula capture (app:theory, flat in N).
F9  f9_model_scaling.pdf    : median rho / capture / absolute seconds saved per
                              candidate vs model params (six models, four
                              families; log-x; savings panel log-log).
F10 f10_tau_calibration.pdf : capture-vs-speedup tau curves, Wan-1.3B overlaid
                              with CogVideoX-5B (Wan-14B tau points omitted:
                              record counts below threshold; see main()).
F11 f11_method_comparison.pdf: delivered gain + wall-clock bars at a matched
                              ~2x exploration budget, incl. prune+cached (E6).
F11b f11b_baselines.pdf     : capture-vs-speedup positioning of published
                              caching methods (TeaCache / PAB / CFG-Cache,
                              A-lite grids, stats via paper_figs/a1_analysis.py)
                              against our tau sweep, plus the fitted common
                              frontier capture = alpha + beta*ln(speedup) with
                              bootstrap band + R^2; skipped if a1 dirs absent.
F15 f15_engines.pdf         : existing methods as exploration engines on
                              delivered-gain vs wall-clock axes at N=8
                              (fig:engines, 2026-07-14): TeaCache / PAB /
                              CFG-Cache commit points vs full best-of-8 and
                              the no-caching control; "ours" = tab:main row.

Uncertainty (P0-1): every aggregate mark carries a 95% prompt-level bootstrap
CI (percentile, B=10^4; CACHEDSEARCH_BOOT_B overrides), same convention as
bootstrap_ci.py / paper/ci_numbers.json. Marker grammar (P0-5): filled =
directly measured; open = simulated over seed subsets of the measured grid;
dashed = model prediction / fit.

Conventions (MUST match make_figs.py -- style is imported from style.py):
- every number recomputed from RAW jsonl; records deduped by
  (prompt, seed, variant) -- cross-partition shards duplicate records;
- latency stats (Cf, Cc, speedups) use seeds 0-7 ONLY: the N=16
  width-extension shards share the results dirs and their latencies must not
  contaminate the paper grid (make_figs.py bug fix, 2026-07-07);
- model-comparison stats (F9/F10) use the first 8 seeds only, for
  comparability (v0 / cog5b_v0 carry 16 width-extension seeds);
- tau dirs contain CACHED rollouts only; full references are merged from the
  model's v0 dir on (prompt, seed) -- full rollouts are tau-independent;
- subset simulation per b1_simulate.py: all C(S,N) seed subsets, or 2000
  sampled subsets per prompt when C(S,N) > SUBSET_CAP;
- capture conventions: F8/F11 use ratio-of-mean-gains vs best-of-N;
  F9/F10 use the mean per-prompt
  gain-capture ratio (tab:tau / tab:scale, estimator of eq:capture).

Cross-checked against the expected values below. Mismatches print a loud
warning but still plot.

Usage:  python code/paper_figs/make_figs2.py [--out assets]
"""
import argparse
import glob
import itertools
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# canonical rcParams + palette: paper_figs/style.py (design-overhaul Lane 0)
from style import (apply_style, BASE, C_BLUE, C_GREEN, C_RED, C_STRAT, C_TAU,
                   C_VIOLET, C_YELLOW, ERR_KW, INK, MUTED,
                   MODEL_COLORS, MODEL_MARKERS, MODEL_SHORT)
from make_figs import (OUT_DEFAULT, RESULTS, boot_idx, boot_speedup, check,
                       ci95, per_prompt, simulate, simulate_pp, yerr_of)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

apply_style()
import style; style.SHOW_UNCERTAINTY = True  # all make_figs2 plots are small-multiple/point plots: show CI bars

SUBSET_CAP = 20000   # enumerate all C(S,N) subsets up to this; else sample
SUBSET_SAMPLE = 2000
# tag -> (label, params, color, marker); colors/markers are the entity-fixed
# MODEL_COLORS/MODEL_MARKERS (design-overhaul Lane B switch: Cog->^, Hunyuan->v,
# LTX->P, Wan-14B->D; Wan family shares the blue ramp).
_MCKEY = {"v0": "Wan2.1-1.3B", "ltx_v0": "LTX-Video-2B", "wan22_5b": "Wan2.2-5B",
          "cog5b_v0": "CogVideoX-5B", "hunyuan_v0": "HunyuanVideo-13B",
          "wan14b": "Wan2.1-14B"}
MODELS = {
    tag: (lbl, prm, MODEL_COLORS[_MCKEY[tag]], MODEL_MARKERS[_MCKEY[tag]])
    for tag, (lbl, prm) in {
        "v0": ("Wan2.1-1.3B", 1.3e9), "ltx_v0": ("LTX-Video-2B", 2.0e9),
        "wan22_5b": ("Wan2.2-TI2V-5B", 5.0e9), "cog5b_v0": ("CogVideoX-5B", 5.0e9),
        "hunyuan_v0": ("HunyuanVideo-13B", 13.0e9), "wan14b": ("Wan2.1-14B", 14.0e9),
    }.items()}

# ------------------------------------------------------- expected numbers
# Expected analysis values. The Wan-14B row was derived at
# n=40 complete prompts; the dir now holds n>=46 -> tolerance loosened there.
EXPECTED2 = {
    # width: N<=16 over C(16,N) subsets of the 16-seed grid (published tab/fig
    # numbers, theory-table pool); N=32 = the full 32-seed grid (one subset per
    # prompt, exactly as N=16 is the full 16-seed grid) -- 2026-07-08 refresh.
    "width_commit": {4: 0.909, 8: 0.922, 16: 0.957, 32: 0.952},
    "scale": {  # tab:scale, six-model refresh 2026-07-08 (all n=50, seeds 0-7)
        "v0":         dict(med=0.905, p10=0.61, top1=0.64, capture=0.901, speedup=1.97),
        "ltx_v0":     dict(med=0.536, p10=0.14, top1=0.50, capture=0.676, speedup=2.63),
        "wan22_5b":   dict(med=0.881, p10=0.60, top1=0.46, capture=0.860, speedup=2.05),
        "cog5b_v0":   dict(med=0.762, p10=0.35, top1=0.48, capture=0.752, speedup=2.06),
        "hunyuan_v0": dict(med=0.762, p10=0.44, top1=0.54, capture=0.799, speedup=2.19),
        "wan14b":     dict(med=0.905, p10=0.52, top1=0.58, capture=0.875, speedup=2.05),
    },
    "tau": {  # (model, tau) -> (capture_mor, speedup)   [tab:tau / sec:abl-cog]
        ("v0", 0.05): (0.936, 1.58), ("v0", 0.10): (0.901, 1.97), ("v0", 0.20): (0.883, 2.41),
        ("cog5b_v0", 0.05): (0.859, 1.78), ("cog5b_v0", 0.10): (0.752, 2.06),
        ("cog5b_v0", 0.20): (0.647, 2.65),
        # Wan-14B arms were ENQUEUED at tau=0.005/0.020 (records carry these
        # values), NOT the intended 0.05/0.20 -- integrated as measured:
        # 0.005 fires zero skips (bit-identical control), 0.02 is mild reuse.
        ("wan14b", 0.005): (1.000, 1.01), ("wan14b", 0.02): (0.921, 1.21),
        ("wan14b", 0.10): (0.875, 2.05),
        # wave-4 tau-calibration arms (tab:taucal); wan14b 0.05/0.20 = the
        # tau05fix/tau20fix reruns at the INTENDED values (frozen 2026-07-08
        # at full coverage n=50)
        ("ltx_v0", 0.02): (0.796, 1.71), ("ltx_v0", 0.05): (0.716, 2.19),
        ("ltx_v0", 0.10): (0.676, 2.63),
        ("wan14b", 0.05): (0.880, 1.59), ("wan14b", 0.20): (0.847, 2.52),
    },
    "sim_f11": {  # (N, strategy) -> (gain vs single, cost s)   [tab:main]
        (4, "bon_full"): (0.514, 273), (8, "bon_full"): (0.752, 547),
        (8, "cached_commit"): (0.712, 346), (8, "cached_keep"): (0.749, 278),
    },
    # Stacked-search reference values: "175 s / 3.11x end-to-end"; the raw timings
    # show 175 s is the EXPLORATION cost (stage1 + preview + stage2 only) --
    # delivered end-to-end (incl. decode/scoring + the 68 s recommit) is 256 s,
    # 2.1x vs best-of-8. F11 plots the delivered number; the tex was re-labeled.
    "stack": dict(expl=175.0, ratio_bo8_expl=3.11, capture_mor=0.886,
                  survival=0.86, med_regret=0.0),
    # F15/fig:engines: (engine) -> (commit gain vs single, commit cost s at
    # N=8). Engines = the a1_* published-method reproductions used as the
    # exploration stage (full refs from b1_gate_v0); "ours" = the v0 grid
    # itself (tab:main row). Frozen 2026-07-14 from the crux dirs.
    "engines": {"teacache": (0.723, 365), "pab": (0.748, 493),
                "cfgcache": (0.748, 465), "none": (0.752, 621),
                "ours": (0.712, 346)},
}


# ------------------------------------------------------------- data layer
def load_rows(tag):
    """Raw records for b1_gate_<tag>, deduped by (prompt, seed, variant)."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, f"b1_gate_{tag}", "scores_shard*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            rows[(r["prompt"], r["seed"], r["variant"])] = r
    return rows


def latencies(rows):
    """Mean per-variant latency, seeds 0-7 ONLY (width-extension shards share
    the dirs; their latencies must not contaminate Cf/Cc -- make_figs.py fix)."""
    lat = defaultdict(list)
    for (p, s, v), r in rows.items():
        if s < 8:
            lat[v].append(r["latency"])
    return {v: float(np.mean(xs)) for v, xs in lat.items()}


def grid(rows, n_seeds=8, full_rows=None):
    """{prompt: (full[n_seeds], cached[n_seeds])} on seeds 0..n_seeds-1, keeping
    prompts with complete coverage. full_rows: merge full references from the
    model's v0 rows (tau dirs store cached only)."""
    by = defaultdict(dict)
    for (p, s, v), r in rows.items():
        if s < n_seeds:
            by[p][(s, v)] = r["score"]
    if full_rows is not None:
        for (p, s, v), r in full_rows.items():
            if v == "full" and s < n_seeds and p in by:
                by[p][(s, "full")] = r["score"]
    pa = {}
    for p, d in by.items():
        if all((s, v) in d for s in range(n_seeds) for v in ("full", "cached")):
            pa[p] = (np.array([d[(s, "full")] for s in range(n_seeds)]),
                     np.array([d[(s, "cached")] for s in range(n_seeds)]))
    return pa


def capture_mor(pa):
    """Mean per-prompt gain-capture ratio (tab:tau convention, eq:capture)."""
    pp = per_prompt(pa)
    ratios = (pp["rand"] - pp["regret"]) / pp["rand"]
    return float(np.mean(ratios[np.isfinite(ratios)]))


def load_stack():
    """E6 prune+cached records (one per prompt), deduped by prompt."""
    recs = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "b1_stack_v0", "scores_shard*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            recs[r["prompt"]] = r
    return recs


def width_sim(pa, Ns, rng_seed=0):
    """Strategy simulation over seed subsets of size N (b1_simulate.py semantics,
    vectorized). Subsets are sorted index tuples; 'single' = the lowest-index
    member (matching itertools.combinations enumeration in b1_simulate.py).
    -> {N: dict(single, bon, keep, commit, top1, n_subsets, pp)} where the
    scalars are means over prompts x subsets and pp holds the per-PROMPT subset
    means (every prompt contributes the same subset count, so the mean of pp
    equals the pooled mean exactly) for the prompt-level bootstrap."""
    rng = np.random.default_rng(rng_seed)
    prompts = sorted(pa)
    keys = ("single", "bon", "keep", "commit", "top1")
    out = {}
    for N in Ns:
        pp = {k: np.zeros(len(prompts)) for k in keys}
        n_sub_total = 0
        for i, p in enumerate(prompts):
            fu, ca = pa[p]
            S = len(fu)
            n_sub = math.comb(S, N)
            if n_sub <= SUBSET_CAP:
                idx = np.array(list(itertools.combinations(range(S), N)), dtype=int)
            else:  # combinatorics explode -> sampled subsets (sorted, w/o replacement)
                idx = np.sort(np.array([rng.choice(S, size=N, replace=False)
                                        for _ in range(SUBSET_SAMPLE)]), axis=1)
            F, C = fu[idx], ca[idx]                      # (n_sub, N)
            pick = C.argmax(axis=1)
            rows_ = np.arange(len(idx))
            pp["single"][i] = F[:, 0].mean()
            pp["bon"][i] = F.max(axis=1).mean()
            pp["keep"][i] = C[rows_, pick].mean()
            pp["commit"][i] = F[rows_, pick].mean()
            pp["top1"][i] = (pick == F.argmax(axis=1)).mean()
            n_sub_total += len(idx)
        out[N] = {k: float(pp[k].mean()) for k in keys}
        out[N]["n_subsets"] = n_sub_total
        out[N]["pp"] = pp
    return out


def lat_groups(rows, variant):
    """Per-prompt latency groups {prompt: [latencies]}, seeds 0-7, one variant."""
    g = defaultdict(list)
    for (p, s, v), r in rows.items():
        if s < 8 and v == variant:
            g[p].append(r["latency"])
    return dict(g)


def boot_costs(fg, cg, seed=0):
    """Bootstrap sample vectors (Cf_b, Cc_b) from paired per-prompt latency
    groups -- shared draws so composite costs (N*Cc + Cf) resample jointly."""
    common = sorted(set(fg) & set(cg))
    fs = np.array([np.sum(fg[p]) for p in common])
    fn = np.array([len(fg[p]) for p in common], float)
    cs = np.array([np.sum(cg[p]) for p in common])
    cn = np.array([len(cg[p]) for p in common], float)
    idx = boot_idx(len(common), seed=seed)
    return fs[idx].sum(1) / fn[idx].sum(1), cs[idx].sum(1) / cn[idx].sum(1)


# ------------------------------------------------------------- figures
def f8_width(rows_v0, out):
    """F8: capture + top-1 vs N on the 16-seed grid (N=32 when data supports),
    with 95% prompt-bootstrap error bars and the copula-model overlay (P1-10):
    the spread-weighted Gaussian-copula capture sum(sigma_p r_p)/sum(sigma_p)
    of app:theory Eq. weightedcapture -- flat in N by Prop. capture -- computed
    from the same pool via r_p = 2 sin(pi rho_p / 6)."""
    Cf, Cc = (lambda l: (l["full"], l["cached"]))(latencies(rows_v0))
    # N<=16 stays on the 16-seed grid (the published tab/theory-table pool);
    # when 32 complete seeds land, N=32 is APPENDED from the 32-seed grid --
    # the full-pool point, exactly as N=16 is the full 16-seed grid. (A whole-
    # curve recompute on the 32-seed pool would silently shift every published
    # N<=16 number and the theory-vs-measured table's calibration pool.)
    pa = grid(rows_v0, n_seeds=16)
    pa32 = grid(rows_v0, n_seeds=32)
    if len(pa32) >= 40:
        Ns = (2, 4, 8, 16, 32)
        print(f"[f8] N=32 grid available (n={len(pa32)} prompts at 32 seeds) -- "
              f"appended (N<=16 remain on the 16-seed pool)")
        assert sorted(pa32) == sorted(pa), "16- and 32-seed pools cover different prompts"
    else:
        Ns = (2, 4, 8, 16)
        print(f"[f8] N=32 NOT supported yet ({len(pa32)} prompts at 32 seeds; "
              f"using {len(pa)} prompts at 16 seeds, N up to 16)")
    sim = width_sim(pa, tuple(N for N in Ns if N <= 16))
    if 32 in Ns:
        sim[32] = width_sim(pa32, (32,))[32]
    cap_c = [100 * (sim[N]["commit"] - sim[N]["single"]) / (sim[N]["bon"] - sim[N]["single"]) for N in Ns]
    cap_k = [100 * (sim[N]["keep"] - sim[N]["single"]) / (sim[N]["bon"] - sim[N]["single"]) for N in Ns]
    top1 = [100 * sim[N]["top1"] for N in Ns]
    cost_ratio = [100 * (N * Cc + Cf) / (N * Cf) for N in Ns]
    # prompt-level bootstrap CIs (ratio-of-mean-gains resampled jointly)
    idx = boot_idx(len(pa))
    cis = {}
    for N in Ns:
        pp = sim[N]["pp"]
        Sm = pp["single"][idx].mean(1)
        Bm = pp["bon"][idx].mean(1)
        cis[(N, "c")] = ci95(100 * (pp["commit"][idx].mean(1) - Sm) / (Bm - Sm))
        cis[(N, "k")] = ci95(100 * (pp["keep"][idx].mean(1) - Sm) / (Bm - Sm))
        cis[(N, "t")] = ci95(100 * pp["top1"][idx].mean(1))
    # copula-model overlay: spread-weighted capture, flat in N (app:theory,
    # b1_theory_check_n.py model M1) with a 95% prompt-bootstrap band.
    rho_p = np.array([spearmanr(pa[p][0], pa[p][1])[0] for p in sorted(pa)])
    sig_p = np.array([pa[p][0].std() for p in sorted(pa)])
    r_p = 2 * np.sin(np.pi * np.clip(rho_p, -1, 1) / 6)
    m1 = float((sig_p * r_p).sum() / sig_p.sum())
    idxm = boot_idx(len(sig_p), seed=7)
    m1_lo, m1_hi = ci95(100 * (sig_p[idxm] * r_p[idxm]).sum(1) / sig_p[idxm].sum(1))
    print(f"[f8] copula-model overlay: spread-weighted capture = {100*m1:.1f}% "
          f"[{m1_lo:.1f}, {m1_hi:.1f}] (flat in N; b1_theory_check_n.py M1)")

    print("[f8] width-scaling (16-seed re-simulation, all C(16,N) subsets):")
    for N, cc_, ck_, t_, cr_ in zip(Ns, cap_c, cap_k, top1, cost_ratio):
        print(f"  N={N:>2}: commit capture {cc_:5.1f}% [{cis[(N,'c')][0]:.1f},{cis[(N,'c')][1]:.1f}]"
              f"  keep {ck_:5.1f}%  top-1 {t_:4.1f}%"
              f"  commit cost {cr_:4.1f}% of best-of-N  ({sim[N]['n_subsets']} subsets)")
    for N, want in EXPECTED2["width_commit"].items():
        check(f"width N={N} commit capture", cap_c[Ns.index(N)] / 100, want)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(3.9, 1.95))
    for a in (ax, ax2):
        a.set_xscale("log", base=2)
        a.set_xticks(Ns)
        a.set_xticklabels([str(N) for N in Ns])
        a.set_xlabel("search width $N$")
        a.set_xlim(Ns[0] * 0.82, Ns[-1] * 1.22)
        a.minorticks_off()
    # open markers: subset-simulated from the measured grid (paper convention)
    ax.errorbar(Ns, cap_k, yerr=yerr_of(cap_k, [cis[(N, "k")] for N in Ns]),
                fmt="-o", color=C_STRAT["cached_keep"], lw=1.2, ms=3.5, mfc="white",
                zorder=3, **ERR_KW)
    ax.errorbar(Ns, cap_c, yerr=yerr_of(cap_c, [cis[(N, "c")] for N in Ns]),
                fmt="-o", color=C_STRAT["cached_commit"], lw=1.2, ms=3.5, mfc="white",
                zorder=3, **ERR_KW)
    for N, c in zip(Ns, cap_c):
        ax.annotate(f"{c:.1f}", (N, c), xytext=(4, -8.5), textcoords="offset points",
                    ha="center", fontsize=5.8, color=C_STRAT["cached_commit"])
    # direct labels (no legend)
    ax.annotate("keep (nominal)", (Ns[1], cap_k[1]), xytext=(-2, 5),
                textcoords="offset points", fontsize=6.0,
                color=C_STRAT["cached_keep"], ha="left")
    ax.text(11.5, 91.4, "commit", fontsize=6.0,
            color=C_STRAT["cached_commit"], ha="center")
    # gray dashed band: the flat, conservative rank-noise (copula) envelope
    xspan = [Ns[0] * 0.82, Ns[-1] * 1.22]
    ax.fill_between(xspan, m1_lo, m1_hi, color=MUTED, alpha=0.16, lw=0, zorder=1)
    ax.plot(xspan, [100 * m1, 100 * m1], color=MUTED, lw=0.9, ls=(0, (4, 2.5)),
            zorder=1)
    ax.annotate(rf"rank-noise model (conservative): {100*m1:.0f}\%, flat",
                (Ns[0] * 0.86, 100 * m1), xytext=(1, -8), textcoords="offset points",
                fontsize=5.6, color=MUTED, ha="left", va="top")
    ax.set_ylabel("gain capture (%)")
    ax.set_ylim(80, 104.5)
    ax2.errorbar(Ns, top1, yerr=yerr_of(top1, [cis[(N, "t")] for N in Ns]),
                 fmt="-s", color=C_YELLOW, lw=1.2, ms=3.5, mfc="white", **ERR_KW)
    for N, t in zip(Ns, top1):
        ax2.annotate(f"{t:.0f}", (N, t), xytext=(-5, 5), textcoords="offset points",
                     ha="center", fontsize=5.8, color=C_YELLOW)
    ax2.set_ylabel("top-1 agreement (%)")
    ax2.set_ylim(45, 92)
    fig.subplots_adjust(wspace=0.42)
    fig.savefig(os.path.join(out, "f8_width_scaling.pdf"))
    plt.close(fig)
    return dict(Ns=Ns, cap_c=cap_c, cap_k=cap_k, top1=top1, cost_ratio=cost_ratio,
                n_prompts=len(pa), m1=m1)


def f9_models(stats, out):
    """F9 (rebuilt for readability): two distinct stories, one per panel.
    (A) Ranking fidelity (median rho) is set by architecture FAMILY, not size:
        models grouped by family on a categorical axis, the Wan2.1/2.2 trio
        banded high across a 10.8x size range, off-family ports clearly below.
    (B) Absolute seconds saved per candidate scale with model cost (log-log vs
        params): saved = Cf - Cc, ~half of Cf on every backbone. Capture / top-1
        follow the same family split and live in tab:scale. CI bars off."""
    order = ["v0", "wan22_5b", "wan14b", "cog5b_v0", "hunyuan_v0", "ltx_v0"]
    xcat = [0, 1, 2, 3.5, 4.5, 5.5]              # visual gap after the Wan trio
    size = {"v0": "1.3B", "wan22_5b": "5B", "wan14b": "14B",
            "cog5b_v0": "5B", "hunyuan_v0": "13B", "ltx_v0": "2B"}

    def mk(ax, x, y, t, s=48):
        ax.scatter([x], [y], marker=MODELS[t][3], s=s, zorder=3, linewidths=1.0,
                   edgecolors=MODELS[t][2],
                   facecolors="white" if t == "wan22_5b" else MODELS[t][2])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.5, 2.15),
                                   gridspec_kw={"width_ratios": [1.32, 1.0]})
    # ---------- Panel A: ranking fidelity is set by family, not size ----------
    wan_r = [stats[t]["med"] for t in ("v0", "wan22_5b", "wan14b")]
    axA.axhspan(min(wan_r), max(wan_r), color=C_BLUE, alpha=0.09, zorder=0)
    axA.axvline(2.75, color=BASE, lw=0.6, ls=(0, (1, 2.2)), zorder=1)
    for x, t in zip(xcat, order):
        y = stats[t]["med"]
        mk(axA, x, y, t)
        axA.annotate(f"{y:.3f}", (x, y), xytext=(0, 7), textcoords="offset points",
                     ha="center", fontsize=5.6, color=MODELS[t][2])
    axA.set_xticks(xcat)
    axA.set_xticklabels([size[t] for t in order], fontsize=6.6)
    axA.set_xlim(-0.7, 6.2)
    axA.set_ylim(0.46, 1.14)
    axA.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axA.set_ylabel(r"median $\rho$  (ranking fidelity)", fontsize=7)
    axA.set_xlabel("parameters, grouped by family")
    axA.annotate("Wan2.1/2.2 family:\nflat across $10.8\\times$ size",
                 (1.0, 1.12), ha="center", va="top", fontsize=6.0,
                 color=MODEL_COLORS["Wan2.1-1.3B"])
    axA.annotate("off-family:\nlower at every size", (4.3, 0.66), ha="center",
                 va="top", fontsize=5.8, color=MUTED)
    # ---------- Panel B: absolute savings scale with model cost ----------
    off = {"v0": (0, -9, "center"), "wan22_5b": (0, 7, "center"),
           "wan14b": (0, -9, "center"), "cog5b_v0": (8, -1, "left"),
           "hunyuan_v0": (8, 2, "left"), "ltx_v0": (8, -1, "left")}
    for t in order:
        y = stats[t]["Cf"] - stats[t]["Cc"]
        mk(axB, MODELS[t][1], y, t, s=42)
        dx, dy, ha = off[t]
        axB.annotate(f"{y:.0f}s", (MODELS[t][1], y), xytext=(dx, dy),
                     textcoords="offset points", ha=ha, va="center",
                     fontsize=5.6, color=MODELS[t][2])
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xticks([1.3e9, 5e9, 14e9])
    axB.set_xticklabels(["1.3B", "5B", "14B"])
    axB.set_yticks([10, 30, 100, 300])
    axB.set_yticklabels(["10", "30", "100", "300"])
    axB.minorticks_off()
    axB.set_xlim(0.9e9, 22e9)
    axB.set_ylim(4, 340)
    axB.set_ylabel("seconds saved per candidate", fontsize=7)
    axB.set_xlabel("parameters")
    axB.annotate(r"${\approx}\,\frac{1}{2}C_f$ per candidate;"
                 "\nabsolute saving\ngrows with size",
                 (0.04, 0.97), xycoords="axes fraction", ha="left", va="top",
                 fontsize=5.4, color=MUTED)
    # ---------- shared right-side model legend ----------
    names = ["Wan2.1-1.3B", "Wan2.2-5B", "Wan2.1-14B",
             "CogVideoX-5B", "HunyuanVideo-13B", "LTX-Video-2B"]
    handles = [Line2D([], [], marker=MODEL_MARKERS[n], color=MODEL_COLORS[n], ls="",
                      mfc="white" if n == "Wan2.2-5B" else MODEL_COLORS[n],
                      mec=MODEL_COLORS[n], ms=4.5, label=MODEL_SHORT[n])
               for n in names]
    fig.subplots_adjust(wspace=0.42, left=0.09, right=0.82, bottom=0.21, top=0.97)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.835, 0.5),
               frameon=False, fontsize=6.3, handletextpad=0.5, labelspacing=0.62)
    fig.savefig(os.path.join(out, "f9_model_scaling.pdf"))
    plt.close(fig)


def f10_tau(curves, out):
    """F10: capture-vs-speedup tau curves for Wan2.1-1.3B, CogVideoX-5B,
    Wan2.1-14B, and LTX-Video-2B, with 95% prompt-bootstrap error bars on
    capture (y) and speedup (x); direct labels instead of a legend.
    curves[tag][tau] = dict(cap, sp, pp, fg, cg). The 14B curve spans the full
    dial: the historical tau=0.005/0.02 arms (enqueue error, integrated as
    measured; 0.005 fires zero skips = bit-identical control) PLUS the wave-4
    tau05fix/tau20fix reruns at the intended 0.05/0.20. LTX is the boundary
    curve of tab:taucal: every measured arm sits below the 85% band."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    # entity-fixed colors/markers (design-overhaul Lane B): Cog yellow/^, LTX
    # red/P, Wan-14B dark-blue/D (dashed to read as a family arc).
    style = {"v0": (MODEL_COLORS["Wan2.1-1.3B"], MODEL_MARKERS["Wan2.1-1.3B"], "-", "Wan2.1-1.3B"),
             "cog5b_v0": (MODEL_COLORS["CogVideoX-5B"], MODEL_MARKERS["CogVideoX-5B"], "-", "CogVideoX-5B"),
             "wan14b": (MODEL_COLORS["Wan2.1-14B"], MODEL_MARKERS["Wan2.1-14B"], (0, (4, 2)), "Wan2.1-14B"),
             "ltx_v0": (MODEL_COLORS["LTX-Video-2B"], MODEL_MARKERS["LTX-Video-2B"], "-", "LTX-Video-2B")}
    for tag, pts in curves.items():
        c, m, ls, lab = style[tag]
        taus = sorted(pts)
        xs = [pts[t]["sp"] for t in taus]
        ys = [100 * pts[t]["cap"] for t in taus]
        ycis = [ci95(100 * pts[t]["pp"][boot_idx(len(pts[t]["pp"]), seed=2)].mean(1))
                for t in taus]
        xcis = [boot_speedup(pts[t]["fg"], pts[t]["cg"], seed=2) for t in taus]
        ax.errorbar(xs, ys, yerr=yerr_of(ys, ycis), xerr=yerr_of(xs, xcis),
                    fmt=m, ls=ls, color=c, lw=1.2, ms=4,
                    mfc="white" if tag == "wan14b" else None, zorder=3, **ERR_KW)
        # tau labels only on the Wan-1.3B reference curve (it shows the dial);
        # the other backbones' tau values are in the caption and Table 10.
        # Model identity is carried by the side legend, so no inline model-name
        # labels crowd the curves.
        if tag == "v0":
            tau_off = {0.05: (0, 6, "center"), 0.10: (0, 6, "center"),
                       0.20: (6, -9, "left")}  # last point sits below the legend
            for t in taus:
                dx, dy, ha = tau_off.get(t, (0, 6, "center"))
                ax.annotate(rf"$\tau{{=}}{t:g}$",
                            (pts[t]["sp"], 100 * pts[t]["cap"]),
                            xytext=(dx, dy), textcoords="offset points",
                            ha=ha, fontsize=5.8, color=c)
    # calibration guide: Wan-1.3B's most aggressive operating point
    wan_min = 100 * min(d["cap"] for d in curves["v0"].values())
    ax.axhline(wan_min, color=BASE, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.annotate("Wan-1.3B's operating band", xy=(1.36, wan_min),
                xytext=(0.98, 82.6), textcoords="data", fontsize=5.8,
                color=MUTED, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                                shrinkA=1, shrinkB=1))
    ax.annotate("same $\\tau$, different\neffective aggressiveness",
                xy=(1.32, 64.5), xycoords="data",
                fontsize=6.2, color=INK, ha="left")
    # side legend: four backbones, entity-fixed color + marker (Wan-14B dashed
    # to read as a family arc); placed in the empty upper-right corner.
    leg_order = ["v0", "wan14b", "cog5b_v0", "ltx_v0"]
    handles = [Line2D([], [], marker=style[t][1], color=style[t][0],
                      ls=style[t][2], lw=1.2,
                      mfc="white" if t == "wan14b" else style[t][0],
                      mec=style[t][0], ms=4.2, label=style[t][3])
               for t in leg_order]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=6.0,
              handletextpad=0.5, labelspacing=0.5, borderaxespad=0.2)
    ax.set_xlabel(r"candidate speedup ($\times$)")
    ax.set_ylabel("search-gain capture (%)")
    ax.set_xlim(0.93, 2.85)
    ax.set_ylim(56, 103)
    fig.savefig(os.path.join(out, "f10_tau_calibration.pdf"))
    plt.close(fig)


def f11_methods(sim, stack_stats, Cf, out, gain_cis=None, cost_cis=None):
    """F11: delivered gain + wall-clock bars at matched ~2x exploration budget.
    gain_cis/cost_cis: per-bar (lo, hi) 95% prompt-bootstrap CIs (or None)."""
    single = {N: sim[(N, "single")][0] for N in (4, 8)}
    bars = [  # (label, gain, cost, color, hatch)
        ("single", 0.0, sim[(4, "single")][1], C_STRAT["single"], None),
        ("best-of-4 full", sim[(4, "bon_full")][0] - single[4],
         sim[(4, "bon_full")][1], C_STRAT["bon_full"], None),
        ("commit $N{=}8$", sim[(8, "cached_commit")][0] - single[8],
         sim[(8, "cached_commit")][1], C_STRAT["cached_commit"], None),
        ("keep $N{=}8$ (nominal)", sim[(8, "cached_keep")][0] - single[8],
         sim[(8, "cached_keep")][1], C_STRAT["cached_keep"], "///"),
        ("prune+cached", stack_stats["gain"], stack_stats["e2e"], C_YELLOW, None),
    ]
    bo8_gain = sim[(8, "bon_full")][0] - single[8]
    bo8_cost = sim[(8, "bon_full")][1]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(4.4, 2.0))
    x = np.arange(len(bars))
    for i, (lab, g, c, col, hat) in enumerate(bars):
        ax.bar(i, g, 0.62, color=col, hatch=hat, edgecolor="white", linewidth=0.4)
        ax2.bar(i, c, 0.62, color=col, hatch=hat, edgecolor="white", linewidth=0.4)
        if gain_cis is not None and gain_cis[i] is not None:
            ax.errorbar([i], [g], yerr=yerr_of([g], [gain_cis[i]]), fmt="none",
                        ecolor=INK, **ERR_KW)
        if cost_cis is not None and cost_cis[i] is not None:
            ax2.errorbar([i], [c], yerr=yerr_of([c], [cost_cis[i]]), fmt="none",
                         ecolor=INK, **ERR_KW)
        if g > 0:  # value inside the bar top (solid chip masks the hatch)
            ax.annotate(f"+{g:.2f}", (i, g - 0.06), va="top", ha="center",
                        fontsize=5.8, color="white",
                        bbox=dict(fc=col, ec="none", pad=0.6))
        else:
            ax.annotate("0", (i, 0.008), va="bottom", ha="center", fontsize=5.8,
                        color=MUTED)
        ax2.annotate(f"{c:.0f}", (i, c), xytext=(-9, 2), textcoords="offset points",
                     ha="center", fontsize=5.8, color=col)
    for a, ref, lab in ((ax, bo8_gain, f"best-of-8 full: +{bo8_gain:.2f}"),
                        (ax2, bo8_cost, f"best-of-8 full: {bo8_cost:.0f} s")):
        a.axhline(ref, color=C_STRAT["bon_full"], lw=0.8, ls=(0, (4, 2)), zorder=1)
        a.annotate(lab, (-0.45, ref), xytext=(1, 2.5), textcoords="offset points",
                   fontsize=5.6, color=C_STRAT["bon_full"], ha="left")
    for a in (ax, ax2):
        a.set_xticks(x)
        a.set_xticklabels([b[0] for b in bars], fontsize=5.8, rotation=24,
                          ha="right", rotation_mode="anchor")
        a.grid(axis="x", visible=False)
    ax.set_ylabel("delivered gain over single")
    ax.set_ylim(0, 1.02)
    ax2.set_ylabel("wall-clock per prompt (s)")
    ax2.set_ylim(0, 640)
    fig.subplots_adjust(wspace=0.38)
    fig.savefig(os.path.join(out, "f11_method_comparison.pdf"))
    plt.close(fig)


def frontier_fit(sp, cap, pp_caps, lat_pairs, seed=3):
    """Fit the common frontier as log-linear capture = alpha + beta*ln(speedup)
    across heterogeneous caching methods (P1-11; a power law 1 - a(s-1)^b was
    tried first and fits worse, R^2 ~ 0.33 vs ~0.92 -- the tau-sweep tail is
    flatter than a power law), with a 95% prompt-bootstrap band: per replicate,
    each point's capture is re-estimated from its per-prompt gain-capture
    ratios and its speedup from its per-prompt latency groups, then the
    frontier is refit. -> alpha, beta, R2, (xg, lo, hi) band over the fitted
    support (no extrapolation to speedup -> 1, where the form exceeds 100%)."""
    from make_figs import B_BOOT
    sp, cap = np.asarray(sp, float), np.asarray(cap, float)

    def fit(s_, c_):
        A = np.vstack([np.ones_like(s_), np.log(s_)]).T
        return np.linalg.lstsq(A, c_, rcond=None)[0]  # alpha, beta

    alpha, beta = fit(sp, cap)
    resid = cap - (alpha + beta * np.log(sp))
    r2 = 1.0 - (resid ** 2).sum() / ((cap - cap.mean()) ** 2).sum()
    # bootstrap band (joint resample of capture + speedup per point)
    rng = np.random.default_rng(seed)
    xg = np.linspace(sp.min() * 0.97, sp.max() * 1.04, 120)
    curves = np.empty((B_BOOT, len(xg)))
    boots = []
    for ppc, (fg, cg) in zip(pp_caps, lat_pairs):
        common = sorted(set(fg) & set(cg))
        fs = np.array([np.sum(fg[p]) for p in common])
        fn = np.array([len(fg[p]) for p in common], float)
        cs = np.array([np.sum(cg[p]) for p in common])
        cn = np.array([len(cg[p]) for p in common], float)
        idx_c = rng.integers(0, len(ppc), size=(B_BOOT, len(ppc)))
        idx_l = rng.integers(0, len(common), size=(B_BOOT, len(common)))
        cap_b = ppc[idx_c].mean(1)
        sp_b = (fs[idx_l].sum(1) / fn[idx_l].sum(1)) / (cs[idx_l].sum(1) / cn[idx_l].sum(1))
        boots.append((sp_b, cap_b))
    for k in range(B_BOOT):
        s_ = np.array([sb[k] for sb, _ in boots])
        c_ = np.array([cb[k] for _, cb in boots])
        a_, b_ = fit(s_, c_)
        curves[k] = a_ + b_ * np.log(xg)
    lo, hi = np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0)
    return float(alpha), float(beta), float(r2), (xg, lo, hi)


def f11b_baselines(curves_v0, bstats, fg_v0, out):
    """F11b: published caching methods (A-lite reproductions, B1-gate protocol)
    as capture-vs-speedup points against our tau sweep, with 95% prompt-bootstrap
    error bars and the fitted common frontier (P1-11).
    curves_v0[tau] = dict(cap, sp, pp, fg, cg); bstats: a1_analysis.method_stats
    dicts (extended with pp_capture / lat_groups); fg_v0: v0 full latency groups."""
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    taus = sorted(curves_v0)
    xs = [curves_v0[t]["sp"] for t in taus]
    ys = [100 * curves_v0[t]["cap"] for t in taus]
    ycis = [ci95(100 * curves_v0[t]["pp"][boot_idx(len(curves_v0[t]["pp"]), seed=3)].mean(1))
            for t in taus]
    xcis = [boot_speedup(curves_v0[t]["fg"], curves_v0[t]["cg"], seed=3) for t in taus]
    ax.errorbar(xs, ys, yerr=yerr_of(ys, ycis), xerr=yerr_of(xs, xcis), fmt="-o",
                color=C_BLUE, lw=1.2, ms=4, zorder=4, **ERR_KW)
    ax.annotate(r"ours, $\tau$ sweep", (xs[0], ys[0]), xytext=(-8, 10),
                textcoords="offset points", ha="right", fontsize=6.0, color=C_BLUE)
    for t in taus:
        ax.annotate(rf"$\tau{{=}}{t:.2f}$", (curves_v0[t]["sp"], 100 * curves_v0[t]["cap"]),
                    xytext=(6, 4), textcoords="offset points", ha="left",
                    fontsize=5.8, color=C_BLUE)
    marks = {  # method -> (label, color, marker, annot offset)
        "teacache": ("TeaCache", C_VIOLET, "^", (7, -2.5)),
        "pab": ("PAB", C_GREEN, "s", (0, 5)),
        "cfgcache": ("CFG-Cache", C_YELLOW, "v", (8, -9)),
        "none": ("no caching", MUTED, "*", (0, -11)),
    }
    for m, (lab, c, mk, (dx, dy)) in marks.items():
        st = bstats.get(m)
        if st is None:
            continue
        x, y = st["speedup"], 100 * st["capture"]
        yci = ci95(100 * st["pp_capture"][boot_idx(len(st["pp_capture"]), seed=3)].mean(1))
        xci = boot_speedup(fg_v0, st["lat_groups"], seed=3)
        ax.errorbar([x], [y], yerr=yerr_of([y], [yci]), xerr=yerr_of([x], [xci]),
                    fmt="none", ecolor=c, zorder=4, **ERR_KW)
        ax.scatter([x], [y], marker=mk, s=26 if mk != "*" else 46, color=c, zorder=4)
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
                    fontsize=5.8, color=c)
    # fitted common frontier: capture = 1 - a*(speedup-1)^b across our tau sweep
    # + the published methods (the "none" control lies on the form exactly)
    fit_pts = ([(curves_v0[t]["sp"], curves_v0[t]["cap"], curves_v0[t]["pp"],
                 (curves_v0[t]["fg"], curves_v0[t]["cg"])) for t in taus]
               + [(bstats[m]["speedup"], bstats[m]["capture"], bstats[m]["pp_capture"],
                   (fg_v0, bstats[m]["lat_groups"]))
                  for m in ("pab", "cfgcache", "teacache") if bstats.get(m)])
    a_, b_, r2, (xg, lo, hi) = frontier_fit([p[0] for p in fit_pts],
                                            [p[1] for p in fit_pts],
                                            [p[2] for p in fit_pts],
                                            [p[3] for p in fit_pts])
    print(f"[f11b] frontier fit: capture(%) = {100*a_:.1f} {100*b_:+.1f} ln(speedup), "
          f"R^2 = {r2:.3f} ({len(fit_pts)} points)")
    ax.fill_between(xg, 100 * lo, 100 * hi, color=MUTED, alpha=0.15, lw=0, zorder=1)
    ax.plot(xg, 100 * (a_ + b_ * np.log(xg)), color=MUTED, lw=0.9,
            ls=(0, (4, 2.5)), zorder=2)
    ax.annotate(rf"fit: ${100*a_:.0f} {100*b_:+.1f}\,\ln x$" + "\n"
                + rf"$R^2 = {r2:.2f}$", (0.06, 0.06), xycoords="axes fraction",
                fontsize=5.8, color=MUTED, ha="left")
    st = bstats.get("easycache-ours")
    if st is not None:  # independent rerun of our default (x = validation marker)
        ax.scatter([st["speedup"]], [100 * st["capture"]], marker="x", s=26,
                   color=C_BLUE, linewidths=0.9, zorder=5)
        ax.annotate("rerun", (st["speedup"], 100 * st["capture"]), xytext=(-4, -10),
                    textcoords="offset points", fontsize=5.8, color=C_BLUE, ha="right")
    ax.set_xlabel(r"candidate speedup ($\times$)")
    ax.set_ylabel("search-gain capture (%)")
    ax.set_xlim(0.9, 2.75)
    ax.set_ylim(82, 103)
    fig.savefig(os.path.join(out, "f11b_baselines.pdf"))
    plt.close(fig)
    return a_, b_, r2


def f15_engines(rows_v0, bstats, out):
    """F15 (fig:engines): existing caching methods as exploration engines, on
    delivered-gain vs wall-clock axes at fixed N=8. Each published method's
    a1_* reproduction ranks the 8 candidates; the delivered video is the
    full-compute rollout of its argmax seed (commit), so cost = 8*Cc_m + Cf
    with Cc_m from that method's measured latencies (speedup vs the v0 full
    latency groups, f11b convention). References: full best-of-8 (dashed line
    + CI band) and the a1 no-caching control (protocol overhead, 114% cost).
    "ours" = the b1_gate_v0 grid itself (tab:main commit row). All points are
    directly measured selections on the 8-seed grid (filled markers)."""
    lat = latencies(rows_v0)
    Cf, Cc = lat["full"], lat["cached"]
    ref = {(p, s): r for (p, s, v), r in rows_v0.items() if v == "full" and s < 8}
    fg_v0 = lat_groups(rows_v0, "full")
    pa8 = grid(rows_v0, 8)
    # per-engine delivered score per prompt (commit = full rollout of the
    # engine's argmax seed); "ours" delivers from the v0 cached column itself
    deliver = {"ours": {p: pa8[p][0][int(np.argmax(pa8[p][1]))] for p in pa8}}
    cost = {"ours": 8 * Cc + Cf}
    for m in ("teacache", "pab", "cfgcache", "none"):
        if not (bstats or {}).get(m):
            print(f"  [f15] engine {m}: no a1 records -- skipped")
            continue
        rows_m = {}
        for f in sorted(glob.glob(os.path.join(RESULTS, f"a1_{m}", "scores_shard*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                rows_m[(r["prompt"], r["seed"], r["variant"])] = r
        by, latg_m = defaultdict(dict), defaultdict(list)
        for (p, s, v), r in rows_m.items():
            by[p][s] = r
            latg_m[p].append(r["latency"])
        deliver[m] = {}
        for p, seeds in by.items():
            if all(s in seeds and (p, s) in ref for s in range(8)):
                meth = np.array([seeds[s]["score"] for s in range(8)])
                deliver[m][p] = ref[(p, int(np.argmax(meth)))]["score"]
        common_l = sorted(set(fg_v0) & set(latg_m))
        sp = (np.mean([np.mean(fg_v0[p]) for p in common_l])
              / np.mean([np.mean(latg_m[p]) for p in common_l]))
        cost[m] = 8 * (Cf / sp) + Cf
    # paired capture (ratio of mean gains vs full best-of-8, tab:main
    # estimator) over each engine's prompts, with a JOINT prompt bootstrap so
    # the CI reflects the paired comparison, not raw between-prompt spread
    pts = {}
    for m, dv in deliver.items():
        ps = sorted(set(dv) & set(pa8))
        s_ = np.array([pa8[p][0][0] for p in ps])
        b_ = np.array([pa8[p][0].max() for p in ps])
        d_ = np.array([dv[p] for p in ps])
        gain = float((d_ - s_).mean())
        cap = 100 * gain / float((b_ - s_).mean())
        idm = boot_idx(len(ps), seed=5)
        cap_ci = ci95(100 * (d_[idm].mean(1) - s_[idm].mean(1))
                      / (b_[idm].mean(1) - s_[idm].mean(1)))
        pts[m] = (cap, cost[m], cap_ci)
        print(f"  [f15] engine {m:10s} gain {gain:+.3f}  capture {cap:5.1f}% "
              f"[{cap_ci[0]:.1f},{cap_ci[1]:.1f}]  cost {cost[m]:.0f}s "
              f"({100 * cost[m] / (8 * Cf):.0f}% of best-of-8)")
        want = EXPECTED2["engines"].get(m)
        if want:
            check(f"f15 engine {m} gain", gain, want[0])
            check(f"f15 engine {m} cost", cost[m], want[1], tol=3.0)

    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    emarks = {"ours": (r"ours ($\tau_c{=}0.10$)", C_BLUE, "o", (0, -13)),
              "teacache": ("TeaCache", C_VIOLET, "^", (26, -3)),
              "pab": ("PAB", C_GREEN, "s", (16, -12)),
              "cfgcache": ("CFG-Cache", C_YELLOW, "v", (-16, -13)),
              "none": ("no caching", MUTED, "*", (-6, -14))}
    ax.axhline(100, color=C_STRAT["bon_full"], lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.annotate(rf"full best-of-8: $100\%$ gain @ ${8 * Cf:.0f}$ s",
                (0.03, 100), xycoords=("axes fraction", "data"),
                xytext=(0, 2.5), textcoords="offset points", fontsize=5.8,
                color=C_STRAT["bon_full"], ha="left")
    for m, (lab, col, mk, (dx, dy)) in emarks.items():
        if m not in pts:
            continue
        cap, c, ci = pts[m]
        ax.errorbar([c], [cap], yerr=yerr_of([cap], [ci]), fmt="none",
                    ecolor=col, zorder=3, **ERR_KW)
        ax.scatter([c], [cap], marker=mk, s=30 if mk != "*" else 52, color=col,
                   zorder=4, edgecolors="white", linewidths=0.4)
        ax.annotate(lab, (c, cap), xytext=(dx, dy), textcoords="offset points",
                    fontsize=5.8, color=col, ha="center")
        ax.annotate(f"{100 * c / (8 * Cf):.0f}% cost", (c, cap),
                    xytext=(dx, dy - 6.5 if dy < 0 else dy + 6.5),
                    textcoords="offset points", fontsize=5.4, color=MUTED,
                    ha="center")
    # (no direction arrow: removed per user 2026-07-14; the caption carries
    # the up-left-is-better reading)
    ax.set_xlabel(r"wall-clock cost per prompt (s), $N{=}8$")
    ax.set_ylabel("delivered gain (% of best-of-8's)")
    ax.set_xlim(300, 665)
    ax.set_ylim(88, 104)
    fig.savefig(os.path.join(out, "f15_engines.pdf"))
    plt.close(fig)


# ------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=== data load (dedup by (prompt,seed,variant); latencies seeds 0-7) ===")
    rows = {tag: load_rows(tag) for tag in
            ["v0", "tau005", "tau020", "cog5b_v0", "cog5b_tau005", "cog5b_tau020",
             "wan14b", "wan14b_tau005", "wan14b_tau020",
             "wan14b_tau05fix", "wan14b_tau20fix",
             "ltx_v0", "ltx_ltx_tau002", "ltx_ltx_tau005",
             "wan22_5b", "hunyuan_v0"]}
    for tag, r in rows.items():
        print(f"  b1_gate_{tag:<15} {len(r):>5} records")

    # ---------------- F9: model scaling (first 8 seeds per model) -------------
    print("=== F9 model scaling (seeds 0-7 per model; six models, four families) ===")
    stats = {}
    for tag in ("v0", "ltx_v0", "wan22_5b", "cog5b_v0", "hunyuan_v0", "wan14b"):
        pa = grid(rows[tag], n_seeds=8)
        lat = latencies(rows[tag])
        pp = per_prompt(pa)
        ratios = (pp["rand"] - pp["regret"]) / pp["rand"]
        stats[tag] = dict(med=float(np.median(pp["rho"])),
                          p10=float(np.percentile(pp["rho"], 10)),
                          top1=float(pp["top1"].mean()),
                          capture=capture_mor(pa),
                          speedup=lat["full"] / lat["cached"],
                          n=len(pa), Cf=lat["full"], Cc=lat["cached"],
                          fg=lat_groups(rows[tag], "full"),
                          cg=lat_groups(rows[tag], "cached"),
                          pp=dict(rho=pp["rho"], top1=pp["top1"].astype(float),
                                  capt=ratios[np.isfinite(ratios)]))
        e, tol = EXPECTED2["scale"][tag], 0.011
        print(f"[{tag}] n={len(pa)} prompts  Cf={lat['full']:.1f}s Cc={lat['cached']:.1f}s")
        for k in ("med", "p10", "top1", "capture", "speedup"):
            check(f"{tag} {k}", stats[tag][k], e[k], tol=tol if k != "speedup" else 0.03)

    # ---------------- F10: tau calibration curves ------------------------------
    print("=== F10 tau curves (cached-only tau dirs joined with model full refs) ===")
    print("[f10] NOTE: the Wan-14B arms were enqueued at tau=0.005/0.020 (records "
          "carry these values), not the 1.3B sweep's 0.05/0.20 -- plotted as "
          "measured; tau=0.005 fires zero skips (bit-identical control, 1.01x)")
    curves = {}
    # wave-4: the 14B curve is completed by the tau05fix/tau20fix reruns at the
    # INTENDED 0.05/0.20 (full dial 0.005..0.20); LTX joins as the boundary
    # curve (its whole curve sits below the healthy band -- tab:taucal).
    for model, taus in [("v0", {0.05: "tau005", 0.10: None, 0.20: "tau020"}),
                        ("cog5b_v0", {0.05: "cog5b_tau005", 0.10: None, 0.20: "cog5b_tau020"}),
                        ("wan14b", {0.005: "wan14b_tau005", 0.02: "wan14b_tau020",
                                    0.05: "wan14b_tau05fix", 0.10: None,
                                    0.20: "wan14b_tau20fix"}),
                        ("ltx_v0", {0.02: "ltx_ltx_tau002", 0.05: "ltx_ltx_tau005",
                                    0.10: None})]:
        Cf = stats[model]["Cf"]
        fg = lat_groups(rows[model], "full")
        pts = {}
        for tau, tdir in taus.items():
            if tdir is None:
                pa, Cc = grid(rows[model], 8), stats[model]["Cc"]
                cg = lat_groups(rows[model], "cached")
            else:
                pa = grid(rows[tdir], 8, full_rows=rows[model])
                Cc = latencies(rows[tdir])["cached"]
                cg = lat_groups(rows[tdir], "cached")
            ppr = per_prompt(pa)
            ratios = (ppr["rand"] - ppr["regret"]) / ppr["rand"]
            pts[tau] = dict(cap=capture_mor(pa), sp=Cf / Cc,
                            pp=ratios[np.isfinite(ratios)], fg=fg, cg=cg,
                            n=len(pa))
            exp = EXPECTED2["tau"].get((model, tau))
            if exp is None:
                print(f"  [{model} tau={tau:g}] n={len(pa)} capture "
                      f"{pts[tau]['cap']:.3f} speedup {pts[tau]['sp']:.2f}x "
                      f"(new point, no paper expectation yet)")
            else:
                cap_e, sp_e = exp
                check(f"{model} tau={tau} capture", pts[tau]["cap"], cap_e)
                check(f"{model} tau={tau} speedup", pts[tau]["sp"], sp_e, tol=0.03)
            if model == "wan14b" and tdir is not None:
                print(f"  [wan14b tau={tau:g}] n={len(pa)} complete prompts "
                      f"(cached-only arms; full refs from b1_gate_wan14b)")
        curves[model] = pts

    # ---------------- F8: width scaling (16-seed grid) -------------------------
    print("=== F8 width scaling ===")
    f8_stats = f8_width(rows["v0"], args.out)

    # ---------------- F11: method comparison incl. E6 stack --------------------
    print("=== F11 method comparison (E6 stack from results/b1_stack_v0) ===")
    pa_v0 = grid(rows["v0"], 8)
    Cf, Cc = stats["v0"]["Cf"], stats["v0"]["Cc"]
    sim = simulate(pa_v0, Cf, Cc, Ns=(4, 8))
    for (N, strat), (g_e, c_e) in EXPECTED2["sim_f11"].items():
        check(f"F11 N={N} {strat} gain", sim[(N, strat)][0] - sim[(N, "single")][0], g_e)
        check(f"F11 N={N} {strat} cost", sim[(N, strat)][1], c_e, tol=1.0)
    stack = load_stack()
    common = [p for p in pa_v0 if p in stack]
    print(f"[stack] {len(stack)} records; {len(common)} joined with v0 grid")
    single8 = sim[(8, "single")][0]
    commit_scores = np.array([stack[p]["commit_score"] for p in common])
    e2e = np.array([stack[p]["timings"]["e2e_s"] for p in common])
    expl = np.array([stack[p]["timings"]["stage1_s"] + stack[p]["timings"]["preview_s"]
                     + stack[p]["timings"]["stage2_s"] for p in common])
    regret = np.array([pa_v0[p][0].max() - stack[p]["commit_score"] for p in common])
    rand = np.array([pa_v0[p][0].max() - pa_v0[p][0].mean() for p in common])
    survival = np.mean([int(np.argmax(pa_v0[p][0])) in stack[p]["survivors"] for p in common])
    st = dict(gain=float(commit_scores.mean()) - single8, e2e=float(e2e.mean()),
              expl=float(expl.mean()),
              capture_mor=float(np.mean(1 - regret / rand)),
              capture_rog=(float(commit_scores.mean()) - single8)
                          / (sim[(8, "bon_full")][0] - single8),
              survival=float(survival), med_regret=float(np.median(regret)))
    e = EXPECTED2["stack"]
    check("stack exploration cost (s)", st["expl"], e["expl"], tol=2.0)
    check("stack expl x vs bo8", sim[(8, "bon_full")][1] / st["expl"], e["ratio_bo8_expl"], tol=0.03)
    check("stack capture (mor)", st["capture_mor"], e["capture_mor"])
    check("stack true-best survival", st["survival"], e["survival"])
    check("stack median regret", st["med_regret"], e["med_regret"], tol=0.001)
    print(f"  stack gain vs single {st['gain']:+.3f} (ratio-of-gains capture "
          f"{100*st['capture_rog']:.1f}%; mor {100*st['capture_mor']:.1f}%)")
    print(f"  stack DELIVERED end-to-end {st['e2e']:.1f}s (incl. decode/scoring + "
          f"recommit) = {sim[(8, 'bon_full')][1] / st['e2e']:.2f}x vs best-of-8 -- "
          f"F11 plots this; the 175s/3.11x in sec:abl-stack is exploration-only")

    # F11 error bars: 95% prompt-bootstrap CIs on gains (per-prompt subset means,
    # resampled jointly with the single baseline) and costs (paired latency boot)
    pp_sim = simulate_pp(pa_v0, Ns=(4, 8))
    idxp = boot_idx(len(pp_sim[(4, "single")]), seed=4)

    def g_ci(N, strat):
        return ci95(pp_sim[(N, strat)][idxp].mean(1) - pp_sim[(N, "single")][idxp].mean(1))

    prompts_v0 = sorted(pa_v0)
    j = np.array([prompts_v0.index(p) for p in common])
    idxs = boot_idx(len(common), seed=4)
    single8_pp = pp_sim[(8, "single")]
    stack_gain_ci = ci95(commit_scores[idxs].mean(1) - single8_pp[j][idxs].mean(1))
    Cf_b, Cc_b = boot_costs(lat_groups(rows["v0"], "full"),
                            lat_groups(rows["v0"], "cached"), seed=4)
    gain_cis = [None, g_ci(4, "bon_full"), g_ci(8, "cached_commit"),
                g_ci(8, "cached_keep"), stack_gain_ci]
    cost_cis = [ci95(Cf_b), ci95(4 * Cf_b), ci95(8 * Cc_b + Cf_b), ci95(8 * Cc_b),
                ci95(e2e[idxs].mean(1))]

    # ---------------- F11b: published caching baselines (A-lite grids) ---------
    print("=== F11b caching-method baselines (a1_* dirs via a1_analysis.py) ===")
    from a1_analysis import all_stats as a1_all_stats
    bstats, _, a1_Cf = a1_all_stats()
    have_baselines = all(bstats.get(m) for m in ("teacache", "pab", "cfgcache"))
    if have_baselines:
        for m, s_ in bstats.items():
            if s_ is None:
                print(f"  [{m}] no records -- omitted")
                continue
            print(f"  [{m:14s}] n={s_['n_records']:3d} rec / {s_['n_prompts']:2d} prompts  "
                  f"{s_['speedup']:.2f}x  capture {100*s_['capture']:.1f}%  "
                  f"med rho {s_['rho_med']:.3f}")
        if bstats.get("easycache-ours"):  # rerun must reproduce the v0 gate point
            check("a1 easycache-ours capture", bstats["easycache-ours"]["capture"],
                  EXPECTED2["tau"][("v0", 0.10)][0])
            check("a1 easycache-ours speedup", bstats["easycache-ours"]["speedup"],
                  EXPECTED2["tau"][("v0", 0.10)][1], tol=0.03)
        if bstats.get("none"):  # determinism control
            check("a1 none speedup", bstats["none"]["speedup"], 1.0, tol=0.02)
            check("a1 none median rho", bstats["none"]["rho_med"], 1.0, tol=0.001)
    else:
        print("  baseline dirs incomplete -- f11b skipped")

    print("=== figures ===")
    style.SHOW_UNCERTAINTY = False  # Fig 19 is a dense scaling plot: CI bars clutter it (uncertainty in text + appendix CI table)
    f9_models(stats, args.out)
    style.SHOW_UNCERTAINTY = True   # restore for the remaining small point plots
    # f10_tau RETIRED: the standalone tau-calibration figure duplicated Fig 16
    # panel B (fig:scaling-master), which now carries the operating band + the
    # full Wan-14B dial. The function is kept below for reference, not generated.
    # f10_tau(curves, args.out)
    # f11_methods RETIRED from the paper (design-overhaul Lane B, D1): its unique
    # prune+cached (E6) point is now folded into fig:pareto (make_figs.f3_pareto);
    # the bars duplicated tab:main. The stack stats above are still computed for
    # the verification checks. Uncomment to regenerate the standalone bar figure.
    # f11_methods(sim, st, Cf, args.out, gain_cis=gain_cis, cost_cis=cost_cis)
    if have_baselines:
        f11b_baselines(curves["v0"], bstats, lat_groups(rows["v0"], "full"), args.out)
    print("=== F15 existing methods as engines (fig:engines) ===")
    f15_engines(rows["v0"], bstats if have_baselines else None, args.out)
    for f in sorted(glob.glob(os.path.join(args.out, "f1[015]*_*.pdf"))
                    + sorted(glob.glob(os.path.join(args.out, "f[89]_*.pdf")))):
        print(" wrote", f)


if __name__ == "__main__":
    main()
