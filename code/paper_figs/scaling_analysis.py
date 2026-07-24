"""Unified scaling-behavior analysis for the CachedSearch paper (wave-4, agent S).

Produces the consolidated "Scaling behavior" exhibits:

F13 f13_scaling_master.pdf : THE master operating-points figure -- every measured
      operating point of the paper on one capture-vs-speedup plane (panel a):
      Wan2.1-1.3B tau sweep (3), published caching methods (PAB/CFG-Cache/
      TeaCache + rerun + no-caching control, a1 grids), schedule lengths
      T in {25,50,100}, 720p resolution, six models at tau=0.10, CogVideoX-5B
      tau sweep, Wan2.1-14B tau arms. Color = model, marker = axis/method
      (entity-fixed); the tab:baselines frontier (fitted on the six tau/method
      points, sec:abl-baselines) is drawn with its 95% bootstrap band and
      EVALUATED on everything else (predict -> validate). Panel (b): search
      width N in {2..32} on the delivered capture-vs-cost plane (marker size
      grows with N).
F14 f14_schedule_scaling.pdf : redundancy-vs-schedule-length trend -- skip
      fraction and capture vs T with log-linear trend fits (a + b ln T, 95%
      prompt-bootstrap band; THREE schedule points -> reported as a trend, not
      a law) and the 720p point as the invariance contrast.

Printed analysis (quoted in sec:abl-scaling prose; grep "PROSE"):
  - per-axis R^2 of the SINGLE Wan-1.3B frontier: fit points (tau+methods),
    held-out Wan-1.3B configs (schedule/resolution/rerun), model axis;
  - per-family fits (CogVideoX; Wan-14B with the degenerate tau=0.005 caveat);
  - pooled-all-points fit (does ONE frontier fit every axis?);
  - capture-vs-skip-fraction plane (is reuse amount the sufficient statistic?);
  - schedule trend slopes with CIs; width saturating fit c_inf - beta/N.

Conventions (MUST match make_figs.py / make_figs2.py / bootstrap_ci.py):
- every number recomputed from RAW jsonl; records deduped by
  (prompt, seed, variant); scores AND latencies restricted to seeds 0-7;
- cross-model stats on the first 8 seeds; tau dirs are cached-only and join
  full references from the model's own v0 dir;
- capture = mean per-prompt gain-capture ratio (eq:capture, tab:tau/tab:scale
  convention) on the candidate plane; the width panel uses the
  ratio-of-mean-gains vs best-of-N convention (tab:main / f8);
- skip fraction = pooled skips / (skips + computes) over the cached records'
  method-internal counters (seeds 0-7);
- 95% CIs: prompt-level bootstrap, B=10^4 (CACHEDSEARCH_BOOT_B overrides).

Cross-checked against the paper's quoted values (EXPECTED3 below); mismatches
print a loud WARNING but still plot (res720 is a LIVE dir -- its row is
refreshed to this script's snapshot coverage whenever it moves).

Usage:  MALLOC_ARENA_MAX=2 \
        python code/paper_figs/scaling_analysis.py [--out paper/figs]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# canonical rcParams + palette + grammar helpers: paper_figs/style.py (Lane 0)
from style import (apply_style, BASE, C_BLUE, C_GREEN, C_RED, C_STRAT,
                   C_VIOLET, C_YELLOW, ERR_KW, INK, MUTED,
                   MODEL_COLORS, MODEL_MARKERS, MODEL_SHORT,
                   callout, direct_label, star_point)
from make_figs import (B_BOOT, OUT_DEFAULT, boot_idx, boot_speedup, check,
                       ci95, per_prompt, yerr_of)
from make_figs2 import (capture_mor, frontier_fit, grid, lat_groups, latencies,
                        load_rows, width_sim)

import matplotlib.pyplot as plt

apply_style()

# ------------------------------------------------------------- point registry
# key -> (model, axis, records dir tag, full-reference tag or None if self)
# axis in {tau, method, schedule, resolution, model}; methods come from the a1
# dirs via a1_analysis (their full refs are the b1_gate_v0 references).
CONFIGS = [
    # Wan2.1-1.3B: the reference backbone -- four axes measured on ONE model
    ("wan13_tau005",   "Wan2.1-1.3B", "tau",        "tau005",        "v0"),
    ("wan13_tau010",   "Wan2.1-1.3B", "tau",        "v0",            None),
    ("wan13_tau020",   "Wan2.1-1.3B", "tau",        "tau020",        "v0"),
    ("wan13_steps25",  "Wan2.1-1.3B", "schedule",   "steps25",       None),
    ("wan13_steps100", "Wan2.1-1.3B", "schedule",   "steps100",      None),
    ("wan13_res720",   "Wan2.1-1.3B", "resolution", "res720",        None),
    # cross-model grid at fixed tau = 0.10 (six models, four families)
    ("ltx",            "LTX-Video-2B",     "model", "ltx_v0",        None),
    ("wan22",          "Wan2.2-TI2V-5B",   "model", "wan22_5b",      None),
    ("cog_tau010",     "CogVideoX-5B",     "model", "cog5b_v0",      None),
    ("hunyuan",        "HunyuanVideo-13B", "model", "hunyuan_v0",    None),
    ("wan14b_tau010",  "Wan2.1-14B",       "model", "wan14b",        None),
    # per-model tau sweeps (CogVideoX; Wan-14B arms ran at 0.005/0.02 -- the
    # conservative end; 0.005 fires zero skips = degenerate 100%-capture control)
    ("cog_tau005",     "CogVideoX-5B", "tau", "cog5b_tau005",   "cog5b_v0"),
    ("cog_tau020",     "CogVideoX-5B", "tau", "cog5b_tau020",   "cog5b_v0"),
    ("wan14b_tau0005", "Wan2.1-14B",   "tau", "wan14b_tau005",  "wan14b"),
    ("wan14b_tau002",  "Wan2.1-14B",   "tau", "wan14b_tau020",  "wan14b"),
    # off-family tau sweeps (so every family shows its own slope in Fig 16
    # panel B; same 50-prompt x 8-seed coverage as the tau=0.10 points above)
    # Wan2.2 uses the corrected reruns (the plain wan22_tau005/020 dirs are the
    # invalid missing-native-res-flag runs: capture ~0, speedup 6-9x).
    ("wan22_tau005",   "Wan2.2-TI2V-5B",   "tau", "wan22_tau005fix",    "wan22_5b"),
    ("wan22_tau020",   "Wan2.2-TI2V-5B",   "tau", "wan22_tau020fix",    "wan22_5b"),
    ("hunyuan_tau005", "HunyuanVideo-13B", "tau", "hunyuan_hun_tau005", "hunyuan_v0"),
    ("hunyuan_tau020", "HunyuanVideo-13B", "tau", "hunyuan_hun_tau020", "hunyuan_v0"),
    ("ltx_tau002",     "LTX-Video-2B",     "tau", "ltx_ltx_tau002",     "ltx_v0"),
    ("ltx_tau005",     "LTX-Video-2B",     "tau", "ltx_ltx_tau005",     "ltx_v0"),
    # Wan-14B full dial: the tau=0.05/0.20 fix reruns complete the arc so Fig 16
    # panel B subsumes the retired standalone tau-calibration figure.
    ("wan14b_tau05",   "Wan2.1-14B",       "tau", "wan14b_tau05fix",    "wan14b"),
    ("wan14b_tau20",   "Wan2.1-14B",       "tau", "wan14b_tau20fix",    "wan14b"),
]
TAU_OF = {"wan13_tau005": 0.05, "wan13_tau010": 0.10, "wan13_tau020": 0.20,
          "cog_tau005": 0.05, "cog_tau010": 0.10, "cog_tau020": 0.20,
          "wan14b_tau0005": 0.005, "wan14b_tau002": 0.02, "wan14b_tau010": 0.10,
          "wan13_steps25": 0.10, "wan13_steps100": 0.10, "wan13_res720": 0.10,
          "ltx": 0.10, "wan22": 0.10, "hunyuan": 0.10}
T_OF = {"wan13_steps25": 25, "wan13_tau010": 50, "wan13_steps100": 100}

# expected values currently quoted in the paper (tab:tau / tab:scale /
# tab:schedule / tab:baselines / f10 caption); res720 is a LIVE dir -- its
# expectation is the last frozen quote and a drift there is REPORTED, not fatal.
EXPECTED3 = {
    "wan13_tau005": (0.936, 1.58), "wan13_tau010": (0.901, 1.97),
    "wan13_tau020": (0.883, 2.41),
    "wan13_steps25": (0.958, 1.41), "wan13_steps100": (0.825, 2.80),
    "wan13_res720": (0.908, 2.05),   # refreshed at final coverage n=50 (wave-4)
    "ltx": (0.676, 2.63), "wan22": (0.860, 2.05), "cog_tau010": (0.752, 2.06),
    "hunyuan": (0.799, 2.19), "wan14b_tau010": (0.875, 2.05),
    "cog_tau005": (0.859, 1.78), "cog_tau020": (0.647, 2.65),
    "wan14b_tau0005": (1.000, 1.01), "wan14b_tau002": (0.921, 1.21),
    "wan22_tau005": (0.893, 1.63), "wan22_tau020": (0.831, 2.37),
    "hunyuan_tau005": (0.851, 1.77), "hunyuan_tau020": (0.797, 2.48),
    "ltx_tau002": (0.796, 1.71), "ltx_tau005": (0.716, 2.19),
    "wan14b_tau05": (0.880, 1.59), "wan14b_tau20": (0.847, 2.52),
    "a1_pab": (0.995, 1.29), "a1_cfgcache": (0.996, 1.38),
    "a1_teacache": (0.932, 1.84), "a1_easycache-ours": (0.901, 1.96),
    "width_commit": {2: 0.911, 4: 0.909, 8: 0.922, 16: 0.957, 32: 0.952},
    "width_cost":   {4: 0.758, 8: 0.633, 16: 0.571, 32: 0.539},
}


# ------------------------------------------------------------------ data layer
def skip_fraction(rows):
    """Pooled skip fraction over the cached records' internal counters
    (seeds 0-7): sum(skips) / sum(skips + computes) across all branches."""
    sk = tot = 0
    for (p, s, v), r in rows.items():
        if v == "cached" and s < 8 and r.get("stats"):
            for b in r["stats"].values():
                sk += b["skips"]
                tot += b["skips"] + b["computes"]
    return sk / tot if tot else float("nan")


def load_points():
    """-> {key: dict(model, axis, cap, sp, skip, n, pp, fg, cg)} for CONFIGS."""
    tags = sorted({c[3] for c in CONFIGS} | {c[4] for c in CONFIGS if c[4]})
    rows = {t: load_rows(t) for t in tags}
    pts = {}
    for key, model, axis, tag, full_src in CONFIGS:
        r = rows[tag]
        if full_src is None:
            pa = grid(r, 8)
            fg = lat_groups(r, "full")
            Cf = latencies(r)["full"]
        else:
            pa = grid(r, 8, full_rows=rows[full_src])
            fg = lat_groups(rows[full_src], "full")
            Cf = latencies(rows[full_src])["full"]
        cg = lat_groups(r, "cached")
        Cc = latencies(r)["cached"]
        pp = per_prompt(pa)
        ratios = (pp["rand"] - pp["regret"]) / pp["rand"]
        pts[key] = dict(model=model, axis=axis, cap=capture_mor(pa), sp=Cf / Cc,
                        skip=skip_fraction(r), n=len(pa), Cf=Cf, Cc=Cc,
                        pp=ratios[np.isfinite(ratios)], fg=fg, cg=cg)
        if key in EXPECTED3:
            cap_e, sp_e = EXPECTED3[key]
            check(f"{key} capture", pts[key]["cap"], cap_e)
            check(f"{key} speedup", pts[key]["sp"], sp_e, tol=0.03)
    return pts, rows


def load_methods(fg_v0):
    """a1 published-method points (capture/speedup vs the v0 full refs)."""
    from a1_analysis import all_stats
    bstats, _, _ = all_stats()
    out = {}
    for m in ("pab", "cfgcache", "teacache", "easycache-ours", "none"):
        st = bstats.get(m)
        if st is None:
            print(f"  [a1_{m}] no records -- omitted")
            continue
        out[m] = dict(model="Wan2.1-1.3B", axis="method", cap=st["capture"],
                      sp=st["speedup"], skip=float("nan"), n=st["n_prompts"],
                      pp=st["pp_capture"], fg=fg_v0, cg=st["lat_groups"])
        if f"a1_{m}" in EXPECTED3:
            cap_e, sp_e = EXPECTED3[f"a1_{m}"]
            check(f"a1_{m} capture", st["capture"], cap_e)
            check(f"a1_{m} speedup", st["speedup"], sp_e, tol=0.03)
    return out


# ------------------------------------------------------------------ fits
def lsq_loglin(sp, cap):
    """capture = alpha + beta ln(speedup); -> alpha, beta, R^2 (fractions)."""
    sp, cap = np.asarray(sp, float), np.asarray(cap, float)
    A = np.vstack([np.ones_like(sp), np.log(sp)]).T
    alpha, beta = np.linalg.lstsq(A, cap, rcond=None)[0]
    resid = cap - (alpha + beta * np.log(sp))
    ss_tot = ((cap - cap.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return float(alpha), float(beta), float(r2)


def r2_vs_curve(sp, cap, alpha, beta):
    """R^2 of a FIXED curve alpha + beta ln(s) on points (no refit); for a
    single point this is undefined -- returns (r2 or nan, residuals)."""
    sp, cap = np.asarray(sp, float), np.asarray(cap, float)
    resid = cap - (alpha + beta * np.log(sp))
    if len(sp) < 2:
        return float("nan"), resid
    ss_tot = ((cap - cap.mean()) ** 2).sum()
    return float(1.0 - (resid ** 2).sum() / ss_tot), resid


def trend_fit_T(pts, keys, field, scale=100.0, seed=11):
    """Log-linear trend y = a + b ln(T) over the three schedule points, with a
    95% prompt-bootstrap band (resample prompts within each T independently,
    recompute the three points, refit). field: 'pp' (capture) or 'skip_pp'.
    -> a, b, (b_lo, b_hi), (Tg, lo, hi) band."""
    Ts = np.array([T_OF[k] for k in keys], float)
    ys = np.array([scale * np.mean(pts[k][field]) for k in keys])
    A = np.vstack([np.ones_like(Ts), np.log(Ts)]).T
    a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    rng = np.random.default_rng(seed)
    Tg = np.linspace(Ts.min() * 0.9, Ts.max() * 1.1, 100)
    curves = np.empty((B_BOOT, len(Tg)))
    bs = np.empty(B_BOOT)
    draws = []
    for k in keys:
        v = pts[k][field]
        idx = rng.integers(0, len(v), size=(B_BOOT, len(v)))
        draws.append(scale * v[idx].mean(1))
    for i in range(B_BOOT):
        yi = np.array([d[i] for d in draws])
        ai, bi = np.linalg.lstsq(A, yi, rcond=None)[0]
        curves[i] = ai + bi * np.log(Tg)
        bs[i] = bi
    lo, hi = np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0)
    return float(a), float(b), ci95(bs), (Tg, lo, hi)


def skip_pp(rows):
    """Per-prompt skip fraction (cached records, seeds 0-7)."""
    from collections import defaultdict
    sk, tot = defaultdict(float), defaultdict(float)
    for (p, s, v), r in rows.items():
        if v == "cached" and s < 8 and r.get("stats"):
            for b in r["stats"].values():
                sk[p] += b["skips"]
                tot[p] += b["skips"] + b["computes"]
    return np.array([sk[p] / tot[p] for p in sorted(tot)])


# ------------------------------------------------------------------ figures
def f13_master(pts, meth, frontier, cogfit, width, out):
    """SPLIT master figure (design-overhaul Lane B). Panel A: the WITHIN-Wan2.1-1.3B
    frontier -- tau sweep + published methods + schedule + resolution, ONE
    log-linear fit + 95% band, <=6 direct labels (the rest faint, no labels).
    Panel B: ACROSS families -- per-family short fits, entity-fixed
    MODEL_COLORS/MODEL_MARKERS, <=2 labels each. The exhaustive per-point table
    is sections/x_operating_points.tex; the width panel moved to
    fig:width-scaling (f8)."""
    (alpha, beta, r2, (xg, blo, bhi)) = frontier
    ca, cb, cr2 = cogfit
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 2.9))

    P = {}
    for k, d in {**pts, **{f"a1_{m}": d for m, d in meth.items()}}.items():
        P[k] = (d["sp"], 100 * d["cap"], d)

    def draw(ax, x, y, d, color, marker, mfc, seed, ms=4.4, z=4):
        yci = ci95(100 * d["pp"][boot_idx(len(d["pp"]), seed=seed)].mean(1))
        xci = boot_speedup(d["fg"], d["cg"], seed=seed)
        ax.errorbar([x], [y], yerr=yerr_of([y], [yci]), xerr=yerr_of([x], [xci]),
                    fmt=marker, color=color, ms=ms, mfc=mfc, zorder=z, **ERR_KW)

    # ============================ Panel A: within Wan-1.3B ====================
    axA.fill_between(xg, 100 * blo, 100 * bhi, color=MUTED, alpha=0.16, lw=0, zorder=1)
    axA.plot(xg, 100 * (alpha + beta * np.log(xg)), color=MUTED, lw=1.1,
             ls=(0, (4, 2.5)), zorder=2)
    axA.annotate(rf"one frontier: ${100*alpha:.0f} {100*beta:+.1f}\,\ln x$"
                 + "\n" + rf"($R^2\,{{=}}\,{r2:.2f}$; held-out $R^2\,{{=}}\,0.90$)",
                 (0.97, 0.05), xycoords="axes fraction", fontsize=6.2,
                 color=MUTED, ha="right", va="bottom")
    # faint, unlabeled: the published caching methods + rerun + zero-skip control
    for m in ("pab", "cfgcache", "teacache"):
        x, y, _ = P[f"a1_{m}"]
        axA.scatter([x], [y], marker="s", s=15, color=C_BLUE, alpha=0.32, zorder=3)
    x, y, _ = P["a1_easycache-ours"]
    axA.scatter([x], [y], marker="x", s=20, color=C_BLUE, alpha=0.45, linewidths=0.8, zorder=3)
    x, y, _ = P["a1_none"]
    axA.scatter([x], [y], marker="*", s=34, color=MUTED, alpha=0.55, zorder=3)
    axA.annotate("caching methods\n(PAB/CFG-Cache/TeaCache),\nrerun, zero-skip control",
                 (0.045, 0.06), xycoords="axes fraction", fontsize=5.6,
                 color=MUTED, ha="left", va="bottom")
    # primary (labeled) points: tau sweep (connected), schedule, resolution
    tau13 = ["wan13_tau005", "wan13_tau010", "wan13_tau020"]
    axA.plot([P[k][0] for k in tau13], [P[k][1] for k in tau13], color=C_BLUE,
             lw=1.2, zorder=4)
    for i, k in enumerate(tau13):
        x, y, d = P[k]
        draw(axA, x, y, d, C_BLUE, "o", C_BLUE, 20 + i)
    for k, mk, s_ in (("wan13_steps25", "P", 40), ("wan13_steps100", "P", 41),
                      ("wan13_res720", "D", 42)):
        x, y, d = P[k]
        draw(axA, x, y, d, C_BLUE, mk, "white", s_)
    labA = {"wan13_tau005": (r"$\tau{=}.05$", (-6, 6), "right"),
            "wan13_tau010": (r"$\tau{=}.10$ (default)", (-6, -8), "right"),
            "wan13_tau020": (r"$\tau{=}.20$", (5, 5), "left"),
            "wan13_steps25": (r"$T{=}25$", (4, 6), "left"),
            "wan13_steps100": (r"$T{=}100$", (4, 4), "left"),
            "wan13_res720": ("720p", (6, -3), "left")}
    for k, (t, (dx, dy), ha) in labA.items():
        x, y, _ = P[k]
        direct_label(axA, x, y, t, C_BLUE, dx=dx, dy=dy, ha=ha, fontsize=6.2)
    axA.set_title("within Wan2.1-1.3B: one backbone, four axes", fontsize=7.2, pad=3)
    axA.set_xlabel(r"candidate speedup ($\times$)")
    axA.set_ylabel("search-gain capture (%)")
    axA.set_xlim(0.92, 2.95)
    axA.set_ylim(78, 102)

    # ============================ Panel B: across families ===================
    # Every family shows its own measured tau-sweep curve (same 50-prompt x
    # 8-seed coverage as the tau=0.10 grid); no abstract fit overlays -- the
    # measured slopes carry the "architecture sets slope" point directly.
    MB = {  # model-color key -> (tau-ascending key list, linestyle)
        "Wan2.1-1.3B":      (["wan13_tau005", "wan13_tau010", "wan13_tau020"], "-"),
        "Wan2.2-5B":        (["wan22_tau005", "wan22", "wan22_tau020"], "-"),
        "Wan2.1-14B":       (["wan14b_tau0005", "wan14b_tau002", "wan14b_tau05", "wan14b_tau010", "wan14b_tau20"], (0, (4, 2))),
        "CogVideoX-5B":     (["cog_tau005", "cog_tau010", "cog_tau020"], "-"),
        "HunyuanVideo-13B": (["hunyuan_tau005", "hunyuan", "hunyuan_tau020"], "-"),
        "LTX-Video-2B":     (["ltx_tau002", "ltx_tau005", "ltx"], "-"),
    }
    for mkey, (keys, ls) in MB.items():
        col = MODEL_COLORS[mkey]
        axB.plot([P[k][0] for k in keys], [P[k][1] for k in keys], color=col,
                 lw=1.1, ls=ls, alpha=0.9, zorder=3)
        for i, k in enumerate(keys):
            x, y, d = P[k]
            draw(axB, x, y, d, col, MODEL_MARKERS[mkey], col, 80 + i, ms=4.0)
    # Wan-1.3B's most aggressive operating point = the calibration band (folded
    # in from the retired standalone tau-calibration figure): to calibrate a new
    # architecture, sweep tau until its curve reaches this line.
    wan_band = min(P[k][1] for k in ["wan13_tau005", "wan13_tau010", "wan13_tau020"])
    axB.axhline(wan_band, color=BASE, lw=0.7, ls=(0, (4, 3)), zorder=1)
    axB.annotate("Wan-1.3B operating band", xy=(0.95, wan_band), xytext=(2, 3),
                 textcoords="offset points", fontsize=5.6, color=MUTED,
                 ha="left", va="bottom")
    # <=2 labels per family (entity-fixed colors carry the identity)
    # Model identity now comes from the shared bottom legend, so panel B keeps
    # only the two informative annotations (slope and the over-driven LTX skip).
    labB = {
        "cog_tau020": (r"$2.7\times$ steeper", MODEL_COLORS["CogVideoX-5B"], (3, -8), "left"),
        "ltx": ("64% skip", MODEL_COLORS["LTX-Video-2B"], (5, 9), "left"),
    }
    for k, (t, c, (dx, dy), ha) in labB.items():
        x, y, _ = P[k]
        direct_label(axB, x, y, t, c, dx=dx, dy=dy, ha=ha, fontsize=6.0)
    axB.annotate(r"each family its own frontier ($R^2{=}{-}0.86$ across models)",
                 (0.5, 0.03), xycoords="axes fraction", fontsize=6.0, color=INK,
                 ha="center", va="bottom")
    axB.set_title("across families: architecture sets level and slope", fontsize=7.2, pad=3)
    axB.set_xlabel(r"candidate speedup ($\times$)")
    axB.set_ylabel("search-gain capture (%)")
    axB.set_xlim(0.92, 2.95)
    axB.set_ylim(57, 104.5)

    # Shared bottom color/marker key: same-family models share a hue (Wan =
    # blue ramp, light 5B -> mid 1.3B -> dark 14B); each off-family port owns
    # one accent + marker. Lets the reader decode every line/point in panel B.
    from matplotlib.lines import Line2D
    leg_order = ["Wan2.1-1.3B", "Wan2.2-5B", "Wan2.1-14B",
                 "CogVideoX-5B", "HunyuanVideo-13B", "LTX-Video-2B"]
    leg_handles = [Line2D([], [], marker=MODEL_MARKERS[n], color=MODEL_COLORS[n],
                          ls="", mfc=MODEL_COLORS[n], mec=MODEL_COLORS[n], ms=4.5,
                          label=MODEL_SHORT[n]) for n in leg_order]
    fig.legend(handles=leg_handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=6, frameon=False, fontsize=6.4, handletextpad=0.4,
               columnspacing=1.4)
    fig.subplots_adjust(wspace=0.28, left=0.075, right=0.99, top=0.90, bottom=0.17)
    fig.savefig(os.path.join(out, "f13_scaling_master.pdf"))
    plt.close(fig)


def f14_schedule(pts, skfit, capfit, out):
    """Skip fraction and capture vs schedule length T (log2 x), with the
    log-linear trend fits + 95% bootstrap bands and 720p as the contrast."""
    keys = ["wan13_steps25", "wan13_tau010", "wan13_steps100"]
    Ts = [T_OF[k] for k in keys]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(4.15, 1.95))
    for a in (ax, ax2):
        a.set_xscale("log", base=2)
        a.set_xticks(Ts)
        a.set_xticklabels([str(t) for t in Ts])
        a.set_xlabel("denoising steps $T$")
        a.set_xlim(21, 122)
        a.minorticks_off()

    # panel 1: skip fraction
    a1, b1, (b1lo, b1hi), (Tg, lo, hi) = skfit
    ax.fill_between(Tg, lo, hi, color=C_BLUE, alpha=0.13, lw=0, zorder=1)
    ax.plot(Tg, a1 + b1 * np.log(Tg), color=C_BLUE, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
    ys = [100 * np.mean(pts[k]["skip_pp"]) for k in keys]
    ycis = [ci95(100 * pts[k]["skip_pp"][boot_idx(len(pts[k]["skip_pp"]), seed=13)].mean(1))
            for k in keys]
    ax.errorbar(Ts, ys, yerr=yerr_of(ys, ycis), fmt="o", color=C_BLUE, ms=4, zorder=4,
                **ERR_KW)
    for t, y in zip(Ts, ys):
        dx, dy = (2, -9) if t == 50 else (-1, 6)   # 720p marker sits just above
        ax.annotate(f"{y:.0f}%", (t, y), xytext=(dx, dy), textcoords="offset points",
                    ha="right", fontsize=5.8, color=C_BLUE)
    y720 = 100 * np.mean(pts["wan13_res720"]["skip_pp"])
    ax.errorbar([50], [y720], fmt="D", color=C_BLUE, mfc="white", ms=3.6, zorder=4)
    ax.annotate(f"720p: {y720:.0f}%", (50, y720), xytext=(6, -5),
                textcoords="offset points", fontsize=5.6, color=C_BLUE, ha="left")
    ax.annotate(rf"trend: ${a1:.0f} {b1:+.1f}\,\ln T$"
                + f"\n(slope CI [{b1lo:+.1f}, {b1hi:+.1f}])",
                (0.05, 0.86), xycoords="axes fraction", fontsize=5.6, color=MUTED)
    ax.set_ylabel("steps skipped (%)")
    ax.set_ylim(24, 78)

    # panel 2: capture
    a2, b2, (b2lo, b2hi), (Tg2, lo2, hi2) = capfit
    ax2.fill_between(Tg2, lo2, hi2, color=C_STRAT["cached_commit"], alpha=0.13,
                     lw=0, zorder=1)
    ax2.plot(Tg2, a2 + b2 * np.log(Tg2), color=C_STRAT["cached_commit"], lw=0.9,
             ls=(0, (4, 2.5)), zorder=2)
    ys2 = [100 * np.mean(pts[k]["pp"]) for k in keys]
    ycis2 = [ci95(100 * pts[k]["pp"][boot_idx(len(pts[k]["pp"]), seed=14)].mean(1))
             for k in keys]
    ax2.errorbar(Ts, ys2, yerr=yerr_of(ys2, ycis2), fmt="o",
                 color=C_STRAT["cached_commit"], ms=4, zorder=4, **ERR_KW)
    for t, y in zip(Ts, ys2):
        dx, dy = (-2, -9) if t == 50 else (3, 5)   # 720p label sits to the right
        ax2.annotate(f"{y:.1f}", (t, y), xytext=(dx, dy), textcoords="offset points",
                     ha="left", fontsize=5.8, color=C_STRAT["cached_commit"])
    y720c = 100 * np.mean(pts["wan13_res720"]["pp"])
    ax2.errorbar([50], [y720c], fmt="D", color=C_STRAT["cached_commit"], mfc="white",
                 ms=3.6, zorder=4)
    ax2.annotate(f"720p: {y720c:.1f}", (50, y720c), xytext=(6, 2),
                 textcoords="offset points", fontsize=5.6,
                 color=C_STRAT["cached_commit"], ha="left")
    ax2.annotate(rf"trend: ${a2:.0f} {b2:.1f}\,\ln T$"
                 + f"\n(slope CI [{b2lo:.1f}, {b2hi:.1f}])",
                 (0.05, 0.13), xycoords="axes fraction", fontsize=5.6, color=MUTED)
    ax2.set_ylabel("gain capture (%)")
    ax2.set_ylim(70, 102)
    fig.subplots_adjust(wspace=0.42)
    fig.savefig(os.path.join(out, "f14_schedule_scaling.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=== operating points (dedup; seeds 0-7 scores+latencies) ===")
    pts, rows = load_points()
    meth = load_methods(lat_groups(rows["v0"], "full"))
    for k, d in list(pts.items()) + [(f"a1_{m}", d) for m, d in meth.items()]:
        print(f"  {k:<18} n={d['n']:3d}  {d['sp']:5.2f}x  capture {100*d['cap']:5.1f}%"
              f"  skip {100*d['skip']:5.1f}%" if np.isfinite(d["skip"]) else
              f"  {k:<18} n={d['n']:3d}  {d['sp']:5.2f}x  capture {100*d['cap']:5.1f}%")

    # per-prompt skip fractions for the schedule trend
    for k, tag in (("wan13_steps25", "steps25"), ("wan13_tau010", "v0"),
                   ("wan13_steps100", "steps100"), ("wan13_res720", "res720")):
        pts[k]["skip_pp"] = skip_pp(rows[tag])

    # ---------------- the frontier (fitted on tau + methods ONLY, = f11b) ----
    print("=== frontier fit (six tau/method points, = fig:baselines) ===")
    fit_keys = ["wan13_tau005", "wan13_tau010", "wan13_tau020"]
    fit_pts = ([(pts[k]["sp"], pts[k]["cap"], pts[k]["pp"],
                 (pts[k]["fg"], pts[k]["cg"])) for k in fit_keys]
               + [(meth[m]["sp"], meth[m]["cap"], meth[m]["pp"],
                   (meth[m]["fg"], meth[m]["cg"]))
                  for m in ("pab", "cfgcache", "teacache")])
    alpha, beta, r2, band = frontier_fit([p[0] for p in fit_pts],
                                         [p[1] for p in fit_pts],
                                         [p[2] for p in fit_pts],
                                         [p[3] for p in fit_pts])
    print(f"[PROSE] frontier: capture(%) = {100*alpha:.1f} {100*beta:+.1f} ln(sp),"
          f" R^2 = {r2:.3f} (fit points: 3 taus + PAB/CFG-Cache/TeaCache)")

    # ---------------- per-axis evaluation of the ONE frontier -----------------
    print("=== per-axis R^2 against the FIXED Wan-1.3B frontier ===")
    axes_eval = {
        "tau (Wan-1.3B, in-fit)": ["wan13_tau005", "wan13_tau010", "wan13_tau020"],
        "method (in-fit + rerun)": ["a1_pab", "a1_cfgcache", "a1_teacache",
                                    "a1_easycache-ours"],
        "schedule T=25/50/100 (held out)": ["wan13_steps25", "wan13_tau010",
                                            "wan13_steps100"],
        "resolution 720p (held out)": ["wan13_res720"],
        "ALL Wan-1.3B held-out": ["wan13_steps25", "wan13_steps100",
                                  "wan13_res720", "a1_easycache-ours"],
        "model axis (6 models @ tau=.10)": ["wan13_tau010", "ltx", "wan22",
                                            "cog_tau010", "hunyuan", "wan14b_tau010"],
    }
    all_p = {**pts, **{f"a1_{m}": d for m, d in meth.items()}}
    for name, keys in axes_eval.items():
        sp = [all_p[k]["sp"] for k in keys]
        cap = [100 * all_p[k]["cap"] for k in keys]
        r2a, resid = r2_vs_curve(sp, cap, 100 * alpha, 100 * beta)
        rtxt = " ".join(f"{r:+.1f}" for r in resid)
        print(f"[PROSE] {name:<36} R^2 = {r2a:6.3f}  residuals(pts): {rtxt}")

    # pooled: does ONE refit frontier cover every axis/model?
    pool_keys = [k for k in all_p if k not in ("a1_none", "wan14b_tau0005")]
    a_all, b_all, r2_all = lsq_loglin([all_p[k]["sp"] for k in pool_keys],
                                      [100 * all_p[k]["cap"] for k in pool_keys])
    print(f"[PROSE] pooled ALL non-degenerate points ({len(pool_keys)}): refit "
          f"{a_all:.1f} {b_all:+.1f} ln(sp), R^2 = {r2_all:.3f}")
    wan13_keys = [k for k in pool_keys if all_p[k]["model"] == "Wan2.1-1.3B"]
    a13, b13, r13 = lsq_loglin([all_p[k]["sp"] for k in wan13_keys],
                               [100 * all_p[k]["cap"] for k in wan13_keys])
    print(f"[PROSE] pooled Wan-1.3B, all four axes ({len(wan13_keys)} pts): "
          f"{a13:.1f} {b13:+.1f} ln(sp), R^2 = {r13:.3f}")

    # per-family fits
    ca, cb, cr2 = lsq_loglin([pts[k]["sp"] for k in ("cog_tau005", "cog_tau010", "cog_tau020")],
                             [pts[k]["cap"] for k in ("cog_tau005", "cog_tau010", "cog_tau020")])
    print(f"[PROSE] CogVideoX own fit: {100*ca:.1f} {100*cb:+.1f} ln(sp), R^2 = {cr2:.3f}"
          f" (slope {100*cb:.1f} vs Wan's {100*beta:.1f} -- steeper)")
    w14 = ["wan14b_tau0005", "wan14b_tau002", "wan14b_tau010"]
    wa, wb, wr2 = lsq_loglin([pts[k]["sp"] for k in w14], [pts[k]["cap"] for k in w14])
    print(f"[PROSE] Wan-14B 3-pt fit (incl. degenerate tau=.005 control): "
          f"{100*wa:.1f} {100*wb:+.1f} ln(sp), R^2 = {wr2:.3f} -- caveat only")

    # capture-vs-skip plane (is reuse amount the sufficient statistic?)
    skip_keys = [k for k in pts if np.isfinite(pts[k]["skip"]) and k != "wan14b_tau0005"]
    wan13_sk = [k for k in skip_keys if pts[k]["model"] == "Wan2.1-1.3B"]
    for name, keys in (("skip-plane, Wan-1.3B only", wan13_sk),
                       ("skip-plane, ALL models", skip_keys)):
        sk = np.array([100 * pts[k]["skip"] for k in keys])
        cp = np.array([100 * pts[k]["cap"] for k in keys])
        A = np.vstack([np.ones_like(sk), sk]).T
        a_, b_ = np.linalg.lstsq(A, cp, rcond=None)[0]
        r2s = 1 - ((cp - (a_ + b_ * sk)) ** 2).sum() / ((cp - cp.mean()) ** 2).sum()
        print(f"[PROSE] {name:<28} capture = {a_:.1f} {b_:+.2f} skip%,"
              f" R^2 = {r2s:.3f} ({len(keys)} pts)")

    # ---------------- schedule trend fits -------------------------------------
    print("=== schedule trend (three points -> trend, not law) ===")
    sk_keys = ["wan13_steps25", "wan13_tau010", "wan13_steps100"]
    skfit = trend_fit_T(pts, sk_keys, "skip_pp", seed=11)
    capfit = trend_fit_T(pts, sk_keys, "pp", seed=12)
    print(f"[PROSE] skip%(T)    = {skfit[0]:.1f} {skfit[1]:+.1f} ln T,"
          f" slope CI [{skfit[2][0]:+.1f}, {skfit[2][1]:+.1f}] pts per ln-step")
    print(f"[PROSE] capture%(T) = {capfit[0]:.1f} {capfit[1]:+.1f} ln T,"
          f" slope CI [{capfit[2][0]:+.1f}, {capfit[2][1]:+.1f}]")
    print(f"[PROSE] res720 contrast: skip {100*np.mean(pts['wan13_res720']['skip_pp']):.1f}%"
          f" vs 50-step default {100*np.mean(pts['wan13_tau010']['skip_pp']):.1f}%;"
          f" capture {100*pts['wan13_res720']['cap']:.1f}% (n={pts['wan13_res720']['n']})")

    # ---------------- width: delivered plane + saturation -----------------------
    print("=== width (16-seed pool N<=16; N=32 = full 32-seed grid) ===")
    pa16 = grid(rows["v0"], 16)
    pa32 = grid(rows["v0"], 32)
    lat = latencies(rows["v0"])
    Cf, Cc = lat["full"], lat["cached"]
    Ns = (2, 4, 8, 16, 32)
    sim = width_sim(pa16, (2, 4, 8, 16))
    sim[32] = width_sim(pa32, (32,))[32]
    cap_c, cost_ratio, cis = [], [], {}
    idx = boot_idx(len(pa16), seed=21)
    for N in Ns:
        pp = sim[N]["pp"]
        c = 100 * (sim[N]["commit"] - sim[N]["single"]) / (sim[N]["bon"] - sim[N]["single"])
        cap_c.append(c)
        cost_ratio.append(100 * (N * Cc + Cf) / (N * Cf))
        i = idx if N < 32 else boot_idx(len(pa32), seed=21)
        Sm, Bm = pp["single"][i].mean(1), pp["bon"][i].mean(1)
        cis[N] = ci95(100 * (pp["commit"][i].mean(1) - Sm) / (Bm - Sm))
        check(f"width N={N} commit capture", c / 100, EXPECTED3["width_commit"][N])
        if N in EXPECTED3["width_cost"]:
            check(f"width N={N} cost ratio", cost_ratio[-1] / 100, EXPECTED3["width_cost"][N])
    # saturating fit c_inf - beta/N (stated as consistency, not law)
    x = 1.0 / np.array(Ns, float)
    A = np.vstack([np.ones_like(x), x]).T
    cinf, mb = np.linalg.lstsq(A, np.array(cap_c), rcond=None)[0]
    r2w = 1 - ((np.array(cap_c) - (cinf + mb * x)) ** 2).sum() / \
        ((np.array(cap_c) - np.mean(cap_c)) ** 2).sum()
    # bootstrap c_inf (joint prompt resample across N<=16; N=32 pool separate)
    idx32 = boot_idx(len(pa32), seed=21)
    caps_b = []
    for N in Ns:
        pp = sim[N]["pp"]
        i = idx if N < 32 else idx32
        Sm, Bm = pp["single"][i].mean(1), pp["bon"][i].mean(1)
        caps_b.append(100 * (pp["commit"][i].mean(1) - Sm) / (Bm - Sm))
    caps_b = np.array(caps_b)          # (5, B)
    cinf_b = np.array([np.linalg.lstsq(A, caps_b[:, i], rcond=None)[0][0]
                       for i in range(caps_b.shape[1])])
    print(f"[PROSE] width saturating fit: capture(N) = {cinf:.1f} {mb:+.1f}/N,"
          f" c_inf = {cinf:.1f}% CI [{ci95(cinf_b)[0]:.1f}, {ci95(cinf_b)[1]:.1f}],"
          f" R^2 = {r2w:.2f} (5 pts; consistency statement only)")
    width = dict(Ns=Ns, cap_c=cap_c, cost_ratio=cost_ratio, cis=cis)

    # ---------------- absolute savings per candidate (f9 panel 3) --------------
    print("=== absolute savings per candidate (tau=0.10, per model) ===")
    for k in ("wan13_tau010", "ltx", "wan22", "cog_tau010", "hunyuan", "wan14b_tau010"):
        d = pts[k]
        print(f"[PROSE] {d['model']:<18} Cf {d['Cf']:6.1f}s  Cc {d['Cc']:6.1f}s  "
              f"saved {d['Cf']-d['Cc']:6.1f}s/candidate  (= {100*(1-d['Cc']/d['Cf']):.0f}% of Cf)")

    # ---------------- figures ---------------------------------------------------
    print("=== figures ===")
    import style
    # master plot is crowded (~20 operating points) -> no CI bars; the schedule
    # trend has 3 points with room -> CI bars on (user directive 2026-07-09).
    style.SHOW_UNCERTAINTY = False; f13_master(pts, meth, (alpha, beta, r2, band), (ca, cb, cr2), width, args.out)
    style.SHOW_UNCERTAINTY = True;  f14_schedule(pts, skfit, capfit, args.out)
    for f in ("f13_scaling_master.pdf", "f14_schedule_scaling.pdf"):
        print(" wrote", os.path.join(args.out, f))


if __name__ == "__main__":
    main()
