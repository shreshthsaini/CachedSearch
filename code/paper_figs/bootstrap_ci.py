"""Prompt-level bootstrap CIs for every headline number of the CachedSearch paper.

Method: nonparametric bootstrap over PROMPTS (the independent sampling unit of
every experiment): resample the prompt set with replacement B=10,000 times,
recompute each statistic on every resample, report the 2.5/97.5 percentiles as
the 95% CI. Point estimates are recomputed from the raw jsonl with EXACTLY the
conventions of make_figs.py / make_figs2.py / a1_analysis.py (records deduped by
(prompt, seed, variant); scores AND latencies restricted to seeds 0-7 -- the
N=16 width-extension shards share the results dirs; tau dirs are cached-only and
join full references from the model's v0 dir) and are cross-checked against the
values currently quoted in the paper (PAPER dict below): any difference beyond
quoting precision prints a loud MISMATCH and is flagged in the JSON.

Latency-derived quantities (speedups, costs) are bootstrapped by resampling
prompts and pooling their per-record latencies (so the identity resample
reproduces the pooled mean exactly); full/cached draws are paired on the prompt
whenever both sides cover the same prompt set, independent otherwise (a1
baseline dirs with partial coverage).

Covers: gate grid (3 taus) - official VBench (n=944) - tab:baselines methods -
tab:scale models - CogVideoX tau curve (f10) - width scaling N in {2,4,8,16}
(16-seed grid, f8) - tab:main strategy gains/captures/costs (8-seed grid) -
E6 stacking (sec:abl-stack) - per-category table (tab:categories) - wave-4:
per-architecture tau-calibration arms (tab:taucal; Hunyuan/LTX/Wan2.2-fix/
Wan-14B-fix) and the VBench-2.0 suite (n=1,012 prompts).

Output: paper/ci_numbers.json  (point + 95% CI + paper value + match flag for
every quantity) and a printed table.

Usage:  MALLOC_ARENA_MAX=2 \
        python code/paper_figs/bootstrap_ci.py [--out paper/ci_numbers.json] [-B 10000]
Reproducibility: one np.random.default_rng(0) consumed in a fixed order; rerun
reproduces every interval bit-exactly.
"""
import argparse
import datetime
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figs import RESULTS, per_prompt                      # noqa: E402
from make_figs2 import grid, latencies, load_rows, load_stack  # noqa: E402
from a1_analysis import load_dedup                             # noqa: E402
from appendix_tables import categorize                         # noqa: E402

PAPER_JSON_DEFAULT = os.environ.get(
    "CACHEDSEARCH_CI_JSON",
    os.path.join(os.environ.get("CACHEDSEARCH_RESULTS", "./results"), "ci_numbers.json"),
)

# ---------------------------------------------------------------- paper values
# Every currently quoted headline value, with its quoting precision (decimals).
# A recomputed point estimate "matches" iff |point - quoted| <= 0.5 * 10^-dec.
# Sources: tab:main, tab:tau, tab:baselines, tab:scale, tab:categories,
# sec:exp-{ranking,main,regret,vbench}, sec:abl-{tau,cog,stack,scalingfigs}, f8.
PAPER = {
    "gate_tau=0.05": dict(median_rho=(0.905, 3), p10_rho=(0.738, 3), top1=(0.70, 2),
                          mean_regret=(0.025, 3), capture_mor=(0.936, 3), speedup=(1.58, 2)),
    # mean_regret sits at the 0.0395 rounding boundary; capture_rom is quoted in
    # the paper as 1 - 0.040/0.657 computed FROM rounded values (true 0.9399):
    # tolerance widened for exactly these rounding-chain artifacts (see NOTES).
    "gate_tau=0.10": dict(median_rho=(0.905, 3), mean_rho=(0.820, 3), p10_rho=(0.614, 3),
                          top1=(0.64, 2), mean_regret=(0.040, 3, 6e-4), zero_regret=(0.64, 2),
                          rand_regret=(0.657, 3), capture_mor=(0.901, 3),
                          capture_rom=(0.939, 3, 1e-3), speedup=(1.97, 2)),
    "gate_tau=0.20": dict(median_rho=(0.857, 3), p10_rho=(0.474, 3), top1=(0.52, 2),
                          mean_regret=(0.057, 3), capture_mor=(0.883, 3), speedup=(2.41, 2)),
    "vbench": dict(median_rho=(0.905, 3), mean_rho=(0.859, 3), p10_rho=(0.690, 3),
                   top1=(0.72, 2), mean_regret=(0.056, 3), zero_regret=(0.72, 2),
                   rand_regret=(0.79, 2), capture_mor=(0.902, 3), speedup=(1.95, 2)),
    # tab:baselines (a1 dirs; "ours" = independent easycache-ours rerun)
    "a1_none": dict(median_rho=(1.000, 3), p10_rho=(1.000, 3), top1=(1.00, 2),
                    mean_regret=(0.000, 3), capture_mor=(1.000, 3), speedup=(0.99, 2)),
    # PAB / CFG-Cache / control rows refreshed 2026-07-08 at final coverage
    # (all a1 dirs now 400 records = 50 complete prompts; stragglers landed)
    "a1_pab": dict(median_rho=(0.976, 3), p10_rho=(0.950, 3), top1=(0.90, 2),
                   mean_regret=(0.004, 3), capture_mor=(0.995, 3), speedup=(1.29, 2)),
    "a1_cfgcache": dict(median_rho=(1.000, 3), p10_rho=(0.929, 3), top1=(0.92, 2),
                        mean_regret=(0.004, 3), capture_mor=(0.996, 3), speedup=(1.38, 2)),
    # TeaCache row refreshed by the a1-analysis agent at final grid coverage
    # (2026-07-08, mid-CI-session): its last shard landed AFTER the original
    # row was derived; refreshed values match this script's recomputation.
    "a1_teacache": dict(median_rho=(0.929, 3), p10_rho=(0.783, 3), top1=(0.68, 2),
                        mean_regret=(0.029, 3), capture_mor=(0.932, 3), speedup=(1.84, 2)),
    "a1_easycache-ours": dict(median_rho=(0.905, 3), p10_rho=(0.614, 3), top1=(0.64, 2),
                              mean_regret=(0.039, 3), capture_mor=(0.901, 3), speedup=(1.96, 2)),
    # tab:scale (v0 row == gate_tau=0.10; six-model refresh 2026-07-08, all n=50)
    "scale_cog5b_v0": dict(median_rho=(0.762, 3), p10_rho=(0.35, 2), top1=(0.48, 2),
                           capture_mor=(0.752, 3), speedup=(2.06, 2)),
    "scale_wan14b": dict(median_rho=(0.905, 3), p10_rho=(0.52, 2), top1=(0.58, 2),
                         capture_mor=(0.875, 3), speedup=(2.05, 2)),
    "scale_ltx_v0": dict(median_rho=(0.536, 3), p10_rho=(0.14, 2), top1=(0.50, 2),
                         capture_mor=(0.676, 3), speedup=(2.63, 2)),
    "scale_wan22_5b": dict(median_rho=(0.881, 3), p10_rho=(0.60, 2), top1=(0.46, 2),
                           capture_mor=(0.860, 3), speedup=(2.05, 2)),
    "scale_hunyuan_v0": dict(median_rho=(0.762, 3), p10_rho=(0.44, 2), top1=(0.54, 2),
                             capture_mor=(0.799, 3), speedup=(2.19, 2)),
    # Wan-14B tau arms: ENQUEUED at tau=0.005/0.020 (records carry these
    # values), not the 1.3B sweep's 0.05/0.20 -- quoted as measured;
    # tau=0.005 fires zero skips (cached == full bit-exactly)
    "wan14b_tau=0.005": dict(median_rho=(1.000, 3), capture_mor=(1.000, 3),
                             speedup=(1.01, 2)),
    "wan14b_tau=0.02": dict(median_rho=(0.917, 3), top1=(0.75, 2),
                            capture_mor=(0.921, 3), speedup=(1.21, 2)),
    # schedule-length / resolution ablation (Wan2.1-1.3B, tau=0.10)
    "steps25": dict(median_rho=(0.964, 3), p10_rho=(0.855, 3), top1=(0.86, 2),
                    mean_regret=(0.019, 3), capture_mor=(0.958, 3), speedup=(1.41, 2)),
    "steps100": dict(median_rho=(0.798, 3), p10_rho=(0.464, 3), top1=(0.62, 2),
                     mean_regret=(0.126, 3), capture_mor=(0.825, 3), speedup=(2.80, 2)),
    # res720 refreshed at FINAL coverage n=50 (wave-4 pass 2026-07-08; the dir
    # completed -- coverage is frozen for good now); tab:schedule row + prose +
    # scaling_analysis.py EXPECTED3 updated together
    "res720": dict(median_rho=(0.881, 3), p10_rho=(0.69, 2), top1=(0.66, 2),
                   mean_regret=(0.059, 3), capture_mor=(0.908, 3), speedup=(2.05, 2)),
    # CogVideoX tau curve (sec:abl-cog prose; f10)
    "cog_tau=0.05": dict(median_rho=(0.810, 3), top1=(0.64, 2),
                         capture_mor=(0.859, 3), speedup=(1.78, 2)),
    "cog_tau=0.20": dict(capture_mor=(0.647, 3), speedup=(2.65, 2)),
    # wave-4 tau-calibration arms (tab:taucal + f10; all n=50)
    "hunyuan_tau=0.05": dict(median_rho=(0.810, 3), capture_mor=(0.851, 3),
                             speedup=(1.77, 2)),
    "hunyuan_tau=0.20": dict(median_rho=(0.690, 3), capture_mor=(0.797, 3),
                             speedup=(2.48, 2)),
    "ltx_tau=0.02": dict(median_rho=(0.786, 3), capture_mor=(0.796, 3),
                         speedup=(1.71, 2)),
    "ltx_tau=0.05": dict(median_rho=(0.619, 3), capture_mor=(0.716, 3),
                         speedup=(2.19, 2)),
    "wan22_tau=0.05": dict(median_rho=(0.905, 3), capture_mor=(0.893, 3),
                           speedup=(1.63, 2)),
    "wan22_tau=0.20": dict(median_rho=(0.833, 3), capture_mor=(0.831, 3),
                           speedup=(2.37, 2)),
    "wan14b_tau=0.05": dict(median_rho=(0.881, 3), capture_mor=(0.880, 3),
                            speedup=(1.59, 2)),
    "wan14b_tau=0.20": dict(median_rho=(0.881, 3), capture_mor=(0.847, 3),
                            speedup=(2.52, 2)),
    # VBench-2.0 cross-suite replication (tab:vbench2; FINAL at full coverage:
    # all 1,013 unique suite prompts complete = 16,208 records; the prompts
    # file has 1,013 lines -- wc -l undercounts by one, no trailing newline)
    "vbench2": dict(median_rho=(0.881, 3), mean_rho=(0.848, 3), p10_rho=(0.690, 3),
                    top1=(0.69, 2), mean_regret=(0.060, 3), zero_regret=(0.69, 2),
                    capture_mor=(0.893, 3), speedup=(1.98, 2)),
    # width scaling, 16-seed grid (sec:exp-main, sec:abl-scalingfigs, f8);
    # N=32 = the full 32-seed grid (one subset per prompt, f8 extension)
    "width_N=4": dict(capture_commit=(0.909, 3)),
    "width_N=8": dict(capture_commit=(0.922, 3), top1=(0.70, 2)),
    "width_N=16": dict(capture_commit=(0.957, 3)),
    "width_N=2": dict(top1=(0.86, 2)),
    "width_N=32": dict(capture_commit=(0.952, 3), top1=(0.66, 2)),
    # tab:main (8-seed grid, tau=0.10; capture = ratio of gains vs best-of-N)
    "main_N=2": dict(gain_bon=(0.304, 3), gain_keep=(0.286, 3), gain_commit=(0.274, 3),
                     capture_keep=(0.941, 3), capture_commit=(0.899, 3)),
    "main_N=4": dict(gain_bon=(0.514, 3), gain_keep=(0.496, 3), gain_commit=(0.461, 3),
                     capture_keep=(0.965, 3), capture_commit=(0.896, 3)),
    # capture_keep in tab:main was computed from the rounded gains 0.749/0.752
    # = 0.996; the unrounded ratio is 0.9966 (rounding-chain artifact, see NOTES)
    "main_N=8": dict(gain_bon=(0.752, 3), gain_keep=(0.749, 3), gain_commit=(0.712, 3),
                     capture_keep=(0.996, 3, 1e-3), capture_commit=(0.947, 3)),
    # E6 stacking (sec:abl-stack, fig:methods)
    "stack": dict(capture_mor=(0.886, 3), capture_rog=(0.941, 3), gain=(0.708, 3),
                  survival=(0.86, 2), e2e_s=(256, 0), expl_s=(175, 0),
                  median_regret=(0.0, 3)),
    # tab:categories (gate grid, tau=0.10)
    "cat_humans": dict(median_rho=(0.881, 3), mean_regret=(0.045, 3), top1=(0.56, 2),
                       mean_spread=(0.53, 2)),
    "cat_animals": dict(median_rho=(0.833, 3), mean_regret=(0.008, 3), top1=(0.78, 2),
                        mean_spread=(0.51, 2)),
    "cat_nature": dict(median_rho=(0.929, 3), mean_regret=(0.005, 3), top1=(0.80, 2),
                       mean_spread=(0.53, 2)),
    "cat_objects": dict(median_rho=(0.857, 3), mean_regret=(0.066, 3), top1=(0.56, 2),
                        mean_spread=(0.43, 2)),
    "cat_stylized": dict(median_rho=(0.917, 3), mean_regret=(0.112, 3), top1=(0.50, 2),
                         mean_spread=(0.54, 2)),
}

RNG = np.random.default_rng(0)
B = 10000
RESULTS_JSON = {}
MISMATCHES = []


def _ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def record(group, name, point, samples, n):
    """Store point + CI; cross-check against the quoted paper value."""
    lo, hi = _ci(samples)
    entry = dict(point=float(point), lo=lo, hi=hi, n_prompts=int(n))
    quoted = PAPER.get(group, {}).get(name)
    flag = ""
    if quoted is not None:
        val, dec = quoted[0], quoted[1]
        tol = quoted[2] if len(quoted) > 2 else 0.5 * 10 ** (-dec) + 1e-9
        entry["paper_value"] = val
        entry["match"] = bool(abs(point - val) <= tol)
        if not entry["match"]:
            flag = f"  <<< MISMATCH vs paper {val}"
            MISMATCHES.append((group, name, float(point), val))
    RESULTS_JSON.setdefault(group, {})[name] = entry
    print(f"  {group:<22} {name:<16} {point:9.4f}  [{lo:8.4f}, {hi:8.4f}]{flag}")


# ---------------------------------------------------------------- bootstrap ops
def boot_pp(group, pa):
    """Per-prompt ranking/regret metrics for a {prompt: (full[8], cached[8])} grid."""
    pp = per_prompt(pa)
    rho, top1 = pp["rho"], pp["top1"].astype(float)
    regret, rand = pp["regret"], pp["rand"]
    ratios = np.where(rand > 0, (rand - regret) / np.where(rand > 0, rand, 1.0), np.nan)
    n = len(rho)
    idx = RNG.integers(0, n, size=(B, n))
    R = rho[idx]
    record(group, "median_rho", np.median(rho), np.median(R, axis=1), n)
    record(group, "mean_rho", rho.mean(), R.mean(axis=1), n)
    record(group, "p10_rho", np.percentile(rho, 10), np.percentile(R, 10, axis=1), n)
    record(group, "top1", top1.mean(), top1[idx].mean(axis=1), n)
    record(group, "mean_regret", regret.mean(), regret[idx].mean(axis=1), n)
    record(group, "zero_regret", (regret == 0).mean(), (regret[idx] == 0).mean(axis=1), n)
    record(group, "rand_regret", rand.mean(), rand[idx].mean(axis=1), n)
    record(group, "capture_mor", np.nanmean(ratios), np.nanmean(ratios[idx], axis=1), n)
    record(group, "capture_rom", 1 - regret.mean() / rand.mean(),
           1 - regret[idx].mean(axis=1) / rand[idx].mean(axis=1), n)


def lat_groups(rows, variant=None):
    """prompt -> (latency sum, count) over deduped records, seeds 0-7 only."""
    g = defaultdict(lambda: [0.0, 0])
    for (p, s, v), r in rows.items():
        if s < 8 and (variant is None or v == variant):
            g[p][0] += r["latency"]
            g[p][1] += 1
    prompts = sorted(g)
    return (prompts,
            np.array([g[p][0] for p in prompts]),
            np.array([g[p][1] for p in prompts], dtype=float))


def boot_lat_ratio(fsum, fcnt, csum, ccnt, paired):
    """Bootstrap samples of (pooled full latency)/(pooled cached latency)."""
    point = (fsum.sum() / fcnt.sum()) / (csum.sum() / ccnt.sum())
    if paired:
        idx = RNG.integers(0, len(fsum), size=(B, len(fsum)))
        Cf = fsum[idx].sum(axis=1) / fcnt[idx].sum(axis=1)
        Cc = csum[idx].sum(axis=1) / ccnt[idx].sum(axis=1)
    else:
        i1 = RNG.integers(0, len(fsum), size=(B, len(fsum)))
        i2 = RNG.integers(0, len(csum), size=(B, len(csum)))
        Cf = fsum[i1].sum(axis=1) / fcnt[i1].sum(axis=1)
        Cc = csum[i2].sum(axis=1) / ccnt[i2].sum(axis=1)
    return point, Cf / Cc, Cf, Cc


def boot_speedup(group, full_rows, cached_rows, cached_variant="cached"):
    """Speedup CI; full/cached paired on prompt when the prompt sets coincide."""
    pf, fsum, fcnt = lat_groups(full_rows, "full")
    pc, csum, ccnt = lat_groups(cached_rows, cached_variant)
    if pf == pc:
        point, s, Cf, Cc = boot_lat_ratio(fsum, fcnt, csum, ccnt, paired=True)
    else:
        point, s, Cf, Cc = boot_lat_ratio(fsum, fcnt, csum, ccnt, paired=False)
    record(group, "speedup", point, s, len(pc))
    return Cf, Cc


# ---------------------------------------------------------------- width / main
def width_pp(pa, Ns):
    """Per-prompt subset-mean strategy values over all C(S,N) seed subsets."""
    prompts = sorted(pa)
    res = {N: {k: np.zeros(len(prompts))
               for k in ("single", "bon", "keep", "commit", "top1")} for N in Ns}
    for i, p in enumerate(prompts):
        fu, ca = pa[p]
        S = len(fu)
        for N in Ns:
            idx = np.array(list(itertools.combinations(range(S), N)), dtype=int)
            F, C = fu[idx], ca[idx]
            pick = C.argmax(axis=1)
            rows_ = np.arange(len(idx))
            res[N]["single"][i] = F[:, 0].mean()
            res[N]["bon"][i] = F.max(axis=1).mean()
            res[N]["keep"][i] = C[rows_, pick].mean()
            res[N]["commit"][i] = F[rows_, pick].mean()
            res[N]["top1"][i] = (pick == F.argmax(axis=1)).mean()
    return prompts, res


def boot_width(group, resN, n, idx):
    s, b = resN["single"], resN["bon"]
    c, k, t = resN["commit"], resN["keep"], resN["top1"]
    Sm, Bm = s[idx].mean(axis=1), b[idx].mean(axis=1)
    Cm, Km = c[idx].mean(axis=1), k[idx].mean(axis=1)
    cap_c = (c.mean() - s.mean()) / (b.mean() - s.mean())
    cap_k = (k.mean() - s.mean()) / (b.mean() - s.mean())
    record(group, "capture_commit", cap_c, (Cm - Sm) / (Bm - Sm), n)
    record(group, "capture_keep", cap_k, (Km - Sm) / (Bm - Sm), n)
    record(group, "top1", t.mean(), t[idx].mean(axis=1), n)


def boot_main(group, resN, n, idx, N, Cf_s, Cc_s, Cf_pt, Cc_pt):
    """tab:main row family at width N: gains, ratio-of-gain captures, costs."""
    s, b, c, k = resN["single"], resN["bon"], resN["commit"], resN["keep"]
    Sm, Bm = s[idx].mean(axis=1), b[idx].mean(axis=1)
    Cm, Km = c[idx].mean(axis=1), k[idx].mean(axis=1)
    record(group, "gain_bon", b.mean() - s.mean(), Bm - Sm, n)
    record(group, "gain_keep", k.mean() - s.mean(), Km - Sm, n)
    record(group, "gain_commit", c.mean() - s.mean(), Cm - Sm, n)
    record(group, "capture_keep",
           (k.mean() - s.mean()) / (b.mean() - s.mean()), (Km - Sm) / (Bm - Sm), n)
    record(group, "capture_commit",
           (c.mean() - s.mean()) / (b.mean() - s.mean()), (Cm - Sm) / (Bm - Sm), n)
    # costs from the paired latency bootstrap (deterministic given Cf, Cc)
    record(group, "cost_bon_s", N * Cf_pt, N * Cf_s, n)
    record(group, "cost_keep_s", N * Cc_pt, N * Cc_s, n)
    record(group, "cost_commit_s", N * Cc_pt + Cf_pt, N * Cc_s + Cf_s, n)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=PAPER_JSON_DEFAULT)
    ap.add_argument("-B", type=int, default=10000)
    args = ap.parse_args()
    global B
    B = args.B

    print(f"=== prompt-level bootstrap, B={B}, percentile 95% CIs, rng seed 0 ===")
    rows = {tag: load_rows(tag) for tag in
            ["v0", "tau005", "tau020", "vbench", "cog5b_v0", "cog5b_tau005",
             "cog5b_tau020", "wan14b"]}

    # ---- gate grid: 3 taus ------------------------------------------------
    pa_v0 = grid(rows["v0"], 8)
    for group, tag, tdir in [("gate_tau=0.05", "tau005", True),
                             ("gate_tau=0.10", "v0", False),
                             ("gate_tau=0.20", "tau020", True)]:
        pa = grid(rows[tag], 8, full_rows=rows["v0"]) if tdir else pa_v0
        boot_pp(group, pa)
        boot_speedup(group, rows["v0"], rows[tag])

    # ---- official VBench suite (n=944) ------------------------------------
    boot_pp("vbench", grid(rows["vbench"], 8))
    boot_speedup("vbench", rows["vbench"], rows["vbench"])

    # ---- tab:baselines (a1 dirs, full refs from v0) ------------------------
    ref = {(p, s): r for (p, s, v), r in rows["v0"].items() if v == "full" and s < 8}
    for method in ("none", "pab", "cfgcache", "teacache", "easycache-ours"):
        a1 = load_dedup(os.path.join(RESULTS, f"a1_{method}", "scores_shard*.jsonl"))
        by = defaultdict(dict)
        for (p, s, v), r in a1.items():
            by[p][s] = r
        pa = {}
        for p, seeds in by.items():
            if all(s in seeds and (p, s) in ref for s in range(8)):
                pa[p] = (np.array([ref[(p, s)]["score"] for s in range(8)]),
                         np.array([seeds[s]["score"] for s in range(8)]))
        boot_pp(f"a1_{method}", pa)
        boot_speedup(f"a1_{method}", rows["v0"], a1, cached_variant=None)

    # ---- tab:scale models + CogVideoX tau curve (f9/f10) -------------------
    for group, tag in [("scale_cog5b_v0", "cog5b_v0"), ("scale_wan14b", "wan14b")]:
        boot_pp(group, grid(rows[tag], 8))
        boot_speedup(group, rows[tag], rows[tag])
    for group, tag in [("cog_tau=0.05", "cog5b_tau005"), ("cog_tau=0.20", "cog5b_tau020")]:
        boot_pp(group, grid(rows[tag], 8, full_rows=rows["cog5b_v0"]))
        boot_speedup(group, rows["cog5b_v0"], rows[tag])

    # ---- width scaling on the 16-seed grid (f8) -----------------------------
    pa16 = grid(rows["v0"], 16)
    prompts16, res16 = width_pp(pa16, (2, 4, 8, 16))
    idx16 = RNG.integers(0, len(prompts16), size=(B, len(prompts16)))
    for N in (2, 4, 8, 16):
        boot_width(f"width_N={N}", res16[N], len(prompts16), idx16)

    def _cap(resN):  # commit-capture point + bootstrap samples (shared idx16 -> paired)
        s, b, c = resN["single"], resN["bon"], resN["commit"]
        Sm, Bm, Cm = s[idx16].mean(axis=1), b[idx16].mean(axis=1), c[idx16].mean(axis=1)
        return (c.mean() - s.mean()) / (b.mean() - s.mean()), (Cm - Sm) / (Bm - Sm)
    (p16, s16), (p4, s4) = _cap(res16[16]), _cap(res16[4])
    # paired difference: the "capture rises with width" claim (sec:theory / f8)
    record("width_diff", "capture_commit_N16_minus_N4", p16 - p4, s16 - s4, len(prompts16))

    # ---- tab:main strategy table on the 8-seed grid -------------------------
    prompts8, res8 = width_pp(pa_v0, (2, 4, 8))
    idx8 = RNG.integers(0, len(prompts8), size=(B, len(prompts8)))
    lat_v0 = latencies(rows["v0"])
    Cf_pt, Cc_pt = lat_v0["full"], lat_v0["cached"]
    pf, fsum, fcnt = lat_groups(rows["v0"], "full")
    pc, csum, ccnt = lat_groups(rows["v0"], "cached")
    assert pf == pc == prompts8
    Cf_s = fsum[idx8].sum(axis=1) / fcnt[idx8].sum(axis=1)   # paired with res8 draws
    Cc_s = csum[idx8].sum(axis=1) / ccnt[idx8].sum(axis=1)
    record("main_costs", "Cf_s", Cf_pt, Cf_s, len(prompts8))
    record("main_costs", "Cc_s", Cc_pt, Cc_s, len(prompts8))
    for N in (2, 4, 8):
        boot_main(f"main_N={N}", res8[N], len(prompts8), idx8, N, Cf_s, Cc_s, Cf_pt, Cc_pt)

    # ---- E6 stacking (joint resample with the v0 grid + latencies) ----------
    stack = load_stack()
    common = [p for p in prompts8 if p in stack]
    j = np.array([prompts8.index(p) for p in common])
    cs = np.array([stack[p]["commit_score"] for p in common])
    e2e = np.array([stack[p]["timings"]["e2e_s"] for p in common])
    expl = np.array([stack[p]["timings"]["stage1_s"] + stack[p]["timings"]["preview_s"]
                     + stack[p]["timings"]["stage2_s"] for p in common])
    regret = np.array([pa_v0[p][0].max() - stack[p]["commit_score"] for p in common])
    rand = np.array([pa_v0[p][0].max() - pa_v0[p][0].mean() for p in common])
    surv = np.array([float(int(np.argmax(pa_v0[p][0])) in stack[p]["survivors"])
                     for p in common])
    n = len(common)
    idx = RNG.integers(0, n, size=(B, n))
    s8, b8 = res8[8]["single"][j], res8[8]["bon"][j]
    Sm, Bm = s8[idx].mean(axis=1), b8[idx].mean(axis=1)
    Cm = cs[idx].mean(axis=1)
    record("stack", "gain", cs.mean() - s8.mean(), Cm - Sm, n)
    record("stack", "capture_rog",
           (cs.mean() - s8.mean()) / (b8.mean() - s8.mean()), (Cm - Sm) / (Bm - Sm), n)
    record("stack", "capture_mor", np.mean(1 - regret / rand),
           (1 - regret / rand)[idx].mean(axis=1), n)
    record("stack", "median_regret", np.median(regret), np.median(regret[idx], axis=1), n)
    record("stack", "survival", surv.mean(), surv[idx].mean(axis=1), n)
    record("stack", "e2e_s", e2e.mean(), e2e[idx].mean(axis=1), n)
    record("stack", "expl_s", expl.mean(), expl[idx].mean(axis=1), n)
    # speedups vs full best-of-8 (paired: same prompts carry the v0 latencies)
    fs, fc = fsum[j], fcnt[j]
    Cf8 = 8 * fs[idx].sum(axis=1) / fc[idx].sum(axis=1)
    record("stack", "speedup_expl_vs_bo8", 8 * fs.sum() / fc.sum() / expl.mean(),
           Cf8 / expl[idx].mean(axis=1), n)
    record("stack", "speedup_e2e_vs_bo8", 8 * fs.sum() / fc.sum() / e2e.mean(),
           Cf8 / e2e[idx].mean(axis=1), n)

    # ---- tab:categories (gate grid, tau=0.10) -------------------------------
    pp = per_prompt(pa_v0)
    cats = defaultdict(list)
    # NB: per_prompt iterates pa.values() in dict-insertion order -- index by
    # the same order (NOT sorted), or category indices point at wrong prompts.
    for i, p in enumerate(pa_v0.keys()):
        cats[categorize(p)].append(i)
    spread = pp["spread"]
    for cat in ("humans", "animals", "nature", "objects", "stylized"):
        ii = np.array(cats[cat])
        n = len(ii)
        idx = RNG.integers(0, n, size=(B, n))
        rho, reg = pp["rho"][ii], pp["regret"][ii]
        t1, sp = pp["top1"][ii].astype(float), spread[ii]
        g = f"cat_{cat}"
        record(g, "median_rho", np.median(rho), np.median(rho[idx], axis=1), n)
        record(g, "mean_regret", reg.mean(), reg[idx].mean(axis=1), n)
        record(g, "top1", t1.mean(), t1[idx].mean(axis=1), n)
        record(g, "mean_spread", sp.mean(), sp[idx].mean(axis=1), n)

    # ---- 2026-07-08 final-integration additions ------------------------------
    # Appended AFTER all prior groups so the RNG stream of previously published
    # intervals is disturbed only where the underlying DATA changed (a1 grids
    # and wan14b filled to 50 prompts).
    rows2 = {tag: load_rows(tag) for tag in
             ["ltx_v0", "wan22_5b", "hunyuan_v0", "wan14b_tau005",
              "wan14b_tau020", "steps25", "steps100", "res720"]}
    # three new tab:scale models (self-contained full+cached dirs)
    for group, tag in [("scale_ltx_v0", "ltx_v0"), ("scale_wan22_5b", "wan22_5b"),
                       ("scale_hunyuan_v0", "hunyuan_v0")]:
        boot_pp(group, grid(rows2[tag], 8))
        boot_speedup(group, rows2[tag], rows2[tag])
    # schedule-length ablation (Wan2.1-1.3B, tau=0.10)
    for group in ("steps25", "steps100"):
        boot_pp(group, grid(rows2[group], 8))
        boot_speedup(group, rows2[group], rows2[group])
    # Wan-14B tau arms -- ENQUEUED at tau=0.005/0.020 (records carry these
    # values), NOT the 1.3B sweep's 0.05/0.20; cached-only dirs, full refs
    # merged from b1_gate_wan14b (f10)
    for group, tag in [("wan14b_tau=0.005", "wan14b_tau005"),
                       ("wan14b_tau=0.02", "wan14b_tau020")]:
        boot_pp(group, grid(rows2[tag], 8, full_rows=rows["wan14b"]))
        boot_speedup(group, rows["wan14b"], rows2[tag])
    # width N=32: the full 32-seed grid (one subset per prompt; N<=16 keep the
    # published 16-seed pool above -- see make_figs2.f8_width)
    pa32 = grid(rows["v0"], 32)
    prompts32, res32 = width_pp(pa32, (32,))
    idx32 = RNG.integers(0, len(prompts32), size=(B, len(prompts32)))
    boot_width("width_N=32", res32[32], len(prompts32), idx32)
    # resolution ablation LAST in the stream: b1_gate_res720 was still filling
    # at integration time (n grows), so keeping it last means a rerun at higher
    # coverage disturbs no other group's draws
    # (2026-07-08 wave-4: the dir COMPLETED at n=50 -- coverage is now frozen,
    # so groups appended below are stable too; res720 quotes refreshed to n=50)
    boot_pp("res720", grid(rows2["res720"], 8))
    boot_speedup("res720", rows2["res720"], rows2["res720"])

    # ---- 2026-07-08 wave-4 additions: tau calibration (tab:taucal) + VBench-2.0
    # Appended AFTER res720 so every previously published interval keeps its
    # draws bit-exactly. New tau arms are cached-only dirs joining full refs
    # from the model's own v0 grid (wan22_tau{005,020}fix are the corrected
    # reruns at native 704x1280x121f. The first arms omitted the resolution
    # flags and are invalid.
    def load_tolerant(tag):
        # local tolerant loader for the wave-4 dirs: skips torn tail lines of
        # live shards (the canonical final run sees complete, parseable dirs;
        # tolerance only matters for draft passes while the fleet writes)
        import glob as _glob
        out = {}
        for f in sorted(_glob.glob(os.path.join(RESULTS, f"b1_gate_{tag}",
                                                "scores_shard*.jsonl"))):
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[(r["prompt"], r["seed"], r["variant"])] = r
        return out

    rows3 = {tag: load_tolerant(tag) for tag in
             ["hunyuan_hun_tau005", "hunyuan_hun_tau020",
              "ltx_ltx_tau002", "ltx_ltx_tau005",
              "wan22_tau005fix", "wan22_tau020fix",
              "wan14b_tau05fix", "wan14b_tau20fix", "vbench2"]}
    for group, tag, ref in [
            ("hunyuan_tau=0.05", "hunyuan_hun_tau005", "hunyuan_v0"),
            ("hunyuan_tau=0.20", "hunyuan_hun_tau020", "hunyuan_v0"),
            ("ltx_tau=0.02", "ltx_ltx_tau002", "ltx_v0"),
            ("ltx_tau=0.05", "ltx_ltx_tau005", "ltx_v0"),
            ("wan22_tau=0.05", "wan22_tau005fix", "wan22_5b"),
            ("wan22_tau=0.20", "wan22_tau020fix", "wan22_5b"),
            ("wan14b_tau=0.05", "wan14b_tau05fix", "wan14b"),
            ("wan14b_tau=0.20", "wan14b_tau20fix", "wan14b")]:
        full = rows2[ref] if ref in rows2 else rows[ref]
        pa = grid(rows3[tag], 8, full_rows=full)
        if len(pa) < 40:  # live dir mid-fill: skip on draft passes only
            print(f"  {group:<22} SKIPPED ({len(pa)} complete prompts -- live dir)")
            continue
        boot_pp(group, pa)
        boot_speedup(group, full, rows3[tag])
    # VBench-2.0 suite (1,013 prompts x 8 seeds, full+cached at tau=0.10).
    # FINAL 2026-07-08: full coverage, 16,208 records = 1,013 x 16 (a top-off
    # wave completed the tail after an interim freeze; data now frozen).
    pa_vb2 = grid(rows3["vbench2"], 8)
    if len(pa_vb2) >= 900:
        boot_pp("vbench2", pa_vb2)
        boot_speedup("vbench2", rows3["vbench2"], rows3["vbench2"])
    else:
        print(f"  vbench2                SKIPPED ({len(pa_vb2)} complete prompts -- live dir)")

    # ---- write JSON ---------------------------------------------------------
    out = dict(
        _meta=dict(
            method="nonparametric bootstrap over prompts (the independent unit); "
                   "B resamples with replacement; 95% CI = 2.5/97.5 percentiles",
            B=B, rng_seed=0, ci_level=0.95,
            data_root=RESULTS,
            conventions="records deduped by (prompt,seed,variant); scores+latencies "
                        "seeds 0-7 only (width-extension shards excluded); tau dirs "
                        "join full refs from the model's v0 dir; latency ratios "
                        "bootstrapped by pooling per-prompt record groups, paired on "
                        "prompt where both sides share the prompt set",
            generated=datetime.datetime.now().isoformat(timespec="seconds"),
            script="code/paper_figs/bootstrap_ci.py",
        ),
        **RESULTS_JSON,
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")
    if MISMATCHES:
        print(f"\n<<< {len(MISMATCHES)} point estimates DIVERGE from quoted paper values:")
        for g, k, got, want in MISMATCHES:
            print(f"    {g}.{k}: recomputed {got:.4f} vs paper {want}")
    else:
        print("\nALL point estimates match the quoted paper values (within quoting precision).")


if __name__ == "__main__":
    main()
