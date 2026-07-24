"""Claims-hardening analysis for the CachedSearch paper (STYLE-GUIDE P0-2/P0-3/P0-4/P0-6).

Recomputes, on the OFFICIAL VBench suite (results/b1_gate_vbench, n=944 fully
covered prompts, seeds 0-7, cross-partition shards deduped by
(prompt, seed, variant)) and on the 50-prompt gate grid:

  P0-2  corr(spread, rho) with 95% bootstrap CI + p-value (asymptotic and
        permutation), plus the regret-vs-rho / regret-vs-spread relationships
        and regret-concentration statistics (how much regret the corrupted
        prompts actually carry).
  P0-3  Wilson 95% binomial CIs for every proportion quoted in the paper
        (top-1 agreement, zero-regret rate, rho<0.7 rate, prune survival),
        plus gate-vs-official significance tests (Fisher exact for
        proportions; two-sample bootstrap for mean rho / p10 rho / capture).
  P0-4  paired bootstrap over prompts for adaptive-vs-fixed tau at matched
        exploration speedup (E5), giving Delta = adaptive - fixed with 95% CI.
  P0-6  timing provenance: n / mean +- sd of the per-rollout latencies behind
        C_f and C_c on every grid, paired-bootstrap 95% CI and sd for each
        speedup ratio, and total measured generation GPU-hours (sum of
        per-rollout latencies over all paper grids).

Data conventions follow code/paper_figs/make_figs.py exactly (its loaders are
imported). Pure CPU; no new GPU runs.

Usage: python code/paper_figs/corruption_n944.py
"""
import glob
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, fisher_exact, mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figs import RESULTS, load_all, load_vbench, per_prompt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from b1_adaptive_tau import load_grid, capture_ratio, N as N_SEEDS  # noqa: E402

B = 10_000
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    """Wilson 95% score interval for a binomial proportion."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - hw, c + hw


def boot_stat(vals, stat=np.mean, B=B):
    vals = np.asarray(vals)
    idx = RNG.integers(0, len(vals), (B, len(vals)))
    s = np.array([stat(vals[i]) for i in idx])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def boot_spearman(x, y, B=B):
    """Percentile bootstrap CI (over prompts) for Spearman corr(x, y)."""
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    rs = []
    for _ in range(B):
        i = RNG.integers(0, n, n)
        r, _ = spearmanr(x[i], y[i])
        rs.append(r)
    rs = np.array(rs)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def perm_pvalue(x, y, B=B):
    """Two-sided permutation p for Spearman corr(x, y)."""
    r_obs, _ = spearmanr(x, y)
    y = np.asarray(y).copy()
    cnt = 0
    for _ in range(B):
        r, _ = spearmanr(x, RNG.permutation(y))
        if abs(r) >= abs(r_obs):
            cnt += 1
    return (cnt + 1) / (B + 1)


def two_sample_boot_p(a, b, stat=np.mean, B=B):
    """Two-sided bootstrap p for stat(a) != stat(b) (unpaired; different prompt sets)."""
    a, b = np.asarray(a), np.asarray(b)
    d_obs = stat(a) - stat(b)
    ds = []
    for _ in range(B):
        ds.append(stat(a[RNG.integers(0, len(a), len(a))])
                  - stat(b[RNG.integers(0, len(b), len(b))]))
    ds = np.array(ds)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    # p from the bootstrap distribution of the difference vs 0
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return float(d_obs), float(lo), float(hi), float(max(p, 1 / B))


def fmt_pct_ci(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}] (k={k}, n={n})"


# ---------------------------------------------------------------- P0-2
def corruption_block(pp, name):
    rho, spread, regret, rand = pp["rho"], pp["spread"], pp["regret"], pp["rand"]
    n = len(rho)
    print(f"\n=== P0-2 corruption analysis: {name} (n={n}) ===")

    r, p_asym = spearmanr(spread, rho)
    lo, hi = boot_spearman(spread, rho)
    p_perm = perm_pvalue(spread, rho)
    print(f"corr(spread, rho)   Spearman = {r:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"  p_asym = {p_asym:.2e}  p_perm = {p_perm:.2e}")

    r2, p2 = spearmanr(spread, regret)
    lo2, hi2 = boot_spearman(spread, regret)
    print(f"corr(spread, regret) Spearman = {r2:+.3f}  95% CI [{lo2:+.3f}, {hi2:+.3f}]  p = {p2:.2e}")
    r3, p3 = spearmanr(rho, regret)
    lo3, hi3 = boot_spearman(rho, regret)
    print(f"corr(rho, regret)   Spearman = {r3:+.3f}  95% CI [{lo3:+.3f}, {hi3:+.3f}]  p = {p3:.2e}")

    # regret concentration on the corrupted tail
    low = rho < 0.7
    k = int(low.sum())
    print(f"rho<0.7: {fmt_pct_ci(k, n)}")
    if k:
        print(f"  mean regret | rho<0.7  = {regret[low].mean():.3f}   (zero-regret rate "
              f"{fmt_pct_ci(int((regret[low]==0).sum()), k)})")
        print(f"  mean regret | rho>=0.7 = {regret[~low].mean():.3f}")
        print(f"  share of TOTAL regret carried by rho<0.7 prompts: "
              f"{100*regret[low].sum()/regret.sum():.1f}% (they are {100*k/n:.1f}% of prompts)")
        u, p_mw = mannwhitneyu(spread[low], spread[~low], alternative="less")
        print(f"  median spread: rho<0.7 {np.median(spread[low]):.3f} vs rho>=0.7 "
              f"{np.median(spread[~low]):.3f}  (MW one-sided p = {p_mw:.2e})")
        below_med = (spread[low] < np.median(spread)).mean()
        print(f"  fraction of rho<0.7 prompts in the bottom half of spread: {100*below_med:.0f}%")
    # where does the regret live?
    hi_reg = regret >= np.percentile(regret, 90)
    print(f"top-decile-regret prompts: median rho = {np.median(rho[hi_reg]):.3f}, "
          f"median spread = {np.median(spread[hi_reg]):.3f} "
          f"(suite median spread {np.median(spread):.3f})")
    q = np.quantile(spread, [0.25, 0.5, 0.75])
    for lab, m in [("Q1 (lowest spread)", spread <= q[0]), ("Q4 (highest spread)", spread > q[2])]:
        print(f"  spread {lab}: median rho {np.median(rho[m]):.3f}, mean regret {regret[m].mean():.3f}, "
              f"share of total regret {100*regret[m].sum()/regret.sum():.1f}%")
    cap = (rand - regret) / rand
    if k:
        cl, ch = cap[low & np.isfinite(cap)], cap[~low & np.isfinite(cap)]
        print(f"  capture | rho<0.7 = {100*cl.mean():.1f}%   capture | rho>=0.7 = {100*ch.mean():.1f}%")
    cap = cap[np.isfinite(cap)]
    clo, chi = boot_stat(cap)
    print(f"capture (mean of per-prompt ratios) = {100*cap.mean():.1f}% "
          f"[{100*clo:.1f}, {100*chi:.1f}]")
    mlo, mhi = boot_stat(rho, stat=np.median)
    print(f"median rho = {np.median(rho):.3f} [{mlo:.3f}, {mhi:.3f}]   "
          f"mean rho = {rho.mean():.3f} [{boot_stat(rho)[0]:.3f}, {boot_stat(rho)[1]:.3f}]")
    rlo, rhi = boot_stat(regret)
    print(f"mean regret = {regret.mean():.4f} [{rlo:.4f}, {rhi:.4f}]  "
          f"(random baseline {rand.mean():.3f})")
    return dict(rho=rho, spread=spread, regret=regret, rand=rand, cap=cap)


# ---------------------------------------------------------------- P0-3
def proportions_block(pp_gate, pp_vb):
    print("\n=== P0-3 proportion CIs (Wilson 95%) + gate-vs-official tests ===")
    ng, nv = len(pp_gate["rho"]), len(pp_vb["rho"])
    rows = [
        ("top-1 agreement", int(pp_gate["top1"].sum()), ng, int(pp_vb["top1"].sum()), nv),
        ("zero-regret rate", int((pp_gate["regret"] == 0).sum()), ng,
         int((pp_vb["regret"] == 0).sum()), nv),
        ("rho<0.7 rate", int((pp_gate["rho"] < 0.7).sum()), ng,
         int((pp_vb["rho"] < 0.7).sum()), nv),
    ]
    for name, kg, ngg, kv, nvv in rows:
        _, p = fisher_exact([[kg, ngg - kg], [kv, nvv - kv]])
        print(f"{name:<18} gate {fmt_pct_ci(kg, ngg)}   official {fmt_pct_ci(kv, nvv)}   "
              f"Fisher p = {p:.3f}")
    print(f"prune survival     gate {fmt_pct_ci(43, 50)}  (86% quoted in sec:abl-stack)")
    print(f"keep>=commit       gate {fmt_pct_ci(30, 50)}  (~60% quoted in sec:abl-temporal)")
    for name, tag_k in [("top-1 tau=0.05", 35), ("top-1 tau=0.20", 26)]:
        print(f"{name:<18} gate {fmt_pct_ci(tag_k, 50)}")
    # continuous gate-vs-official comparisons (unpaired bootstrap; prompt sets differ)
    for name, stat in [("mean rho", np.mean), ("p10 rho", lambda v: np.percentile(v, 10)),
                       ("median rho", np.median)]:
        d, lo, hi, p = two_sample_boot_p(pp_vb["rho"], pp_gate["rho"], stat=stat)
        print(f"{name:<18} official-gate Delta = {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  boot p = {p:.3f}")
    cg = (pp_gate["rand"] - pp_gate["regret"]) / pp_gate["rand"]
    cv = (pp_vb["rand"] - pp_vb["regret"]) / pp_vb["rand"]
    d, lo, hi, p = two_sample_boot_p(cv[np.isfinite(cv)], cg[np.isfinite(cg)])
    print(f"{'capture (mor)':<18} official-gate Delta = {100*d:+.1f}pts "
          f"[{100*lo:+.1f}, {100*hi:+.1f}]  boot p = {p:.3f}")


# ---------------------------------------------------------------- P0-4
def adaptive_block():
    print("\n=== P0-4 adaptive-vs-fixed tau: paired bootstrap over prompts ===")
    prompts, full, cached, _, _ = load_grid()
    # load_grid() pools latencies over ALL seeds incl. the later 16-seed width
    # extension shards; the paper's operating points (1.58/1.97/2.41x) are the
    # seeds 0-7 grid -> recompute Cf/Cc with the seed filter.
    lat = {}
    for tag, tau in [("v0", 0.10), ("tau005", 0.05), ("tau020", 0.20)]:
        d = load_latency_pairs(tag)
        lat[tau] = {v: float(np.mean([x[v] for x in d.values() if v in x]))
                    for v in ("full", "cached") if any(v in x for x in d.values())}
    Cf = lat[0.10]["full"]
    Cc = {t: lat[t]["cached"] for t in (0.05, 0.10, 0.20)}
    P = len(prompts)
    cap_fixed = {t: np.array([capture_ratio(full[i], int(np.argmax(cached[t][i])))
                              for i in range(P)]) for t in (0.05, 0.10, 0.20)}
    sp_fixed = {t: Cf / Cc[t] for t in (0.05, 0.10, 0.20)}
    pairs = list(itertools.combinations(range(N_SEEDS), 2))
    cap_ag = cap_fixed[0.20]
    cap_co = cap_fixed[0.05]
    cost_ag = N_SEEDS * Cc[0.20]
    cost_co = 2 * Cc[0.20] + N_SEEDS * Cc[0.05]
    sp_hat = np.array([[abs(cached[0.20][i, a] - cached[0.20][i, b]) for a, b in pairs]
                       for i in range(P)])  # [P, 28]
    pooled = sp_hat.ravel()

    # trace the frontier finely; pick the threshold whose speedup best matches fixed tau=0.10
    best = None
    for q in np.arange(0, 100.25, 0.25):
        t = np.percentile(pooled, q)
        agg = sp_hat >= t
        cost = np.where(agg, cost_ag, cost_co).mean()
        speedup = N_SEEDS * Cf / cost
        cap_pp = np.where(agg, cap_ag[:, None], cap_co[:, None]).mean(axis=1)  # per-prompt
        cand = (abs(speedup - sp_fixed[0.10]), q, t, speedup, cap_pp)
        if best is None or cand[0] < best[0]:
            best = cand
    _, q, t, speedup, cap_adp = best
    print(f"fixed tau=0.10 : capture {100*cap_fixed[0.10].mean():.1f}%  speedup {sp_fixed[0.10]:.2f}x")
    print(f"adaptive match : capture {100*cap_adp.mean():.1f}%  speedup {speedup:.2f}x  "
          f"(threshold = p{q:.2f} of pooled probe spreads, t = {t:.3f})")
    delta = cap_adp - cap_fixed[0.10]
    idx = RNG.integers(0, P, (B, P))
    dmeans = delta[idx].mean(axis=1)
    lo, hi = np.percentile(dmeans, [2.5, 97.5])
    print(f"paired Delta (adaptive - fixed) = {100*delta.mean():+.2f} pts  "
          f"95% CI [{100*lo:+.2f}, {100*hi:+.2f}]  (paired bootstrap over {P} prompts, B={B})")
    print(f"  P(Delta >= 0 in bootstrap) = {float((dmeans >= 0).mean()):.3f}")
    # conservative endpoint for the second claim in sec:abl-adaptive
    agg0 = sp_hat >= np.inf  # never aggressive -> all-conservative arm
    cost0 = np.where(agg0, cost_ag, cost_co).mean()
    print(f"conservative endpoint: capture {100*cap_co.mean():.1f}% at {N_SEEDS*Cf/cost0:.2f}x "
          f"(fixed tau=0.05: {100*cap_co.mean():.1f}% at {sp_fixed[0.05]:.2f}x)")


# ---------------------------------------------------------------- P0-6
def load_latency_pairs(tag, dedupe=False):
    """-> dict[(prompt, seed)] = {variant: latency}, seeds 0-7 only."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, f"b1_gate_{tag}", "scores_shard*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r["seed"] >= 8:
                continue
            rows[(r["prompt"], r["seed"], r["variant"])] = r["latency"]
    d = defaultdict(dict)
    for (p, s, v), lat in rows.items():
        d[(p, s)][v] = lat
    return d


def speedup_ci(pairs_full, pairs_cached, paired_keys=None):
    """Ratio-of-means speedup with bootstrap 95% CI and sd."""
    f, c = np.asarray(pairs_full), np.asarray(pairs_cached)
    if paired_keys is not None:  # paired resampling over (prompt, seed) units
        n = len(f)
        idx = RNG.integers(0, n, (B, n))
        r = f[idx].mean(axis=1) / c[idx].mean(axis=1)
    else:
        r = np.array([f[RNG.integers(0, len(f), len(f))].mean()
                      / c[RNG.integers(0, len(c), len(c))].mean() for _ in range(B)])
    return f.mean() / c.mean(), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), float(r.std())


def timing_block():
    print("\n=== P0-6 timing provenance ===")
    v0 = load_latency_pairs("v0")
    keys = sorted(k for k, d in v0.items() if "full" in d and "cached" in d)
    lf = np.array([v0[k]["full"] for k in keys])
    lc = np.array([v0[k]["cached"] for k in keys])
    print(f"gate v0 (tau=0.10): n={len(lf)} full, n={len(lc)} cached (paired)")
    print(f"  C_f = {lf.mean():.1f} +- {lf.std():.1f} s   C_c = {lc.mean():.1f} +- {lc.std():.1f} s")
    s, lo, hi, sd = speedup_ci(lf, lc, paired_keys=True)
    print(f"  speedup = {s:.2f}x  95% CI [{lo:.2f}, {hi:.2f}]  (bootstrap sd {sd:.3f})")
    for tag, tau in [("tau005", 0.05), ("tau020", 0.20)]:
        d = load_latency_pairs(tag)
        lc2 = np.array([v["cached"] for v in d.values() if "cached" in v])
        print(f"gate {tag} (tau={tau}): n={len(lc2)} cached, C_c = {lc2.mean():.1f} +- {lc2.std():.1f} s, "
              f"speedup vs v0 C_f = {lf.mean()/lc2.mean():.2f}x")
    vb = load_latency_pairs("vbench")
    keys = sorted(k for k, d in vb.items() if "full" in d and "cached" in d)
    lfv = np.array([vb[k]["full"] for k in keys])
    lcv = np.array([vb[k]["cached"] for k in keys])
    print(f"official vbench (tau=0.10, deduped, paired (prompt,seed)): n={len(lfv)} each")
    print(f"  C_f = {lfv.mean():.1f} +- {lfv.std():.1f} s   C_c = {lcv.mean():.1f} +- {lcv.std():.1f} s")
    s, lo, hi, sd = speedup_ci(lfv, lcv, paired_keys=True)
    print(f"  speedup = {s:.2f}x  95% CI [{lo:.2f}, {hi:.2f}]  (bootstrap sd {sd:.3f})")

    # total measured generation GPU-hours: sum of per-rollout latencies, all paper grids
    dirs = ["b1_gate_v0", "b1_gate_tau005", "b1_gate_tau020", "b1_gate_vbench",
            "b1_gate_wan14b", "b1_gate_wan14b_tau005", "b1_gate_wan14b_tau020",
            "b1_gate_cog5b_v0", "b1_gate_cog5b_tau005", "b1_gate_cog5b_tau020",
            "b1_stack_v0", "b1_temporal",
            "a1_none", "a1_pab", "a1_cfgcache", "a1_teacache", "a1_easycache-ours"]
    total = 0.0
    print("measured generation time by grid (sum of recorded per-rollout latencies):")
    for d in dirs:
        secs = 0.0
        seen = set()
        for f in sorted(glob.glob(os.path.join(RESULTS, d, "*.jsonl"))):
            if not os.path.isfile(f):
                continue
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r.get("prompt"), r.get("seed"), r.get("variant"), r.get("tau"))
                if key in seen:  # cross-partition duplicates count once
                    continue
                seen.add(key)
                for field in ("latency", "gen_latency", "latency_s"):
                    if isinstance(r.get(field), (int, float)):
                        secs += r[field]
                        break
        if secs:
            print(f"  {d:<24} {secs/3600:8.1f} h")
            total += secs
    print(f"  {'TOTAL (generation only)':<24} {total/3600:8.1f} h")


# ---------------------------------------------------------------- main
def main():
    data, cost = load_all()
    vb, vb_cost = load_vbench()
    pp_gate = per_prompt(data[0.10])
    pp_vb = per_prompt(vb)
    print(f"loaded: gate n={len(pp_gate['rho'])}, official n={len(pp_vb['rho'])} "
          f"(fully covered, seeds 0-7, deduped)")

    corruption_block(pp_gate, "gate grid, tau=0.10")
    corruption_block(pp_vb, "OFFICIAL VBench suite, tau=0.10")
    proportions_block(pp_gate, pp_vb)
    adaptive_block()
    timing_block()


if __name__ == "__main__":
    main()
