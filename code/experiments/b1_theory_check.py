"""Numerical verification of CachedSearch theory appendix (paper/sections/theory.tex).

Checks:
 1. e_N = E[max of N std normals] table.
 2. Prop 1: E[Z_{i-hat}] = r * e_N  (selection value under Gaussian-copula noise);
    expected regret sigma*(1-r)*e_N; capture ratio = r, independent of N.
 3. Spearman link r = 2 sin(pi * rho_S / 6).
 4. Prop 2: top-1 agreement p1(N, r) by Monte Carlo; median regret = 0 iff p1 >= 1/2.
 5. Prop 3 (heterogeneous prompts): aggregate capture = spread-weighted mean of r_p;
    checked on REAL gate data (b1_gate_v0), incl. predicted vs measured capture/regret.
"""
import glob, json, itertools, os
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

rng = np.random.default_rng(0)
M = 2_000_000  # MC samples

print("=" * 70)
print("1) e_N = E[max of N iid std normals]")
for N in (2, 4, 8, 16):
    Z = rng.standard_normal((M // 4, N))
    print(f"   N={N:2d}: MC {Z.max(1).mean():.4f}")
print("   closed forms: e_2 = 1/sqrt(pi) =", 1 / np.sqrt(np.pi))

print("=" * 70)
print("2) Prop 1: E[Z_selected] = r*e_N; capture = r for all N")
for r in (0.832, 0.912):
    for N in (2, 4, 8):
        Zh = rng.standard_normal((M // 8, N))          # noisy (cached) latent
        eta = rng.standard_normal((M // 8, N))
        Z = r * Zh + np.sqrt(1 - r * r) * eta          # true (full) latent
        idx = Zh.argmax(1)
        sel = Z[np.arange(len(Z)), idx].mean()
        eN = Z.max(1).mean()
        print(f"   r={r} N={N}: E[Z_sel] MC {sel:.4f} vs r*e_N {r*eN:.4f}"
              f" | capture MC {sel/eN:.4f} (pred {r})")

print("=" * 70)
print("3) Spearman <-> copula param: r = 2 sin(pi*rho_S/6)")
for rho_s in (0.820, 0.905):
    print(f"   rho_S={rho_s} -> r = {2*np.sin(np.pi*rho_s/6):.4f}")
# sanity: simulate and measure spearman
r = 0.912
Zh = rng.standard_normal(500_000); Z = r * Zh + np.sqrt(1-r*r)*rng.standard_normal(500_000)
print(f"   check: r=0.912 -> Spearman MC {spearmanr(Z, Zh)[0]:.4f} (target 0.905)")

print("=" * 70)
print("4) Prop 2: top-1 agreement p1(N,r), MC")
for r in (0.832, 0.912):
    for N in (2, 4, 8):
        Zh = rng.standard_normal((M // 8, N))
        Z = r * Zh + np.sqrt(1 - r * r) * rng.standard_normal((M // 8, N))
        p1 = (Z.argmax(1) == Zh.argmax(1)).mean()
        print(f"   r={r} N={N}: p1 = {p1:.3f}")

print("=" * 70)
print("5) REAL DATA (b1_gate_v0): per-prompt (sigma_p, rho_p); predicted vs measured")
RESULTS = os.environ.get("CACHEDSEARCH_RESULTS", "./results")
by = defaultdict(dict)
for f in glob.glob(os.path.join(RESULTS, "b1_gate_v0", "scores_shard*.jsonl")):
    for line in open(f):
        rec = json.loads(line)
        by[rec["prompt"]][(rec["seed"], rec["variant"])] = rec["score"]

sig, rho, reg_meas, gap_meas = [], [], [], []
cap_num, cap_den = 0.0, 0.0
e8 = 1.4236  # E[max of 8 std normals]
pred_reg = []
N = 8
for prompt, d in by.items():
    seeds = sorted({s for s, v in d if (s, "full") in d and (s, "cached") in d})
    if len(seeds) < 8:
        continue
    fu = np.array([d[(s, "full")] for s in seeds])
    ca = np.array([d[(s, "cached")] for s in seeds])
    rs, _ = spearmanr(fu, ca)
    r_p = 2 * np.sin(np.pi * rs / 6)          # copula param from measured Spearman
    s_p = fu.std(ddof=0)
    sig.append(s_p); rho.append(rs)
    # measured regret over the full N=8 pool
    reg_meas.append(fu.max() - fu[ca.argmax()])
    gap_meas.append(fu.max() - fu.mean())     # oracle gain over random pick
    # model predictions
    pred_reg.append(s_p * (1 - r_p) * e8)
    cap_num += s_p * r_p * e8
    cap_den += s_p * e8
sig, rho, reg_meas, gap_meas, pred_reg = map(np.array, (sig, rho, reg_meas, gap_meas, pred_reg))
r_all = 2 * np.sin(np.pi * rho / 6)
print(f"   prompts: {len(sig)}")
print(f"   corr(spread, rho) Spearman = {spearmanr(sig, rho)[0]:+.3f}  (expected: +0.25)")
print(f"   mean rho {rho.mean():.3f}, median rho {np.median(rho):.3f}")
print(f"   MEASURED  mean regret {reg_meas.mean():.3f}, median {np.median(reg_meas):.3f}, "
      f"zero-regret frac {(reg_meas<=1e-9).mean():.2f}")
print(f"   PREDICTED mean regret sigma*(1-r)*e_8 = {pred_reg.mean():.3f}")
print(f"   MEASURED  aggregate capture 1 - sum(reg)/sum(gap) = {1 - reg_meas.sum()/gap_meas.sum():.3f}")
print(f"   PREDICTED capture, spread-weighted r  = {cap_num/cap_den:.3f}")
print(f"   PREDICTED capture, unweighted mean r  = {r_all.mean():.3f}")
print(f"   regret share on below-median-spread prompts: "
      f"measured {reg_meas[sig < np.median(sig)].sum()/max(reg_meas.sum(),1e-9):.2f}")

print("=" * 70)
print("6) Cost / break-even (measured constants)")
Cf, Cc = 68.3, 34.7
g = Cc / Cf
print(f"   gamma = Cc/Cf = {g:.3f}; break-even N* = 1/(1-gamma) = {1/(1-g):.2f}")
for N in (2, 4, 8):
    print(f"   N={N}: commit {N*Cc+Cf:.0f}s vs full BoN {N*Cf:.0f}s "
          f"({(N*Cc+Cf)/(N*Cf)*100:.1f}%) ; keep {N*Cc:.0f}s ({Cc/Cf*100:.1f}%)")
