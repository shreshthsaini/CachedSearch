"""Why does measured gain capture RISE with search width N?

Proposition (capture) of paper/sections/theory.tex predicts capture(N, r) = r,
independent of N, under a Gaussian copula WITH Gaussian marginals. The 16-seed
re-simulation measures capture rising: ~91 -> 92 -> 96 % at N = 4/8/16
(sections/experiments.tex, fig f8). This script diagnoses the gap on the REAL
b1_gate_v0 grid (50 prompts x 16 seeds x {full, cached}, tau=0.10).

Candidate mechanisms tested:
  M0  homogeneous Gaussian copula + Gaussian marginal ......... capture = r, FLAT
  M1  per-prompt heterogeneity (sigma_p, r_p), Gaussian marginals:
      aggregate capture = sum_p sigma_p r_p / sum_p sigma_p ... STILL FLAT
      (the width factor e_N multiplies numerator and denominator alike and
      cancels; heterogeneity alone CANNOT produce any N-dependence).
  M2g CONTROL: full plug-in machinery (finite 16-pool, subset estimator,
      per-prompt r_p) but with each prompt's scores replaced by Gaussianized
      values mu_p + sigma_p * (normal scores of the same ranks).
      If M2g is flat, neither the estimator nor the finite pool nor the
      heterogeneity drives the rise.
  M2  SEMI-EMPIRICAL PLUG-IN: keep each prompt's OBSERVED 16 full scores
      (empirical marginal F_p) and its calibrated copula r_p; model only the
      cached RANKING as Gaussian rank noise; run the identical C(16,N)-subset
      estimator. If M2 rises like the data, the mechanism is the non-Gaussian
      (bounded / near-tied-at-the-top) score marginal interacting with
      max-selection: gain keeps growing with N while the value lost to rank
      noise saturates, because a larger pool contains several near-best
      candidates and surfacing ANY of them is enough.
  M3  iid variant of M2 (fresh iid draws from the empirical quantile function
      instead of without-replacement subsets) -- separates marginal shape from
      the finite-pool/subset-estimator structure.
  M4  SCORE-ADDITIVE NOISE: cached ranking modeled as rank(fu + eps_p * xi),
      xi ~ N(0,1), with eps_p calibrated per prompt so the model's expected
      Spearman matches the measured rho_p (same calibration target as M2).
      On a GAUSSIAN marginal M4 and M2 are the SAME model (additive noise on
      a Gaussian is the Gaussian copula); on the real, tie-heavy marginals
      they differ in WHERE the rank noise lands: additive noise corrupts the
      near-tied mid-pack (driving Spearman down) but rarely flips candidates
      separated by more than ~eps -- so top selection stays accurate and
      capture rises with N while global Spearman is unchanged.

Also reported: measured mean regret by N vs the Gaussian prediction
sigma(1-r)e_N (which GROWS ~ e_N; the data saturates instead), a brute-force
check that the exact subset-weight formulas reproduce b1_simulate.py's
enumeration, and a prompt-level bootstrap CI on the measured capture curve.

No GPU. Data: $CACHEDSEARCH_RESULTS/b1_gate_v0/scores_shard*.jsonl.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.special import comb, ndtr, ndtri
from scipy.stats import spearmanr

RESULTS = os.environ.get("CACHEDSEARCH_RESULTS", "./results")
NS = (2, 4, 8, 16)
S = 16                    # pool size (seeds per prompt)
R_REPS = 4000             # noise replicates per prompt for M2/M2g
R_IID = 4000              # replicates for M3
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- data
def load_pool():
    by = defaultdict(dict)
    for f in glob.glob(os.path.join(RESULTS, "b1_gate_v0", "scores_shard*.jsonl")):
        for line in open(f):
            rec = json.loads(line)
            by[rec["prompt"]][(rec["seed"], rec["variant"])] = rec["score"]
    prompts = []
    for prompt, d in sorted(by.items()):
        seeds = sorted({s for s, v in d if (s, "full") in d and (s, "cached") in d})
        if len(seeds) < S:
            continue
        seeds = seeds[:S]
        fu = np.array([d[(s, "full")] for s in seeds])
        ca = np.array([d[(s, "cached")] for s in seeds])
        prompts.append((prompt, fu, ca))
    return prompts


# ------------------------------------------- exact subset-mean machinery
# For subsets of size N drawn uniformly from a pool of S items:
#   P[item i is the argmax of X over the subset | i's X-rank is k (1=lowest)]
#     * P[i in subset]  summed over subsets  =  C(k-1, N-1) / C(S, N)
# so  E_subsets[ Y[argmax_subset X] ] = sum_i Y_i * C(rank_X(i)-1, N-1) / C(S,N).
# "single" in b1_simulate.py is Y[min-seed member]: weight C(S-j, N-1)/C(S,N)
# for the item at ascending-seed position j (1..S). Both verified brute-force.
W_ARGMAX = {N: comb(np.arange(S) - 0.0, N - 1) for N in NS}  # placeholder, built below


def _build_weights():
    for N in NS:
        k = np.arange(1, S + 1)                       # rank 1..S (1 = lowest)
        W_ARGMAX[N] = comb(k - 1, N - 1) / comb(S, N)  # weight by rank of X


_build_weights()


def e_subset_argmax(y, x, N):
    """E over all C(S,N) subsets of y[argmax of x within subset]."""
    ranks = np.argsort(np.argsort(x)) + 1             # 1 = lowest
    return float(np.sum(y * W_ARGMAX[N][ranks - 1]))


def e_subset_single(y, N):
    """E over subsets of y[first (min-seed) member]; y is in ascending-seed order."""
    j = np.arange(1, S + 1)
    w = comb(S - j, N - 1) / comb(S, N)
    return float(np.sum(y * w))


def capture_curve(prompts, ca_of):
    """Aggregate capture at each N; ca_of(p_idx, fu, ca) -> cached-score vector."""
    out = {}
    for N in NS:
        num = den = 0.0
        for i, (_, fu, ca) in enumerate(prompts):
            c = ca_of(i, fu, ca)
            sng = e_subset_single(fu, N)
            num += e_subset_argmax(fu, c, N) - sng
            den += e_subset_argmax(fu, fu, N) - sng
        out[N] = num / den
    return out


# ---------------------------------------------------------------- main
prompts = load_pool()
P = len(prompts)
print(f"loaded {P} prompts x {S} seeds (full+cached) from b1_gate_v0")
assert P == 50 and all(len(fu) == S for _, fu, _ in prompts)

# per-prompt calibration
rho_p = np.array([spearmanr(fu, ca)[0] for _, fu, ca in prompts])
r_p = np.clip(2 * np.sin(np.pi * rho_p / 6), -0.999, 0.999)
sig_p = np.array([fu.std(ddof=0) for _, fu, _ in prompts])

print("=" * 74)
print("0) sanity: exact subset weights vs brute-force enumeration (b1_simulate)")
import itertools
for N in (2, 4):
    diffs = []
    for _, fu, ca in prompts[:6]:
        acc_sel = acc_max = acc_sng = 0.0
        idx = range(S)
        n_sub = 0
        for sub in itertools.combinations(idx, N):
            sub = list(sub)
            f, c = fu[sub], ca[sub]
            acc_sel += f[int(np.argmax(c))]
            acc_max += f.max()
            acc_sng += f[0]
            n_sub += 1
        diffs.append(abs(acc_sel / n_sub - e_subset_argmax(fu, ca, N))
                     + abs(acc_max / n_sub - e_subset_argmax(fu, fu, N))
                     + abs(acc_sng / n_sub - e_subset_single(fu, N)))
    print(f"   N={N}: max |enumeration - closed form| = {max(diffs):.2e}")

print("=" * 74)
print("1) MEASURED capture curve (identical estimator to b1_simulate.py)")
meas = capture_curve(prompts, lambda i, fu, ca: ca)
print("   " + "  ".join(f"N={N}: {meas[N]*100:.1f}%" for N in NS))

# 8-seed restriction should reproduce tab:main (89.9 / 89.6 / 94.7 at N=2/4/8)
by8 = []
for _, fu, ca in prompts:
    by8.append((None, fu[:8], ca[:8]))
print("   8-seed restriction (seeds 0-7), brute force, vs tab:main:")
for N in (2, 4, 8):
    num = den = 0.0
    for _, fu, ca in by8:
        sel = mx = sng = 0.0
        n_sub = 0
        for sub in itertools.combinations(range(8), N):
            f, c = fu[list(sub)], ca[list(sub)]
            sel += f[int(np.argmax(c))]; mx += f.max(); sng += f[0]; n_sub += 1
        num += (sel - sng) / n_sub
        den += (mx - sng) / n_sub
    print(f"     N={N}: {num/den*100:.1f}%")

# prompt-level bootstrap CI on the measured 16-seed curve
per_prompt = {N: [] for N in NS}
for _, fu, ca in prompts:
    for N in NS:
        sng = e_subset_single(fu, N)
        per_prompt[N].append((e_subset_argmax(fu, ca, N) - sng,
                              e_subset_argmax(fu, fu, N) - sng))
boot = {N: [] for N in NS}
for _ in range(4000):
    pick = rng.integers(0, P, P)
    for N in NS:
        arr = np.array(per_prompt[N])[pick]
        boot[N].append(arr[:, 0].sum() / arr[:, 1].sum())
print("   bootstrap 95% CI (prompt resampling):")
for N in NS:
    lo, hi = np.percentile(boot[N], [2.5, 97.5])
    print(f"     N={N:2d}: {meas[N]*100:.1f}%  [{lo*100:.1f}, {hi*100:.1f}]")

print("=" * 74)
print("2) M0/M1: any Gaussian-marginal model is FLAT in N")
print(f"   M0 homogeneous: capture = r  (median-cal r = {2*np.sin(np.pi*np.median(rho_p)/6):.3f})")
m1 = float(np.sum(sig_p * r_p) / np.sum(sig_p))
print(f"   M1 heterogeneous (sigma_p, r_p): sum sig*r / sum sig = {m1:.3f} at EVERY N")
print("   (e_N multiplies numerator and denominator alike -> cancels exactly)")

print("=" * 74)
print("3) M2g CONTROL (plug-in machinery, Gaussianized marginals) and")
print("   M2 SEMI-EMPIRICAL PLUG-IN (observed scores + Gaussian rank noise @ r_p)")
# Blom normal scores for ranks 1..S
blom = ndtri((np.arange(1, S + 1) - 0.375) / (S + 0.25))


def model_curve(score_of):
    """score_of(i) -> the 16-vector of 'full scores' used for prompt i."""
    num = {N: 0.0 for N in NS}
    den = {N: 0.0 for N in NS}
    for i, (_, fu, ca) in enumerate(prompts):
        y = score_of(i)
        ranks_true = np.argsort(np.argsort(y))            # 0..S-1, by value
        z = blom[ranks_true]                              # latent by true rank
        eps = rng.standard_normal((R_REPS, S))
        zh = r_p[i] * z[None, :] + np.sqrt(1 - r_p[i] ** 2) * eps
        rk = np.argsort(np.argsort(zh, axis=1), axis=1)   # cached-rank 0..S-1
        for N in NS:
            w = W_ARGMAX[N][rk]                           # (R,S) weights
            sel = (w * y[None, :]).sum(axis=1).mean()
            sng = e_subset_single(y, N)
            num[N] += sel - sng
            den[N] += e_subset_argmax(y, y, N) - sng
    return {N: num[N] / den[N] for N in NS}


mu_p = np.array([fu.mean() for _, fu, _ in prompts])
m2g = model_curve(lambda i: mu_p[i] + sig_p[i] * blom)    # Gaussianized control
m2 = model_curve(lambda i: prompts[i][1])                 # observed marginals
print("   M2g (Gaussian marginals): " + "  ".join(f"N={N}: {m2g[N]*100:.1f}%" for N in NS))
print("   M2  (empirical marginals): " + "  ".join(f"N={N}: {m2[N]*100:.1f}%" for N in NS))

print("=" * 74)
print("4) M3: iid plug-in (fresh draws from empirical quantiles, no shared pool)")
m3 = {}
for N in NS:
    num = den = 0.0
    for i, (_, fu, _) in enumerate(prompts):
        fs = np.sort(fu)
        z = rng.standard_normal((R_IID, N))
        zh = r_p[i] * z + np.sqrt(1 - r_p[i] ** 2) * rng.standard_normal((R_IID, N))
        sv = fs[np.minimum((ndtr(z) * S).astype(int), S - 1)]   # F_p^{-1}(Phi(Z))
        sel = sv[np.arange(R_IID), zh.argmax(axis=1)].mean()
        num += sel - fu.mean()
        den += sv.max(axis=1).mean() - fu.mean()
    m3[N] = num / den
print("   M3: " + "  ".join(f"N={N}: {m3[N]*100:.1f}%" for N in NS))

print("=" * 74)
print("4b) paired bootstrap: is the measured RISE significant?")
for Na, Nb in ((4, 16), (8, 16), (2, 8)):
    diffs = []
    for _ in range(4000):
        pick = rng.integers(0, P, P)
        a = np.array(per_prompt[Na])[pick]
        b = np.array(per_prompt[Nb])[pick]
        diffs.append(b[:, 0].sum() / b[:, 1].sum() - a[:, 0].sum() / a[:, 1].sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"   capture(N={Nb}) - capture(N={Na}) = "
          f"{(meas[Nb]-meas[Na])*100:+.1f} pts  [95% CI {lo*100:+.1f}, {hi*100:+.1f}]")

print("=" * 74)
print("5) M4: score-additive noise, eps_p calibrated to the measured Spearman")


def mean_spearman_additive(fu, eps, reps=768, seed=1):
    g = np.random.default_rng(seed)
    n = len(fu)
    xi = g.standard_normal((reps, n))
    ch = fu[None, :] + eps * xi
    rk_f = np.argsort(np.argsort(fu))
    rk_c = np.argsort(np.argsort(ch, axis=1), axis=1)
    d2 = ((rk_c - rk_f[None, :]) ** 2).sum(axis=1)
    return float((1 - 6 * d2 / (n * (n * n - 1))).mean())


eps_p = np.zeros(P)
for i, (_, fu, _) in enumerate(prompts):
    target = rho_p[i]
    if target >= 0.9995:
        eps_p[i] = 0.0
        continue
    lo, hi = 1e-6, 10.0 * (fu.std(ddof=0) + 1e-9)
    while mean_spearman_additive(fu, hi) > target:
        hi *= 2
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if mean_spearman_additive(fu, mid) > target:
            lo = mid
        else:
            hi = mid
    eps_p[i] = 0.5 * (lo + hi)
print(f"   calibrated eps_p: median {np.median(eps_p):.3f}, "
      f"mean {eps_p.mean():.3f} reward units "
      f"(median eps/sigma = {np.median(eps_p/np.maximum(sig_p,1e-9)):.2f})")

m4 = {N: [0.0, 0.0] for N in NS}
reg4 = {N: 0.0 for N in NS}
top1_m4 = 0.0
for i, (_, fu, _) in enumerate(prompts):
    xi = rng.standard_normal((R_REPS, S))
    ch = fu[None, :] + eps_p[i] * xi
    rk = np.argsort(np.argsort(ch, axis=1), axis=1)
    top1_m4 += (ch.argmax(axis=1) == fu.argmax()).mean() / P
    for N in NS:
        w = W_ARGMAX[N][rk]
        sel = (w * fu[None, :]).sum(axis=1).mean()
        sng = e_subset_single(fu, N)
        mx = e_subset_argmax(fu, fu, N)
        m4[N][0] += sel - sng
        m4[N][1] += mx - sng
        reg4[N] += mx - sel
m4 = {N: v[0] / v[1] for N, v in m4.items()}
print("   M4 capture: " + "  ".join(f"N={N}: {m4[N]*100:.1f}%" for N in NS))
print("   M4 mean regret by N: "
      + "  ".join(f"N={N}: {reg4[N]/P:.3f}" for N in NS))

print("=" * 74)
print("5b) noise placement: does the cached argmax stay in the true top of the")
print("    pool more than Gaussian-copula rank noise predicts? (pool of 16)")
top1_meas = np.mean([float(ca.argmax() == fu.argmax()) for _, fu, ca in prompts])
topk_meas = {k: np.mean([float(np.argsort(np.argsort(fu))[ca.argmax()] >= S - k)
                         for _, fu, ca in prompts]) for k in (1, 2, 3)}
top1_m2 = 0.0
topk_m2 = {k: 0.0 for k in (1, 2, 3)}
for i, (_, fu, _) in enumerate(prompts):
    ranks_true = np.argsort(np.argsort(fu))
    z = blom[ranks_true]
    eps = rng.standard_normal((R_REPS, S))
    zh = r_p[i] * z[None, :] + np.sqrt(1 - r_p[i] ** 2) * eps
    am = zh.argmax(axis=1)
    top1_m2 += (am == fu.argmax()).mean() / P
    for k in (1, 2, 3):
        topk_m2[k] += (ranks_true[am] >= S - k).mean() / P
print(f"   pool top-1 hit:  measured {top1_meas:.2f} | M2 Gaussian-copula "
      f"{top1_m2:.2f} | M4 additive {top1_m4:.2f}")
print(f"   argmax-ca in true top-k (measured vs M2): "
      + "  ".join(f"top-{k}: {topk_meas[k]:.2f}/{topk_m2[k]:.2f}" for k in (1, 2, 3)))

print("=" * 74)
print("5c) M5: t-copula plug-in -- same per-prompt (F_p, Spearman rho_p)")
print("    calibration as M2, but the rank-noise copula has upper-tail")
print("    dependence (bivariate t, one GLOBAL dof parameter nu). nu is")
print("    selected to match the measured pool top-1/top-2 concordance")
print("    (NOT the capture curve), then the capture curve is PREDICTED.")
from scipy.special import stdtr, stdtrit

blom_u = (np.arange(1, S + 1) - 0.375) / (S + 0.25)


def t_spearman_curve(nu, r_grid, n=200_000, seed=7):
    """Population Spearman of the bivariate-t copula on a grid of r."""
    g = np.random.default_rng(seed)
    w1 = g.standard_normal(n); w2 = g.standard_normal(n)
    mix = np.sqrt(g.chisquare(nu, n) / nu)
    out = []
    for r in r_grid:
        x = w1 / mix
        xh = (r * w1 + np.sqrt(1 - r * r) * w2) / mix
        out.append(12 * np.mean(stdtr(nu, x) * stdtr(nu, xh)) - 3)
    return np.array(out)


def m5_curve(nu):
    r_grid = np.linspace(0.0, 0.999, 60)
    rho_grid = t_spearman_curve(nu, r_grid)
    r_t = np.interp(rho_p, rho_grid, r_grid)          # per-prompt r for this nu
    x_pool = stdtrit(nu, blom_u)                      # latent by true rank
    cap = {N: [0.0, 0.0] for N in NS}
    top1 = top2 = 0.0
    for i, (_, fu, _) in enumerate(prompts):
        ranks_true = np.argsort(np.argsort(fu))
        x = x_pool[ranks_true]
        r = r_t[i]
        # conditional law of the t-copula: Xhat | X=x ~ r x + s(x) t_{nu+1}
        scale = np.sqrt((nu + x ** 2) * (1 - r * r) / (nu + 1))
        tdraw = rng.standard_t(nu + 1, (R_REPS, S))
        xh = r * x[None, :] + scale[None, :] * tdraw
        rk = np.argsort(np.argsort(xh, axis=1), axis=1)
        am = xh.argmax(axis=1)
        top1 += (am == fu.argmax()).mean() / P
        top2 += (ranks_true[am] >= S - 2).mean() / P
        for N in NS:
            w = W_ARGMAX[N][rk]
            sel = (w * fu[None, :]).sum(axis=1).mean()
            sng = e_subset_single(fu, N)
            cap[N][0] += sel - sng
            cap[N][1] += e_subset_argmax(fu, fu, N) - sng
    return {N: v[0] / v[1] for N, v in cap.items()}, top1, top2


best = None
for nu in (3, 4, 5, 6, 8, 12, 20):
    c, t1, t2 = m5_curve(nu)
    err = abs(t1 - top1_meas) + abs(t2 - topk_meas[2])
    print(f"   nu={nu:>2}: top-1 {t1:.2f} top-2 {t2:.2f} "
          f"(measured {top1_meas:.2f}/{topk_meas[2]:.2f}) | capture "
          + " ".join(f"{c[N]*100:.1f}" for N in NS))
    if best is None or err < best[0]:
        best = (err, nu, c, t1, t2)
_, nu_star, m5, t1_star, t2_star = best
lam_u = {}  # upper-tail dependence at the median-calibrated r, for reporting
r_med = 2 * np.sin(np.pi * np.median(rho_p) / 6)
lam = 2 * stdtr(nu_star + 1, -np.sqrt((nu_star + 1) * (1 - r_med) / (1 + r_med)))
print(f"   selected nu = {nu_star} (top-concordance match; lambda_U at "
      f"median r = {lam:.2f})")
print("   M5 capture: " + "  ".join(f"N={N}: {m5[N]*100:.1f}%" for N in NS))

print("=" * 74)
print("6) regret growth: Gaussian model says regret ~ sigma(1-r)e_N (GROWS);")
print("   the data saturates (this is the same mechanism seen from the loss side)")
eN = {2: 0.5642, 4: 1.0294, 8: 1.4236, 16: 1.7660}
for N in NS:
    reg = gain = 0.0
    for _, fu, ca in prompts:
        sng = e_subset_single(fu, N)
        reg += e_subset_argmax(fu, fu, N) - e_subset_argmax(fu, ca, N)
        gain += e_subset_argmax(fu, fu, N) - sng
    pred = float(np.sum(sig_p * (1 - r_p)) * eN[N])
    print(f"   N={N:2d}: measured mean regret {reg/P:.3f} (gain {gain/P:.3f})"
          f" | Gaussian-marginal prediction {pred/P:.3f}")

print("=" * 74)
print("7) small-sample calibration fix: with n=16 pairs the sample Spearman is")
print("   attenuated, E[r_S] = ((n-2) rho_S + 3 tau)/(n+1)  [Hoeffding/Kendall,")
print("   exact for iid continuous pairs]. Calibrating r_p from the raw sample")
print("   Spearman therefore UNDERSTATES r_p; invert the identity instead.")
n_ = S


def corrected_r_gauss(s_obs):
    lo, hi = 0.0, 0.9999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        rho = (6 / np.pi) * np.arcsin(mid / 2)
        tau = (2 / np.pi) * np.arcsin(mid)
        e_rs = ((n_ - 2) * rho + 3 * tau) / (n_ + 1)
        if e_rs < s_obs:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


r_pc = np.array([corrected_r_gauss(s) if s < 0.9995 else 0.9995 for s in rho_p])
print(f"   median r_p: raw-calibrated {np.median(r_p):.3f} -> "
      f"bias-corrected {np.median(r_pc):.3f}")


def m2_curve(rvec, collect_reps=False):
    cap = {N: [0.0, 0.0] for N in NS}
    reps = {N: np.zeros(R_REPS) for N in NS}
    den_tot = {N: 0.0 for N in NS}
    top1 = top2 = 0.0
    for i, (_, fu, _) in enumerate(prompts):
        ranks_true = np.argsort(np.argsort(fu))
        z = blom[ranks_true]
        eps = rng.standard_normal((R_REPS, S))
        zh = rvec[i] * z[None, :] + np.sqrt(1 - rvec[i] ** 2) * eps
        rk = np.argsort(np.argsort(zh, axis=1), axis=1)
        am = zh.argmax(axis=1)
        top1 += (am == fu.argmax()).mean() / P
        top2 += (ranks_true[am] >= S - 2).mean() / P
        for N in NS:
            w = W_ARGMAX[N][rk]
            sel_r = (w * fu[None, :]).sum(axis=1)         # per-replicate
            sng = e_subset_single(fu, N)
            mx = e_subset_argmax(fu, fu, N)
            cap[N][0] += sel_r.mean() - sng
            cap[N][1] += mx - sng
            reps[N] += sel_r - sng
            den_tot[N] += mx - sng
    curve = {N: v[0] / v[1] for N, v in cap.items()}
    dist = {N: reps[N] / den_tot[N] for N in NS} if collect_reps else None
    return curve, top1, top2, dist


m2c, t1_m2c, t2_m2c, m2c_dist = m2_curve(r_pc, collect_reps=True)
print("   M2c (Gaussian copula, corrected r_p): "
      + "  ".join(f"N={N}: {m2c[N]*100:.1f}%" for N in NS))
print(f"   M2c pool top-1/top-2: {t1_m2c:.2f}/{t2_m2c:.2f} "
      f"(measured {top1_meas:.2f}/{topk_meas[2]:.2f})")


def m5c_curve(nu, collect_reps=False):
    r_grid = np.linspace(0.0, 0.999, 60)
    rho_pop = t_spearman_curve(nu, r_grid)
    tau_pop = (2 / np.pi) * np.arcsin(r_grid)          # elliptical: exact
    e_rs_grid = ((n_ - 2) * rho_pop + 3 * tau_pop) / (n_ + 1)
    r_t = np.interp(rho_p, e_rs_grid, r_grid)
    x_pool = stdtrit(nu, blom_u)
    cap = {N: [0.0, 0.0] for N in NS}
    reps = {N: np.zeros(R_REPS) for N in NS}
    den_tot = {N: 0.0 for N in NS}
    top1 = top2 = 0.0
    for i, (_, fu, _) in enumerate(prompts):
        ranks_true = np.argsort(np.argsort(fu))
        x = x_pool[ranks_true]
        r = r_t[i]
        scale = np.sqrt((nu + x ** 2) * (1 - r * r) / (nu + 1))
        tdraw = rng.standard_t(nu + 1, (R_REPS, S))
        xh = r * x[None, :] + scale[None, :] * tdraw
        rk = np.argsort(np.argsort(xh, axis=1), axis=1)
        am = xh.argmax(axis=1)
        top1 += (am == fu.argmax()).mean() / P
        top2 += (ranks_true[am] >= S - 2).mean() / P
        for N in NS:
            w = W_ARGMAX[N][rk]
            sel_r = (w * fu[None, :]).sum(axis=1)
            sng = e_subset_single(fu, N)
            mx = e_subset_argmax(fu, fu, N)
            cap[N][0] += sel_r.mean() - sng
            cap[N][1] += mx - sng
            reps[N] += sel_r - sng
            den_tot[N] += mx - sng
    curve = {N: v[0] / v[1] for N, v in cap.items()}
    dist = {N: reps[N] / den_tot[N] for N in NS} if collect_reps else None
    return curve, top1, top2, dist


best = None
for nu in (3, 4, 5, 6, 8, 12, 20):
    c, t1, t2, _ = m5c_curve(nu)
    err = abs(t1 - top1_meas) + abs(t2 - topk_meas[2])
    print(f"   M5c nu={nu:>2}: top-1 {t1:.2f} top-2 {t2:.2f} | capture "
          + " ".join(f"{c[N]*100:.1f}" for N in NS))
    if best is None or err < best[0]:
        best = (err, nu)
nu_star_c = best[1]
m5c, t1_m5c, t2_m5c, m5c_dist = m5c_curve(nu_star_c, collect_reps=True)
print(f"   selected nu = {nu_star_c}")
print("   M5c capture: " + "  ".join(f"N={N}: {m5c[N]*100:.1f}%" for N in NS))

print("=" * 74)
print("8) is the measured curve inside the models' ESTIMATOR noise? (one noise")
print("   realization per prompt, 50 prompts -- distribution across replicates)")
for name, dist in (("M2c", m2c_dist), ("M5c", m5c_dist)):
    for N in NS:
        d = dist[N]
        q = float((d < meas[N]).mean())
        print(f"   {name} N={N:2d}: model mean {d.mean()*100:.1f}% "
              f"sd {d.std()*100:.1f} | measured {meas[N]*100:.1f}% "
              f"is at quantile {q:.3f}")

print("=" * 74)
print("9) consistency numbers for the paper")
# (a) MC check of the finite-n Spearman identity at n=16, r=0.913
r_chk, reps_chk = 0.913, 40_000
g = np.random.default_rng(11)
z1 = g.standard_normal((reps_chk, n_))
z2 = r_chk * z1 + np.sqrt(1 - r_chk**2) * g.standard_normal((reps_chk, n_))
rk1 = np.argsort(np.argsort(z1, axis=1), axis=1)
rk2 = np.argsort(np.argsort(z2, axis=1), axis=1)
d2 = ((rk1 - rk2) ** 2).sum(axis=1)
rs_mc = float((1 - 6 * d2 / (n_ * (n_ * n_ - 1))).mean())
rho_pop = (6 / np.pi) * np.arcsin(r_chk / 2)
tau_pop = (2 / np.pi) * np.arcsin(r_chk)
rs_form = ((n_ - 2) * rho_pop + 3 * tau_pop) / (n_ + 1)
print(f"   (a) E[sample r_S] at n=16, r=0.913: MC {rs_mc:.4f} vs formula "
      f"{rs_form:.4f} (population rho_S {rho_pop:.4f})")
# (b) corrected analytic flat capture (spread-weighted, Prop spread)
flat_c = float(np.sum(sig_p * r_pc) / np.sum(sig_p))
print(f"   (b) corrected spread-weighted flat capture sum sig*r_c/sum sig = "
      f"{flat_c:.3f} (raw: {m1:.3f})")
# (c) corrected Gaussian regret prediction sigma(1-r_c)e_N vs measured
print("   (c) mean regret, measured vs corrected Gaussian prediction:")
for N in NS:
    reg = 0.0
    for _, fu, ca in prompts:
        reg += e_subset_argmax(fu, fu, N) - e_subset_argmax(fu, ca, N)
    pred = float(np.sum(sig_p * (1 - r_pc)) * eN[N]) / P
    print(f"       N={N:2d}: measured {reg/P:.3f} | corrected prediction {pred:.3f}")
# (d) revisit rem:conservative on the 8-seed pool with corrected calibration
rho8 = np.array([spearmanr(fu[:8], ca[:8])[0] for _, fu, ca in prompts])
def corrected_r_gauss_n(s_obs, n):
    lo, hi = 0.0, 0.9999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        e_rs = ((n - 2) * (6/np.pi)*np.arcsin(mid/2) + 3*(2/np.pi)*np.arcsin(mid)) / (n + 1)
        (lo, hi) = (mid, hi) if e_rs < s_obs else (lo, mid)
    return 0.5 * (lo + hi)
r8_raw = np.clip(2 * np.sin(np.pi * rho8 / 6), -0.999, 0.999)
r8_c = np.array([corrected_r_gauss_n(s, 8) if s < 0.9995 else 0.9995 for s in rho8])
sig8 = np.array([fu[:8].std(ddof=0) for _, fu, _ in prompts])
reg8_meas = np.mean([fu[:8].max() - fu[:8][ca[:8].argmax()] for _, fu, ca in prompts])
print(f"   (d) 8-seed pool (rem:conservative): measured mean regret {reg8_meas:.3f}")
print(f"       predicted sigma(1-r)e_8: raw {np.mean(sig8*(1-r8_raw))*eN[8]:.3f} "
      f"(x{np.mean(sig8*(1-r8_raw))*eN[8]/reg8_meas:.1f}) -> corrected "
      f"{np.mean(sig8*(1-r8_c))*eN[8]:.3f} "
      f"(x{np.mean(sig8*(1-r8_c))*eN[8]/reg8_meas:.1f})")
print(f"       median rho8 {np.median(rho8):.3f}: raw r {np.median(r8_raw):.3f} "
      f"-> corrected {np.median(r8_c):.3f}")

print("=" * 74)
print("SUMMARY  (capture, %)")
print(f"   {'N':>3} {'measured':>9} {'M0/M1 flat':>11} {'M2 raw':>7}"
      f" {'M4 additive':>12} {'M5 raw':>7} {'M2c':>6} {'M5c':>6}")
for N in NS:
    print(f"   {N:>3} {meas[N]*100:>9.1f} {m1*100:>11.1f} {m2[N]*100:>7.1f}"
          f" {m4[N]*100:>12.1f} {m5[N]*100:>7.1f} {m2c[N]*100:>6.1f}"
          f" {m5c[N]*100:>6.1f}")
