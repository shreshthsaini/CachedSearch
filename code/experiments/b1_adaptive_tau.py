"""E5: adaptive per-prompt caching threshold ; pure simulation from EXISTING 3-tau data.

All three gate datasets (b1_gate_{tau005,v0,tau020}) share the same 50 prompts and
8 seeds; tau005/tau020 store cached rollouts only, the full-compute reference is
merged from v0 (full rollouts are tau-independent and seed-deterministic).

Policy (probe-then-commit-to-a-tau), motivated by the corruption analysis
(corr(spread, rho) = +0.25: HIGH-spread prompts keep their ranking under caching,
LOW-spread prompts corrupt ; cheaply, but conservatism there is also cheap to buy):
  1. Probe: run K=2 candidates at the cheapest setting tau=0.20; estimate the
     prompt's score spread as |s1 - s2| of the two cached scores.
  2. If spread_hat >= threshold t: AGGRESSIVE arm ; run the remaining 6 candidates
     at tau=0.20 and select on all 8 tau=0.20 cached scores.
     Exploration cost = 8 * Cc(0.20).
  3. Else: CONSERVATIVE arm ; rerun all 8 candidates at tau=0.05 and select on the
     tau=0.05 cached scores (the two probes are sunk cost).
     Exploration cost = 2 * Cc(0.20) + 8 * Cc(0.05).

Metrics per prompt: gain capture = (full[pick] - mean full) / (max full - mean full)
(the estimator of the paper's Eq. capture); exploration speedup = 8*Cf / cost.
Probe-pair choice is averaged over all C(8,2)=28 seed pairs (the selection inside
each arm does not depend on which pair probed ; only the arm decision does).
Sweeping t over quantiles of spread_hat traces the adaptive frontier between the
fixed tau=0.05 and tau=0.20 operating points.
"""
import argparse
import glob
import itertools
import json
import os
from collections import defaultdict

import numpy as np

RESULTS = os.environ.get("CACHEDSEARCH_RESULTS", "./results")
TAGS = {0.05: "tau005", 0.10: "v0", 0.20: "tau020"}
N = 8


def load_grid():
    """-> prompts P; full[P,8]; cached {tau: [P,8]}; Cf; Cc {tau}."""
    by = {tau: defaultdict(dict) for tau in TAGS}
    lat = {tau: [] for tau in TAGS}
    lat_full = []
    for tau, tag in TAGS.items():
        for f in glob.glob(os.path.join(RESULTS, f"b1_gate_{tag}", "scores_shard*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if r["seed"] >= 8:  # width-extension / wave-3 shards share the dirs;
                    continue        # the paper grid (scores AND latencies) is seeds 0-7
                by[tau][r["prompt"]][(r["seed"], r["variant"])] = r["score"]
                (lat[tau] if r["variant"] == "cached" else lat_full).append(r["latency"])
    full_ref = {p: {s: sc for (s, v), sc in d.items() if v == "full"}
                for p, d in by[0.10].items()}
    prompts = sorted(p for p in full_ref
                     if all(len({s for (s, v) in by[tau][p] if v == "cached"}) >= N
                            for tau in TAGS) and len(full_ref[p]) >= N)
    seeds = sorted(full_ref[prompts[0]])[:N]
    full = np.array([[full_ref[p][s] for s in seeds] for p in prompts])
    cached = {tau: np.array([[by[tau][p][(s, "cached")] for s in seeds] for p in prompts])
              for tau in TAGS}
    return prompts, full, cached, float(np.mean(lat_full)), {t: float(np.mean(v)) for t, v in lat.items()}


def capture_ratio(full_row, pick_idx):
    rand = full_row.max() - full_row.mean()
    return (full_row[pick_idx] - full_row.mean()) / rand if rand > 0 else 1.0


def fixed_point(full, cached_tau, Cf, Cc_tau):
    caps = [capture_ratio(full[i], int(np.argmax(cached_tau[i]))) for i in range(len(full))]
    return float(np.mean(caps)), Cf / Cc_tau  # capture, exploration speedup (= 8Cf / 8Cc)


def adaptive_points(full, cached, Cf, Cc, thresholds):
    """For each threshold t return (mean capture, mean exploration speedup, aggressive frac,
    regret-weighted capture 1 - mean regret / mean random regret)."""
    P = len(full)
    pairs = list(itertools.combinations(range(N), 2))
    # per-prompt, arm-level quantities (independent of probe pair)
    pick_ag = np.array([int(np.argmax(cached[0.20][i])) for i in range(P)])
    pick_co = np.array([int(np.argmax(cached[0.05][i])) for i in range(P)])
    cap_ag = np.array([capture_ratio(full[i], pick_ag[i]) for i in range(P)])
    cap_co = np.array([capture_ratio(full[i], pick_co[i]) for i in range(P)])
    reg_ag = np.array([full[i].max() - full[i, pick_ag[i]] for i in range(P)])
    reg_co = np.array([full[i].max() - full[i, pick_co[i]] for i in range(P)])
    rand = np.array([full[i].max() - full[i].mean() for i in range(P)])
    cost_ag = N * Cc[0.20]
    cost_co = 2 * Cc[0.20] + N * Cc[0.05]
    # spread estimates for every probe pair: [P, n_pairs]
    sp = np.array([[abs(cached[0.20][i, a] - cached[0.20][i, b]) for a, b in pairs]
                   for i in range(P)])
    out = []
    for t in thresholds:
        agg = sp >= t                                        # [P, n_pairs] arm decision
        cap = np.where(agg, cap_ag[:, None], cap_co[:, None]).mean()
        cost = np.where(agg, cost_ag, cost_co).mean()
        rom = 1 - np.where(agg, reg_ag[:, None], reg_co[:, None]).mean() / rand.mean()
        out.append((float(cap), N * Cf / cost, float(agg.mean()), float(rom)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantiles", default="0,5,10,20,30,40,50,60,70,80,90,95,100",
                    help="threshold sweep as percentiles of the pooled spread estimates")
    args = ap.parse_args()

    prompts, full, cached, Cf, Cc = load_grid()
    print(f"prompts {len(prompts)} | Cf {Cf:.1f}s | Cc " +
          " ".join(f"tau={t}:{c:.1f}s" for t, c in sorted(Cc.items())))

    print("\nfixed-tau operating points (N=8, capture = mean per-prompt gain-capture):")
    for tau in (0.05, 0.10, 0.20):
        cap, sp = fixed_point(full, cached[tau], Cf, Cc[tau])
        print(f"  tau={tau:.2f}: capture {cap*100:5.1f}%  exploration speedup {sp:.2f}x")

    pooled = np.abs(cached[0.20][:, :, None] - cached[0.20][:, None, :])[
        :, *np.triu_indices(N, 1)].ravel()
    qs = [float(q) for q in args.quantiles.split(",")]
    thresholds = np.percentile(pooled, qs)
    pts = adaptive_points(full, cached, Cf, Cc, thresholds)
    print("\nadaptive policy (probe K=2 at tau=0.20; aggressive tau=0.20 if spread>=t else tau=0.05):")
    print(f"  {'pctl':>5} {'t':>6}  {'capture':>8} {'speedup':>8} {'frac aggressive':>16} {'regret-wt cap':>14}")
    for q, t, (cap, sp, fr, rom) in zip(qs, thresholds, pts):
        print(f"  {q:>5.0f} {t:>6.3f}  {cap*100:>7.1f}% {sp:>7.2f}x {fr*100:>15.1f}% {rom*100:>13.1f}%")
    print("\nconclusion: the adaptive frontier tracks, and slightly under-runs, the fixed-tau")
    print("frontier under both capture conventions (sunk probe cost + noisy 2-sample spread")
    print("estimate consume the headroom left by the already-flat fixed-tau curve).")


if __name__ == "__main__":
    main()
