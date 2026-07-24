"""Simulate CachedSearch strategies from measured gate data (no GPU needed).

Strategies per prompt, over seed subsets of size N (bootstrap all C(8,N) or sampled):
  single      : one full-compute generation (no search)          cost = C_f
  bon_full    : best-of-N, all candidates full compute           cost = N*C_f
  cached_keep : cached candidates, keep the cached winner        cost = N*C_c
  cached_commit: cached candidates, re-generate winner at full   cost = N*C_c + C_f
Value of a chosen seed = its FULL-compute score for {single, bon_full, cached_commit}
(recommit reproduces the full trajectory deterministically from the seed);
= its CACHED score for cached_keep.

Also: per-prompt ranking-corruption analysis (rho vs score spread etc).
"""
import argparse, glob, json, itertools, os
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from videogen1.gen import RESULTS


def load(tag):
    by, lat = defaultdict(dict), defaultdict(list)
    for f in glob.glob(os.path.join(RESULTS, f"b1_gate_{tag}", "scores_shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            by[r["prompt"]][(r["seed"], r["variant"])] = r["score"]
            lat[r["variant"]].append(r["latency"])
    return by, {k: float(np.mean(v)) for k, v in lat.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--Ns", default="2,4,8")
    args = ap.parse_args()

    by, lat = load(args.tag)
    Cf, Cc = lat["full"], lat["cached"]
    print(f"measured latencies: full {Cf:.1f}s | cached {Cc:.1f}s (ratio {Cf/Cc:.2f}x)\n")

    header = f"{'N':>2} {'strategy':<14} {'reward':>7} {'cost(s)':>8} {'vs single':>9}"
    print(header); print("-" * len(header))
    results = {}
    for N in [int(x) for x in args.Ns.split(",")]:
        vals = defaultdict(list); costs = {}
        for prompt, d in by.items():
            seeds = sorted({s for s, v in d if (s, "full") in d and (s, "cached") in d})
            if len(seeds) < 8: continue
            full = {s: d[(s, "full")] for s in seeds}
            cach = {s: d[(s, "cached")] for s in seeds}
            for subset in itertools.combinations(seeds, N):
                fu = np.array([full[s] for s in subset])
                ca = np.array([cach[s] for s in subset])
                vals["single"].append(fu[0])
                vals["bon_full"].append(fu.max())
                vals["cached_keep"].append(ca[int(np.argmax(ca))])
                vals["cached_commit"].append(fu[int(np.argmax(ca))])
        costs = {"single": Cf, "bon_full": N * Cf,
                 "cached_keep": N * Cc, "cached_commit": N * Cc + Cf}
        for strat in ("single", "bon_full", "cached_keep", "cached_commit"):
            v = float(np.mean(vals[strat]))
            base = float(np.mean(vals["single"]))
            print(f"{N:>2} {strat:<14} {v:>+7.3f} {costs[strat]:>8.0f} {v-base:>+9.3f}")
            results[(N, strat)] = (v, costs[strat])
        print()

    # headline: cached_commit vs bon_full gain capture at each N
    print("gain capture (cached_commit gain / bon_full gain) & cost ratio:")
    for N in [int(x) for x in args.Ns.split(",")]:
        s = results[(N, "single")][0]
        bf, cbf = results[(N, "bon_full")]
        cc, ccc = results[(N, "cached_commit")]
        cap = (cc - s) / (bf - s) if bf > s else float("nan")
        print(f"  N={N}: capture {cap*100:5.1f}% at {ccc/cbf*100:4.1f}% of best-of-N cost "
              f"(cached_commit {ccc:.0f}s vs bon_full {cbf:.0f}s)")

    # corruption analysis: what predicts per-prompt rho?
    print("\ncorruption analysis (per-prompt):")
    rhos, spreads, means = [], [], []
    for prompt, d in by.items():
        seeds = sorted({s for s, v in d if (s, "full") in d and (s, "cached") in d})
        if len(seeds) < 8: continue
        fu = np.array([d[(s, "full")] for s in seeds])
        ca = np.array([d[(s, "cached")] for s in seeds])
        rho, _ = spearmanr(fu, ca)
        rhos.append(rho); spreads.append(fu.std()); means.append(fu.mean())
    rhos, spreads, means = map(np.array, (rhos, spreads, means))
    r1, _ = spearmanr(spreads, rhos)
    r2, _ = spearmanr(means, rhos)
    print(f"  corr(score spread, rho) = {r1:+.3f}   <- low-spread prompts corrupt more?")
    print(f"  corr(mean score,  rho) = {r2:+.3f}   <- hard prompts corrupt more?")
    print(f"  prompts with rho<0.7: {int((rhos<0.7).sum())}/{len(rhos)}")


if __name__ == "__main__":
    main()
