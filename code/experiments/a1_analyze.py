"""Analyze A1 baselines: ranking preservation per method vs b1_gate_v0 full references.

For each method dir results/a1_<method>[_<tag>]/ join records with the full-compute
references (variant=="full") in results/b1_gate_<ref-tag>/ on (prompt, seed), then
report the same statistics as the B1 gate (b1_analyze.py conventions):
  - per-prompt Spearman rho of candidate ranking (method scores vs full scores)
  - top-1 agreement (argmax over seeds matches)
  - regret: full_score(best-by-method) vs max full_score; capture % vs the
    random-pick baseline regret (max - mean), as in the B1 regret analysis
  - latency speedup vs the full references' latency
  - mean method-internal skip/compute counters

Usage:  python a1_analyze.py                       # all methods, default dirs
        python a1_analyze.py --methods teacache,pab --tag smoke
"""
import argparse, glob, json, os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from videogen1.gen import RESULTS

METHODS = ("teacache", "pab", "cfgcache", "easycache-ours", "none")


def load_rows(pattern):
    rows = []
    for f in glob.glob(pattern):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--tag", default="", help="suffix of the a1 results dirs (e.g. 'smoke')")
    ap.add_argument("--ref-tag", default="v0", help="b1_gate_<ref-tag> supplies full references")
    ap.add_argument("--min-seeds", type=int, default=4)
    args = ap.parse_args()

    ref = {}   # (prompt, seed) -> full score
    ref_lat = []
    for r in load_rows(os.path.join(RESULTS, f"b1_gate_{args.ref_tag}", "scores_shard*.jsonl")):
        if r["variant"] == "full":
            ref[(r["prompt"], r["seed"])] = r["score"]
            ref_lat.append(r["latency"])
    print(f"full references: {len(ref)} records from b1_gate_{args.ref_tag} "
          f"(mean latency {np.mean(ref_lat):.1f}s)\n")

    header = (f"{'method':16s} {'n_rec':>5s} {'prompts':>7s} {'rho_med':>7s} {'rho_mean':>8s} "
              f"{'rho_p10':>7s} {'top1%':>6s} {'regret':>7s} {'capture%':>8s} {'lat_s':>6s} {'speedup':>7s}")
    print(header)
    for method in args.methods.split(","):
        method = method.strip()
        d = os.path.join(RESULTS, f"a1_{method}" + (f"_{args.tag}" if args.tag else ""))
        rows = load_rows(os.path.join(d, "scores_shard*.jsonl"))
        if not rows:
            print(f"{method:16s}  (no records in {d})")
            continue
        by = defaultdict(dict)  # prompt -> seed -> method score
        lat = []
        for r in rows:
            by[r["prompt"]][r["seed"]] = r["score"]
            lat.append(r["latency"])
        rhos, top1, regrets, rand_regrets = [], [], [], []
        for prompt, d_seeds in by.items():
            seeds = sorted(s for s in d_seeds if (prompt, s) in ref)
            if len(seeds) < args.min_seeds:
                continue
            full = np.array([ref[(prompt, s)] for s in seeds])
            meth = np.array([d_seeds[s] for s in seeds])
            rho, _ = spearmanr(full, meth)
            rhos.append(rho)
            top1.append(int(np.argmax(full) == np.argmax(meth)))
            regrets.append(full.max() - full[int(np.argmax(meth))])
            rand_regrets.append(full.max() - full.mean())
        if not rhos:
            print(f"{method:16s} {len(rows):5d}  (not enough overlapping seeds vs refs ; "
                  f"need >= {args.min_seeds}/prompt)")
            continue
        rhos = np.array(rhos)
        capture = (1 - np.mean(regrets) / max(np.mean(rand_regrets), 1e-9)) * 100
        speedup = np.mean(ref_lat) / np.mean(lat)
        print(f"{method:16s} {len(rows):5d} {len(rhos):7d} {np.median(rhos):7.3f} {rhos.mean():8.3f} "
              f"{np.percentile(rhos, 10):7.3f} {np.mean(top1)*100:6.1f} {np.mean(regrets):7.3f} "
              f"{capture:8.1f} {np.mean(lat):6.1f} {speedup:7.2f}")

    print("\n(rho computed per prompt over seeds present in BOTH the method run and the full refs; "
          "capture = 1 - mean_regret/mean_random_regret, B1 convention)")


if __name__ == "__main__":
    main()
