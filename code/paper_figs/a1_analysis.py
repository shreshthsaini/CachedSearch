"""A-lite caching-baseline analysis for the CachedSearch paper (tab:baselines, f11b).

Compares published training-free caching methods under the B1 gate protocol
implemented by experiments/a1_baselines.py.
RANKING PRESERVATION vs the b1_gate_v0 full-compute references:

  results/a1_{teacache,pab,cfgcache,easycache-ours,none}/scores_shard*.jsonl
    50 prompts x 8 seeds, ONE cached-equivalent variant per method
  results/b1_gate_v0/scores_shard*.jsonl, variant=="full", seeds 0-7
    = the same full references used by tab:main / tab:tau (Cf = 68.3 s)

Number-integrity conventions:
- every number recomputed from raw jsonl; records deduped by (prompt, seed, variant);
- seeds 0-7 ONLY, scores AND latencies (b1_gate_v0 carries 16 width-extension seeds);
- per-prompt stats on prompts with COMPLETE 8/8-seed coverage in both the method run
  and the refs (incomplete grids -> smaller n, reported honestly, never imputed);
- capture = mean per-prompt gain-capture ratio (estimator of eq:capture, the
  tab:tau convention, at N=8); ratio-of-means also printed for cross-checking;
- speedup = Cf / mean method latency.

Sanity row: 'none' re-generates full compute through the (mode="off") wrapper --
seed determinism means it must reproduce the reference scores (rho ~ 1, speedup ~ 1).
If it does not, distrust the whole table and investigate determinism first; the
script prints the per-record |score - ref score| distribution for exactly this check.

Usage:  python code/paper_figs/a1_analysis.py
        (module also imported by make_figs2.py for the f11b baseline figure)
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figs import RESULTS

METHODS = ("teacache", "pab", "cfgcache", "easycache-ours", "none")
LABELS = {
    "teacache": "TeaCache (thresh 0.08)",
    "pab": "PAB (self 2 / cross 6)",
    "cfgcache": "CFG-Cache (interval 5)",
    "easycache-ours": "ours (EasyCache-style, tau 0.10)",
    "none": "no caching (control)",
}


def load_dedup(pattern, seed_max=8):
    """Raw records deduped by (prompt, seed, variant), seeds < seed_max only."""
    rows = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # torn tail line of a live shard
                continue
            if r["seed"] >= seed_max:
                continue
            rows[(r["prompt"], r["seed"], r["variant"])] = r
    return rows


def full_refs():
    """(prompt, seed) -> full record, from b1_gate_v0 (seeds 0-7); plus Cf."""
    rows = load_dedup(os.path.join(RESULTS, "b1_gate_v0", "scores_shard*.jsonl"))
    ref = {(p, s): r for (p, s, v), r in rows.items() if v == "full"}
    Cf = float(np.mean([r["latency"] for r in ref.values()]))
    return ref, Cf


def method_stats(method, ref, Cf, n_seeds=8, tag=""):
    """Ranking-preservation stats for one a1 method dir; None if no data."""
    d = os.path.join(RESULTS, f"a1_{method}" + (f"_{tag}" if tag else ""))
    rows = load_dedup(os.path.join(d, "scores_shard*.jsonl"))
    if not rows:
        return None
    by = defaultdict(dict)
    for (p, s, v), r in rows.items():
        by[p][s] = r
    rho, top1, regret, rand, dscore = [], [], [], [], []
    for p, seeds in by.items():
        if not all(s in seeds and (p, s) in ref for s in range(n_seeds)):
            continue  # complete 8/8 prompts only
        full = np.array([ref[(p, s)]["score"] for s in range(n_seeds)])
        meth = np.array([seeds[s]["score"] for s in range(n_seeds)])
        r_, _ = spearmanr(full, meth)
        rho.append(r_)
        top1.append(int(np.argmax(full) == np.argmax(meth)))
        regret.append(full.max() - full[int(np.argmax(meth))])
        rand.append(full.max() - full.mean())
        dscore.extend(np.abs(meth - full))
    lat = np.array([r["latency"] for r in rows.values()])
    # method-internal counters (jsonl "stats" field, nested dicts flattened),
    # averaged over records
    def flat(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v, prefix + k + ".")
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                yield prefix + k, v
    counters = defaultdict(list)
    for r in rows.values():
        for k, v in flat(r.get("stats") or {}):
            counters[k].append(v)
    rho, regret, rand = np.array(rho), np.array(regret), np.array(rand)
    ratios = (rand - regret) / rand
    latg = defaultdict(list)  # per-prompt latency groups (speedup bootstrap, f11b)
    for (p, s, v), r in rows.items():
        latg[p].append(r["latency"])
    return dict(
        method=method, n_records=len(rows), n_prompts=len(rho),
        pp_capture=ratios[np.isfinite(ratios)],  # per-prompt gain-capture (bootstrap)
        lat_groups=dict(latg),
        lat=float(lat.mean()), speedup=Cf / float(lat.mean()),
        rho_med=float(np.median(rho)), rho_mean=float(rho.mean()),
        rho_p10=float(np.percentile(rho, 10)),
        top1=float(np.mean(top1)), regret=float(regret.mean()),
        rand=float(rand.mean()),
        capture=float(np.mean(ratios[np.isfinite(ratios)])),          # eq:capture / tab:tau
        capture_rom=1.0 - float(regret.mean()) / float(rand.mean()),  # ratio of means
        dscore=float(np.mean(dscore)) if dscore else float("nan"),
        n_exact=int(np.sum(np.array(dscore) == 0)) if dscore else 0,
        counters={k: float(np.mean(v)) for k, v in counters.items()},
    )


def all_stats(tag=""):
    ref, Cf = full_refs()
    return {m: method_stats(m, ref, Cf, tag=tag) for m in METHODS}, ref, Cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix of the a1 dirs (e.g. 'smoke')")
    args = ap.parse_args()

    ref, Cf = full_refs()
    print(f"full refs: {len(ref)} records (b1_gate_v0, variant=full, seeds 0-7), "
          f"Cf = {Cf:.1f} s\n")
    hdr = (f"{'method':16s} {'n_rec':>5s} {'n_pr':>4s} {'lat_s':>6s} {'speedup':>7s} "
           f"{'rho_med':>7s} {'rho_mn':>6s} {'rho_p10':>7s} {'top1%':>5s} "
           f"{'regret':>6s} {'capt%':>6s} {'capt_rom%':>9s}")
    print(hdr)
    stats = {}
    for m in METHODS:
        st = method_stats(m, ref, Cf, tag=args.tag)
        stats[m] = st
        if st is None:
            print(f"{m:16s}  (no records)")
            continue
        print(f"{m:16s} {st['n_records']:5d} {st['n_prompts']:4d} {st['lat']:6.1f} "
              f"{st['speedup']:6.2f}x {st['rho_med']:7.3f} {st['rho_mean']:6.3f} "
              f"{st['rho_p10']:7.3f} {100*st['top1']:5.1f} {st['regret']:6.3f} "
              f"{100*st['capture']:6.1f} {100*st['capture_rom']:9.1f}")
    print("\n(capt% = mean per-prompt gain-capture ratio at N=8, eq:capture/tab:tau "
          "convention; capt_rom% = 1 - mean_regret/mean_rand; prompts require "
          "complete 8/8-seed coverage)")

    for m in METHODS:
        st = stats.get(m)
        if st and st["counters"]:
            cs = "  ".join(f"{k}={v:g}" for k, v in sorted(st["counters"].items()))
            print(f"  [{m}] mean internal counters: {cs}")

    st = stats.get("none")
    if st:
        print(f"\n'none' determinism check: mean |score - ref| = {st['dscore']:.5f} "
              f"over {8*st['n_prompts']} records ({st['n_exact']} exactly equal); "
              f"rho_med = {st['rho_med']:.3f}, speedup = {st['speedup']:.3f}x")
        if st["rho_med"] < 0.99:
            print("  <<< WARNING: 'none' does NOT reproduce the full references "
                  "(expected rho ~ 1). Investigate seed determinism before "
                  "trusting ANY row of this table.")
    return stats


if __name__ == "__main__":
    main()
