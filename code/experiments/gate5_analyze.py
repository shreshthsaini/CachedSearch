"""gate5 running analysis ; CachedSearch gate metrics on the 3 model-breadth rungs
(Wan2.2-TI2V-5B, LTX-Video 2B, HunyuanVideo 13B) + reference rungs for context.

Semantics MIRROR paper_figs/make_figs2.py's F9 loader exactly (first-8-seeds
convention, tab:scale / eq:capture estimator):
  - records deduped by (prompt, seed, variant) across ALL scores_shard*.jsonl
    of the tag dir (cross-partition shards may duplicate records);
  - the per-prompt grid uses seeds 0..7 ONLY and keeps prompts with complete
    full+cached coverage on all 8 seeds;
  - latencies (Cf, Cc, speedup) use seeds 0..7 ONLY;
  - rho/top1/regret/rand per make_figs.per_prompt: regret = full best - full
    score at cached argmax; rand = full best - full mean;
  - capture = mean per-prompt gain-capture ratio mean((rand - regret)/rand)
    over finite values (F9/tab:tau/tab:scale convention, estimator of
    eq:capture) ; NOT the ratio-of-mean-gains used by F8/tab:main.

Usage:  python experiments/gate5_analyze.py [--tags ...]
Prints a running table; safe on partial data (prompts without full 8-seed
coverage are simply not counted yet).
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

RESULTS = os.environ.get("CACHEDSEARCH_RESULTS", "./results")

# tag -> display label (gate5 rungs first; reference rungs for cross-model context)
DEFAULT_TAGS = [
    ("wan22_5b", "Wan2.2-TI2V-5B"),
    ("ltx_v0", "LTX-Video-2B"),
    ("hunyuan_v0", "HunyuanVideo-13B"),
    ("v0", "Wan2.1-1.3B (ref)"),
    ("cog5b_v0", "CogVideoX-5B (ref)"),
]


def load_rows(tag):
    """Raw records for b1_gate_<tag>, deduped by (prompt, seed, variant)."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, f"b1_gate_{tag}", "scores_shard*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # torn tail line of a live shard file
                continue
            rows[(r["prompt"], r["seed"], r["variant"])] = r
    return rows


def latencies(rows):
    """Mean per-variant latency, seeds 0-7 ONLY (make_figs2 convention)."""
    lat = defaultdict(list)
    for (p, s, v), r in rows.items():
        if s < 8:
            lat[v].append(r["latency"])
    return {v: float(np.mean(xs)) for v, xs in lat.items() if xs}


def grid(rows, n_seeds=8):
    """{prompt: (full[n_seeds], cached[n_seeds])}, complete-coverage prompts only."""
    by = defaultdict(dict)
    for (p, s, v), r in rows.items():
        if s < n_seeds:
            by[p][(s, v)] = r["score"]
    pa = {}
    for p, d in by.items():
        if all((s, v) in d for s in range(n_seeds) for v in ("full", "cached")):
            pa[p] = (np.array([d[(s, "full")] for s in range(n_seeds)]),
                     np.array([d[(s, "cached")] for s in range(n_seeds)]))
    return pa


def per_prompt(pa):
    """Mirror of make_figs.per_prompt."""
    rho, top1, regret, rand = [], [], [], []
    for fu, ca in pa.values():
        r, _ = spearmanr(fu, ca)
        rho.append(r)
        top1.append(int(np.argmax(fu) == np.argmax(ca)))
        regret.append(fu.max() - fu[int(np.argmax(ca))])
        rand.append(fu.max() - fu.mean())
    return {k: np.array(v) for k, v in
            zip(["rho", "top1", "regret", "rand"], [rho, top1, regret, rand])}


def capture_mor(pp):
    """Mean per-prompt gain-capture ratio (eq:capture estimator, F9/tab:scale)."""
    ratios = (pp["rand"] - pp["regret"]) / pp["rand"]
    ratios = ratios[np.isfinite(ratios)]
    return float(np.mean(ratios)) if len(ratios) else float("nan")


def analyze_tag(tag):
    rows = load_rows(tag)
    if not rows:
        return None
    pa = grid(rows, n_seeds=8)
    lat = latencies(rows)
    out = dict(n_records=len(rows), n_prompts=len(pa),
               Cf=lat.get("full", float("nan")), Cc=lat.get("cached", float("nan")))
    out["speedup"] = out["Cf"] / out["Cc"] if out["Cc"] == out["Cc"] and out["Cc"] else float("nan")
    if pa:
        pp = per_prompt(pa)
        out.update(med_rho=float(np.median(pp["rho"])),
                   p10_rho=float(np.percentile(pp["rho"], 10)),
                   top1=float(np.mean(pp["top1"])),
                   capture=capture_mor(pp),
                   mean_regret=float(np.mean(pp["regret"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None,
                    help="tags to analyze (default: gate5 rungs + reference rungs)")
    args = ap.parse_args()
    tags = [(t, t) for t in args.tags] if args.tags else DEFAULT_TAGS

    hdr = (f"{'model':<20} {'recs':>5} {'prompts':>7} {'med rho':>8} {'p10 rho':>8} "
           f"{'top-1':>6} {'capture':>8} {'regret':>7} {'Cf(s)':>7} {'Cc(s)':>7} {'speed':>6}")
    print(hdr)
    print("-" * len(hdr))
    for tag, label in tags:
        r = analyze_tag(tag)
        if r is None:
            print(f"{label:<20} {'--':>5} {'no records yet':>7}")
            continue
        if "med_rho" not in r:
            print(f"{label:<20} {r['n_records']:>5} {r['n_prompts']:>7}"
                  f" {'(no complete 8-seed prompts yet)':>34}"
                  f"  Cf {r['Cf']:.1f} Cc {r['Cc']:.1f}" if r['Cc'] == r['Cc'] else "")
            continue
        print(f"{label:<20} {r['n_records']:>5} {r['n_prompts']:>7} {r['med_rho']:>8.3f} "
              f"{r['p10_rho']:>8.3f} {r['top1']:>5.0%} {r['capture']:>7.1%} "
              f"{r['mean_regret']:>7.3f} {r['Cf']:>7.1f} {r['Cc']:>7.1f} {r['speedup']:>5.2f}x")
    print("\nconventions: seeds 0-7 only; complete-coverage prompts; capture = mean per-prompt")
    print("gain-capture ratio (eq:capture / tab:scale); latencies seeds 0-7 only (Cf/Cc).")


if __name__ == "__main__":
    main()
