"""Analyze B1 gate results: per-prompt Spearman rho (cached vs full candidate ranking),
top-1 agreement, latency ratio. GO if median rho > 0.8."""
import argparse, glob, json, os
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from videogen1.gen import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0")
    args = ap.parse_args()

    rows = []
    for f in glob.glob(os.path.join(RESULTS, f"b1_gate_{args.tag}", "scores_shard*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    by = defaultdict(dict)   # prompt -> {(seed, variant): score}
    lat = defaultdict(list)
    for r in rows:
        by[r["prompt"]][(r["seed"], r["variant"])] = r["score"]
        lat[r["variant"]].append(r["latency"])

    rhos, top1 = [], []
    for prompt, d in by.items():
        seeds = sorted({s for s, v in d if (s, "full") in d and (s, "cached") in d})
        if len(seeds) < 4:
            continue
        full = [d[(s, "full")] for s in seeds]
        cach = [d[(s, "cached")] for s in seeds]
        rho, _ = spearmanr(full, cach)
        rhos.append(rho)
        top1.append(int(np.argmax(full) == np.argmax(cach)))

    rhos = np.array(rhos)
    print(f"prompts analyzed : {len(rhos)}")
    print(f"Spearman rho     : median {np.median(rhos):.3f} | mean {rhos.mean():.3f} | p10 {np.percentile(rhos,10):.3f}")
    print(f"top-1 agreement  : {np.mean(top1)*100:.1f}%")
    if lat["full"] and lat["cached"]:
        print(f"latency          : full {np.mean(lat['full']):.1f}s | cached {np.mean(lat['cached']):.1f}s "
              f"| speedup {np.mean(lat['full'])/np.mean(lat['cached']):.2f}x")
    verdict = "GO" if np.median(rhos) > 0.8 else "NO-GO (analyze failure modes / lower tau)"
    print(f"VERDICT          : {verdict}")


if __name__ == "__main__":
    main()
