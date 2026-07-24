"""Appendix tables for Paper 1 (owner: paper-appendix agent).

Computes, from RAW jsonl only (no numbers typed in from docs):
  1. Headline cross-checks for the theory-vs-measurement table (app:theory-vs-meas):
     median/mean/p10 rho, top-1 agreement, mean/median regret, random-pick
     baseline, capture (both conventions) on b1_gate_v0 (Wan2.1-1.3B, tau=0.10).
  2. Per-prompt-category analysis (app:categories): keyword bucketing of the 50
     gate prompts into {humans, animals, nature, objects/vehicles, stylized} and
     per-category median rho / mean regret / top-1 / spread. Emits LaTeX rows.
  3. E4 first cross-scale data point (app:scaling): same statistics on
     b1_gate_cog5b_v0 (CogVideoX-5B, tau=0.10, 49f/480x720/50 steps, CFG 6.0).

Run:  MALLOC_ARENA_MAX=2 python code/paper_figs/appendix_tables.py
Data: $CACHEDSEARCH_RESULTS/b1_gate_{v0,cog5b_v0}/scores_shard*.jsonl
"""
import glob, json, os
import numpy as np
from scipy.stats import spearmanr

RESULTS = os.environ.get("CACHEDSEARCH_RESULTS", "./results")
PROMPTS = os.path.join(os.path.dirname(__file__), "..", "experiments", "prompts_gate50.txt")

# --- keyword bucketing (priority order matters; first hit wins) -------------
CATEGORIES = [
    ("stylized", ["made of light", "watercolor", "stop-motion", "origami", "clay",
                  "cyberpunk"]),
    ("humans", ["chef", "children", "ballerina", "barista", "painter", "musician",
                "player", "blacksmith", "cowboy", "woman", "skateboarder", "cyclist",
                "surfer", "dancer", "librarian", "toddler", "knight", "astronaut"]),
    ("animals", ["retriever", "dog", "hummingbird", "fox", "cat", "jellyfish",
                 "eagle", "penguin", "bees", "dinosaur", "fish", "birds"]),
    ("objects", ["locomotive", "balloon", "firework", "sailboat", "robot", "bus",
                 "wine", "projector", "car", "submarine", "train"]),
    ("nature", ["waves", "clouds", "lava", "sunflower", "leaves", "rain", "snow",
                "fog", "campfire", "storm", "lake", "rice field", "desert"]),
]

def categorize(prompt: str) -> str:
    import re
    p = prompt.lower()
    for name, kws in CATEGORIES:
        if any(re.search(rf"\b{re.escape(k)}s?\b", p) for k in kws):  # s? = optional plural
            return name
    return "objects"  # fallback


def load_tag(tag):
    """-> {prompt: {seed: {variant: (score, latency)}}}"""
    d = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, f"b1_gate_{tag}", "scores_shard*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d.setdefault(r["prompt"], {}).setdefault(r["seed"], {})[r["variant"]] = (
                r["score"], r["latency"])
    return d


def per_prompt_stats(data):
    """-> list of dicts (prompt, rho, top1, regret, spread, base=max-mean)."""
    rows = []
    for prompt, seeds in data.items():
        ss = sorted(seeds)
        fu = np.array([seeds[s]["full"][0] for s in ss])
        ca = np.array([seeds[s]["cached"][0] for s in ss])
        assert len(fu) == 8, (prompt, len(fu))
        rho = spearmanr(fu, ca).statistic
        top1 = int(np.argmax(fu) == np.argmax(ca))
        regret = float(fu.max() - fu[int(np.argmax(ca))])
        rows.append(dict(prompt=prompt, rho=float(rho), top1=top1, regret=regret,
                         spread=float(fu.std()),  # convention of b1_simulate/make_figs
                         base=float(fu.max() - fu.mean())))
    return rows


def latencies(data):
    fu = [v["full"][1] for s in data.values() for v in s.values()]
    ca = [v["cached"][1] for s in data.values() for v in s.values()]
    return float(np.mean(fu)), float(np.mean(ca))


def headline(rows, name):
    rho = np.array([r["rho"] for r in rows])
    reg = np.array([r["regret"] for r in rows])
    base = np.array([r["base"] for r in rows])
    cap_ratio_means = 1 - reg.mean() / base.mean()
    cap_mean_ratio = float(np.mean(1 - reg / base))
    print(f"\n== {name} (n={len(rows)} prompts) ==")
    print(f"  rho: median {np.median(rho):.3f}  mean {rho.mean():.3f}  p10 {np.percentile(rho, 10):.3f}")
    print(f"  rho<0.7: {int((rho < 0.7).sum())}/{len(rows)}")
    print(f"  top-1 agreement: {100 * np.mean([r['top1'] for r in rows]):.0f}%")
    print(f"  regret: mean {reg.mean():.3f}  median {np.median(reg):.3f}  zero-regret {100 * np.mean(reg == 0):.0f}%")
    print(f"  random-pick baseline (mean max-mean): {base.mean():.3f}")
    print(f"  capture: ratio-of-means {100 * cap_ratio_means:.1f}%  mean per-prompt ratio {100 * cap_mean_ratio:.1f}%")
    print(f"  relative regret (mean regret / mean spread): {100 * reg.mean() / np.mean([r['spread'] for r in rows]):.1f}%")
    print(f"  corr(spread, rho) Spearman: {spearmanr([r['spread'] for r in rows], rho).statistic:+.3f}")


def category_table(rows):
    print("\n== per-category (b1_gate_v0, tau=0.10, N=8) ==")
    cats = {}
    for r in rows:
        cats.setdefault(categorize(r["prompt"]), []).append(r)
    order = ["humans", "animals", "nature", "objects", "stylized"]
    print(f"{'category':<10} {'n':>3} {'med rho':>8} {'mean regret':>12} {'top-1':>6} {'zero-reg':>9} {'mean spread':>12}")
    latex = []
    for c in order:
        rs = cats.get(c, [])
        rho = np.array([r["rho"] for r in rs])
        reg = np.array([r["regret"] for r in rs])
        top1 = 100 * np.mean([r["top1"] for r in rs])
        spread = np.mean([r["spread"] for r in rs])
        print(f"{c:<10} {len(rs):>3} {np.median(rho):>8.3f} {reg.mean():>12.3f} {top1:>5.0f}% {100 * np.mean(reg == 0):>8.0f}% {spread:>12.3f}")
        latex.append(f"{c} & {len(rs)} & ${np.median(rho):.3f}$ & ${reg.mean():.3f}$ & "
                     f"${top1:.0f}\\%$ & ${spread:.2f}$ \\\\")
    print("\nLaTeX rows:")
    print("\n".join(latex))
    print("\nassignments:")
    for c in order:
        print(f"  {c}: " + " | ".join(r["prompt"][:44] for r in cats.get(c, [])))


def main():
    for tag, name in [("v0", "Wan2.1-T2V-1.3B (b1_gate_v0)"),
                      ("cog5b_v0", "CogVideoX-5B (b1_gate_cog5b_v0)")]:
        data = load_tag(tag)
        rows = per_prompt_stats(data)
        headline(rows, name)
        cf, cc = latencies(data)
        print(f"  latency: full {cf:.1f}s  cached {cc:.1f}s  speedup {cf / cc:.2f}x  gamma {cc / cf:.3f}")
        if tag == "v0":
            category_table(rows)


if __name__ == "__main__":
    main()
