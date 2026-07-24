"""E3 verifier-robustness analysis (paper sec:abl-verifier).

Consumes the multi-verifier rescoring produced by experiments/b1_rescore.py in
$CACHEDSEARCH_RESULTS/b1_verifiers_vbench/ (scores_shard*.jsonl; resumable shards,
last line per file wins). Two video populations live in that directory:

  E1 winner pairs : results/b1_temporal/videos, files {hash}_{tau}_{keep|commit}.mp4
                    (50 prompts x 3 tau x 2 arms = 300 videos)
  E2 official run : results/b1_gate_vbench/videos, files {hash}_{seed}_{full|cached}.mp4
                    (cached = tau0.10; scored incrementally by e3-scale-s* tasks)

Analyses (all printed; no figures):
  0. VALIDATION + FILTER  E3's re-scored ImageReward vs the gate's original
                 in-memory ImageReward, joined on (prompt, seed, variant).
                 The prompt index is CLIP retrieval (hash(prompt) was
                 per-process-randomized, see b1_rescore.py docstring) and the
                 official 946-prompt suite contains many near-duplicate
                 prompts, so retrieval misassigns a minority of hash groups
                 (low index_margin). The gate join is ground truth: if the
                 assigned prompt is right, the re-scored IR must match the
                 gate score up to mp4-round-trip noise (~0.02). We therefore
                 KEEP only hash groups whose median |IR_e3 - IR_gate| < 0.10
                 and report the retained fraction; all downstream analyses use
                 validated groups only.
  1. CROSS-VERIFIER CORRELATION  per-video ImageReward vs VideoScore dims,
                 pooled and within-prompt (the decision-relevant granularity).
  2. RANKING PRESERVATION UNDER VIDEOSCORE  per-prompt rho/top-1/regret/capture
                 with VideoScore-avg as the verifier, on prompts with complete
                 8 full + 8 cached coverage (seeds 0-7); the E3-rescored
                 ImageReward on the SAME videos is the apples-to-apples control.
  3. CROSS-VERIFIER SELECTION  the cached ImageReward pick evaluated under
                 full-compute VideoScore (does IR-guided cached search
                 over-optimize IR at VideoScore's expense?).
  4. KEEP-VS-COMMIT UNDER VIDEOSCORE  paired deltas per dimension per tau on
                 the E1 winner pairs (does the frame-verifier's mild preference
                 for cached outputs survive a video-native verifier, and does
                 the dynamic dimension see the motion dampening?).

Run:  MALLOC_ARENA_MAX=2 python paper_figs/e3_analysis.py
"""
from __future__ import annotations
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

RESULTS = os.environ.get("CACHEDSEARCH_RESULTS", "./results")
E3_DIR = os.path.join(RESULTS, "b1_verifiers_vbench")
GATE_DIR = os.path.join(RESULTS, "b1_gate_vbench")

VS_DIMS = ["videoscore_visual", "videoscore_temporal", "videoscore_dynamic",
           "videoscore_alignment", "videoscore_factual", "videoscore_avg"]
TAUS = [("tau005", 0.05), ("v0", 0.10), ("tau020", 0.20)]  # E1 tags -> tau values
GROUP_DIFF_TOL = 0.10  # median per-group |IR_e3 - IR_gate| accepted as "assignment correct"


def read_jsonl(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # torn tail line of a live shard
    return out


def load_e3():
    """Dedupe by file (last line wins, matching b1_rescore.py resume logic)."""
    recs = {}
    for f in sorted(glob.glob(os.path.join(E3_DIR, "scores_shard*.jsonl"))):
        for r in read_jsonl(f):
            recs[r["file"]] = r
    e1 = [r for r in recs.values() if "arm" in r]
    e2 = [r for r in recs.values() if "seed" in r]
    return e1, e2


def load_gate():
    rows = {}
    for f in sorted(glob.glob(os.path.join(GATE_DIR, "scores_shard*.jsonl"))):
        for r in read_jsonl(f):
            rows[(r["prompt"], r["seed"], r["variant"])] = r["score"]
    return rows


def per_prompt_stats(groups, key):
    """groups: {prompt: {(seed, variant): rec}} complete 8f+8c. Returns arrays in
    the conventions of paper_figs/make_figs.py (regret on FULL scores of `key`,
    selection by CACHED scores of `key`)."""
    rho, top1, regret, rand = [], [], [], []
    for d in groups.values():
        seeds = sorted({s for s, v in d if v == "full"})
        fu = np.array([d[(s, "full")][key] for s in seeds])
        ca = np.array([d[(s, "cached")][key] for s in seeds])
        rho.append(spearmanr(fu, ca).statistic)
        top1.append(int(np.argmax(fu) == np.argmax(ca)))
        regret.append(fu.max() - fu[int(np.argmax(ca))])
        rand.append(fu.max() - fu.mean())
    return {k: np.array(v) for k, v in
            zip(["rho", "top1", "regret", "rand"], [rho, top1, regret, rand])}


def capture(pp):
    ratios = (pp["rand"] - pp["regret"]) / pp["rand"]
    return 100 * float(np.mean(ratios[np.isfinite(ratios)]))


def summarize(pp, label):
    print(f"  {label:28s} median rho {np.median(pp['rho']):.3f}  "
          f"mean {pp['rho'].mean():.3f}  p10 {np.percentile(pp['rho'], 10):.3f}  "
          f"top-1 {100 * pp['top1'].mean():.0f}%  "
          f"mean regret {pp['regret'].mean():.4f} (rand {pp['rand'].mean():.4f})  "
          f"capture {capture(pp):.1f}%  zero-regret {100 * (pp['regret'] == 0).mean():.0f}%")


def main():
    e1, e2 = load_e3()
    print(f"E3 records: {len(e1)} E1 winner-pair videos, {len(e2)} E2 official videos "
          f"(both verifiers: {sum(1 for r in e2 if 'videoscore_avg' in r and 'imagereward' in r)})")

    # ---------------------------------------------------------------- 0. validation
    print("\n[0] VALIDATION + FILTER: re-scored ImageReward vs gate's original, per hash group")
    gate = load_gate()
    scored = [r for r in e2 if "imagereward" in r and "videoscore_avg" in r]
    by_hash = defaultdict(list)
    for r in scored:
        by_hash[r["hash"]].append(r)
    kept_groups, dropped = {}, 0
    for h, rs in by_hash.items():
        diffs = [abs(r["imagereward"] - gate[(r["prompt"], r["seed"], r["variant"])])
                 for r in rs if (r["prompt"], r["seed"], r["variant"]) in gate]
        if diffs and float(np.median(diffs)) < GROUP_DIFF_TOL:
            kept_groups[h] = rs
        else:
            dropped += 1
    kept = [r for rs in kept_groups.values() for r in rs]
    print(f"  hash groups scored: {len(by_hash)}  validated: {len(kept_groups)}  "
          f"dropped (prompt-index misassignment): {dropped} "
          f"({100 * dropped / len(by_hash):.1f}% of groups)")
    x = np.array([r["imagereward"] for r in kept])
    y = np.array([gate[(r["prompt"], r["seed"], r["variant"])] for r in kept])
    print(f"  validated records n={len(x)}  pearson {pearsonr(x, y).statistic:.4f}  "
          f"spearman {spearmanr(x, y).statistic:.4f}  mean|diff| {np.abs(x - y).mean():.4f} "
          f"(mp4 round-trip + frame-sampling noise)")

    # deduplicate validated records to (prompt, seed, variant); prefer higher margin
    by_psv = {}
    for r in kept:
        k = (r["prompt"], r["seed"], r["variant"])
        r["gate_ir"] = gate[k] if k in gate else None  # original in-memory IR
        if k not in by_psv or r["index_margin"] > by_psv[k]["index_margin"]:
            by_psv[k] = r
    print(f"  unique (prompt, seed, variant) after dedup: {len(by_psv)}")

    # ---------------------------------------------------------------- 1. correlations
    print("\n[1] CROSS-VERIFIER PER-VIDEO CORRELATION (E2 official videos)")
    ir = np.array([r["imagereward"] for r in by_psv.values()])
    print(f"  pooled over n={len(ir)} videos:")
    for dim in VS_DIMS:
        v = np.array([r[dim] for r in by_psv.values()])
        print(f"    IR vs {dim:24s} pearson {pearsonr(ir, v).statistic:+.3f}  "
              f"spearman {spearmanr(ir, v).statistic:+.3f}")

    # within-prompt (prompts with >= 8 full videos): rank agreement across candidates
    groups_all = defaultdict(dict)
    for (p, s, v), r in by_psv.items():
        if s < 8:
            groups_all[p][(s, v)] = r
    wp = defaultdict(list)
    for p, d in groups_all.items():
        for variant in ("full", "cached"):
            recs = [d[(s, vv)] for (s, vv) in d if vv == variant]
            if len(recs) >= 8:
                a = np.array([r["imagereward"] for r in recs])
                b = np.array([r["videoscore_avg"] for r in recs])
                wp[variant].append(spearmanr(a, b).statistic)
    for variant, v in wp.items():
        v = np.array(v)
        print(f"  within-prompt spearman(IR, VS-avg), {variant:6s} candidates: "
              f"median {np.median(v):+.3f}  mean {np.nanmean(v):+.3f}  (n={len(v)} prompts)")

    # ---------------------------------------------------------------- 2. ranking preservation
    print("\n[2] RANKING PRESERVATION cached-vs-full, complete 8f+8c groups (seeds 0-7)")
    complete = {}
    for p, d in groups_all.items():
        if (len({s for (s, v) in d if v == "full"}) == 8
                and len({s for (s, v) in d if v == "cached"}) == 8):
            complete[p] = d
    print(f"  complete prompt groups: n={len(complete)}")
    pp_vs = per_prompt_stats(complete, "videoscore_avg")
    pp_ir = per_prompt_stats(complete, "imagereward")
    # within-prompt score spread (std of the 8 FULL scores; make_figs 'spread'
    # convention) -- context for the weaker VS ranking numbers (spread mechanism)
    sp_vs, sp_ir = [], []
    for d in complete.values():
        seeds = sorted({s for s, v in d if v == "full"})
        sp_vs.append(np.std([d[(s, "full")]["videoscore_avg"] for s in seeds]))
        sp_ir.append(np.std([d[(s, "full")]["imagereward"] for s in seeds]))
    print(f"  within-prompt FULL-score spread (std over 8 seeds): "
          f"VideoScore-avg median {np.median(sp_vs):.3f} (scale [1,4])  "
          f"ImageReward median {np.median(sp_ir):.3f} (scale ~[-2.3,2.3])")
    if all(r["gate_ir"] is not None for d in complete.values() for r in d.values()):
        summarize(per_prompt_stats(complete, "gate_ir"), "gate ImageReward (paper E2)")
    summarize(pp_ir, "ImageReward re-scored (ctrl)")
    summarize(pp_vs, "VideoScore-avg")
    for dim in ["videoscore_visual", "videoscore_temporal", "videoscore_alignment"]:
        summarize(per_prompt_stats(complete, dim), dim)

    # ---------------------------------------------------------------- 3. cross-verifier selection
    print("\n[3] CROSS-VERIFIER SELECTION: cached-IR pick judged by full-compute VideoScore")
    regret_vs, rand_vs = [], []
    for d in complete.values():
        seeds = sorted({s for s, v in d if v == "full"})
        ca_ir = np.array([d[(s, "cached")]["imagereward"] for s in seeds])
        fu_vs = np.array([d[(s, "full")]["videoscore_avg"] for s in seeds])
        regret_vs.append(fu_vs.max() - fu_vs[int(np.argmax(ca_ir))])
        rand_vs.append(fu_vs.max() - fu_vs.mean())
    regret_vs, rand_vs = np.array(regret_vs), np.array(rand_vs)
    ratios = (rand_vs - regret_vs) / rand_vs
    print(f"  mean VS-regret of cached-IR pick {regret_vs.mean():.4f} "
          f"(random-pick baseline {rand_vs.mean():.4f})  "
          f"VS-gain capture {100 * np.mean(ratios[np.isfinite(ratios)]):.1f}%  "
          f"zero-regret {100 * (regret_vs == 0).mean():.0f}%  (n={len(regret_vs)})")
    # reference: the FULL-compute IR pick judged by VS (verifier disagreement floor,
    # no caching involved)
    regret_f, keep_rand = [], []
    for d in complete.values():
        seeds = sorted({s for s, v in d if v == "full"})
        fu_ir = np.array([d[(s, "full")]["imagereward"] for s in seeds])
        fu_vs = np.array([d[(s, "full")]["videoscore_avg"] for s in seeds])
        regret_f.append(fu_vs.max() - fu_vs[int(np.argmax(fu_ir))])
        keep_rand.append(fu_vs.max() - fu_vs.mean())
    regret_f, keep_rand = np.array(regret_f), np.array(keep_rand)
    ratios_f = (keep_rand - regret_f) / keep_rand
    print(f"  reference (no caching): full-IR pick VS-regret {regret_f.mean():.4f}  "
          f"VS-gain capture {100 * np.mean(ratios_f[np.isfinite(ratios_f)]):.1f}%")
    # paired prompt-level bootstrap on the caching-attributable selection loss
    # (cached-IR pick vs full-IR pick, both judged by full-compute VideoScore;
    # B=10^4, seed 0 -- convention of paper_figs/bootstrap_ci.py)
    fin = np.isfinite(ratios) & np.isfinite(ratios_f)
    dcap = ratios[fin] - ratios_f[fin]
    dreg = regret_vs - regret_f
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(dcap), size=(10000, len(dcap)))
    lo_c, hi_c = np.percentile(dcap[idx].mean(axis=1), [2.5, 97.5])
    idx2 = rng.integers(0, len(dreg), size=(10000, len(dreg)))
    lo_r, hi_r = np.percentile(dreg[idx2].mean(axis=1), [2.5, 97.5])
    print(f"  PAIRED caching-attributable loss (cached-IR pick minus full-IR pick, "
          f"95% prompt-bootstrap CI, B=10^4):")
    print(f"    VS-capture difference {100 * dcap.mean():+.1f} pts "
          f"[{100 * lo_c:+.1f}, {100 * hi_c:+.1f}]  (n={fin.sum()})")
    print(f"    VS-regret  difference {dreg.mean():+.4f} [{lo_r:+.4f}, {hi_r:+.4f}]")

    # ---------------------------------------------------------------- 4. keep vs commit
    print("\n[4] KEEP-VS-COMMIT UNDER VIDEOSCORE (E1 winner pairs, paired per prompt hash)")
    pairs = defaultdict(dict)
    for r in e1:
        pairs[(r["hash"], r["tau_tag"])][r["arm"]] = r
    for tau, tau_val in TAUS:
        tp = {k: v for k, v in pairs.items() if k[1] == tau and len(v) == 2}
        print(f"  tau={tau_val} (tag {tau}, n={len(tp)} pairs):")
        for dim in ["imagereward"] + VS_DIMS:
            dk = np.array([v["keep"][dim] for v in tp.values()])
            dc = np.array([v["commit"][dim] for v in tp.values()])
            d = dk - dc
            try:
                pval = wilcoxon(dk, dc).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                pval = float("nan")
            print(f"    {dim:24s} mean keep-commit {d.mean():+.4f}  "
                  f"keep>=commit {100 * (d >= 0).mean():.0f}%  wilcoxon p={pval:.3f}")


if __name__ == "__main__":
    main()
