"""E1: temporal-quality check ; does keep-draft (cached winner video) survive
scrutiny vs recommit (full-compute regeneration of the same winner seed)?

For each (prompt, tau): winner seed = argmax cached score (from existing gate data).
Generate: keep = cached rollout @ winner seed; commit = full rollout @ same seed.
Record LPIPS(keep, commit) (fidelity of the delivered draft to its full-compute twin),
rescore both with ImageReward, SAVE both videos for VBench-temporal scoring later.

Sharded + resumable. Usage: python b1_temporal.py --shard i --num-shards 10
"""
import argparse, glob, json, os
import numpy as np
from collections import defaultdict

from videogen1.gen import load_pipe, generate, FrameRewardScorer, video_lpips, append_jsonl, RESULTS
from videogen1.video_io import write_video
from videogen1.caching import CacheConfig, wrap_pipeline

TAUS = {"tau005": 0.05, "v0": 0.10, "tau020": 0.20}


def winners():
    out = []  # (prompt, tau_tag, tau, winner_seed)
    for tag, tau in TAUS.items():
        by = defaultdict(dict)
        for f in glob.glob(os.path.join(RESULTS, f"b1_gate_{tag}", "scores_shard*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if r["variant"] == "cached":
                    by[r["prompt"]][r["seed"]] = r["score"]
        for prompt, d in sorted(by.items()):
            if len(d) >= 8:
                seed = max(d, key=d.get)
                out.append((prompt, tag, tau, seed))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--frames", type=int, default=81)
    args = ap.parse_args()

    work = winners()[args.shard::args.num_shards]
    outdir = os.path.join(RESULTS, "b1_temporal")
    vdir = os.path.join(outdir, "videos")
    log = os.path.join(outdir, f"records_shard{args.shard}.jsonl")
    done = set()
    if os.path.exists(log):
        done = {(r["prompt"], r["tau_tag"]) for r in map(json.loads, open(log))}

    pipe = load_pipe()
    scorer = FrameRewardScorer()
    cache = wrap_pipeline(pipe, CacheConfig(mode="off", total_steps=args.steps))

    for prompt, tag, tau, seed in work:
        if (prompt, tag) in done:
            continue
        h = abs(hash(prompt)) % 10**8
        # keep-draft: cached rollout at winner seed
        cache.cfg = CacheConfig(mode="adaptive", tau=tau, total_steps=args.steps)
        cache.reset()
        keep, lat_k = generate(pipe, prompt, seed, frames=args.frames, steps=args.steps)
        # commit: full rollout, same seed (deterministic twin)
        cache.cfg = CacheConfig(mode="off", total_steps=args.steps)
        cache.reset()
        commit, lat_c = generate(pipe, prompt, seed, frames=args.frames, steps=args.steps)

        lp = video_lpips(keep, commit, k=16)
        rec = dict(prompt=prompt, tau_tag=tag, tau=tau, seed=seed,
                   lpips_keep_vs_commit=lp,
                   reward_keep=scorer.score(keep, prompt),
                   reward_commit=scorer.score(commit, prompt),
                   lat_keep=lat_k, lat_commit=lat_c)
        write_video(keep, os.path.join(vdir, f"{h}_{tag}_keep.mp4"))
        write_video(commit, os.path.join(vdir, f"{h}_{tag}_commit.mp4"))
        append_jsonl(log, rec)
        print(f"[{tag}] lpips={lp:.4f} rk={rec['reward_keep']:+.3f} rc={rec['reward_commit']:+.3f} :: {prompt[:50]}")

    print("shard done:", log)


if __name__ == "__main__":
    main()
