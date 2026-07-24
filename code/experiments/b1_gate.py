"""B1 GATE EXPERIMENT (Paper 1 go/no-go).

Question: does training-free caching corrupt candidate RANKING in test-time search?
Protocol: per prompt, N seeds; generate each candidate twice (full compute vs cached);
score both with the verifier; compute per-prompt Spearman rho + top-1 agreement.
GO if median rho > 0.8 across prompts.

Sharded for the drip-feed:  python b1_gate.py --shard 0 --num-shards 10
Resumable: skips (prompt, seed, variant) triples already in the output jsonl.
"""
import argparse, os, itertools, json
import numpy as np
import torch

from videogen1.gen import load_pipe, generate, save_video, FrameRewardScorer, append_jsonl, RESULTS
from videogen1.caching import CacheConfig, wrap_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts_gate50.txt"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--variants", default="full,cached",
                    help="comma list; use 'cached' only for tau sweeps (full is tau-independent, reuse v0)")
    ap.add_argument("--save-videos", action="store_true")
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--model-id", default=None, help="HF model id (default Wan2.1-1.3B)")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    prompts = [p.strip() for p in open(args.prompts) if p.strip()]
    prompts = prompts[args.shard::args.num_shards]
    outdir = os.path.join(RESULTS, f"b1_gate_{args.tag}")
    # write to our own shard file (no write races), but build the done-set from ALL
    # shard files in the tag dir ; lets different shard partitions (29-way array,
    # 64-way multi-node sweep) coexist without recomputing each other's work
    import glob as _glob
    suffix = f"_p{args.num_shards}" if args.num_shards != 10 else ""
    log = os.path.join(outdir, f"scores_shard{args.shard}{suffix}.jsonl")
    done = set()
    for f in _glob.glob(os.path.join(outdir, "scores_shard*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["prompt"], r["seed"], r["variant"]))

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "videogen1"),
                         name=f"b1-gate-{args.tag}-s{args.shard}", config=vars(args))

    pipe = load_pipe(model_id=args.model_id)
    scorer = FrameRewardScorer()
    cache = wrap_pipeline(pipe, CacheConfig(mode="off", total_steps=args.steps))

    def refresh_done():
        d = set()
        for f in _glob.glob(os.path.join(outdir, "scores_shard*.jsonl")):
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d.add((r["prompt"], r["seed"], r["variant"]))
        return d

    variants = tuple(v.strip() for v in args.variants.split(","))
    n_done_local = 0
    for prompt, seed in itertools.product(prompts, range(args.seeds)):
        for variant in variants:
            # periodic refresh: with heterogeneous parallel jobs (8/16/64-node +
            # singles) chewing the same tag, re-glob so we skip others' progress
            if n_done_local and n_done_local % 25 == 0:
                done = refresh_done()
            if (prompt, seed, variant) in done:
                continue
            cache.cfg = CacheConfig(
                mode="off" if variant == "full" else "adaptive",
                tau=args.tau, total_steps=args.steps)
            cache.reset()
            video, latency = generate(pipe, prompt, seed,
                                      height=args.height, width=args.width,
                                      frames=args.frames, steps=args.steps)
            score = scorer.score(video, prompt)
            rec = dict(prompt=prompt, seed=seed, variant=variant, score=score,
                       latency=latency, tau=args.tau, stats=cache.stats())
            append_jsonl(log, rec)
            n_done_local += 1
            if args.save_videos:
                h = abs(hash(prompt)) % 10**8
                save_video(video, os.path.join(outdir, "videos", f"{h}_{seed}_{variant}.mp4"))
            if run:
                run.log({f"latency/{variant}": latency, f"score/{variant}": score})
            print(f"[{variant:6s}] seed={seed} {latency:6.1f}s score={score:+.3f} :: {prompt[:60]}")

    print("shard done:", log)


if __name__ == "__main__":
    main()
