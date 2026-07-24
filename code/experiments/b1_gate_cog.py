"""B1 GATE EXPERIMENT on CogVideoX-5B (E4: second model-ladder rung for Paper 1).

Same protocol/CLI as b1_gate.py, ported to CogVideoX-5B via videogen1.gen_cog:
per prompt, N seeds; generate each candidate twice (full vs cached); score both;
b1_analyze.py computes per-prompt Spearman rho + top-1 agreement.

KEY DIFFERENCE vs Wan: CogVideoX does CFG by batch-concat in a SINGLE transformer
call per step, so the cache wrapper runs with num_branches=1 (enforced by
gen_cog.cog_cache_config). Defaults are CogVideoX-5b's: 49 frames, 480x720,
50 steps, guidance 6.0.

Sharded for the taskq/drip-feed:  python b1_gate_cog.py --shard 0 --num-shards 10
Resumable: skips (prompt, seed, variant) triples already in ANY shard file of the
tag dir. Results dir: results/b1_gate_cog5b_<tag>/ (analyze: b1_analyze.py --tag cog5b_<tag>).
"""
import argparse, os, itertools, json

from videogen1.gen import save_video, FrameRewardScorer, append_jsonl, RESULTS
from videogen1.gen_cog import load_pipe_cog, generate_cog, wrap_pipeline_cog, cog_cache_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=os.environ.get("CACHEDSEARCH_PROMPTS", "prompts.txt"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--variants", default="full,cached",
                    help="comma list; use 'cached' only for tau sweeps (full is tau-independent, reuse v0)")
    ap.add_argument("--save-videos", action="store_true")
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    prompts = [p.strip() for p in open(args.prompts) if p.strip()]
    prompts = prompts[args.shard::args.num_shards]
    outdir = os.path.join(RESULTS, f"b1_gate_cog5b_{args.tag}")
    # write to our own shard file (no write races), but build the done-set from ALL
    # shard files in the tag dir ; lets different shard partitions coexist without
    # recomputing each other's work (same convention as b1_gate.py)
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
                         name=f"b1-gate-cog5b-{args.tag}-s{args.shard}", config=vars(args))

    pipe = load_pipe_cog()
    scorer = FrameRewardScorer()
    cache = wrap_pipeline_cog(pipe, cog_cache_config(mode="off", total_steps=args.steps))

    variants = tuple(v.strip() for v in args.variants.split(","))
    for prompt, seed in itertools.product(prompts, range(args.seeds)):
        for variant in variants:
            if (prompt, seed, variant) in done:
                continue
            cache.cfg = cog_cache_config(
                mode="off" if variant == "full" else "adaptive",
                tau=args.tau, total_steps=args.steps)
            cache.reset()
            video, latency = generate_cog(pipe, prompt, seed,
                                          height=args.height, width=args.width,
                                          frames=args.frames, steps=args.steps,
                                          guidance=args.guidance)
            score = scorer.score(video, prompt)
            rec = dict(prompt=prompt, seed=seed, variant=variant, score=score,
                       latency=latency, tau=args.tau, stats=cache.stats())
            append_jsonl(log, rec)
            if args.save_videos:
                h = abs(hash(prompt)) % 10**8
                save_video(video, os.path.join(outdir, "videos", f"{h}_{seed}_{variant}.mp4"))
            if run:
                run.log({f"latency/{variant}": latency, f"score/{variant}": score})
            print(f"[{variant:6s}] seed={seed} {latency:6.1f}s score={score:+.3f} :: {prompt[:60]}")

    print("shard done:", log)


if __name__ == "__main__":
    main()
