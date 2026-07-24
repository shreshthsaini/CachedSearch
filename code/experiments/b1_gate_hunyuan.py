"""B1 GATE EXPERIMENT on HunyuanVideo-13B (gate5: model-breadth rung for Paper 1).

Same protocol/CLI as b1_gate.py / b1_gate_cog.py, ported to HunyuanVideo via
videogen1.gen_hunyuan: per prompt, N seeds; generate each candidate twice (full
vs cached); score both; b1_analyze.py computes per-prompt Spearman rho + top-1.

KEY DIFFERENCE vs Wan: HunyuanVideo is GUIDANCE-DISTILLED ; one transformer
call per step with an embedded guidance scale; NO uncond pass at all
(true_cfg_scale=1.0). The cache wrapper therefore runs with num_branches=1
(enforced by gen_hunyuan.hunyuan_cache_config). Defaults: 61 frames, 480x720,
50 steps, embedded guidance 6.0, whole pipe bf16 + VAE tiling (13B + Llama-8B
text encoder ~= 43 GB weights; decode is the peak-memory step).

Sharded for the taskq/drip-feed:  python b1_gate_hunyuan.py --shard 0 --num-shards 5
Resumable: skips (prompt, seed, variant) triples already in ANY shard file of
the tag dir. Results dir: results/b1_gate_hunyuan_<tag>/ (analyze:
b1_analyze.py --tag hunyuan_<tag>).
"""
import argparse, os, itertools, json

from videogen1.gen import save_video, FrameRewardScorer, append_jsonl, RESULTS
from videogen1.gen_hunyuan import (load_pipe_hunyuan, generate_hunyuan,
                                   wrap_pipeline_hunyuan, hunyuan_cache_config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts_gate50.txt"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--frames", type=int, default=61)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--guidance", type=float, default=6.0,
                    help="EMBEDDED distilled guidance (not true CFG)")
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--variants", default="full,cached",
                    help="comma list; use 'cached' only for tau sweeps (full is tau-independent, reuse v0)")
    ap.add_argument("--save-videos", action="store_true")
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    prompts = [p.strip() for p in open(args.prompts) if p.strip()]
    prompts = prompts[args.shard::args.num_shards]
    outdir = os.path.join(RESULTS, f"b1_gate_hunyuan_{args.tag}")
    # write to our own shard file (no write races), but build the done-set from ALL
    # shard files in the tag dir ; lets different shard partitions coexist without
    # recomputing each other's work (same convention as b1_gate.py)
    import glob as _glob
    suffix = f"_p{args.num_shards}" if args.num_shards != 10 else ""
    log = os.path.join(outdir, f"scores_shard{args.shard}{suffix}.jsonl")

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

    done = refresh_done()

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "videogen1"),
                         name=f"b1-gate-hunyuan-{args.tag}-s{args.shard}", config=vars(args))

    pipe = load_pipe_hunyuan()
    scorer = FrameRewardScorer()
    cache = wrap_pipeline_hunyuan(pipe, hunyuan_cache_config(mode="off", total_steps=args.steps))

    variants = tuple(v.strip() for v in args.variants.split(","))
    n_done_local = 0
    for prompt, seed in itertools.product(prompts, range(args.seeds)):
        for variant in variants:
            if n_done_local and n_done_local % 25 == 0:
                done = refresh_done()
            if (prompt, seed, variant) in done:
                continue
            cache.cfg = hunyuan_cache_config(
                mode="off" if variant == "full" else "adaptive",
                tau=args.tau, total_steps=args.steps)
            cache.reset()
            video, latency = generate_hunyuan(pipe, prompt, seed,
                                              height=args.height, width=args.width,
                                              frames=args.frames, steps=args.steps,
                                              guidance=args.guidance)
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
