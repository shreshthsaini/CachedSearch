"""B1 GATE EXPERIMENT on LTX-Video 2B (gate5: model-breadth rung for Paper 1).

Same protocol/CLI as b1_gate.py / b1_gate_cog.py, ported to LTX-Video via
videogen1.gen_ltx: per prompt, N seeds; generate each candidate twice (full vs
cached); score both; b1_analyze.py computes per-prompt Spearman rho + top-1.

KEY DIFFERENCES vs Wan:
  - LTX does CFG by batch-concat in a SINGLE transformer call per step
    => num_branches=1 (enforced by gen_ltx.ltx_cache_config).
  - The diffusers-format checkpoint of Lightricks/LTX-Video is the BASE 2B
    (50-step CFG model, guidance 3.0) ; NOT the distilled few-step variant.
    Defaults here are its canonical doc-example sampling: 161 frames, 704x480,
    50 steps, guidance 3.0, standard negative prompt.
  - --distilled runs the FEW-STEP boundary probe: swaps in the 2b-0.9.8
    distilled transformer+VAE (single-file ckpt, same repo) and samples the
    official 7-step custom-timestep schedule with NO CFG; warmup/cooldown
    auto-scale to 2/2 (gen_ltx.scaled_warmup_cooldown). This measures whether
    adaptive caching has anything to skip on few-step schedules.

Sharded for the taskq/drip-feed:  python b1_gate_ltx.py --shard 0 --num-shards 5
Resumable: skips (prompt, seed, variant) triples already in ANY shard file of
the tag dir. Results dir: results/b1_gate_ltx_<tag>/ (analyze: b1_analyze.py
--tag ltx_<tag>).
"""
import argparse, os, itertools, json

from videogen1.gen import save_video, FrameRewardScorer, append_jsonl, RESULTS
from videogen1.gen_ltx import (load_pipe_ltx, generate_ltx, wrap_pipeline_ltx,
                               ltx_cache_config, LTX_DISTILLED_TIMESTEPS,
                               LTX_DISTILLED_GUIDANCE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts_gate50.txt"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--frames", type=int, default=161)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=704)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--distilled", action="store_true",
                    help="few-step boundary probe: 2b-0.9.8-distilled weights, "
                         "7 custom timesteps, no CFG, warmup/cooldown 2/2")
    ap.add_argument("--variants", default="full,cached",
                    help="comma list; use 'cached' only for tau sweeps (full is tau-independent, reuse v0)")
    ap.add_argument("--save-videos", action="store_true")
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    timesteps = None
    if args.distilled:
        timesteps = LTX_DISTILLED_TIMESTEPS
        args.steps = len(timesteps)
        args.guidance = LTX_DISTILLED_GUIDANCE

    prompts = [p.strip() for p in open(args.prompts) if p.strip()]
    prompts = prompts[args.shard::args.num_shards]
    outdir = os.path.join(RESULTS, f"b1_gate_ltx_{args.tag}")
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
                         name=f"b1-gate-ltx-{args.tag}-s{args.shard}", config=vars(args))

    pipe = load_pipe_ltx(distilled=args.distilled)
    scorer = FrameRewardScorer()
    cache = wrap_pipeline_ltx(pipe, ltx_cache_config(mode="off", total_steps=args.steps))

    variants = tuple(v.strip() for v in args.variants.split(","))
    n_done_local = 0
    for prompt, seed in itertools.product(prompts, range(args.seeds)):
        for variant in variants:
            if n_done_local and n_done_local % 25 == 0:
                done = refresh_done()
            if (prompt, seed, variant) in done:
                continue
            cache.cfg = ltx_cache_config(
                mode="off" if variant == "full" else "adaptive",
                tau=args.tau, total_steps=args.steps)
            cache.reset()
            video, latency = generate_ltx(
                pipe, prompt, seed, height=args.height, width=args.width,
                frames=args.frames, steps=args.steps, guidance=args.guidance,
                timesteps=timesteps,
                decode_timestep=0.05 if args.distilled else 0.0,
                decode_noise_scale=0.025 if args.distilled else None)
            score = scorer.score(video, prompt)
            rec = dict(prompt=prompt, seed=seed, variant=variant, score=score,
                       latency=latency, tau=args.tau, stats=cache.stats(),
                       steps=args.steps, distilled=args.distilled)
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
