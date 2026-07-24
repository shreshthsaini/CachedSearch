"""A1 UNIFIED-PROTOCOL BASELINE RUNNER (Pillar A-lite; baselines for Paper 1).

Reproduce published training-free caching methods under ONE protocol (identical to
the B1 gate: Wan2.1-T2V-1.3B, 480x832, 81 frames, 50 UniPC steps, ImageReward
verifier, prompts_gate50.txt x 8 seeds). For a given --method we generate ONLY the
method (cached-equivalent) variant; full-compute references are REUSED from
results/b1_gate_v0/ (variant=="full"), which ran the exact same protocol.

Methods (see videogen1/baselines.py for provenance):
  teacache       TeaCache, official Wan2.1 rule + published 1.3B coefficients
  pab            PAB-style attention broadcast (self range 2, cross range 6)
  cfgcache       FasterCache's CFG-Cache component (uncond-branch reuse via FFT)
  easycache-ours our EasyCache-style wrapper from videogen1/caching.py (tau)
  none           no caching (fresh full-compute control / latency reference)

Same CLI/sharding/resume/done-set conventions as b1_gate.py:
  python a1_baselines.py --method teacache --shard 0 --num-shards 5
Results -> results/a1_<method>/ (or a1_<method>_<tag> when --tag is set).
Analyze with a1_analyze.py (ranking preservation vs b1_gate_v0 full refs).
"""
import argparse, glob, hashlib, itertools, json, os

from videogen1.gen import load_pipe, generate, save_video, FrameRewardScorer, append_jsonl, RESULTS

METHODS = ("teacache", "pab", "cfgcache", "easycache-ours", "none")


def build_wrapper(pipe, method, args):
    """Attach the selected acceleration to the pipeline; return (wrapper, knobs-dict)."""
    if method in ("easycache-ours", "none"):
        from videogen1.caching import CacheConfig, wrap_pipeline
        cfg = CacheConfig(mode="adaptive" if method == "easycache-ours" else "off",
                          tau=args.tau, total_steps=args.steps)
        return wrap_pipeline(pipe, cfg), {"tau": args.tau} if method == "easycache-ours" else {}
    from videogen1.baselines import apply_baseline
    model_size = "14B" if (args.model_id and "14B" in args.model_id) else "1.3B"
    if method == "teacache":
        knobs = {"thresh": args.teacache_thresh, "use_ret_steps": args.teacache_ret_steps}
    elif method == "pab":
        knobs = {"self_range": args.pab_self_range, "cross_range": args.pab_cross_range,
                 "self_threshold": tuple(args.pab_threshold), "cross_threshold": tuple(args.pab_threshold)}
    elif method == "cfgcache":
        knobs = {"uncond_interval": args.cfgcache_interval}
    else:
        raise ValueError(method)
    return apply_baseline(pipe, method, num_steps=args.steps, model_size=model_size, **knobs), knobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts_gate50.txt"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    # method knobs (defaults = published operating points, see videogen1/baselines.py)
    ap.add_argument("--tau", type=float, default=0.10, help="easycache-ours tolerance (B1 default)")
    ap.add_argument("--teacache-thresh", type=float, default=0.08)
    ap.add_argument("--teacache-ret-steps", action="store_true")
    ap.add_argument("--pab-self-range", type=int, default=2)
    ap.add_argument("--pab-cross-range", type=int, default=6)
    ap.add_argument("--pab-threshold", type=int, nargs=2, default=[100, 850])
    ap.add_argument("--cfgcache-interval", type=int, default=5)
    ap.add_argument("--save-videos", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the results dir (e.g. 'smoke')")
    ap.add_argument("--model-id", default=None, help="HF model id (default Wan2.1-1.3B)")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    prompts = [p.strip() for p in open(args.prompts) if p.strip()]
    prompts = prompts[args.shard::args.num_shards]
    outdir = os.path.join(RESULTS, f"a1_{args.method}" + (f"_{args.tag}" if args.tag else ""))
    # own shard file (no write races); done-set built from ALL shard files in the dir
    # so heterogeneous shard partitions coexist without recomputing (b1_gate convention)
    suffix = f"_p{args.num_shards}" if args.num_shards != 5 else ""
    log = os.path.join(outdir, f"scores_shard{args.shard}{suffix}.jsonl")

    def refresh_done():
        d = set()
        for f in glob.glob(os.path.join(outdir, "scores_shard*.jsonl")):
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
                         name=f"a1-{args.method}{'-' + args.tag if args.tag else ''}-s{args.shard}",
                         config=vars(args))

    pipe = load_pipe(model_id=args.model_id)
    scorer = FrameRewardScorer()
    wrapper, knobs = build_wrapper(pipe, args.method, args)

    variant = args.method
    n_done_local = 0
    for prompt, seed in itertools.product(prompts, range(args.seeds)):
        if n_done_local and n_done_local % 25 == 0:
            done = refresh_done()
        if (prompt, seed, variant) in done:
            continue
        wrapper.reset()
        video, latency = generate(pipe, prompt, seed,
                                  height=args.height, width=args.width,
                                  frames=args.frames, steps=args.steps)
        score = scorer.score(video, prompt)
        rec = dict(prompt=prompt, seed=seed, variant=variant, score=score,
                   latency=latency, method=args.method, knobs=knobs,
                   stats=wrapper.stats())
        append_jsonl(log, rec)
        n_done_local += 1
        if args.save_videos:
            h = hashlib.md5(prompt.encode()).hexdigest()[:10]  # stable across processes
            save_video(video, os.path.join(outdir, "videos", f"{h}_{seed}_{variant}.mp4"))
        if run:
            run.log({f"latency/{variant}": latency, f"score/{variant}": score})
        print(f"[{variant:14s}] seed={seed} {latency:6.1f}s score={score:+.3f} :: {prompt[:60]}")

    print("shard done:", log)


if __name__ == "__main__":
    main()
