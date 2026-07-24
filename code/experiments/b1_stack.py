"""B1 E6: CachedSearch STACKS ON mid-trajectory pruning (LatSearch-style lever).

Cached rollouts and early candidate pruning combine multiplicatively.
CachedSearch acts as a multiplier on search methods.

Protocol per prompt (N seeds):
  stage 1  run ALL N candidates CACHED, but only to step K (= --prune-step) of --steps;
  preview  VAE-decode the x0 estimate at step K (UniPC flow_prediction keeps
           x0_pred = x_t - sigma_t*v in model_outputs[-1]; see gen_resumable.py) for
           --preview-frames evenly spaced frames only; score with ImageReward-frames;
  prune    keep top --topk candidates;
  stage 2  resume ONLY survivors (exact UniPC+cache state restore) to full --steps,
           decode + score full videos; method's pick = argmax cached score;
  commit   regenerate the picked seed at FULL compute via the untouched pipeline path
           (deterministic; gives arm (c) score + measured commit wall-clock).

Arms recorded per prompt:
  (a) prune+cached  = this script (candidates/preview/final fields + timings)
  (b) cached-only all-N-full-length = REUSED from results/b1_gate_v0 (join on prompt
      at analysis time; same prompts/seeds/steps/frames/tau)
  (c) commit_score  = full-compute score of the final pick (regret vs the known v0
      full scores: regret = max_seed full_v0(prompt,seed) - commit_score)

Cost accounting: wall-clock per stage; e2e_s = encode+stage1+preview+stage2+final
scoring+commit; cached_allN_cost_est_s = N * C_c where C_c = mean full-length cached
candidate cost measured in-run (stage1/K-step segment scaled by survivors' stage-2
segment + decode + score).

Sharded/resumable like b1_gate.py (record granularity = prompt):
  python experiments/b1_stack.py --shard 0 --num-shards 5
Smoke:  python experiments/b1_stack.py --smoke   (2 prompts, 2 seeds, 20 steps, K=8)
Results -> $CACHEDSEARCH_RESULTS/b1_stack_<tag>/scores_shard*.jsonl
"""
import argparse, glob, hashlib, json, os, time

import numpy as np
import torch

from videogen1.gen import (RESULTS, FrameRewardScorer, append_jsonl, generate,
                           load_pipe, save_video)
from videogen1.gen_resumable import ResumableWanRunner
from videogen1.caching import CacheConfig, wrap_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=os.environ.get("CACHEDSEARCH_PROMPTS", "prompts.txt"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--prune-step", type=int, default=20, help="K: pause/preview/prune step")
    ap.add_argument("--topk", type=int, default=4, help="candidates surviving the prune")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--preview-frames", type=int, default=4)
    ap.add_argument("--save-videos", action="store_true")
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="2 prompts, 2 seeds, 20 steps, K=8, topk=1, 33 frames, tag=smoke")
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.steps, args.prune_step, args.topk = 2, 20, 8, 1
        args.frames = 33
        if args.tag == "v0":
            args.tag = "smoke"

    prompts = [p.strip() for p in open(args.prompts) if p.strip()]
    if args.smoke:
        prompts = prompts[:2]
    prompts = prompts[args.shard::args.num_shards]

    outdir = os.path.join(RESULTS, f"b1_stack_{args.tag}")
    log = os.path.join(outdir, f"scores_shard{args.shard}.jsonl")

    def done_prompts():
        d = set()
        for f in glob.glob(os.path.join(outdir, "scores_shard*.jsonl")):
            for line in open(f):
                try:
                    d.add(json.loads(line)["prompt"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return d

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "videogen1"),
                         name=f"b1-stack-{args.tag}-s{args.shard}", config=vars(args))

    pipe = load_pipe()
    scorer = FrameRewardScorer()
    cache = wrap_pipeline(pipe, CacheConfig(mode="adaptive", tau=args.tau, total_steps=args.steps))
    runner = ResumableWanRunner(pipe, height=args.height, width=args.width,
                                frames=args.frames, steps=args.steps)

    done = done_prompts()
    for prompt in prompts:
        if prompt in done:
            continue
        done = done_prompts()  # re-glob: other shards/reruns may have progressed
        if prompt in done:
            continue

        cache.cfg = CacheConfig(mode="adaptive", tau=args.tau, total_steps=args.steps)

        t0 = time.time()
        embeds = runner.encode_prompt(prompt)
        t_encode = time.time() - t0

        # ---- stage 1: all N candidates, cached, to step K -------------------------
        states, stage1_times = [], []
        for seed in range(args.seeds):
            st = runner.new_state(seed)
            t1 = time.time()
            runner.run(st, embeds, args.prune_step)
            stage1_times.append(time.time() - t1)
            states.append(st)
        t_stage1 = float(sum(stage1_times))

        # ---- preview: decode x0 estimate (few frames) + score ---------------------
        t2 = time.time()
        preview_scores = []
        k_orig = scorer.k
        scorer.k = args.preview_frames
        for st in states:
            frames = runner.decode_preview(st.x0_pred, n_frames=args.preview_frames)
            preview_scores.append(scorer.score(frames, prompt))
        scorer.k = k_orig
        t_preview = time.time() - t2

        order = np.argsort(preview_scores)[::-1]
        survivors = sorted(int(i) for i in order[:args.topk])
        pruned = sorted(int(i) for i in order[args.topk:])

        # ---- stage 2: resume survivors to full length, decode + score -------------
        stage2_times, decode_times, score_times = [], [], []
        final_scores = {}
        videos = {}
        survivor_cache_stats = {}
        for i in survivors:
            st = states[i]
            t3 = time.time()
            runner.run(st, embeds, args.steps)
            stage2_times.append(time.time() - t3)
            survivor_cache_stats[str(st.seed)] = cache.stats()  # cumulative over both segments
            t4 = time.time()
            video = runner.decode_full(st.latents)
            decode_times.append(time.time() - t4)
            t5 = time.time()
            final_scores[st.seed] = scorer.score(video, prompt)
            score_times.append(time.time() - t5)
            if args.save_videos:
                videos[st.seed] = video
        t_stage2 = float(sum(stage2_times))
        t_decode = float(sum(decode_times))
        t_score_final = float(sum(score_times))

        picked_seed = max(final_scores, key=final_scores.get)

        # ---- commit: regenerate winner at FULL compute (arm c) --------------------
        cache.cfg = CacheConfig(mode="off", total_steps=args.steps)
        cache.reset()
        full_video, t_commit = generate(pipe, prompt, picked_seed,
                                        height=args.height, width=args.width,
                                        frames=args.frames, steps=args.steps)
        commit_score = scorer.score(full_video, prompt)
        cache.cfg = CacheConfig(mode="adaptive", tau=args.tau, total_steps=args.steps)

        h = hashlib.md5(prompt.encode()).hexdigest()[:8]
        if args.save_videos:
            for seed, video in videos.items():
                save_video(video, os.path.join(outdir, "videos", f"{h}_{seed}_stacked.mp4"))
            save_video(full_video, os.path.join(outdir, "videos", f"{h}_{picked_seed}_commit.mp4"))

        # ---- cost accounting -------------------------------------------------------
        e2e = t_encode + t_stage1 + t_preview + t_stage2 + t_decode + t_score_final + t_commit
        # C_c: full-length cached candidate cost (K-step segment + rest + decode + score)
        c_c = (float(np.mean(stage1_times)) + float(np.mean(stage2_times))
               + float(np.mean(decode_times)) + float(np.mean(score_times)))
        rec = dict(
            prompt=prompt, prompt_hash=h, tag=args.tag, tau=args.tau,
            seeds=args.seeds, steps=args.steps, prune_step=args.prune_step,
            topk=args.topk, frames=args.frames, preview_frames=args.preview_frames,
            preview_scores=[float(s) for s in preview_scores],
            survivors=survivors, pruned=pruned,
            final_scores={str(k): float(v) for k, v in final_scores.items()},
            picked_seed=int(picked_seed), commit_score=float(commit_score),
            timings=dict(encode_s=t_encode, stage1_s=t_stage1, stage1_per_cand=stage1_times,
                         preview_s=t_preview, stage2_s=t_stage2, stage2_per_cand=stage2_times,
                         decode_s=t_decode, score_final_s=t_score_final,
                         commit_s=t_commit, e2e_s=e2e,
                         cached_candidate_cost_est_s=c_c,
                         cached_allN_cost_est_s=args.seeds * c_c),
            cache_stats=survivor_cache_stats,
        )
        append_jsonl(log, rec)
        if run:
            run.log({"e2e_s": e2e, "commit_score": commit_score,
                     "allN_est_s": args.seeds * c_c})
        print(f"[stack] pick=seed{picked_seed} commit={commit_score:+.3f} "
              f"e2e={e2e:.0f}s vs allN~{args.seeds * c_c:.0f}s+commit :: {prompt[:60]}")

    print("shard done:", log)


if __name__ == "__main__":
    main()
