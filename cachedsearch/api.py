"""The user-facing CachedSearch API.

Three functions:

    cached_search(pipe, prompt, verifier, n=8)   run search on one prompt
    cached_search_batch(pipe, prompts, verifier) run search on many prompts
    calibrate_tau(pipe, prompts, verifier)       find the threshold for a new model

Everything is training-free. The only model-specific quantity is the caching
threshold tau, which `calibrate_tau` measures in about two GPU-hours.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from videogen1.caching import CacheConfig, wrap_pipeline

# A verifier maps (video_frames, prompt) to a scalar; higher is better.
Verifier = Callable[[object, str], float]


@dataclass
class SearchResult:
    video: object          # the delivered video (full-compute sample in commit mode)
    seed: int              # the seed that produced it
    draft_scores: list     # verifier score of every cached draft
    explore_seconds: float
    commit_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.explore_seconds + self.commit_seconds


def _generate(pipe, prompt: str, seed: int, steps: int, gen_kwargs: dict):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(prompt=prompt, num_inference_steps=steps, generator=generator, **gen_kwargs)
    return out.frames[0]


def cached_search(
    pipe,
    prompt: str,
    verifier: Verifier,
    n: int = 8,
    tau: float = 0.10,
    steps: int = 50,
    mode: str = "commit",
    seeds: Sequence[int] | None = None,
    gen_kwargs: dict | None = None,
) -> SearchResult:
    """Best-of-n search where exploration runs under caching.

    Args:
        pipe: any diffusers video pipeline exposing `.transformer` (Wan, CogVideoX,
            HunyuanVideo, LTX-Video all work).
        prompt: the text prompt.
        verifier: callable (frames, prompt) -> float, higher is better. Any scorer
            works; we used ImageReward averaged over 8 uniformly spaced frames.
        n: search width (number of candidates). Recommit pays off for n >= 4.
        tau: caching threshold. 0.10 is calibrated for Wan2.1; run `calibrate_tau`
            for a new architecture (do not copy this constant across families).
        steps: denoising steps.
        mode: "commit" regenerates the winner at full compute (recommended; the
            delivered video is a genuine full-compute sample). "keep" returns the
            cached draft directly: cheaper, but caching artifacts reach the user
            and motion is mildly dampened.
        seeds: explicit seeds; defaults to range(n).
        gen_kwargs: extra pipeline kwargs (height, width, num_frames, guidance...).

    Returns:
        SearchResult with the delivered video, its seed, draft scores, and timings.
    """
    seeds = list(seeds if seeds is not None else range(n))
    gen_kwargs = dict(gen_kwargs or {})

    # ---- explore: every candidate under caching -------------------------------
    cache = wrap_pipeline(pipe, CacheConfig(mode="adaptive", tau=tau, total_steps=steps))
    drafts, scores = [], []
    t0 = time.time()
    for seed in seeds:
        cache.reset()                      # cache state must not leak across candidates
        frames = _generate(pipe, prompt, seed, steps, gen_kwargs)
        drafts.append(frames)
        scores.append(float(verifier(frames, prompt)))
    explore_s = time.time() - t0

    best = int(np.argmax(scores))
    winner_seed = seeds[best]

    # ---- commit: regenerate only the winner at full compute -------------------
    if mode == "keep":
        return SearchResult(drafts[best], winner_seed, scores, explore_s, 0.0)

    wrap_pipeline(pipe, CacheConfig(mode="off", total_steps=steps))
    t0 = time.time()
    video = _generate(pipe, prompt, winner_seed, steps, gen_kwargs)
    commit_s = time.time() - t0
    return SearchResult(video, winner_seed, scores, explore_s, commit_s)


def cached_search_batch(
    pipe, prompts: Iterable[str], verifier: Verifier, **kwargs
) -> list[SearchResult]:
    """cached_search over many prompts; returns one SearchResult per prompt."""
    return [cached_search(pipe, p, verifier, **kwargs) for p in prompts]


def calibrate_tau(
    pipe,
    prompts: Sequence[str],
    verifier: Verifier,
    taus: Sequence[float] = (0.05, 0.10, 0.20),
    n: int = 8,
    steps: int = 50,
    target_capture: float = 0.90,
    gen_kwargs: dict | None = None,
) -> dict:
    """Measure the speed-fidelity frontier of a NEW model and pick tau.

    Protocol (this is the paper's protocol in miniature): generate every
    (prompt, seed) pair twice, once at full compute and once under caching,
    score both, and report how much of full-compute search's gain the cached
    ranking retains. Roughly two GPU-hours with 25 prompts on a 1.3B model.

    Returns {tau: {"speedup", "capture", "spearman"}} plus "recommended": the
    largest tau whose capture is at or above target_capture.
    """
    prompts = list(prompts)
    gen_kwargs = dict(gen_kwargs or {})
    seeds = list(range(n))

    # full-compute references, shared across all tau values
    wrap_pipeline(pipe, CacheConfig(mode="off", total_steps=steps))
    full_scores, full_latency = {}, []
    for p in prompts:
        row = []
        for s in seeds:
            t0 = time.time()
            frames = _generate(pipe, p, s, steps, gen_kwargs)
            full_latency.append(time.time() - t0)
            row.append(float(verifier(frames, p)))
        full_scores[p] = np.asarray(row)
    cf = float(np.mean(full_latency))

    report = {}
    for tau in taus:
        cache = wrap_pipeline(pipe, CacheConfig(mode="adaptive", tau=tau, total_steps=steps))
        rhos, captures, latency = [], [], []
        for p in prompts:
            row = []
            for s in seeds:
                cache.reset()
                t0 = time.time()
                frames = _generate(pipe, p, s, steps, gen_kwargs)
                latency.append(time.time() - t0)
                row.append(float(verifier(frames, p)))
            cached = np.asarray(row)
            full = full_scores[p]
            ra, rb = np.argsort(np.argsort(full)), np.argsort(np.argsort(cached))
            rhos.append(float(np.corrcoef(ra, rb)[0, 1]))
            pick, spread = int(np.argmax(cached)), full.max() - full.mean()
            captures.append((full[pick] - full.mean()) / spread if spread > 1e-9 else 1.0)
        report[tau] = {
            "speedup": cf / float(np.mean(latency)),
            "capture": float(np.mean(captures)),
            "spearman": float(np.median(rhos)),
        }

    ok = [t for t in taus if report[t]["capture"] >= target_capture]
    report["recommended"] = max(ok) if ok else min(taus)
    return report
