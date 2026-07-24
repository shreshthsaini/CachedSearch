"""P0 smoke: one full + one cached generation, latency + cache stats + saved mp4s,
plus verifier scoring (covers the full gate-experiment dependency surface).
Run on gh-dev. Success = both videos exist, cached is faster, no NaNs, scores finite."""
import time, os, torch
from videogen1.gen import load_pipe, generate, save_video, FrameRewardScorer, RESULTS
from videogen1.caching import CacheConfig, wrap_pipeline

STEPS, FRAMES = 30, 33  # small smoke config

pipe = load_pipe()
scorer = FrameRewardScorer()
prompt = "A red fox trotting through fresh snow in a pine forest"
cache = wrap_pipeline(pipe, CacheConfig(mode="off", total_steps=STEPS))

for variant, mode in (("full", "off"), ("cached", "adaptive")):
    cache.cfg = CacheConfig(mode=mode, tau=0.10, total_steps=STEPS)
    cache.reset()
    video, latency = generate(pipe, prompt, seed=0, frames=FRAMES, steps=STEPS)
    import numpy as np
    arr = np.asarray(video[0])
    assert not np.isnan(arr.astype(float)).any(), "NaN frames!"
    score = scorer.score(video, prompt)
    assert np.isfinite(score), "non-finite reward score!"
    save_video(video, os.path.join(RESULTS, "smoke", f"{variant}.mp4"))
    print(f"SMOKE {variant}: {latency:.1f}s score={score:+.3f} stats={cache.stats()}")
print("SMOKE_OK")
