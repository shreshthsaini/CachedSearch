"""Find the throughput-optimal candidate batch size on GH200 (96GB).
Measures videos/min + peak VRAM at batch in {1,2,4,6,8}, full and cached,
at the study config (480x832/81f/50 steps). Also checks per-seed determinism:
batch[0] must match a single-run of the same seed (max |diff| on frame 0)."""
import torch, numpy as np
from videogen1.gen import load_pipe, generate, generate_batch
from videogen1.caching import CacheConfig, wrap_pipeline

STEPS, FRAMES, H, W = 50, 81, 480, 832
PROMPT = "A red fox trotting through fresh snow in a pine forest"

pipe = load_pipe()
cache = wrap_pipeline(pipe, CacheConfig(mode="off", total_steps=STEPS))

# determinism reference (single run, seed 0)
cache.cfg = CacheConfig(mode="off", total_steps=STEPS); cache.reset()
ref, _ = generate(pipe, PROMPT, 0, height=H, width=W, frames=FRAMES, steps=STEPS)
ref0 = np.asarray(ref[0], dtype=np.float32)

for mode in ("off", "adaptive"):
    label = "full" if mode == "off" else "cached"
    for bs in (1, 2, 4, 6, 8):
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            cache.cfg = CacheConfig(mode=mode, tau=0.10, total_steps=STEPS)
            cache.reset()
            videos, lat = generate_batch(pipe, PROMPT, list(range(bs)),
                                         height=H, width=W, frames=FRAMES, steps=STEPS)
            mem = torch.cuda.max_memory_allocated() / 2**30
            thr = bs / lat * 60
            drift = float(np.abs(np.asarray(videos[0][0], dtype=np.float32) - ref0).max()) if mode == "off" else -1
            print(f"BENCH {label} bs={bs}: {lat:6.1f}s  {thr:5.2f} vids/min  peak {mem:5.1f} GB  seed0drift {drift:.4f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"BENCH {label} bs={bs}: OOM", flush=True)
            torch.cuda.empty_cache()
            break
print("BENCH_DONE")
