"""Recompute LPIPS(keep, commit) from SAVED E1 videos (post video_lpips bugfix). CPU-safe."""
import glob, json, os
import numpy as np
from videogen1.video_io import read_video_np
from videogen1.gen import video_lpips, RESULTS

vdir = os.path.join(RESULTS, "b1_temporal", "videos")
out = os.path.join(RESULTS, "b1_temporal", "lpips_fixed.jsonl")
done = set()
if os.path.exists(out):
    done = {json.loads(l)["key"] for l in open(out)}
keeps = sorted(glob.glob(os.path.join(vdir, "*_keep.mp4")))
print(f"{len(keeps)} pairs, {len(done)} done")
for kp in keeps:
    key = os.path.basename(kp)[:-9]
    if key in done:
        continue
    cp = kp.replace("_keep.mp4", "_commit.mp4")
    if not os.path.exists(cp):
        continue
    a, b = read_video_np(kp), read_video_np(cp)
    lp = video_lpips(a, b, k=16, device="cuda")
    with open(out, "a") as f:
        f.write(json.dumps({"key": key, "lpips": lp}) + "\n")
    print(f"{key} lpips={lp:.4f}", flush=True)
print("RECOMPUTE_DONE")
