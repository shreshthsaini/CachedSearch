"""End-to-end example: best-of-8 search on Wan2.1 for the price of about 5 rollouts.

    python examples/run_wan.py --prompt "a red fox running through deep snow"

Swap the pipeline for CogVideoX, HunyuanVideo, or LTX-Video and the only thing
that changes is tau (run examples/calibrate_new_model.py once for a new family).
"""
import argparse

import torch
from diffusers import AutoencoderKLWan, WanPipeline

from cachedsearch import cached_search


def load_imagereward_verifier(device="cuda", num_frames=8):
    """ImageReward averaged over uniformly spaced frames: our default verifier.

    Any callable (frames, prompt) -> float works here. Use whatever scorer your
    workflow already trusts; CachedSearch never assumes a particular verifier.
    """
    import ImageReward as RM
    import numpy as np
    from PIL import Image

    model = RM.load("ImageReward-v1.0", device=device)

    def verifier(frames, prompt):
        idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
        scores = [
            model.score(prompt, Image.fromarray(np.asarray(frames[i]).astype("uint8")))
            for i in idx
        ]
        return float(np.mean(scores))

    return verifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    ap.add_argument("--n", type=int, default=8, help="search width")
    ap.add_argument("--tau", type=float, default=0.10, help="caching threshold")
    ap.add_argument("--mode", default="commit", choices=["commit", "keep"])
    ap.add_argument("--out", default="delivered.mp4")
    args = ap.parse_args()

    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16).to("cuda")

    result = cached_search(
        pipe,
        args.prompt,
        verifier=load_imagereward_verifier(),
        n=args.n,
        tau=args.tau,
        mode=args.mode,
        gen_kwargs=dict(height=480, width=832, num_frames=81, guidance_scale=5.0),
    )

    print(f"winner seed      : {result.seed}")
    print(f"draft scores     : {[round(s, 3) for s in result.draft_scores]}")
    print(f"explore / commit : {result.explore_seconds:.0f}s / {result.commit_seconds:.0f}s")
    print(f"total            : {result.total_seconds:.0f}s "
          f"(full best-of-{args.n} would cost about "
          f"{args.n * result.commit_seconds:.0f}s)")

    from diffusers.utils import export_to_video
    export_to_video(result.video, args.out, fps=16)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
