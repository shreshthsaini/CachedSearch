"""Shared generation + scoring utilities (Wan 2.1 1.3B v0)."""
from __future__ import annotations
import os, time, json
import torch
import numpy as np

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
RESULTS = os.environ.get("CACHEDSEARCH_RESULTS", "./results")


def load_pipe(device="cuda", dtype=torch.bfloat16, model_id=None):
    from diffusers import WanPipeline, AutoencoderKLWan
    model_id = model_id or MODEL_ID
    # VAE in fp32: known bf16 dtype trap with video VAEs (see user memory: bf16 dtype traps)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
    pipe.to(device)
    return pipe


@torch.no_grad()
def generate(pipe, prompt: str, seed: int, height=480, width=832, frames=81,
             steps=50, guidance=5.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t0 = time.time()
    out = pipe(prompt=prompt, height=height, width=width, num_frames=frames,
               num_inference_steps=steps, guidance_scale=guidance, generator=g)
    latency = time.time() - t0
    video = out.frames[0]  # list/array of HWC uint8-ish frames
    return video, latency


@torch.no_grad()
def generate_batch(pipe, prompt: str, seeds: list[int], height=480, width=832,
                   frames=81, steps=50, guidance=5.0):
    """Batched candidates: N seeds for one prompt in a single batched forward.
    Per-seed generators -> initial latents identical to individual runs (diffusers
    randn_tensor draws per batch element); kernel reduction order may cause tiny
    numeric drift vs batch=1 ; document in methods, don't mix within one study arm.
    NOTE (caching): the adaptive skip indicator becomes batch-averaged -> one shared
    skip schedule for the whole candidate batch (uniform-schedule semantics).
    Throughput rationale: GH200 96GB is ~30GB-used at batch 1; batching fills VRAM
    and amortizes launch/bandwidth overhead (user directive 2026-07-06)."""
    gens = [torch.Generator(device="cuda").manual_seed(s) for s in seeds]
    t0 = time.time()
    out = pipe(prompt=prompt, height=height, width=width, num_frames=frames,
               num_inference_steps=steps, guidance_scale=guidance,
               generator=gens, num_videos_per_prompt=len(seeds))
    latency = time.time() - t0
    return list(out.frames), latency  # [N] videos, wall-clock for the whole batch


from videogen1.video_io import to_uint8, write_video as save_video  # central video I/O (torchcodec-backed reads)


class FrameRewardScorer:
    """v0 verifier: ImageReward averaged over K uniformly sampled frames.
    TODO(P1): add VisionReward + 2-verifier ensemble per reward-hacking caveat."""

    def __init__(self, k=8, device="cuda"):
        import ImageReward as ir
        hf_home = os.environ.get("HF_HOME", "./hf_cache")
        root = os.environ.get("IMAGEREWARD_ROOT",
                              os.path.join(hf_home, "ImageReward"))
        self.model = ir.load("ImageReward-v1.0", device=device, download_root=root)
        self.k = k

    @torch.no_grad()
    def score(self, video, prompt: str) -> float:
        from PIL import Image
        idx = np.linspace(0, len(video) - 1, self.k).astype(int)
        scores = []
        for i in idx:
            img = Image.fromarray(to_uint8(video[i]))
            scores.append(self.model.score(prompt, img))
        return float(np.mean(scores))


def video_lpips(video_a, video_b, k=8, device="cuda"):
    """Fidelity of cached vs full output, LPIPS mean over K frames (lazy import)."""
    import lpips  # optional dep; install if fidelity metric needed on node
    loss = lpips.LPIPS(net="alex").to(device)
    idx = np.linspace(0, min(len(video_a), len(video_b)) - 1, k).astype(int)
    vals = []
    with torch.no_grad():
        for i in idx:
            # to_uint8 first: raw pipeline frames are float in [0,1] and would
            # collapse to ~-1 under the uint8 normalization (bug caught 2026-07-06)
            a = torch.from_numpy(to_uint8(video_a[i])).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
            b = torch.from_numpy(to_uint8(video_b[i])).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
            vals.append(loss(a, b).item())
    return float(np.mean(vals))


def append_jsonl(path, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
