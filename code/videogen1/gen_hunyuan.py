"""HunyuanVideo-13B generation utilities (gate5 port of videogen1.gen ; Wan/Cog untouched).

Verified 2026-07-07 against the installed diffusers 0.39.0 source
(pipelines/hunyuan_video/pipeline_hunyuan_video.py,
models/transformers/transformer_hunyuan_video.py) and the cached snapshot
(hunyuanvideo-community/HunyuanVideo @ e8c2aaa):

1. CFG STRUCTURE: HunyuanVideo is GUIDANCE-DISTILLED (transformer config
   guidance_embeds=true). The pipeline makes ONE transformer call per step with
   an embedded guidance tensor (guidance_scale*1000); classic 2-branch CFG only
   happens if true_cfg_scale > 1 AND a negative prompt is given (defaults:
   true_cfg_scale=1.0, no negative prompt => NO uncond pass at all).
   => CacheConfig(num_branches=1), and generate_hunyuan asserts true_cfg==1.0
   (a true-CFG run would be 2 calls/step and need num_branches=2).

2. Wrapper compatible AS-IS: keyword-only call (hidden_states=, timestep=,
   encoder_hidden_states=, encoder_attention_mask=, pooled_projections=,
   guidance=, attention_kwargs=, return_dict=False), returns (sample,) under
   return_dict=False; skip path (x + delta,) indexes fine; cache_context("cond")
   / .config / .dtype resolve via CachedTransformer.__getattr__ delegation.

3. DTYPE: whole pipe bf16. The community README mixes fp16 pipe + bf16
   transformer; we keep everything bf16 for uniformity (no fp16 overflow risk;
   decode_latents casts latents to the VAE dtype ; line 776 ; so any uniform
   dtype is safe). Both gate arms share the VAE, so rankings are unaffected.

4. MEMORY: 13B transformer (~25 GB) + Llama-8B text encoder (~16 GB) + CLIP-L +
   VAE ~= 43 GB weights on a 96 GB GH200 ; fits without offload.
   vae.enable_tiling() IS enabled (community-recommended; the 3D VAE decode of
   61f @ 480x720 is the peak-memory step on this pipeline). Tiling is
   deterministic and identical across both gate arms.

5. CONFIG USED (gate5): 480x720, 61 frames ((61-1)%4==0 ok), 50 steps
   (pipeline/model-card default), embedded guidance 6.0, flow-shift 7.0 from the
   repo scheduler config. Pipeline defaults of 720x1280x129f would be ~87k
   tokens on a 13B model ; far too slow for an 800-generation gate.
"""
from __future__ import annotations
import time
import torch

from videogen1.caching import CacheConfig, wrap_pipeline

MODEL_ID_HUNYUAN = "hunyuanvideo-community/HunyuanVideo"

# 61f/480x720/50 steps: fits 96GB with VAE tiling; guidance is the EMBEDDED
# distilled guidance (not true CFG).
HUNYUAN_DEFAULTS = dict(height=480, width=720, frames=61, steps=50, guidance=6.0)


def hunyuan_cache_config(mode="adaptive", tau=0.10, total_steps=50, **kw) -> CacheConfig:
    """CacheConfig with the HunyuanVideo-correct num_branches=1 (single
    guidance-embedded call per step, see docstring point 1). Always build
    configs through this ; CacheConfig's default num_branches=2 is the Wan
    convention and would silently split the single call stream across two
    states, corrupting warmup/cooldown/skip logic."""
    kw.setdefault("num_branches", 1)
    assert kw["num_branches"] == 1, \
        "HunyuanVideo is guidance-distilled (1 call/step): num_branches must be 1"
    return CacheConfig(mode=mode, tau=tau, total_steps=total_steps, **kw)


def load_pipe_hunyuan(device="cuda", dtype=torch.bfloat16):
    from diffusers import HunyuanVideoPipeline
    pipe = HunyuanVideoPipeline.from_pretrained(MODEL_ID_HUNYUAN, torch_dtype=dtype)
    # Peak memory on this pipeline is the 3D-VAE decode; tiling is deterministic
    # and shared by both gate arms (community-recommended for this model).
    pipe.vae.enable_tiling()
    pipe.to(device)
    return pipe


def wrap_pipeline_hunyuan(pipe, cfg: CacheConfig | None = None):
    """wrap_pipeline with the num_branches=1 guard applied."""
    cfg = cfg or hunyuan_cache_config(mode="off")
    assert cfg.num_branches == 1, \
        "HunyuanVideo is guidance-distilled (1 call/step): num_branches must be 1"
    return wrap_pipeline(pipe, cfg)


@torch.no_grad()
def generate_hunyuan(pipe, prompt: str, seed: int, height=480, width=720, frames=61,
                     steps=50, guidance=6.0, true_cfg=1.0):
    """Mirror of gen.generate for HunyuanVideo. output_type='np' so frames are
    float [0,1] numpy HWC like Wan's ; keeps to_uint8/FrameRewardScorer/
    save_video working unchanged. guidance = EMBEDDED distilled guidance."""
    assert true_cfg == 1.0, \
        "true_cfg>1 adds an uncond pass (2 calls/step) ; wrapper is configured num_branches=1"
    g = torch.Generator(device="cuda").manual_seed(seed)
    t0 = time.time()
    out = pipe(prompt=prompt, height=height, width=width, num_frames=frames,
               num_inference_steps=steps, guidance_scale=guidance,
               true_cfg_scale=true_cfg, generator=g, output_type="np")
    latency = time.time() - t0
    video = out.frames[0]
    return video, latency
