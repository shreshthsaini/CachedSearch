"""LTX-Video generation utilities (gate5 port of videogen1.gen ; Wan/Cog files untouched).

Verified 2026-07-07 against the installed diffusers 0.39.0 source
(pipelines/ltx/pipeline_ltx.py, models/transformers/transformer_ltx.py) and the
cached model snapshot (Lightricks/LTX-Video @ 8984fa25):

1. MODEL IDENTITY (important ; corrects the task premise): the diffusers-format
   folders of Lightricks/LTX-Video (what `LTXPipeline.from_pretrained` loads) are
   the BASE 2B checkpoint (28 layers, 2048 dim, _diffusers_version 0.32.0.dev0,
   VAE without timestep_conditioning = v0.9.x base). It is a normal 50-step CFG
   model, NOT the distilled few-step model. The distilled variants
   (ltxv-2b-0.9.8-distilled.safetensors etc., 7-step custom timesteps, no CFG)
   are separate single-file checkpoints in the same repo. Canonical sampling for
   the diffusers checkpoint per the pipeline's own EXAMPLE_DOC_STRING:
   704x480, 161 frames, 50 steps, guidance 3.0, negative prompt
   "worst quality, inconsistent motion, blurry, jittery, distorted".
   => primary gate runs 50 steps (plenty of skippable middle); the few-step
   caching question is probed separately via `load_pipe_ltx(distilled=True)`.

2. CFG is BATCH-CONCAT, ONE transformer call per step (like CogVideoX, unlike Wan):
     latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
     noise_pred = self.transformer(hidden_states=..., ..., return_dict=False)[0]
     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
   => CacheConfig(num_branches=1); the cached delta tensor has batch dim 2 and
   carries the uncond/cond deltas separately.

3. Wrapper compatible AS-IS: pipeline calls the transformer keyword-only
   (hidden_states=, encoder_hidden_states=, timestep=, encoder_attention_mask=,
   num_frames=, height=, width=, rope_interpolation_scale=, attention_kwargs=,
   return_dict=False), forward returns (output,) under return_dict=False; the
   skip path's (x + delta,) is indexed with [0]; cache_context("cond_uncond") /
   .config / .dtype resolve via CachedTransformer.__getattr__. Latents enter the
   transformer already PACKED as [B, seq, 128] ; the wrapper's norm-based
   indicator works on any shape.

4. Dtype: whole pipe bf16 (official example). LTX VAE scaling_factor=1.0,
   pipeline denormalizes with latents_mean/std internally. No tiling needed:
   2B VAE decode of 161f @ 480x704 fits easily in 96GB.

5. Few-step probe (distilled=True): swaps in transformer+VAE loaded
   from_single_file(ltxv-2b-0.9.8-distilled.safetensors) ; cached in the same
   repo snapshot ; and samples with the official recipe from
   Lightricks/LTX-Video configs/ltxv-2b-0.9.8-distilled.yaml first_pass:
   timesteps [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725] (7 steps),
   guidance_scale=1 (NO CFG), decode_timestep 0.05, decode_noise_scale 0.025.
   With 7 steps the warmup/cooldown must scale: use scaled_warmup_cooldown().
"""
from __future__ import annotations
import time
import torch

from videogen1.caching import CacheConfig, wrap_pipeline

MODEL_ID_LTX = "Lightricks/LTX-Video"

# Canonical sampling from the pipeline's EXAMPLE_DOC_STRING for this exact repo.
LTX_DEFAULTS = dict(height=480, width=704, frames=161, steps=50, guidance=3.0)
LTX_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

# Official distilled 2B recipe (single-scale first_pass of
# configs/ltxv-2b-0.9.8-distilled.yaml, fetched 2026-07-07): sigma*1000.
LTX_DISTILLED_FILE = "ltxv-2b-0.9.8-distilled.safetensors"
LTX_DISTILLED_TIMESTEPS = [1000.0, 993.7, 987.5, 981.2, 975.0, 909.4, 725.0]
LTX_DISTILLED_GUIDANCE = 1.0  # no CFG ; single-batch transformer calls


def scaled_warmup_cooldown(total_steps: int) -> tuple[int, int]:
    """Protocol warmup/cooldown: 5/5 for standard (>=25-step) schedules ; matches
    the Wan/CogVideoX gate protocol ; else ~25% of the schedule on each end
    (e.g. 7 steps -> 2/2), so few-step schedules keep a nonzero middle."""
    if total_steps >= 25:
        return 5, 5
    k = max(1, round(total_steps * 0.25))
    return k, k


def ltx_cache_config(mode="adaptive", tau=0.10, total_steps=50, **kw) -> CacheConfig:
    """CacheConfig with the LTX-correct num_branches=1 (batch-concat CFG, see
    docstring point 2) and schedule-scaled warmup/cooldown. Always build LTX
    configs through this ; CacheConfig's default num_branches=2 is the Wan
    convention and would silently split LTX's single call stream across two
    states, corrupting warmup/cooldown/skip logic."""
    kw.setdefault("num_branches", 1)
    assert kw["num_branches"] == 1, "LTX does batch-concat CFG: num_branches must be 1"
    wu, cd = scaled_warmup_cooldown(total_steps)
    kw.setdefault("warmup_steps", wu)
    kw.setdefault("cooldown_steps", cd)
    return CacheConfig(mode=mode, tau=tau, total_steps=total_steps, **kw)


def load_pipe_ltx(device="cuda", dtype=torch.bfloat16, distilled=False):
    from diffusers import LTXPipeline
    pipe = LTXPipeline.from_pretrained(MODEL_ID_LTX, torch_dtype=dtype)
    if distilled:
        # Few-step probe: distilled 2B transformer + its matching 0.9.8 VAE from
        # the single-file checkpoint cached in the same repo. Best-effort ; if
        # diffusers' single-file config inference rejects the 0.9.8 2B layout,
        # record the failure as a boundary finding (do not silently fall back).
        from diffusers import LTXVideoTransformer3DModel, AutoencoderKLLTXVideo
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(MODEL_ID_LTX, LTX_DISTILLED_FILE)
        pipe.transformer = LTXVideoTransformer3DModel.from_single_file(ckpt, torch_dtype=dtype)
        pipe.vae = AutoencoderKLLTXVideo.from_single_file(ckpt, torch_dtype=dtype)
    pipe.to(device)
    return pipe


def wrap_pipeline_ltx(pipe, cfg: CacheConfig | None = None):
    """wrap_pipeline with the num_branches=1 guard applied."""
    cfg = cfg or ltx_cache_config(mode="off")
    assert cfg.num_branches == 1, "LTX does batch-concat CFG: num_branches must be 1"
    return wrap_pipeline(pipe, cfg)


@torch.no_grad()
def generate_ltx(pipe, prompt: str, seed: int, height=480, width=704, frames=161,
                 steps=50, guidance=3.0, negative_prompt=LTX_NEGATIVE_PROMPT,
                 timesteps=None, decode_timestep=0.0, decode_noise_scale=None):
    """Mirror of gen.generate for LTX-Video. output_type='np' so frames are float
    [0,1] numpy HWC like Wan's ; keeps to_uint8/FrameRewardScorer/save_video
    working unchanged. `timesteps` (list) overrides the uniform schedule for the
    distilled few-step probe; steps must equal len(timesteps) then so the cache
    wrapper's total_steps bookkeeping stays truthful."""
    if timesteps is not None:
        assert steps == len(timesteps), "steps must match len(timesteps) for cache bookkeeping"
    g = torch.Generator(device="cuda").manual_seed(seed)
    kw = {}
    if decode_noise_scale is not None:
        kw["decode_noise_scale"] = decode_noise_scale
    t0 = time.time()
    out = pipe(prompt=prompt, negative_prompt=negative_prompt, height=height,
               width=width, num_frames=frames, num_inference_steps=steps,
               timesteps=timesteps, guidance_scale=guidance, generator=g,
               decode_timestep=decode_timestep, output_type="np", **kw)
    latency = time.time() - t0
    video = out.frames[0]
    return video, latency
