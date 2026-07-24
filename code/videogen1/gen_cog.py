"""CogVideoX-5B generation utilities (E4 port of videogen1.gen ; Wan stays untouched).

Verified 2026-07-06 against the installed diffusers source
(pipelines/cogvideo/pipeline_cogvideox.py, models/transformers/cogvideox_transformer_3d.py):

1. CFG is BATCH-CONCAT, ONE transformer call per step:
     latent_model_input = torch.cat([latents] * 2)   # [uncond, cond] stacked in batch dim
     noise_pred = self.transformer(hidden_states=latent_model_input, ...)[0]
     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
   Unlike WanPipeline (one call per CFG branch), so the cache wrapper MUST use
   CacheConfig(num_branches=1): a single cache state / skip schedule shared by both
   CFG halves ; correct, because the cached delta tensor itself has batch dim 2 and
   carries the uncond and cond deltas separately.

2. Call convention is keyword-only from the pipeline:
     transformer(hidden_states=..., encoder_hidden_states=..., timestep=...,
                 image_rotary_emb=..., attention_kwargs=..., return_dict=False)
   and CogVideoXTransformer3DModel.forward returns a 1-tuple (sample,) under
   return_dict=False (Transformer2DModelOutput otherwise) ; exactly like Wan.
   CachedTransformer.forward(hidden_states, *args, **kwargs) binds hidden_states by
   keyword, its compute path handles both tuple and .sample outputs, and its skip
   path returns (approx,) which the pipeline indexes with [0]. The pipeline's
   `with self.transformer.cache_context("cond_uncond")` and reads of
   .config/.dtype resolve through CachedTransformer.__getattr__ delegation.
   => NO wrapper subclass needed; only the num_branches=1 config differs.

3. VAE dtype: CogVideoX does NOT need the Wan fp32-VAE treatment. The official
   THUDM/diffusers recipe runs the whole 5B pipe (VAE included) in bf16 ; the repo
   ships bf16 weights and there is no documented bf16 quality trap for
   AutoencoderKLCogVideoX (unlike AutoencoderKLWan). Moreover this pipeline's
   decode_latents() does NOT cast latents to the VAE dtype (WanPipeline does), and
   AutoencoderKLCogVideoX.decode has no internal cast either, so a mixed
   fp32-VAE/bf16-latents load would hard-crash on dtype mismatch. => bf16 end-to-end.
   Both gate arms (full/cached) share the same VAE, so rankings are unaffected.
"""
from __future__ import annotations
import time
import torch

from videogen1.caching import CacheConfig, wrap_pipeline

MODEL_ID_COG = "THUDM/CogVideoX-5b"

# CogVideoX-5b defaults (transformer config: sample_height*8=480, sample_width*8=720,
# sample_frames=49; pipeline __call__ defaults: 50 steps, guidance_scale=6).
COG_DEFAULTS = dict(height=480, width=720, frames=49, steps=50, guidance=6.0)


def load_pipe_cog(device="cuda", dtype=torch.bfloat16):
    from diffusers import CogVideoXPipeline
    # Whole pipe (transformer + T5-XXL + VAE) in bf16 ; official CogVideoX-5b recipe;
    # see module docstring point 3 for why the VAE is NOT fp32 here (contra Wan).
    pipe = CogVideoXPipeline.from_pretrained(MODEL_ID_COG, torch_dtype=dtype)
    pipe.to(device)
    # No vae.enable_tiling(): GH200 96GB decodes 49f@480x720 untiled. If a decode
    # OOM ever appears on smaller cards, enable it (deterministic, but changes
    # pixels slightly ; keep it consistent across a study arm).
    return pipe


def cog_cache_config(mode="adaptive", tau=0.10, total_steps=50, **kw) -> CacheConfig:
    """CacheConfig with the CogVideoX-correct num_branches=1 (see docstring point 1).
    Always build configs through this ; CacheConfig's default num_branches=2 is the
    Wan convention and would split CogVideoX's single call stream across two states,
    halving every branch's effective step counter and misfiring warmup/cooldown."""
    kw.setdefault("num_branches", 1)
    assert kw["num_branches"] == 1, "CogVideoX does batch-concat CFG: num_branches must be 1"
    return CacheConfig(mode=mode, tau=tau, total_steps=total_steps, **kw)


def wrap_pipeline_cog(pipe, cfg: CacheConfig | None = None):
    """wrap_pipeline with the num_branches=1 guard applied."""
    cfg = cfg or cog_cache_config(mode="off")
    assert cfg.num_branches == 1, "CogVideoX does batch-concat CFG: num_branches must be 1"
    return wrap_pipeline(pipe, cfg)


@torch.no_grad()
def generate_cog(pipe, prompt: str, seed: int, height=480, width=720, frames=49,
                 steps=50, guidance=6.0):
    """Mirror of gen.generate for CogVideoX. output_type='np' so frames are float
    [0,1] numpy HWC like Wan's (CogVideoX's own default is PIL) ; keeps
    to_uint8 / FrameRewardScorer / save_video working unchanged."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    t0 = time.time()
    out = pipe(prompt=prompt, height=height, width=width, num_frames=frames,
               num_inference_steps=steps, guidance_scale=guidance, generator=g,
               output_type="np")
    latency = time.time() - t0
    video = out.frames[0]
    return video, latency
