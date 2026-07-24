"""Resumable Wan denoising loop: pause at step K, preview the x0 estimate, resume survivors.

Replicates the diffusers WanPipeline.__call__ denoising loop (v6.x, UniPCMultistepScheduler
with flow_prediction) using pipe components, adding:
  1. pause/resume of individual candidates mid-trajectory via CandidateState snapshots
     (latents + UniPC multistep history + CachedTransformer branch states),
  2. a cheap x0 preview: UniPC with predict_x0=True stores the CONVERTED model output
     x0_pred = x_t - sigma_t * v_pred in scheduler.model_outputs[-1] at every step, so
     the flow-matching x0 estimate at the pause step is available for free ; we VAE-decode
     only a few evenly spaced latent frames of it (per-frame T=1 decodes; joint-clip fallback).

Does NOT modify videogen1.gen / videogen1.caching (E6 constraint). Numerics of a
full-length run through this loop match pipe(...) (same prepare_latents/generator,
same per-step ops), modulo kernel nondeterminism.
"""
from __future__ import annotations
import dataclasses
import time

import numpy as np
import torch

from videogen1.caching import CachedTransformer


@dataclasses.dataclass
class CandidateState:
    seed: int
    latents: torch.Tensor                    # fp32 denoising-space latents
    step: int = 0                            # next step index to run
    sched: dict | None = None                # UniPC internal-state snapshot
    cache: tuple | None = None               # (call_count, [branch-state copies])
    x0_pred: torch.Tensor | None = None      # x0 estimate at the pause step (fp32)
    gen_time: float = 0.0                    # accumulated denoise wall-clock (s)


class ResumableWanRunner:
    def __init__(self, pipe, height=480, width=832, frames=81, steps=50, guidance=5.0):
        self.pipe = pipe
        self.height, self.width, self.frames = height, width, frames
        self.steps, self.guidance = steps, guidance
        self.device = pipe._execution_device
        # transformer may be wrapped by CachedTransformer; .dtype/.config delegate
        self.transformer_dtype = pipe.transformer.dtype
        self.num_channels_latents = pipe.transformer.config.in_channels
        self.do_cfg = guidance > 1.0

    # ---------------- prompt encoding (once per prompt, shared across candidates) ----

    @torch.no_grad()
    def encode_prompt(self, prompt: str):
        pipe = self.pipe
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt, negative_prompt=None,
            do_classifier_free_guidance=self.do_cfg,
            num_videos_per_prompt=1, max_sequence_length=512, device=self.device)
        prompt_embeds = prompt_embeds.to(self.transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(self.transformer_dtype)
        return prompt_embeds, negative_prompt_embeds

    # ---------------- candidate lifecycle ---------------------------------------------

    @torch.no_grad()
    def new_state(self, seed: int) -> CandidateState:
        # identical to gen.generate(): fresh cuda generator seeded per candidate
        g = torch.Generator(device="cuda").manual_seed(seed)
        latents = self.pipe.prepare_latents(
            1, self.num_channels_latents, self.height, self.width, self.frames,
            torch.float32, self.device, g, None)
        return CandidateState(seed=seed, latents=latents)

    # ---- UniPC snapshot/restore. Tensors in these lists are replaced (never mutated
    # in place) by scheduler.step, so shallow list copies + references are safe.
    def _snapshot_sched(self) -> dict:
        s = self.pipe.scheduler
        return dict(model_outputs=list(s.model_outputs),
                    timestep_list=list(s.timestep_list),
                    lower_order_nums=s.lower_order_nums,
                    last_sample=s.last_sample,
                    this_order=getattr(s, "this_order", None),
                    step_index=s._step_index)

    def _restore_sched(self, snap: dict):
        s = self.pipe.scheduler
        s.model_outputs = list(snap["model_outputs"])
        s.timestep_list = list(snap["timestep_list"])
        s.lower_order_nums = snap["lower_order_nums"]
        s.last_sample = snap["last_sample"]
        if snap["this_order"] is not None:
            s.this_order = snap["this_order"]
        s._step_index = snap["step_index"]

    # ---- CachedTransformer branch-state snapshot/restore (fields are replaced, not
    # mutated in place -> dataclasses.replace copies are sufficient).
    def _cache_wrapper(self) -> CachedTransformer | None:
        t = self.pipe.transformer
        return t if isinstance(t, CachedTransformer) else None

    def _snapshot_cache(self):
        c = self._cache_wrapper()
        if c is None:
            return None
        return (c._call, [dataclasses.replace(st) for st in c._states])

    def _restore_cache(self, snap):
        c = self._cache_wrapper()
        if c is None or snap is None:
            return
        call, states = snap
        c._call = call
        c._states = [dataclasses.replace(st) for st in states]

    # ---------------- the manual denoising loop ---------------------------------------

    @torch.no_grad()
    def run(self, state: CandidateState, embeds, until_step: int) -> CandidateState:
        """Advance `state` from state.step to `until_step` (exclusive upper bound =
        number of completed steps). Installs/restores per-candidate scheduler + cache
        state, so candidates can be interleaved arbitrarily."""
        assert until_step <= self.steps
        pipe, sched = self.pipe, self.pipe.scheduler
        prompt_embeds, negative_prompt_embeds = embeds

        # deterministic sigma/timestep table for this step count
        sched.set_timesteps(self.steps, device=self.device)
        timesteps = sched.timesteps
        cache = self._cache_wrapper()
        if state.step == 0:
            sched.set_begin_index(0)
            if cache is not None:
                cache.reset()
        else:
            self._restore_sched(state.sched)
            self._restore_cache(state.cache)

        latents = state.latents
        t0 = time.time()
        for i in range(state.step, until_step):
            t = timesteps[i]
            latent_model_input = latents.to(self.transformer_dtype)
            timestep = t.expand(latents.shape[0])

            with pipe.transformer.cache_context("cond"):
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input, timestep=timestep,
                    encoder_hidden_states=prompt_embeds, attention_kwargs=None,
                    return_dict=False)[0]
            if self.do_cfg:
                with pipe.transformer.cache_context("uncond"):
                    noise_uncond = pipe.transformer(
                        hidden_states=latent_model_input, timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds, attention_kwargs=None,
                        return_dict=False)[0]
                noise_pred = noise_uncond + self.guidance * (noise_pred - noise_uncond)

            latents = sched.step(noise_pred, t, latents, return_dict=False)[0]
        state.gen_time += time.time() - t0

        state.latents = latents
        state.step = until_step
        # free x0 estimate: UniPC (predict_x0=True, flow_prediction) stores the
        # converted output x0_pred = x_t - sigma_t * v_pred as model_outputs[-1]
        if sched.model_outputs[-1] is not None:
            state.x0_pred = sched.model_outputs[-1]
        if until_step < self.steps:  # only snapshot if we will resume later
            state.sched = self._snapshot_sched()
            state.cache = self._snapshot_cache()
        else:
            state.sched = state.cache = None
        return state

    # ---------------- decoding --------------------------------------------------------

    def _normalize_for_vae(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self.pipe.vae
        latents = latents.to(vae.dtype)
        latents_mean = (torch.tensor(vae.config.latents_mean)
                        .view(1, vae.config.z_dim, 1, 1, 1)
                        .to(latents.device, latents.dtype))
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
            1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        return latents / latents_std + latents_mean

    @torch.no_grad()
    def decode_full(self, latents: torch.Tensor):
        """Full-length decode, identical to the pipeline tail. Returns [T,H,W,C] float [0,1]."""
        z = self._normalize_for_vae(latents)
        video = self.pipe.vae.decode(z, return_dict=False)[0]
        return self.pipe.video_processor.postprocess_video(video, output_type="np")[0]

    @torch.no_grad()
    def decode_preview(self, latents: torch.Tensor, n_frames: int = 4):
        """Cheap preview: VAE-decode only n_frames evenly spaced latent frames of an
        x0 estimate. Primary path: independent T=1 decodes (each latent frame treated
        as a clip start ; exact for frame 0, approximate elsewhere; fine for reward
        preview). Fallback: joint decode of the n selected frames as one short clip."""
        z = self._normalize_for_vae(latents)
        T = z.shape[2]
        idx = np.linspace(0, T - 1, min(n_frames, T)).astype(int)
        frames = []
        try:
            for j in idx:
                v = self.pipe.vae.decode(z[:, :, j:j + 1], return_dict=False)[0]
                fr = self.pipe.video_processor.postprocess_video(v, output_type="np")[0]
                frames.append(fr[0])
        except Exception:
            v = self.pipe.vae.decode(z[:, :, list(idx)], return_dict=False)[0]
            fr = self.pipe.video_processor.postprocess_video(v, output_type="np")[0]
            sel = np.linspace(0, len(fr) - 1, min(n_frames, len(fr))).astype(int)
            frames = [fr[k] for k in sel]
        return frames
