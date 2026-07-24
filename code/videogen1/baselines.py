"""Published training-free caching baselines for Wan 2.1 (diffusers), A-lite unified protocol.

Three SELECTABLE methods, each wrapping the WanPipeline transformer the same way
videogen1.caching.CachedTransformer does (drop-in `pipe.transformer = Wrapper(...)`,
`.reset()` between generations, `.stats()` for method-internal counters).
This module does NOT touch videogen1/caching.py (our EasyCache-style wrapper).

Provenance (faithfulness > cleverness ; each rule is a minimal port of the official code):

1. TeaCacheTransformer  ; TeaCache (arXiv 2411.19108)
   Ported from the OFFICIAL Wan2.1 integration:
     repo   github.com/ali-vilab/TeaCache
     commit 7c10efc4702c6b619f47805f7abe4a7a08085aa0 (2025-06-08)
     file   TeaCache4Wan2.1/teacache_generate.py  (teacache_forward, L438-582; setup L869-894)
   Rule: per CFG branch (even call = cond, odd = uncond), accumulate a polynomial
   rescale of the rel-L1 change of the timestep embedding (e, or e0 when
   use_ret_steps); if the accumulator stays below `thresh`, skip all transformer
   blocks and reuse the cached token-level residual (x += previous_residual),
   then apply the (timestep-dependent) output head as usual.
   Published Wan2.1 poly coefficients + ret/cutoff steps copied verbatim below.
   Adaptation: their hook patches Wan-repo `WanModel.forward`; here the identical
   logic is inlined into a re-implementation of diffusers 0.39.0
   WanTransformer3DModel.forward (site-packages .../transformers/transformer_wan.py,
   forward() L629-731), where temb == their `e` and timestep_proj == their `e0`.

2. PABTransformer  ; Pyramid Attention Broadcast (arXiv 2408.12588)
   Ported from the OFFICIAL VideoSys implementation (vendored inside the TeaCache
   repo at the same commit; identical to NUS-HPC-AI-Lab/VideoSys):
     file   videosys/core/pab_mgr.py                 (if_broadcast_* rule, L54-91)
     file   videosys/models/transformers/cogvideox_transformer_3d.py (block-level
            attention-output caching, L264-296)
     config videosys/pipelines/cogvideox/pipeline_cogvideox.py::CogVideoXPABConfig
            (full-3D-attention precedent: spatial range 2, threshold (100, 850))
   Rule: reuse the cached attention OUTPUT when count % range != 0 and
   threshold[0] < t < threshold[1]; recompute (and re-cache) otherwise.
   Adaptations for Wan2.1 (no official Wan PAB exists):
   - Wan's unified 3D self-attn plays PAB's "spatial" role (range 2), following
     the official CogVideoX config, the full-attention precedent. Wan additionally
     has cross-attn; PAB broadcasts cross most aggressively (range 6 in the
     official OpenSora config, pipeline_open_sora.py) -> cross range 6 here.
   - diffusers WanPipeline runs CFG as two sequential transformer calls per step
     (cond then uncond), unlike VideoSys's batched CFG, so counters and caches
     are keyed per (block, attn-kind, branch). NOTE: this is also why we do not
     use diffusers' built-in apply_pyramid_attention_broadcast here - its per-call
     counters would broadcast ACROSS the cond/uncond calls for this pipeline.

3. CFGCacheTransformer  ; FasterCache's CFG-Cache component (arXiv 2410.19355)
   Ported from the OFFICIAL implementation:
     repo   github.com/Vchitect/FasterCache
     commit 02d05ef7edb48bdec01d2b4df6edfe7f472a41b9 (2024-12-27)
     file   scripts/cogvideox/fastercache_sample_cogvideox.py
            (fft() L335-350, fastercache_dit_forward L354-424; 50-step config in
            configs/cogvideox/fastercache_sample.yaml -> constants below are
            calibrated at 50 sampling steps, same as our protocol)
   Rule (their counter is 1-indexed, incremented once per denoise step):
   - step >= 18 and step % 5 != 0: compute cond only; reconstruct uncond via
     frequency-domain compensation: uncond ~= ifft2((lf(cond)+delta_lf) +
     (hf(cond)+delta_hf)), where on each skipped step delta_lf *= 1.1 while
     step <= 40 and delta_hf *= 1.1 while step >= 30 (cumulative, as in the
     official code).
   - otherwise compute both branches; when step >= 16 refresh
     delta_{lf,hf} = fft(uncond) - fft(cond).
   Adaptation: their model batches [cond, uncond] in one call; Wan makes two
   sequential calls, so the wrapper intercepts the odd (uncond) call and returns
   the reconstruction without running the transformer. FFT is applied per latent
   frame ((B F) C H W), exactly like their per-frame rearrange. We implement
   ONLY the CFG-Cache component (not their attention-reuse part) - that isolates
   the CFG axis next to PAB (attention axis) and TeaCache/EasyCache (full-model).

All wrappers assume Wan2.1-style inputs (timestep.ndim == 1, no wan2.2 ti2v
expand_timesteps) and diffusers' two-calls-per-step CFG.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# shared plumbing
# --------------------------------------------------------------------------- #
class _WrapperBase(nn.Module):
    """Delegating wrapper skeleton (same contract as caching.CachedTransformer)."""

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.t = transformer
        self.dtype = transformer.dtype
        self.config = transformer.config  # pipeline reads .config

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.t, name)


# --------------------------------------------------------------------------- #
# 1. TeaCache (official Wan2.1 rule)
# --------------------------------------------------------------------------- #
# Verbatim from TeaCache4Wan2.1/teacache_generate.py L881-894 (t2v models):
#   coefficients[model_size][use_ret_steps]
TEACACHE_T2V_COEFFICIENTS = {
    "1.3B": {
        False: [2.39676752e03, -1.31110545e03, 2.01331979e02, -8.29855975e00, 1.37887774e-01],
        True: [-5.21862437e04, 9.23041404e03, -5.28275948e02, 1.36987616e01, -4.99875664e-02],
    },
    "14B": {
        False: [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
        True: [-3.03318725e05, 4.90537029e04, -2.65530556e03, 5.87365115e01, -3.15583525e-01],
    },
}


@dataclass
class TeaCacheConfig:
    # README (TeaCache4Wan2.1) t2v-1.3B operating points: 0.05 (~1.6x) / 0.07 / 0.08 (~2.1x);
    # default 0.08 = their example command for t2v-1.3B, closest to our EasyCache tau=0.10 (1.97x).
    thresh: float = 0.08
    num_steps: int = 50            # sampling steps; internal call counter runs 2*num_steps
    use_ret_steps: bool = False    # README: ret-steps variant is better "except for t2v-1.3B"
    model_size: str = "1.3B"       # {"1.3B", "14B"} -> published coefficient set

    @property
    def coefficients(self):
        return TEACACHE_T2V_COEFFICIENTS[self.model_size][self.use_ret_steps]

    @property
    def ret_steps(self):           # teacache_generate.py L886/L893
        return 5 * 2 if self.use_ret_steps else 1 * 2

    @property
    def cutoff_steps(self):        # teacache_generate.py L887/L894
        return self.num_steps * 2 if self.use_ret_steps else self.num_steps * 2 - 2


@dataclass
class _TeaBranchState:
    prev_mod: torch.Tensor | None = None       # previous_e0_{even,odd}
    residual: torch.Tensor | None = None       # previous_residual_{even,odd}
    accum: float = 0.0                          # accumulated_rel_l1_distance_{even,odd}
    skips: int = 0
    computes: int = 0


class TeaCacheTransformer(_WrapperBase):
    """TeaCache for diffusers WanTransformer3DModel (Wan2.1, T2V/I2V, CFG = 2 calls/step).

    forward() re-implements diffusers 0.39.0 WanTransformer3DModel.forward (inference
    path, Wan2.1 branch) with the official TeaCache skip rule inlined around the
    block loop. The output head (norm_out/proj_out, timestep-modulated) is always
    applied at the CURRENT timestep, exactly like the official hook (head after
    `x += previous_residual`).
    """

    def __init__(self, transformer: nn.Module, cfg: TeaCacheConfig):
        super().__init__(transformer)
        self.cfg = cfg
        self.reset()

    def reset(self):
        self._cnt = 0  # counts transformer calls: even=cond, odd=uncond (official self.cnt)
        self._states = [_TeaBranchState(), _TeaBranchState()]

    def stats(self):
        return {
            "method": "teacache",
            "thresh": self.cfg.thresh,
            "use_ret_steps": self.cfg.use_ret_steps,
            **{f"branch{i}": {"skips": s.skips, "computes": s.computes}
               for i, s in enumerate(self._states)},
        }

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs=None,
    ):
        t = self.t
        c = self.cfg
        assert timestep.ndim == 1, "TeaCacheTransformer supports Wan2.1-style timesteps only"

        # ---- verbatim re-implementation of diffusers WanTransformer3DModel.forward ----
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = t.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = t.rope(hidden_states)
        x = t.patch_embedding(hidden_states)
        x = x.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = t.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # ---- TeaCache decision (teacache_forward L519-550); temb==e, timestep_proj==e0 ----
        modulated_inp = timestep_proj if c.use_ret_steps else temb
        st = self._states[self._cnt % 2]
        if self._cnt < c.ret_steps or self._cnt >= c.cutoff_steps:
            should_calc = True
            st.accum = 0.0
        else:
            rescale_func = np.poly1d(c.coefficients)
            st.accum += rescale_func(
                ((modulated_inp - st.prev_mod).abs().mean() / st.prev_mod.abs().mean()).cpu().item()
            )
            if st.accum < c.thresh:
                should_calc = False
            else:
                should_calc = True
                st.accum = 0.0
        st.prev_mod = modulated_inp.clone()

        # ---- blocks with residual cache (teacache_forward L552-568) ----
        if not should_calc:
            x = x + st.residual
            st.skips += 1
        else:
            ori_x = x
            for block in t.blocks:
                x = block(x, encoder_hidden_states, timestep_proj, rotary_emb)
            st.residual = (x - ori_x).detach()
            st.computes += 1

        # ---- output head (diffusers forward L704-727) ----
        shift, scale = (t.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(x.device)
        scale = scale.to(x.device)
        x = (t.norm_out(x.float()) * (1 + scale) + shift).type_as(x)
        x = t.proj_out(x)
        x = x.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = x.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        self._cnt += 1
        if self._cnt >= c.num_steps * 2:  # official auto-reset (teacache_forward L579-581)
            self._cnt = 0

        if not return_dict:
            return (output,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=output)


# --------------------------------------------------------------------------- #
# 2. PAB-style attention broadcast
# --------------------------------------------------------------------------- #
@dataclass
class PABWanConfig:
    num_steps: int = 50
    self_range: int = 2                       # PAB "spatial" role (CogVideoXPABConfig default)
    self_threshold: tuple = (100, 850)        # broadcast only when th[0] < t < th[1]
    cross_range: int = 6                      # OpenSoraPABConfig cross_range default
    cross_threshold: tuple = (100, 850)


class _PABState:
    """Shared mutable state: per-(block, kind, branch) counters and cached attn outputs."""

    def __init__(self, cfg: PABWanConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.branch = 0
        self.timestep = None
        self.counts: dict = {}
        self.cache: dict = {}
        self.skips = {"self": 0, "cross": 0}
        self.computes = {"self": 0, "cross": 0}


class _PABAttn(nn.Module):
    """Wraps one WanAttention; reuses the cached attention output when broadcasting.

    Decision is a verbatim port of pab_mgr.PABManager.if_broadcast_spatial/cross:
        flag = (count % range != 0) and (threshold[0] < timestep < threshold[1])
        count = (count + 1) % steps
    with count/cache keyed per CFG branch (Wan runs cond/uncond sequentially).
    """

    def __init__(self, attn: nn.Module, state: _PABState, kind: str, block_idx: int):
        super().__init__()
        self.attn = attn
        self._state = [state]  # hide from nn.Module registration
        self.kind = kind
        self.block_idx = block_idx

    def forward(self, *args, **kwargs):
        st = self._state[0]
        cfg = st.cfg
        rng = cfg.self_range if self.kind == "self" else cfg.cross_range
        lo, hi = cfg.self_threshold if self.kind == "self" else cfg.cross_threshold
        key = (self.block_idx, self.kind, st.branch)
        count = st.counts.get(key, 0)
        flag = (count % rng != 0) and (st.timestep is not None) and (lo < st.timestep < hi)
        st.counts[key] = (count + 1) % cfg.num_steps
        if flag and key in st.cache:
            st.skips[self.kind] += 1
            return st.cache[key]
        out = self.attn(*args, **kwargs)
        st.cache[key] = out
        st.computes[self.kind] += 1
        return out


class PABTransformer(_WrapperBase):
    """PAB-style attention broadcast for diffusers Wan2.1 (self-attn range 2, cross range 6)."""

    def __init__(self, transformer: nn.Module, cfg: PABWanConfig):
        super().__init__(transformer)
        self.cfg = cfg
        self._state = _PABState(cfg)
        self._call = 0
        for i, block in enumerate(transformer.blocks):
            if not isinstance(block.attn1, _PABAttn):
                block.attn1 = _PABAttn(block.attn1, self._state, "self", i)
                block.attn2 = _PABAttn(block.attn2, self._state, "cross", i)

    def unwrap(self):
        for block in self.t.blocks:
            if isinstance(block.attn1, _PABAttn):
                block.attn1 = block.attn1.attn
                block.attn2 = block.attn2.attn
        return self.t

    def reset(self):
        self._call = 0
        self._state.reset()

    def stats(self):
        return {
            "method": "pab",
            "self_range": self.cfg.self_range,
            "cross_range": self.cfg.cross_range,
            "skips": dict(self._state.skips),
            "computes": dict(self._state.computes),
        }

    def forward(self, hidden_states, timestep=None, *args, **kwargs):
        self._state.branch = self._call % 2  # WanPipeline: cond call then uncond call
        self._call += 1
        if timestep is not None:
            self._state.timestep = int(timestep.flatten()[0].item())  # int(timestep[0]) upstream
        return self.t(hidden_states, timestep, *args, **kwargs)


# --------------------------------------------------------------------------- #
# 3. FasterCache-style CFG-Cache
# --------------------------------------------------------------------------- #
def _fastercache_fft(tensor: torch.Tensor):
    """Verbatim port of fastercache_sample_cogvideox.py::fft (L335-350), (BF) C H W."""
    tensor_fft = torch.fft.fft2(tensor)
    tensor_fft_shifted = torch.fft.fftshift(tensor_fft)
    B, C, H, W = tensor.size()
    radius = min(H, W) // 5
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    center_x, center_y = W // 2, H // 2
    mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius**2
    low_freq_mask = mask.unsqueeze(0).unsqueeze(0).to(tensor.device)
    high_freq_mask = ~low_freq_mask
    low_freq_fft = tensor_fft_shifted * low_freq_mask
    high_freq_fft = tensor_fft_shifted * high_freq_mask
    return low_freq_fft, high_freq_fft


@dataclass
class CFGCacheConfig:
    """Official constants are calibrated for 50 sampling steps (their CogVideoX config);
    for other step counts they are rescaled proportionally (documented deviation)."""
    num_steps: int = 50
    uncond_interval: int = 5      # counter % 5 != 0 -> skip uncond   (L370)
    skip_start: int = 18          # counter >= 18 -> skipping allowed (L370)
    delta_start: int = 16         # counter >= 16 -> refresh deltas   (L413)
    lf_boost_until: int = 40      # delta_lf *= 1.1 while counter <= 40 (L387-388)
    hf_boost_from: int = 30       # delta_hf *= 1.1 while counter >= 30 (L389-390)

    def scaled(self, name: str) -> int:
        v = getattr(self, name)
        if self.num_steps == 50:
            return v
        return max(1, round(v * self.num_steps / 50))


class CFGCacheTransformer(_WrapperBase):
    """FasterCache CFG-Cache for diffusers Wan2.1 (sequential cond/uncond calls).

    Even calls (cond) always run the transformer; the step counter increments there
    (1-indexed, matching the official `self.counter += 1` at the top of the fused
    forward). Odd calls (uncond) are skipped on the official schedule and replaced
    by the frequency-compensated reconstruction from THIS step's cond output.
    """

    def __init__(self, transformer: nn.Module, cfg: CFGCacheConfig):
        super().__init__(transformer)
        self.cfg = cfg
        self.reset()

    def reset(self):
        self._call = 0
        self._step = 0            # official 1-indexed counter (incremented on cond call)
        self._cond_out = None
        self._delta_lf = None
        self._delta_hf = None
        self._skips = 0
        self._computes_cond = 0
        self._computes_uncond = 0

    def stats(self):
        return {
            "method": "cfgcache",
            "uncond_interval": self.cfg.uncond_interval,
            "skips_uncond": self._skips,
            "computes_cond": self._computes_cond,
            "computes_uncond": self._computes_uncond,
        }

    @staticmethod
    def _frames(x: torch.Tensor) -> torch.Tensor:
        # [B, C, F, H, W] -> [(B F), C, H, W], the official per-frame rearrange
        B, C, F, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)

    @staticmethod
    def _unframes(x: torch.Tensor, shape) -> torch.Tensor:
        B, C, F, H, W = shape
        return x.reshape(B, F, C, H, W).permute(0, 2, 1, 3, 4)

    @torch.no_grad()
    def _reconstruct_uncond(self, cond: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        lf_c, hf_c = _fastercache_fft(self._frames(cond).float())
        # cumulative boosts exactly as official (L387-390), mutating the stored deltas
        if self._step <= c.scaled("lf_boost_until"):
            self._delta_lf = self._delta_lf * 1.1
        if self._step >= c.scaled("hf_boost_from"):
            self._delta_hf = self._delta_hf * 1.1
        combined = (self._delta_lf + lf_c) + (self._delta_hf + hf_c)
        recovered = torch.fft.ifft2(torch.fft.ifftshift(combined)).real
        return self._unframes(recovered, cond.shape).to(cond.dtype)

    def forward(self, hidden_states, *args, **kwargs):
        c = self.cfg
        branch = self._call % 2
        self._call += 1

        if branch == 0:  # cond: always computed
            self._step += 1
            out = self.t(hidden_states, *args, **kwargs)
            self._cond_out = out[0] if isinstance(out, tuple) else out.sample
            self._computes_cond += 1
            return out

        # uncond
        skip = (
            self._step >= c.scaled("skip_start")
            and self._step % c.uncond_interval != 0
            and self._delta_lf is not None
        )
        if skip:
            self._skips += 1
            return (self._reconstruct_uncond(self._cond_out),)

        out = self.t(hidden_states, *args, **kwargs)
        uncond = out[0] if isinstance(out, tuple) else out.sample
        self._computes_uncond += 1
        if self._step >= c.scaled("delta_start"):  # refresh deltas (L413-422)
            lf_c, hf_c = _fastercache_fft(self._frames(self._cond_out).float())
            lf_uc, hf_uc = _fastercache_fft(self._frames(uncond).float())
            self._delta_lf = lf_uc - lf_c
            self._delta_hf = hf_uc - hf_c
        return out


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def apply_baseline(pipe, method: str, num_steps: int, model_size: str = "1.3B", **knobs):
    """Wrap pipe.transformer with the selected baseline. Returns the wrapper
    (call .reset() between generations, .stats() after). knobs override the
    method dataclass defaults (e.g. thresh=, self_range=, uncond_interval=)."""
    if method == "teacache":
        cfg = TeaCacheConfig(num_steps=num_steps, model_size=model_size, **knobs)
        wrapper = TeaCacheTransformer(pipe.transformer, cfg)
    elif method == "pab":
        cfg = PABWanConfig(num_steps=num_steps, **knobs)
        wrapper = PABTransformer(pipe.transformer, cfg)
    elif method == "cfgcache":
        cfg = CFGCacheConfig(num_steps=num_steps, **knobs)
        wrapper = CFGCacheTransformer(pipe.transformer, cfg)
    else:
        raise ValueError(f"unknown baseline method: {method}")
    pipe.transformer = wrapper
    return wrapper
