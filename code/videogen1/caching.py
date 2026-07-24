"""Training-free caching wrappers for video DiTs (v0: Wan 2.1 via diffusers).

Two modes:
  - "adaptive": EasyCache-style online rule. Cache the transformation vector
    delta = v(x_t) - x_t; while an accumulated change indicator stays below
    tau, skip the transformer forward and return x_t + delta.
  - "static": fixed skip pattern over a step range (PAB/uniform-style baseline).

CFG handling: diffusers WanPipeline calls the transformer once per branch per
step. We key cache state on call parity (branch index = call_count % branches)
so cond/uncond each get independent state.
"""
from __future__ import annotations
import torch
from dataclasses import dataclass, field


@dataclass
class _BranchState:
    prev_input: torch.Tensor | None = None
    prev_output: torch.Tensor | None = None
    delta: torch.Tensor | None = None
    accum: float = 0.0
    skips: int = 0
    computes: int = 0
    step: int = 0


@dataclass
class CacheConfig:
    mode: str = "adaptive"          # "adaptive" | "static" | "off"
    tau: float = 0.10               # adaptive: accumulated-change tolerance
    warmup_steps: int = 5           # always compute the first K steps
    cooldown_steps: int = 5         # always compute the last K steps (needs total_steps)
    total_steps: int = 50
    static_keep_every: int = 2      # static: compute every k-th step in the middle
    num_branches: int = 2           # CFG cond+uncond


class CachedTransformer(torch.nn.Module):
    """Wraps a diffusers transformer; drop-in via `pipe.transformer = CachedTransformer(pipe.transformer, cfg)`."""

    def __init__(self, transformer: torch.nn.Module, cfg: CacheConfig):
        super().__init__()
        self.t = transformer
        self.cfg = cfg
        self.dtype = transformer.dtype
        self.config = transformer.config  # pipeline reads .config
        self._call = 0
        self._states = [_BranchState() for _ in range(cfg.num_branches)]

    # pipeline may call other attrs; delegate
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.t, name)

    def reset(self):
        self._call = 0
        self._states = [_BranchState() for _ in range(self.cfg.num_branches)]

    def stats(self):
        return {
            f"branch{i}": {"skips": s.skips, "computes": s.computes}
            for i, s in enumerate(self._states)
        }

    def _should_compute(self, st: _BranchState, x: torch.Tensor) -> bool:
        c = self.cfg
        if c.mode == "off" or st.delta is None:
            return True
        if st.step < c.warmup_steps or st.step >= c.total_steps - c.cooldown_steps:
            return True
        if c.mode == "static":
            return (st.step % c.static_keep_every) == 0
        # adaptive: estimate output change rate from input change (EasyCache-style)
        if st.prev_input is None:
            return True
        dx = (x - st.prev_input).float().norm() / max(st.prev_input.float().norm().item(), 1e-8)
        st.accum += dx.item()
        return st.accum > c.tau

    def forward(self, hidden_states, *args, **kwargs):
        st = self._states[self._call % self.cfg.num_branches]
        self._call += 1
        x = hidden_states

        if self._should_compute(st, x):
            out = self.t(hidden_states, *args, **kwargs)
            sample = out[0] if isinstance(out, tuple) else out.sample
            st.delta = (sample - x).detach()
            st.prev_input = x.detach()
            st.prev_output = sample.detach()
            st.accum = 0.0
            st.computes += 1
            st.step += 1
            return out
        # skip: approximate v(x_t) ~= x_t + cached delta
        st.skips += 1
        st.step += 1
        approx = x + st.delta
        return (approx,)


def wrap_pipeline(pipe, cfg: CacheConfig):
    """Attach caching to a diffusers video pipeline. Call .reset() between generations."""
    if isinstance(pipe.transformer, CachedTransformer):
        pipe.transformer.cfg = cfg
        pipe.transformer.reset()
    else:
        pipe.transformer = CachedTransformer(pipe.transformer, cfg)
    return pipe.transformer
