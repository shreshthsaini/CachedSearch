"""Verifiers: the scorer that ranks candidates.

CachedSearch is verifier-agnostic. Any callable ``(frames, prompt) -> float``
works, higher meaning better. If you do not pass one, we use ImageReward
averaged over uniformly spaced frames, which is the verifier all of the
paper's headline numbers use.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

Verifier = Callable[[object, str], float]

_DEFAULT = None


def imagereward_verifier(device: str = "cuda", num_frames: int = 8) -> Verifier:
    """ImageReward averaged over `num_frames` uniformly spaced frames.

    Requires the `image-reward` package; the model downloads on first use.
    """
    import ImageReward as RM
    from PIL import Image

    model = RM.load("ImageReward-v1.0", device=device)

    def score(frames, prompt: str) -> float:
        idx = np.linspace(0, len(frames) - 1, min(num_frames, len(frames))).astype(int)
        vals = [
            model.score(prompt, Image.fromarray(np.asarray(frames[i]).astype("uint8")))
            for i in idx
        ]
        return float(np.mean(vals))

    return score


def default_verifier(device: str = "cuda") -> Verifier:
    """The default scorer, loaded once and reused across calls."""
    global _DEFAULT
    if _DEFAULT is None:
        try:
            _DEFAULT = imagereward_verifier(device=device)
        except ImportError as exc:  # keep the failure actionable
            raise ImportError(
                "No verifier was passed and the default (ImageReward) is not "
                "installed. Either `pip install image-reward`, or pass your own "
                "scorer: cached_search(pipe, prompt, verifier=my_scorer)."
            ) from exc
    return _DEFAULT
