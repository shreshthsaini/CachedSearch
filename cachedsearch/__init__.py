"""CachedSearch: explore cheap, commit full.

Drop-in test-time search for any diffusers video pipeline:

    from cachedsearch import cached_search

    result = cached_search(pipe, "a red fox in snow", n=8)

The verifier defaults to ImageReward averaged over 8 frames, the scorer behind
the paper's numbers. Pass ``verifier=`` to use your own.

The exploration rollouts run under training-free caching; only the winning
seed is regenerated at full compute, so the returned video is a genuine
full-compute sample.
"""

from .api import CacheConfig, cached_search, cached_search_batch, calibrate_tau
from .verifiers import default_verifier, imagereward_verifier

__all__ = [
    "cached_search", "cached_search_batch", "calibrate_tau", "CacheConfig",
    "default_verifier", "imagereward_verifier",
]
