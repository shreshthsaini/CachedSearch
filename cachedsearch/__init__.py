"""CachedSearch: explore cheap, commit full.

Drop-in test-time search for any diffusers video pipeline:

    from cachedsearch import cached_search

    video = cached_search(pipe, "a red fox in snow", verifier=my_scorer, n=8)

The exploration rollouts run under training-free caching; only the winning
seed is regenerated at full compute, so the returned video is a genuine
full-compute sample.
"""

from .api import CacheConfig, cached_search, cached_search_batch, calibrate_tau

__all__ = ["cached_search", "cached_search_batch", "calibrate_tau", "CacheConfig"]
