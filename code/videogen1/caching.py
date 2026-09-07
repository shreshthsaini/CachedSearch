"""Compatibility shim: the cache wrapper now lives in the installable package."""
from cachedsearch.caching import CacheConfig, CachedTransformer, wrap_pipeline  # noqa: F401
