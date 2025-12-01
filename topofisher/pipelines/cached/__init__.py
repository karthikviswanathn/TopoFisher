"""Cached pipelines for saving and loading persistence diagrams."""

from .base import CachedPipeline
from .compression import CachedCompressionPipeline

__all__ = ['CachedPipeline', 'CachedCompressionPipeline']