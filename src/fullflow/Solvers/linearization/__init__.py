"""Public interfaces for FullFlow state-space linearization."""

from .solver import LinearizationError, LinearizationResult

__all__ = ["LinearizationError", "LinearizationResult"]
