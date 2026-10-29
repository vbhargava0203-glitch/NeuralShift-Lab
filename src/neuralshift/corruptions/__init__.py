"""Controlled covariate shifts applied to unnormalized images in [0, 1]."""

from .registry import SEVERITIES, apply_corruption

__all__ = ["SEVERITIES", "apply_corruption"]
