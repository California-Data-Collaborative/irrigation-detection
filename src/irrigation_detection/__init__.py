"""Irrigation detection from hourly smart water meter data."""

from .models import DilatedUNet1D, load_model, WINDOW_SIZE
from .detection import detect_irrigation

__all__ = ["DilatedUNet1D", "load_model", "detect_irrigation", "WINDOW_SIZE"]
