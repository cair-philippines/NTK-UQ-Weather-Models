"""Empirical NTK calibration: kernel construction, decomposition, GP posterior.

``NTKCalibrator`` centers the calibration features, builds the last-layer
empirical NTK kernel, decomposes it with SVD or ICA, estimates the observation
noise from the eigenvalue tail, and learns a per-variable post-hoc scale so that
the Gaussian-process posterior intervals reach the target coverage.
"""

from .ntk import NTKCalibrator

__all__ = ["NTKCalibrator"]
