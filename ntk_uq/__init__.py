"""NTK-UQ-Weather-Models.

Last-layer empirical Neural Tangent Kernel (NTK) uncertainty quantification for
pre-trained AI weather forecasting models. The package provides per-model
last-layer feature extractors (FourCastNetV2, Pangu-Weather, Aurora, AIFS) and a
calibrator that builds the empirical NTK kernel, decomposes it via SVD or ICA,
and returns Gaussian-process posterior uncertainty.

Reference:
    J. M. A. Minoza, R. G. Laylo, and S. C. Ibanez. "Scalable Uncertainty
    Quantification for Extreme Weather Forecasting via Empirical Neural Tangent
    Kernels." KDD 2026.
"""

from .calibration import NTKCalibrator
from .features import FeatureExtractor

__version__ = "0.1.0"
__all__ = ["NTKCalibrator", "FeatureExtractor"]
