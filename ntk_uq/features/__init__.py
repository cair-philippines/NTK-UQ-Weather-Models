"""Per-model last-layer feature extractors for empirical NTK uncertainty.

Each extractor registers a forward hook on the model's last layer, runs an
autoregressive rollout to the requested lead time, and aggregates the captured
activations into a fixed-dimensional feature vector. ``extract_features_dual``
returns both aggregations used in the paper:

    - ``global_avg``: global-average pooling (lower dimension; spatial UQ).
    - ``multi_stat``: six per-channel statistics (mean, std, min, max, q25, q75;
      higher dimension; scalar UQ with stronger discrimination).

Model extractors are imported lazily because each depends on an optional,
model-specific backend (``ai-models-fourcastnetv2``, ``onnxruntime``,
``microsoft-aurora``, ``anemoi-inference``).
"""

from .base import FeatureExtractor


def __getattr__(name):
    if name == "FourCastNetFeatureExtractor":
        from .fourcastnet import FourCastNetFeatureExtractor
        return FourCastNetFeatureExtractor
    if name == "PanguWeatherFeatureExtractor":
        from .pangu import PanguWeatherFeatureExtractor
        return PanguWeatherFeatureExtractor
    if name == "AuroraFeatureExtractor":
        from .aurora import AuroraFeatureExtractor
        return AuroraFeatureExtractor
    if name == "AIFSFeatureExtractor":
        from .aifs import AIFSFeatureExtractor
        return AIFSFeatureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FeatureExtractor",
    "FourCastNetFeatureExtractor",
    "PanguWeatherFeatureExtractor",
    "AuroraFeatureExtractor",
    "AIFSFeatureExtractor",
]
