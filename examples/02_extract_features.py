"""How to extract last-layer empirical NTK features from each weather model.

Each model needs its own backend and pre-trained weights (see the README install
matrix); this script documents the canonical extraction pattern per model rather
than running them all. The common recipe is:

    1. instantiate the extractor;
    2. load the pre-trained checkpoint;
    3. run an autoregressive rollout and capture last-layer activations via a
       forward hook;
    4. aggregate into dual feature vectors:
         - ``global_avg``: global-average pooling (lower-dim; spatial UQ);
         - ``multi_stat``: six per-channel statistics (higher-dim; scalar UQ).

The dual features then feed ``NTKCalibrator`` (see 01_calibrate_synthetic.py).
"""

import torch


def fourcastnetv2(x: torch.Tensor) -> dict:
    """FourCastNetV2 (SFNO). Backend: ai-models-fourcastnetv2.

    Hook target: final SFNO block output (B, 256, H, W).
    Dual dims: global_avg = 256, multi_stat = 256 x 6 = 1536.
    """
    from ntk_uq.features import FourCastNetFeatureExtractor

    ext = FourCastNetFeatureExtractor(device="cuda")
    ext.load_model("path/to/fourcastnetv2/weights")
    return ext.extract_features_dual(x, lead_time_hours=24)


def pangu(x: torch.Tensor) -> dict:
    """Pangu-Weather (3D Swin Transformer, ONNX). Backend: onnxruntime.

    Hook target: 69-channel prediction tensor (global-average pooled).
    Dual dims: global_avg = 69, multi_stat = 414.
    Requires the normalization stats file (see compute_pangu_normalization).
    """
    from ntk_uq.features import PanguWeatherFeatureExtractor

    ext = PanguWeatherFeatureExtractor(device="cuda")
    ext.load_model("path/to/pangu_weather.onnx")
    return ext.extract_features_dual(x, lead_time_hours=24)


def aurora(x: torch.Tensor) -> dict:
    """Aurora (Swin backbone + Perceiver encoder/decoder). Backend: microsoft-aurora.

    Hook target: Perceiver decoder latent (B, 2, 65), averaged over time steps.
    Dual dims: global_avg = 65, multi_stat = 390.
    """
    from ntk_uq.features import AuroraFeatureExtractor

    ext = AuroraFeatureExtractor(device="cuda")
    ext.load_model("microsoft/aurora")
    return ext.extract_features_dual(x, lead_time_hours=24)


def aifs(date: str = "2021-12-16") -> dict:
    """AIFS (GNN encoder + Transformer processor). Backend: anemoi-inference.

    Hook target: final GNN processor output (global-average pooled).
    Dual dims: global_avg = 1024, multi_stat = 6144.
    AIFS pulls ERA5 by date through the Anemoi runner, so extraction is
    date-based rather than tensor-based.
    """
    from ntk_uq.features import AIFSFeatureExtractor

    ext = AIFSFeatureExtractor(device="cuda")
    ext.load_model("ecmwf/aifs-single-1.1")
    return ext.extract_features_from_date_dual(date, lead_time_hours=24)


if __name__ == "__main__":
    print(__doc__)
    print("Edit the weight paths and call the function for the model you have "
          "installed. Each returns {'global_avg': tensor, 'multi_stat': tensor}.")
