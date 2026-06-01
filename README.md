# NTK-UQ-Weather-Models

Last-layer **empirical Neural Tangent Kernel (NTK)** uncertainty quantification
for pre-trained AI weather forecasting models. This is the reference
implementation for:

> J. M. A. Miñoza, R. G. Laylo, and S. C. Ibañez. **"Scalable Uncertainty
> Quantification for Extreme Weather Forecasting via Empirical Neural Tangent
> Kernels."** *Proceedings of the 32nd ACM SIGKDD Conference on Knowledge
> Discovery and Data Mining (KDD '26)*, 2026.

The method is **post-hoc**: it adds calibrated, spatially-adaptive uncertainty to
any frozen deterministic weather model, with no retraining and a single
matrix-vector product at inference. It treats a model's last-layer features as an
empirical NTK, builds the feature kernel, decomposes it (SVD or ICA), and returns
the Gaussian-process posterior variance.

## What this repo gives you

1. **Per-model last-layer feature extractors** for four production weather models
   (the centerpiece — how to pull empirical-NTK features from each architecture).
2. **`NTKCalibrator`** — kernel construction, SVD/ICA decomposition, GP posterior,
   and per-variable post-hoc scaling.
3. The **EM-DAT initialization-date list** (n = 100) used in the paper, plus the
   extraction utility.

## Install

```bash
git clone https://github.com/JomaMinoza/NTK-UQ-Weather-Models.git
cd NTK-UQ-Weather-Models
pip install -e .
```

The core install (kernel + calibration + GP posterior) needs only
`torch`, `numpy`, `scipy`, `scikit-learn`. Each weather model needs its **own
backend** — install only the ones you use:

| Model | Architecture | Backend (extra) | Features `global_avg` / `multi_stat` |
|-------|--------------|-----------------|--------------------------------------|
| FourCastNetV2 | SFNO | `pip install -e ".[fourcastnet]"` | 256 / 1536 |
| Pangu-Weather | 3D Swin Transformer (ONNX) | `pip install -e ".[pangu]"` | 69 / 414 |
| Aurora | Swin backbone + Perceiver I/O | `pip install -e ".[aurora]"` | 65 / 390 |
| AIFS | GNN encoder + Transformer | `pip install -e ".[aifs]"` | 1024 / 6144 |

EM-DAT date extraction needs `pandas`/`openpyxl` (`pip install -e ".[data]"`).
Install everything with `pip install -e ".[all]"`.

## Quickstart (no model weights needed)

The empirical-NTK calibration runs on plain feature arrays. This demonstrates the
core property — inputs dissimilar to the calibration set get higher uncertainty:

```bash
python examples/01_calibrate_synthetic.py
# Mean uncertainty, in-distribution test inputs : 0.03
# Mean uncertainty, out-of-distribution inputs  : 1.55
# Ratio (out / in)                              : 47x
```

```python
import torch
from ntk_uq import NTKCalibrator

features = torch.randn(100, 256)              # last-layer calibration features
cal = NTKCalibrator(model_name="demo", rank_k=10, device="cpu")
cal.calibrate_lead_time(features, lead_time_hours=24)

sigma = cal.compute_uncertainty(torch.randn(8, 256), lead_time_hours=24)
print(sigma["uncertainty"])                   # GP posterior std per sample
```

## Extracting empirical-NTK features from a weather model

Every extractor follows the same recipe: load the checkpoint, register a forward
hook on the last layer, roll out to the lead time, and aggregate into the dual
feature vectors (`global_avg` for spatial UQ, `multi_stat` for scalar UQ). See
`examples/02_extract_features.py` for all four. For example, FourCastNetV2:

```python
from ntk_uq.features import FourCastNetFeatureExtractor

ext = FourCastNetFeatureExtractor(device="cuda")
ext.load_model("path/to/fourcastnetv2/weights")
feats = ext.extract_features_dual(x, lead_time_hours=24)
# feats == {"global_avg": tensor(256,), "multi_stat": tensor(1536,)}
```

AIFS pulls ERA5 by date through the Anemoi runner, so it is date-based:

```python
from ntk_uq.features import AIFSFeatureExtractor

ext = AIFSFeatureExtractor(device="cuda")
ext.load_model("ecmwf/aifs-single-1.1")
feats = ext.extract_features_from_date_dual("2021-12-16", lead_time_hours=24)
```

## EM-DAT initialization dates

```python
from ntk_uq.data import load_calibration_dates, load_precursor_pool

dates = load_calibration_dates()   # 100 init dates (paper n), bundled with the package
pool = load_precursor_pool()       # 208-date EM-DAT precursor pool (3-day lookback)
```

The 100 calibration dates are a 3-day lookback before each EM-DAT event onset,
deduplicated and selected from the 208-date precursor pool. EM-DAT itself is not
redistributed; download it from https://www.emdat.be/ and rebuild the pool with
`ntk_uq.data.extract_precursor_dates(path_to_xlsx)`.

## Paper ↔ code mapping

| Paper | Code |
|-------|------|
| Last-layer feature kernel `K(x,x') = φ(x)ᵀφ(x')` | `NTKCalibrator.calibrate_lead_time` |
| SVD / ICA decomposition (Section 3.4) | `NTKCalibrator._calibrate_svd` / `_calibrate_ica` |
| GP posterior variance (Eq. 4) | `NTKCalibrator.compute_uncertainty` |
| Dual features: `global_avg`, `multi_stat` | `*.extract_features_dual` |
| Post-hoc scaling `α` (Section 3.5) | `NTKCalibrator.calibrate_all_lead_times` |
| FourCastNetV2 (SFNO) | `FourCastNetFeatureExtractor` |
| Pangu-Weather | `PanguWeatherFeatureExtractor` |
| Aurora | `AuroraFeatureExtractor` |
| AIFS | `AIFSFeatureExtractor` |
| EM-DAT events, n = 100 | `ntk_uq.data.load_calibration_dates()` |

## Tests

```bash
pip install pytest
pytest tests/
```

The tests exercise the kernel/decomposition/GP path, the discrimination property,
save/load, and the metrics — all without weather-model weights.

## Citation

```bibtex
@inproceedings{minoza2026ntkuq,
  title     = {Scalable Uncertainty Quantification for Extreme Weather Forecasting via Empirical Neural Tangent Kernels},
  author    = {Mi{\~n}oza, Jose Marie Antonio and Laylo, Rex Gregor and Iba{\~n}ez, Sebastian C.},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year      = {2026},
  doi       = {10.1145/3770855.3818106}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Copyright 2026 Department of Education (Center for AI Research), Philippines.
