"""Tests for the empirical NTK calibration pipeline.

These run without any weather-model weights: they exercise the package imports,
the kernel/decomposition/GP-posterior path on synthetic features, the
discrimination property, save/load round-tripping, and the evaluation metrics.
"""

import numpy as np
import pytest
import torch

from ntk_uq import NTKCalibrator, FeatureExtractor
from ntk_uq.inference import compute_coverage, compute_crps


FEATURE_DIM = 128
LEAD_TIME = 24


def _synthetic_features(n: int, off_manifold: float = 0.0, seed: int = 0):
    """Low-rank features plus optional off-manifold shift (an 'unusual' state)."""
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(FEATURE_DIM, 8, generator=g)
    coeffs = torch.randn(n, 8, generator=g)
    base = coeffs @ basis.T + 0.05 * torch.randn(n, FEATURE_DIM, generator=g)
    if off_manifold:
        base = base + off_manifold * torch.randn(n, FEATURE_DIM, generator=g)
    return base


@pytest.fixture(scope="module")
def calibrator():
    cal = NTKCalibrator(model_name="test", rank_k=10, device="cpu")
    cal.calibrate_lead_time(_synthetic_features(100, seed=1), lead_time_hours=LEAD_TIME)
    return cal


def test_package_imports():
    assert callable(NTKCalibrator)
    assert isinstance(FeatureExtractor, type)


def test_feature_extractors_are_lazy():
    # Importing the package must not require any model backend.
    from ntk_uq.features import FeatureExtractor as FE
    assert isinstance(FE, type)


def test_calibration_produces_uncertainty(calibrator):
    feats = _synthetic_features(16, seed=2)
    out = calibrator.compute_uncertainty(feats, lead_time_hours=LEAD_TIME)
    assert set(["uncertainty", "prior_var", "posterior_var"]).issubset(out.keys())
    assert out["uncertainty"].shape[0] == 16
    assert torch.all(out["uncertainty"] >= 0)


def test_ood_gets_higher_uncertainty(calibrator):
    in_dist = _synthetic_features(20, off_manifold=0.0, seed=3)
    out_dist = _synthetic_features(20, off_manifold=3.0, seed=4)
    s_in = calibrator.compute_uncertainty(in_dist, LEAD_TIME)["uncertainty"].mean()
    s_out = calibrator.compute_uncertainty(out_dist, LEAD_TIME)["uncertainty"].mean()
    assert s_out > s_in


def test_unknown_lead_time_raises(calibrator):
    with pytest.raises(ValueError):
        calibrator.compute_uncertainty(_synthetic_features(4), lead_time_hours=999)


def test_save_load_roundtrip(tmp_path, calibrator):
    calibrator.save(str(tmp_path))
    reloaded = NTKCalibrator.load(str(tmp_path), model_name="test", device="cpu")
    feats = _synthetic_features(8, seed=5)
    a = calibrator.compute_uncertainty(feats, LEAD_TIME)["uncertainty"]
    b = reloaded.compute_uncertainty(feats, LEAD_TIME)["uncertainty"]
    assert torch.allclose(a, b, atol=1e-5)


def test_metrics():
    torch.manual_seed(0)
    n = 500
    targets = torch.zeros(n)
    sigma = torch.ones(n)
    preds = torch.randn(n)  # errors ~ N(0, 1), matching sigma
    cov = compute_coverage(preds, targets, sigma, quantiles=[0.9])
    assert 0.80 <= cov["coverage_90"] <= 1.0
    crps = compute_crps(preds, targets, sigma)
    assert crps > 0
