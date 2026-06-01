"""Runnable demo of empirical NTK calibration on synthetic features.

This example needs no weather-model weights: it shows the core empirical-NTK
pipeline end to end on synthetic last-layer features.

    features  ->  center + build last-layer NTK kernel  ->  SVD/ICA decomposition
              ->  GP posterior variance  ->  post-hoc-scaled uncertainty

The key property it demonstrates: test inputs that are *dissimilar* to the
calibration distribution in feature space receive *higher* epistemic
uncertainty, while inputs similar to the calibration set receive lower
uncertainty. This is exactly the signal the four weather-model extractors
provide on real data.

Run:
    python examples/01_calibrate_synthetic.py
"""

import numpy as np
import torch

from ntk_uq.calibration import NTKCalibrator


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_calib, feature_dim, lead_time = 100, 256, 24

    # Synthetic calibration features living on a low-rank manifold (as real
    # last-layer features do after global pooling). The leading directions carry
    # the structure; the rest is small noise.
    basis = torch.randn(feature_dim, 8)
    coeffs = torch.randn(n_calib, 8)
    calib_features = coeffs @ basis.T + 0.05 * torch.randn(n_calib, feature_dim)

    # Calibrate the empirical NTK for this lead time.
    calibrator = NTKCalibrator(model_name="synthetic", rank_k=10, device=device)
    calibrator.calibrate_lead_time(calib_features, lead_time_hours=lead_time)

    # Two test sets:
    #   - "in-distribution": drawn from the same manifold as calibration.
    #   - "out-of-distribution": shifted off the manifold (an unusual atmospheric
    #     state, analogous to an extreme event).
    in_dist = (torch.randn(20, 8) @ basis.T) + 0.05 * torch.randn(20, feature_dim)
    out_dist = (torch.randn(20, 8) @ basis.T) + 3.0 * torch.randn(20, feature_dim)

    u_in = calibrator.compute_uncertainty(in_dist, lead_time_hours=lead_time)
    u_out = calibrator.compute_uncertainty(out_dist, lead_time_hours=lead_time)

    sigma_in = u_in["uncertainty"].mean().item()
    sigma_out = u_out["uncertainty"].mean().item()

    print(f"Mean uncertainty, in-distribution test inputs : {sigma_in:.4f}")
    print(f"Mean uncertainty, out-of-distribution inputs  : {sigma_out:.4f}")
    print(f"Ratio (out / in)                              : {sigma_out / sigma_in:.2f}x")
    assert sigma_out > sigma_in, "expected higher uncertainty off the manifold"
    print("\nOK: dissimilar inputs receive higher epistemic uncertainty.")


if __name__ == "__main__":
    main()
