"""
Uncertainty Estimation at Inference Time

Fast uncertainty quantification using pre-computed NTK calibration.
Returns per-variable-per-grid-point uncertainty estimates.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from ..calibration import NTKCalibrator
from ..features import FeatureExtractor

logger = logging.getLogger(__name__)


class UncertaintyEstimator:
    """
    Fast uncertainty estimation using pre-computed NTK calibration.

    At inference time:
    1. Extract features from model prediction
    2. Project onto calibration eigenvectors
    3. Compute uncertainty as posterior variance
    """

    def __init__(
        self,
        calibrator: NTKCalibrator,
        feature_extractor: FeatureExtractor,
        output_shape: Tuple[int, int, int] = (73, 721, 1440),
        device: str = "cuda",
    ):
        """
        Initialize uncertainty estimator.

        Args:
            calibrator: Pre-trained NTK calibrator
            feature_extractor: Model feature extractor
            output_shape: (channels, lat, lon) for reshaping
            device: Computation device
        """
        self.calibrator = calibrator
        self.feature_extractor = feature_extractor
        self.output_shape = output_shape
        self.device = device

    def estimate(
        self,
        x: torch.Tensor,
        lead_time_hours: int,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate uncertainty for a forecast.

        Args:
            x: Input tensor (batch, channels, lat, lon)
            lead_time_hours: Forecast lead time
            return_components: Whether to return variance components

        Returns:
            Dict with 'uncertainty' and optionally variance components
        """
        # Extract features
        features = self.feature_extractor.extract_features(
            x,
            lead_time_hours=lead_time_hours,
        )

        # Compute uncertainty using calibrator
        result = self.calibrator.compute_uncertainty(
            features,
            lead_time_hours=lead_time_hours,
        )

        # Reshape to spatial grid if possible
        batch_size = features.shape[0]
        n_channels, lat, lon = self.output_shape

        # Check if we can reshape
        expected_features = n_channels * lat * lon
        actual_features = result["uncertainty"].shape[-1] if result["uncertainty"].dim() > 1 else 1

        output = {"uncertainty": result["uncertainty"]}

        if return_components:
            output["prior_var"] = result["prior_var"]
            output["posterior_var"] = result["posterior_var"]
            output["correction"] = result["correction"]

        return output

    def estimate_with_forecast(
        self,
        x: torch.Tensor,
        lead_time_hours: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run forecast and estimate uncertainty together.

        Args:
            x: Input tensor
            lead_time_hours: Forecast lead time

        Returns:
            Tuple of (forecast, uncertainty)
        """
        x = x.to(self.device)

        # Run model forward pass (features captured by hook)
        with torch.no_grad():
            n_steps = lead_time_hours // 6
            output = x
            for step in range(n_steps):
                output = self.feature_extractor.model(output)

        # Get uncertainty from captured features
        unc_result = self.calibrator.compute_uncertainty(
            self.feature_extractor._features.view(x.shape[0], -1),
            lead_time_hours=lead_time_hours,
        )

        return output, unc_result["uncertainty"]

    def batch_estimate(
        self,
        x: torch.Tensor,
        lead_times: List[int],
    ) -> Dict[int, torch.Tensor]:
        """
        Estimate uncertainty for multiple lead times.

        Args:
            x: Input tensor
            lead_times: List of lead times

        Returns:
            Dict mapping lead_time -> uncertainty
        """
        results = {}

        for lt in lead_times:
            result = self.estimate(x, lt)
            results[lt] = result["uncertainty"]

        return results


class EnsembleUncertaintyEstimator:
    """
    Combine NTK uncertainty with ensemble spread.

    Total uncertainty = sqrt(NTK_var + Ensemble_var)
    """

    def __init__(
        self,
        ntk_estimators: Dict[str, UncertaintyEstimator],
        ensemble_weight: float = 0.5,
    ):
        """
        Initialize ensemble uncertainty estimator.

        Args:
            ntk_estimators: Dict of model_name -> UncertaintyEstimator
            ensemble_weight: Weight for ensemble spread (0-1)
        """
        self.ntk_estimators = ntk_estimators
        self.ensemble_weight = ensemble_weight
        self.ntk_weight = 1 - ensemble_weight

    def estimate(
        self,
        inputs: Dict[str, torch.Tensor],
        lead_time_hours: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate combined uncertainty from ensemble.

        Args:
            inputs: Dict of model_name -> input tensor
            lead_time_hours: Lead time

        Returns:
            Dict with ensemble mean, spread, NTK uncertainty, and total
        """
        forecasts = []
        ntk_vars = []

        for model_name, x in inputs.items():
            if model_name not in self.ntk_estimators:
                logger.warning(f"No NTK estimator for {model_name}")
                continue

            estimator = self.ntk_estimators[model_name]

            # Get forecast and NTK uncertainty
            forecast, unc = estimator.estimate_with_forecast(x, lead_time_hours)
            forecasts.append(forecast)
            ntk_vars.append(unc ** 2)

        if not forecasts:
            raise ValueError("No valid forecasts generated")

        # Stack forecasts
        forecast_stack = torch.stack(forecasts, dim=0)

        # Ensemble mean and spread
        ensemble_mean = forecast_stack.mean(dim=0)
        ensemble_var = forecast_stack.var(dim=0)

        # Average NTK variance
        ntk_var_avg = torch.stack(ntk_vars, dim=0).mean(dim=0)

        # Combined uncertainty
        total_var = self.ntk_weight * ntk_var_avg + self.ensemble_weight * ensemble_var.mean()
        total_uncertainty = torch.sqrt(total_var)

        return {
            "ensemble_mean": ensemble_mean,
            "ensemble_spread": torch.sqrt(ensemble_var),
            "ntk_uncertainty": torch.sqrt(ntk_var_avg),
            "total_uncertainty": total_uncertainty,
        }


def compute_coverage(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    uncertainties: torch.Tensor,
    quantiles: List[float] = [0.5, 0.9, 0.95],
) -> Dict[str, float]:
    """
    Compute prediction interval coverage.

    Args:
        predictions: Model predictions
        targets: Ground truth
        uncertainties: Uncertainty estimates (std dev)
        quantiles: Coverage quantiles to check

    Returns:
        Dict with coverage statistics
    """
    from scipy import stats

    errors = (predictions - targets).abs()
    results = {}

    for q in quantiles:
        # Z-score for quantile
        z = stats.norm.ppf((1 + q) / 2)

        # Check if error is within interval
        within = errors <= (z * uncertainties)
        coverage = within.float().mean().item()

        results[f"coverage_{int(q*100)}"] = coverage
        results[f"expected_{int(q*100)}"] = q

    # Calibration error
    for q in quantiles:
        key = f"coverage_{int(q*100)}"
        expected_key = f"expected_{int(q*100)}"
        results[f"calibration_error_{int(q*100)}"] = abs(results[key] - results[expected_key])

    return results


def compute_crps(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    uncertainties: torch.Tensor,
) -> float:
    """
    Compute Continuous Ranked Probability Score.

    CRPS measures both calibration and sharpness.

    Args:
        predictions: Model predictions
        targets: Ground truth
        uncertainties: Uncertainty estimates (std dev)

    Returns:
        CRPS value (lower is better)
    """
    # For Gaussian predictive distribution:
    # CRPS = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))
    # where z = (y - mu) / sigma

    from scipy import stats

    sigma = uncertainties.cpu().numpy()
    mu = predictions.cpu().numpy()
    y = targets.cpu().numpy()

    z = (y - mu) / (sigma + 1e-8)

    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)

    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))

    return crps.mean()
