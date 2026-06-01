"""
NTK Calibration

Last-layer Neural Tangent Kernel calibration for uncertainty quantification.
Computes eigendecomposition of the kernel matrix for efficient inference.
"""

import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NTKCalibrator:
    """
    Neural Tangent Kernel Calibrator.

    Computes and stores the eigendecomposition of the NTK kernel
    for efficient uncertainty quantification at inference time.
    """

    def __init__(
        self,
        model_name: str,
        rank_k: int = 100,
        device: str = "cuda",
        jitter: float = 1e-6,
        reference_lead_time: int = 24,
    ):
        """
        Initialize NTK calibrator.

        Args:
            model_name: Name of the model being calibrated
            rank_k: Number of top eigenvalues to keep
            device: Computation device
            jitter: Regularization added to kernel diagonal for numerical stability
            reference_lead_time: Reference lead time stored in calibration metadata
        """
        self.model_name = model_name
        self.rank_k = rank_k
        self.device = device
        self.jitter = jitter
        self.reference_lead_time = reference_lead_time

        # Calibration data (per lead time)
        self.calibrations: Dict[int, Dict] = {}

        # Post-hoc calibration scales (per lead time, optionally per variable)
        # These scale raw NTK uncertainties to achieve target coverage.
        # Format: {lead_time: scale} OR {lead_time: {var: scale}}
        self.calibration_scales: Dict[int, Union[float, Dict[str, float]]] = {}

    def calibrate_lead_time(
        self,
        features: torch.Tensor,
        lead_time_hours: int,
    ) -> Dict:
        """
        Calibrate for a specific lead time.

        Uses SVD-based approach first (more stable), falls back to
        feature-space approach if SVD fails.

        Args:
            features: Feature matrix (n_samples, feature_dim)
            lead_time_hours: Lead time in hours

        Returns:
            Dict with calibration data and metadata
        """
        logger.info(f"Calibrating for t+{lead_time_hours}h")
        logger.info(f"Features shape: {features.shape}")

        n_samples, feature_dim = features.shape
        features = features.to(self.device)

        # Normalize features for numerical stability
        feature_norms = torch.norm(features, dim=1, keepdim=True)
        feature_mean_norm = feature_norms.mean()
        features_normalized = features / (feature_mean_norm + 1e-8)
        logger.info(f"Feature norm range: [{feature_norms.min():.4f}, {feature_norms.max():.4f}]")
        logger.info(f"Mean feature norm: {feature_mean_norm:.4f}")

        # Center features before SVD (subtract mean direction)
        # Without centering, lambda_1 captures the mean direction (99%+ variance),
        # masking the true covariance structure needed for UQ discrimination.
        features_mean = features_normalized.mean(dim=0)
        features_centered = features_normalized - features_mean.unsqueeze(0)
        logger.info(f"Feature mean norm (pre-centering): {features_mean.norm():.4f}")

        # Try SVD first (most stable)
        calibration = self._calibrate_svd(features_centered, n_samples, feature_dim, lead_time_hours)

        if calibration is None:
            # Fall back to feature-space approach
            logger.warning("SVD failed, trying feature-space approach...")
            calibration = self._calibrate_feature_space(features_centered, n_samples, feature_dim, lead_time_hours)

        if calibration is None:
            raise RuntimeError(f"All calibration methods failed for t+{lead_time_hours}h")

        # Store normalization and centering parameters for inference
        calibration["feature_mean_norm"] = feature_mean_norm.cpu().item()
        calibration["features_mean"] = features_mean.cpu()

        self.calibrations[lead_time_hours] = calibration

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return calibration

    def _calibrate_svd(
        self,
        features: torch.Tensor,
        n_samples: int,
        feature_dim: int,
        lead_time_hours: int,
    ) -> Optional[Dict]:
        """
        SVD-based calibration (primary method).

        Uses SVD on features directly: phi = U @ S @ V.T
        More numerically stable than eigendecomposition of K.
        """
        logger.info("Trying SVD-based calibration...")

        try:
            # SVD on features: phi = U @ S @ V.T
            # K = phi @ phi.T = U @ S^2 @ U.T
            U, S, Vh = torch.linalg.svd(features, full_matrices=False)

            # S are singular values, S^2 are eigenvalues of K
            eigenvalues = S ** 2

            # Filter out near-zero singular values
            threshold = S.max() * 1e-10
            valid_mask = S > threshold
            n_valid = valid_mask.sum().item()

            if n_valid < 1:
                logger.warning("SVD: No valid singular values found")
                return None

            logger.info(f"SVD succeeded: {n_valid}/{len(S)} valid singular values")
            logger.info(f"Singular value range: [{S.min():.6f}, {S.max():.6f}]")

            # Keep top-k
            k = min(self.rank_k, n_valid, n_samples)

            # U contains left singular vectors (eigenvectors of K)
            U_k = U[:, :k]
            Lambda_k = eigenvalues[:k]
            # V contains right singular vectors (for feature-space projection)
            V_k = Vh[:k, :].T  # (feature_dim, k)

            # Compute variance captured
            total_var = eigenvalues.sum().item()
            captured_var = Lambda_k.sum().item()
            var_ratio = captured_var / total_var if total_var > 0 else 0

            logger.info(f"Total variance: {total_var:.4f}")
            logger.info(f"Captured variance (top-{k}): {captured_var:.4f} ({var_ratio*100:.1f}%)")
            logger.info(f"Largest eigenvalue: {Lambda_k[0].item():.6f}")
            logger.info(f"Smallest kept eigenvalue: {Lambda_k[-1].item():.6f}")

            # Estimate noise variance from eigenvalue tail
            noise_var = self._estimate_noise_variance(eigenvalues, k)
            logger.info(f"Estimated noise variance: {noise_var:.6f}")

            return {
                "method": "svd",
                "U_k": U_k.cpu(),
                "Lambda_k": Lambda_k.cpu(),
                "V_k": V_k.cpu(),  # For feature-space inference
                "S_k": S[:k].cpu(),  # Singular values
                "noise_var": noise_var,
                "n_samples": n_samples,
                "feature_dim": feature_dim,
                "rank_k": k,
                "total_variance": total_var,
                "captured_variance": captured_var,
                "variance_ratio": var_ratio,
                "lead_time_hours": lead_time_hours,
                "calibration_time": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.warning(f"SVD failed: {e}")
            return None

    def _calibrate_feature_space(
        self,
        features: torch.Tensor,
        n_samples: int,
        feature_dim: int,
        lead_time_hours: int,
    ) -> Optional[Dict]:
        """
        Feature-space calibration (fallback method).

        Works in feature space (dxd) instead of sample space (nxn).
        Better when n_samples < feature_dim.

        Computes: G = phi.T @ phi (feature covariance)
        """
        logger.info("Trying feature-space calibration...")

        try:
            # Feature covariance: G = phi.T @ phi (dxd)
            G = features.T @ features  # (feature_dim, feature_dim)

            # Add regularization
            jitter = self.jitter
            max_attempts = 5

            for attempt in range(max_attempts):
                try:
                    # Regularize based on trace
                    trace_G = torch.trace(G)
                    reg = jitter * trace_G / feature_dim
                    G_reg = G + reg * torch.eye(feature_dim, device=G.device, dtype=G.dtype)

                    # Cholesky decomposition (more stable for PSD matrices)
                    L = torch.linalg.cholesky(G_reg)
                    logger.info(f"Feature-space Cholesky succeeded with jitter={jitter:.2e}")

                    # For uncertainty, we need (G + lambdaI)^{-1}
                    # Store L for efficient solve at inference
                    # Also compute eigendecomposition for variance stats
                    eigenvalues, eigenvectors = torch.linalg.eigh(G_reg)

                    # Keep top-k
                    k = min(self.rank_k, feature_dim)
                    idx = torch.argsort(eigenvalues, descending=True)
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]

                    Lambda_k = eigenvalues[:k]
                    V_k = eigenvectors[:, :k]

                    # Compute variance
                    total_var = eigenvalues.sum().item()
                    captured_var = Lambda_k.sum().item()
                    var_ratio = captured_var / total_var if total_var > 0 else 0

                    logger.info(f"Total variance: {total_var:.4f}")
                    logger.info(f"Captured variance (top-{k}): {captured_var:.4f} ({var_ratio*100:.1f}%)")

                    # Estimate noise variance from eigenvalue tail
                    noise_var = self._estimate_noise_variance(eigenvalues, k)
                    logger.info(f"Estimated noise variance: {noise_var:.6f}")

                    return {
                        "method": "feature_space",
                        "L": L.cpu(),  # Cholesky factor
                        "V_k": V_k.cpu(),  # Top-k eigenvectors of G
                        "Lambda_k": Lambda_k.cpu(),  # Top-k eigenvalues
                        # features_mean now stored by calibrate_lead_time()
                        "noise_var": noise_var,
                        "n_samples": n_samples,
                        "feature_dim": feature_dim,
                        "rank_k": k,
                        "total_variance": total_var,
                        "captured_variance": captured_var,
                        "variance_ratio": var_ratio,
                        "lead_time_hours": lead_time_hours,
                        "calibration_time": datetime.now().isoformat(),
                        "regularization": reg.item(),
                    }

                except torch._C._LinAlgError:
                    if attempt < max_attempts - 1:
                        jitter *= 10
                        logger.warning(f"Cholesky failed, retrying with jitter={jitter:.2e}")
                    else:
                        raise

        except Exception as e:
            logger.warning(f"Feature-space calibration failed: {e}")
            return None

    def _estimate_noise_variance(
        self,
        eigenvalues: torch.Tensor,
        k: int,
    ) -> float:
        """
        Estimate noise variance from the eigenvalue spectrum.

        Uses the mean of tail eigenvalues (indices k..d) as sigma^2_n.
        When top-k captures all variance (tail ~ 0), falls back to
        the mean of kept eigenvalues as a regularization nugget.
        This prevents the GP correction from fully cancelling the prior,
        which would destroy sample discrimination.

        Args:
            eigenvalues: All eigenvalues sorted descending
            k: Number of kept eigenvalues

        Returns:
            Estimated noise variance (float)
        """
        d = len(eigenvalues)
        if k >= d:
            # Full rank: use mean of all eigenvalues as nugget
            return eigenvalues.mean().item()

        tail = eigenvalues[k:]
        tail_mean = tail.mean().item()

        if tail_mean > 1e-10:
            return tail_mean

        # Tail is negligible -- fall back to mean of kept eigenvalues as nugget
        return eigenvalues[:k].mean().item()

    def calibrate_all_lead_times(
        self,
        feature_extractor,
        data_loader,
        lead_times: List[int] = [24, 48, 72, 120, 240],
        n_samples: int = 500,
        batch_size: int = 4,
    ) -> None:
        """
        Calibrate for all specified lead times.

        Args:
            feature_extractor: FeatureExtractor instance
            data_loader: WeatherBench2Loader instance
            lead_times: List of lead times in hours
            n_samples: Samples per lead time
            batch_size: Batch size for feature extraction
        """
        for lead_time in lead_times:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing lead time: t+{lead_time}h")
            logger.info(f"{'='*50}")

            # Collect features for this lead time
            features_list = []
            sample_count = 0

            for time_dt, data in tqdm(
                data_loader.iter_samples(
                    year=2021,
                    n_samples=n_samples,
                ),
                total=n_samples,
                desc=f"t+{lead_time}h",
            ):
                # Convert to model input
                x = data_loader.to_model_input(data, model_type=self.model_name)

                # Extract features
                features = feature_extractor.extract_features(
                    x,
                    lead_time_hours=lead_time,
                )
                features_list.append(features.cpu())

                sample_count += 1
                if sample_count >= n_samples:
                    break

            # Stack all features
            all_features = torch.cat(features_list, dim=0)
            logger.info(f"Collected {all_features.shape[0]} samples")

            # Calibrate
            self.calibrate_lead_time(all_features, lead_time)

            # Cleanup
            del features_list, all_features
            gc.collect()

    def save(self, output_dir: str) -> None:
        """
        Save calibration data to disk.

        Args:
            output_dir: Base output directory
        """
        base_dir = Path(output_dir) / self.model_name
        base_dir.mkdir(parents=True, exist_ok=True)

        # Tensor keys to save separately
        tensor_keys = ["U_k", "Lambda_k", "V_k", "S_k", "L", "features_mean"]

        for lead_time, cal in self.calibrations.items():
            lead_dir = base_dir / f"t+{lead_time:03d}h"
            lead_dir.mkdir(exist_ok=True)

            # Save tensors
            for key in tensor_keys:
                if key in cal and cal[key] is not None:
                    torch.save(cal[key], lead_dir / f"{key}.pt")

            # Save metadata (everything except tensors)
            meta = {k: v for k, v in cal.items() if k not in tensor_keys}

            # Add calibration scale if available
            # Can be single float (global) or dict (per-variable)
            if lead_time in self.calibration_scales:
                scale = self.calibration_scales[lead_time]
                if isinstance(scale, dict):
                    meta["calibration_scales_per_var"] = scale
                else:
                    meta["calibration_scale"] = scale

            with open(lead_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(f"Saved calibration ({cal.get('method', 'unknown')}) for t+{lead_time}h to {lead_dir}")

    @classmethod
    def load(
        cls,
        output_dir: str,
        model_name: str,
        device: str = "cuda",
    ) -> "NTKCalibrator":
        """
        Load calibration from disk.

        Args:
            output_dir: Base output directory
            model_name: Model name
            device: Device to load tensors to

        Returns:
            NTKCalibrator instance
        """
        base_dir = Path(output_dir) / model_name

        # Find all lead time directories
        lead_dirs = sorted(base_dir.glob("t+*h"))

        calibrator = cls(model_name=model_name, device=device)

        # Tensor keys to try loading
        tensor_keys = ["U_k", "Lambda_k", "V_k", "S_k", "L", "features_mean"]

        for lead_dir in lead_dirs:
            # Parse lead time from directory name
            lead_time = int(lead_dir.name[2:-1])  # "t+024h" -> 24

            # Load metadata first
            with open(lead_dir / "meta.json", "r") as f:
                meta = json.load(f)

            cal_data = {**meta}

            # Load tensors (only if they exist)
            for key in tensor_keys:
                tensor_path = lead_dir / f"{key}.pt"
                if tensor_path.exists():
                    cal_data[key] = torch.load(tensor_path, map_location=device)

            calibrator.calibrations[lead_time] = cal_data

            # Load calibration scale if available
            # Can be single float (global) or dict (per-variable)
            if "calibration_scales_per_var" in meta:
                calibrator.calibration_scales[lead_time] = meta["calibration_scales_per_var"]
            elif "calibration_scale" in meta:
                calibrator.calibration_scales[lead_time] = meta["calibration_scale"]

            method = meta.get("method", "unknown")
            scale = calibrator.calibration_scales.get(lead_time)
            if scale is None:
                scale_str = ""
            elif isinstance(scale, dict):
                scale_str = f", scales={len(scale)} vars"
            else:
                scale_str = f", scale={scale:.4f}"
            logger.info(f"Loaded calibration ({method}{scale_str}) for t+{lead_time}h")

        return calibrator

    def compute_uncertainty(
        self,
        features: torch.Tensor,
        lead_time_hours: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty for given features.

        Automatically uses the appropriate method based on how
        calibration was performed (SVD or feature-space).

        Args:
            features: Feature tensor (batch, feature_dim)
            lead_time_hours: Lead time for which to compute uncertainty

        Returns:
            Dict with 'uncertainty', 'prior_var', 'posterior_var'
        """
        if lead_time_hours not in self.calibrations:
            available = list(self.calibrations.keys())
            raise ValueError(
                f"No calibration for t+{lead_time_hours}h. "
                f"Available: {available}"
            )

        cal = self.calibrations[lead_time_hours]
        method = cal.get("method", "eigendecomp")  # backward compat

        features = features.to(self.device)

        # Normalize features same way as calibration
        feature_mean_norm = cal.get("feature_mean_norm", 1.0)
        features_normalized = features / (feature_mean_norm + 1e-8)

        # Center features same way as calibration
        if "features_mean" in cal:
            features_mean = cal["features_mean"].to(self.device)
            features_centered = features_normalized - features_mean.unsqueeze(0)
        else:
            # Backward compatibility: old calibrations without centering
            features_centered = features_normalized

        # Prior variance: ||phi_centered||^2
        prior_var = (features_centered ** 2).sum(dim=-1)

        if method == "svd":
            return self._compute_uncertainty_svd(features_centered, cal, prior_var)
        elif method == "feature_space":
            return self._compute_uncertainty_feature_space(features_centered, cal, prior_var)
        else:
            # Eigendecomposition of kernel matrix K = phiphi^T
            return self._compute_uncertainty_eigendecomp(features_centered, cal, prior_var)

    def _compute_uncertainty_svd(
        self,
        features: torch.Tensor,
        cal: Dict,
        prior_var: torch.Tensor,
        use_residual: bool = True,
        residual_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute uncertainty using SVD calibration.

        Uses GP posterior predictive formula:
            sigma^2(x*) = ||phi(x*)||^2 + sigma^2_n - Sigma lambda_j c_j^2 / (lambda_j + sigma^2_n)

        where c_j = phi^Tv_j are projection coefficients and sigma^2_n is the
        noise variance estimated from the eigenvalue tail.

        Additionally computes residual variance (unprojected component) which
        provides sample-specific discrimination when GP posterior collapses.

        Args:
            features: Centered feature tensor (batch, feature_dim)
            cal: Calibration dictionary
            prior_var: Prior variance ||phi_centered||^2 per sample
            use_residual: If True, add residual variance for discrimination
            residual_weight: Weight for residual term (default 1.0)
        """
        V_k = cal["V_k"].to(self.device)  # (feature_dim, k)
        Lambda_k = cal["Lambda_k"].to(self.device)  # (k,)
        noise_var = cal.get("noise_var", 1e-8)

        # Project features onto right singular vectors
        proj = features @ V_k  # (batch, k)

        # GP posterior correction: Sigma lambda_j c_j^2 / (lambda_j + sigma^2_n)
        # Shrinkage weight lambda_j / (lambda_j + sigma^2_n)  in  (0, 1)
        weight = Lambda_k / (Lambda_k + noise_var)
        correction = (weight.unsqueeze(0) * proj ** 2).sum(dim=-1)

        # Posterior variance: ||phi||^2 + sigma^2_n - correction
        # The +sigma^2_n accounts for observation noise in the GP predictive distribution
        posterior_var = torch.clamp(prior_var + noise_var - correction, min=1e-8)

        # Residual variance: ||phi - V_k V_k^T phi||^2 = ||phi||^2 - ||V_k^T phi||^2
        # This captures variance NOT explained by top-k principal components
        proj_norm_sq = (proj ** 2).sum(dim=-1)  # ||V_k^T phi||^2
        residual_var = torch.clamp(prior_var - proj_norm_sq, min=0.0)

        # Combined variance: GP posterior + weighted residual
        # The residual provides sample-specific discrimination
        if use_residual:
            combined_var = posterior_var + residual_weight * residual_var
        else:
            combined_var = posterior_var

        # Uncertainty is sqrt of combined variance
        uncertainty = torch.sqrt(combined_var)

        return {
            "uncertainty": uncertainty,
            "prior_var": prior_var,
            "posterior_var": posterior_var,
            "correction": correction,
            "residual_var": residual_var,
            "combined_var": combined_var,
            "method": "svd",
        }

    def _compute_uncertainty_feature_space(
        self,
        features: torch.Tensor,
        cal: Dict,
        prior_var: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute uncertainty using feature-space calibration.

        Uses GP posterior predictive formula:
            sigma^2(x*) = ||phi(x*)||^2 + sigma^2_n - Sigma lambda_j c_j^2 / (lambda_j + sigma^2_n)
        """
        V_k = cal["V_k"].to(self.device)  # (feature_dim, k)
        Lambda_k = cal["Lambda_k"].to(self.device)  # (k,)
        noise_var = cal.get("noise_var", 1e-8)

        # Project onto principal components
        proj = features @ V_k  # (batch, k)

        # GP posterior correction: Sigma lambda_j c_j^2 / (lambda_j + sigma^2_n)
        weight = Lambda_k / (Lambda_k + noise_var)
        correction = (weight.unsqueeze(0) * proj ** 2).sum(dim=-1)

        # Posterior variance: ||phi||^2 + sigma^2_n - correction
        posterior_var = torch.clamp(prior_var + noise_var - correction, min=1e-8)

        # Uncertainty
        uncertainty = torch.sqrt(posterior_var)

        return {
            "uncertainty": uncertainty,
            "prior_var": prior_var,
            "posterior_var": posterior_var,
            "correction": correction,
            "method": "feature_space",
        }

    def _compute_uncertainty_eigendecomp(
        self,
        features: torch.Tensor,
        cal: Dict,
        prior_var: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Eigendecomposition method using kernel eigenvectors U_k.

        Uses GP posterior predictive formula:
            sigma^2(x*) = ||phi(x*)||^2 + sigma^2_n - Sigma lambda_j c_j^2 / (lambda_j + sigma^2_n)
        """
        U_k = cal["U_k"].to(self.device)
        Lambda_k = cal["Lambda_k"].to(self.device)
        noise_var = cal.get("noise_var", 1e-8)

        # Project features onto eigenvectors
        proj = features @ U_k  # (batch, k)

        # GP posterior correction: Sigma lambda_j c_j^2 / (lambda_j + sigma^2_n)
        weight = Lambda_k / (Lambda_k + noise_var)
        correction = (weight.unsqueeze(0) * proj ** 2).sum(dim=-1)

        # Posterior variance: ||phi||^2 + sigma^2_n - correction
        posterior_var = torch.clamp(prior_var + noise_var - correction, min=1e-8)

        # Uncertainty
        uncertainty = torch.sqrt(posterior_var)

        return {
            "uncertainty": uncertainty,
            "prior_var": prior_var,
            "posterior_var": posterior_var,
            "correction": correction,
            "method": "eigendecomp",
        }
