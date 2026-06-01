"""
Pangu-Weather Feature Extractor for NTK Uncertainty Quantification.

Extracts features from ONNX model intermediate layers.
Uses onnxruntime to get encoder output features.

Model inputs:
- input: (5, 13, 721, 1440) = [z, q, t, u, v] at 13 pressure levels
- input_surface: (4, 721, 1440) = [msl, u10, v10, t2m]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

import numpy as np
import torch

from .base import FeatureExtractor

if TYPE_CHECKING:
    import onnxruntime as ort

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
onnx = None
ort = None

def _ensure_onnx():
    global onnx, ort
    if onnx is None:
        import onnx as _onnx
        onnx = _onnx
    if ort is None:
        import onnxruntime as _ort
        ort = _ort


# Pangu-Weather normalization statistics
# These are computed from ERA5 data and required for proper model I/O
# The ONNX model expects normalized inputs (mean=0, std=1) and outputs normalized values
# Default location: project_root/pangu_stats.npz
_PROJECT_ROOT = Path(__file__).parent.parent.parent
PANGU_STATS_FILE = str(_PROJECT_ROOT / 'pangu_stats.npz') if (_PROJECT_ROOT / 'pangu_stats.npz').exists() else None
_PANGU_STATS_CACHE = None  # Cached stats to avoid repeated loading


def load_pangu_stats(stats_file: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load Pangu normalization statistics.

    Args:
        stats_file: Path to pangu_stats.npz file. If None, uses PANGU_STATS_FILE.

    Returns:
        Dict with keys: weather_mean, weather_std, weather_surface_mean, weather_surface_std
    """
    global _PANGU_STATS_CACHE, PANGU_STATS_FILE

    if _PANGU_STATS_CACHE is not None:
        return _PANGU_STATS_CACHE

    if stats_file is None:
        stats_file = PANGU_STATS_FILE

    if stats_file is None:
        raise ValueError(
            "Pangu normalization stats not found. Please either:\n"
            "1. Set PANGU_STATS_FILE to point to pangu_stats.npz\n"
            "2. Run scripts/compute_pangu_normalization.py to generate stats\n"
            "3. Pass stats_file parameter to load_pangu_stats()"
        )

    stats = np.load(stats_file)
    _PANGU_STATS_CACHE = {
        'weather_mean': stats['weather_mean'],
        'weather_std': stats['weather_std'],
        'weather_surface_mean': stats['weather_surface_mean'],
        'weather_surface_std': stats['weather_surface_std'],
    }

    logger.info(f"Loaded Pangu normalization stats from {stats_file}")
    return _PANGU_STATS_CACHE


class PanguWeatherFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for Pangu-Weather (ONNX model).

    Pangu-Weather uses a 3D Swin Transformer architecture.
    We extract features from the encoder output before the final projection.

    Architecture (from paper):
    - Patch embedding
    - 3D Swin Transformer encoder (8 blocks)
    - 3D Swin Transformer decoder (8 blocks)
    - Output projection

    Feature extraction: encoder output after transformer blocks
    """

    def __init__(self, device: str = "cuda"):
        super().__init__("pangu-weather", device)
        self.session_24h = None
        self.session_6h = None

        # Architecture constants
        self.n_pressure_levels = 13
        self.n_pressure_vars = 5  # z, q, t, u, v
        self.n_surface_vars = 4   # msl, u10, v10, t2m
        self.img_size = (721, 1440)

        # Feature extraction settings
        self.feature_aggregation = "global_avg"
        self.intermediate_layer = None  # Will be set after model inspection

        # Normalization stats (lazy loaded)
        self._stats = None

    def _get_stats(self) -> Dict[str, np.ndarray]:
        """Get normalization stats (lazy load)."""
        if self._stats is None:
            self._stats = load_pangu_stats()
        return self._stats

    def _normalize_input(
        self,
        input_pl: np.ndarray,
        input_sfc: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize inputs to mean=0, std=1 as expected by Pangu ONNX model.

        Args:
            input_pl: (5, 13, H, W) pressure level data in physical units
            input_sfc: (4, H, W) surface data in physical units

        Returns:
            Normalized (input_pl, input_sfc)
        """
        stats = self._get_stats()

        # Z-score normalization
        input_pl_norm = (input_pl - stats['weather_mean']) / (stats['weather_std'] + 1e-8)
        input_sfc_norm = (input_sfc - stats['weather_surface_mean']) / (stats['weather_surface_std'] + 1e-8)

        return input_pl_norm.astype(np.float32), input_sfc_norm.astype(np.float32)

    def _denormalize_output(
        self,
        output_pl: np.ndarray,
        output_sfc: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Denormalize outputs from normalized space to physical units.

        Args:
            output_pl: (5, 13, H, W) normalized pressure level output
            output_sfc: (4, H, W) normalized surface output

        Returns:
            Denormalized (output_pl, output_sfc) in physical units
        """
        stats = self._get_stats()

        # Reverse Z-score normalization
        output_pl_phys = output_pl * stats['weather_std'] + stats['weather_mean']
        output_sfc_phys = output_sfc * stats['weather_surface_std'] + stats['weather_surface_mean']

        return output_pl_phys.astype(np.float32), output_sfc_phys.astype(np.float32)

    def _register_hook(self) -> None:
        """Not used for ONNX models - feature extraction via session.run()."""
        pass  # ONNX uses session.run() instead of hooks

    def _get_target_layer(self):
        """Not used for ONNX models."""
        return None  # ONNX doesn't have PyTorch layers

    def load_model(
        self,
        weights_path_24h: str,
        weights_path_6h: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Load Pangu-Weather ONNX models.

        Args:
            weights_path_24h: Path to pangu_weather_24.onnx
            weights_path_6h: Path to pangu_weather_6.onnx (optional)
        """
        _ensure_onnx()  # Lazy load onnx/onnxruntime
        logger.info(f"Loading Pangu-Weather from {weights_path_24h}")

        # Set up ONNX runtime options
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']

        # Load 24h model
        self.session_24h = self._load_onnx_session(weights_path_24h, providers)

        # Load 6h model if provided (for autoregressive rollout)
        if weights_path_6h:
            self.session_6h = self._load_onnx_session(weights_path_6h, providers)
            logger.info(f"Loaded 6h model from {weights_path_6h}")

        # Inspect model to find intermediate layers
        self._inspect_model(weights_path_24h)

        logger.info("Pangu-Weather loaded successfully")

    def _load_onnx_session(self, path: str, providers: List[str]) -> Any:
        """Load ONNX model from local or GCS path."""
        if path.startswith("gs://"):
            import gcsfs
            import tempfile
            fs = gcsfs.GCSFileSystem()
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                fs.get(path, f.name)
                return ort.InferenceSession(f.name, providers=providers)
        else:
            return ort.InferenceSession(path, providers=providers)

    def _inspect_model(self, weights_path: str) -> None:
        """Inspect ONNX model to find intermediate layers for feature extraction."""
        logger.info("Inspecting ONNX model structure...")

        if weights_path.startswith("gs://"):
            # Already loaded, use session
            input_names = [inp.name for inp in self.session_24h.get_inputs()]
            output_names = [out.name for out in self.session_24h.get_outputs()]
        else:
            model = onnx.load(weights_path)
            input_names = [inp.name for inp in model.graph.input]
            output_names = [out.name for out in model.graph.output]

            # Find intermediate nodes that could be good feature extraction points
            # Look for nodes after encoder blocks
            intermediate_outputs = []
            for node in model.graph.node:
                if 'encoder' in node.name.lower() or 'transformer' in node.name.lower():
                    intermediate_outputs.extend(node.output)

            if intermediate_outputs:
                logger.info(f"Found {len(intermediate_outputs)} potential intermediate layers")
                # Use the last encoder output as feature extraction point
                self.intermediate_layer = intermediate_outputs[-1] if intermediate_outputs else None

        logger.info(f"  Inputs: {input_names}")
        logger.info(f"  Outputs: {output_names}")

        # Get input shapes
        for inp in self.session_24h.get_inputs():
            logger.info(f"  {inp.name}: {inp.shape}")

    def extract_features(
        self,
        x: np.ndarray = None,
        x_surface: np.ndarray = None,
        lead_time_hours: int = 24,
        *,
        x_pressure: np.ndarray = None,
    ) -> np.ndarray:
        """
        Extract features from Pangu-Weather.

        Args:
            x: Stacked input (batch, 69, 721, 1440) or (69, 721, 1440)
               Will be split into pressure (65 ch) and surface (4 ch)
            x_pressure: Pressure level input (batch, 5, 13, 721, 1440) or (5, 13, 721, 1440)
            x_surface: Surface input (batch, 4, 721, 1440) or (4, 721, 1440)
            lead_time_hours: Forecast lead time

        Returns:
            Feature array (batch, feature_dim)
        """
        # Handle both stacked input (x) and separate inputs (x_pressure, x_surface)
        if x is not None and x_pressure is None:
            # Stacked input - split into pressure and surface
            # Pangu format: 65 pressure channels (5 vars x 13 levels) + 4 surface channels = 69
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()

            if x.ndim == 3:
                # (69, 721, 1440) -> add batch
                x = x[np.newaxis, ...]

            # Split: first 4 are surface, last 65 are pressure (5 vars x 13 levels)
            # Data format from weatherbench2: surface (msl, u10, v10, t2m) + upper (z, q, t, u, v x 13 levels)
            x_surface = x[:, :4, :, :]  # (batch, 4, H, W)
            x_pressure = x[:, 4:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])  # (batch, 5, 13, H, W)
            batch_size = x.shape[0]
        elif x_pressure is not None:
            # Separate inputs provided
            if isinstance(x_pressure, torch.Tensor):
                x_pressure = x_pressure.cpu().numpy()
            if isinstance(x_surface, torch.Tensor):
                x_surface = x_surface.cpu().numpy()

            # Handle batch dimension - ONNX model expects no batch dim
            if x_pressure.ndim == 5:
                batch_size = x_pressure.shape[0]
            else:
                x_pressure = x_pressure[np.newaxis, ...]
                x_surface = x_surface[np.newaxis, ...]
                batch_size = 1
        else:
            raise ValueError("Must provide either x (stacked) or x_pressure and x_surface")

        # Determine rollout steps
        if lead_time_hours <= 24:
            n_steps = 1
            session = self.session_24h
        else:
            # Use 24h model for longer forecasts (autoregressive)
            n_steps = lead_time_hours // 24
            session = self.session_24h

        # Run inference
        features_list = []

        for b in range(batch_size):
            # Prepare inputs for single sample - squeeze batch dimension for ONNX
            input_pl = x_pressure[b].astype(np.float32)  # (5, 13, 721, 1440)
            input_sfc = x_surface[b].astype(np.float32)   # (4, 721, 1440)

            # Autoregressive rollout
            for step in range(n_steps):
                outputs = session.run(
                    None,
                    {
                        "input": input_pl,
                        "input_surface": input_sfc
                    }
                )

                # Update inputs for next step
                output_pl, output_sfc = outputs[0], outputs[1]
                input_pl = output_pl
                input_sfc = output_sfc

            # Extract features from final output
            # Add back dimension for aggregation: (1, 5, 13, H, W)
            features = self._aggregate_features(
                output_pl[np.newaxis, ...],
                output_sfc[np.newaxis, ...]
            )
            features_list.append(features)

        return np.stack(features_list, axis=0)

    def prepare_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare initial state from stacked input for multi-leadtime rollout.

        Args:
            x: Stacked input (69, 721, 1440) or (batch, 69, 721, 1440)

        Returns:
            Tuple of (input_pl, input_sfc) ready for step_forward
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        if x.ndim == 3:
            x = x[np.newaxis, ...]

        # Split: first 4 are surface, last 65 are pressure
        x_surface = x[:, :4, :, :].astype(np.float32)  # (batch, 4, H, W)
        x_pressure = x[:, 4:, :, :].reshape(
            x.shape[0], 5, 13, x.shape[2], x.shape[3]
        ).astype(np.float32)  # (batch, 5, 13, H, W)

        # Return single sample (squeeze batch) for ONNX
        return x_pressure[0], x_surface[0]

    def step_forward(
        self,
        state: Tuple[np.ndarray, np.ndarray],
        n_steps: int = 1,
        extract_features: bool = True,
        dual_aggregation: bool = False,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Optional[Any]]:
        """
        Run N autoregressive steps forward and optionally extract features.

        Pangu uses 6h steps, so n_steps=1 means 6 hours forward.

        Args:
            state: Tuple of (input_pl, input_sfc) from prepare_state or previous step
            n_steps: Number of 6h steps to run
            extract_features: Whether to extract features at final step
            dual_aggregation: If True, return both global_avg and multi_stat features

        Returns:
            Tuple of:
                - new_state: (output_pl, output_sfc) for next iteration
                - features: Feature array or dict if extract_features=True, else None
        """
        input_pl, input_sfc = state

        # Use 6h model for stepping
        session = self.session_6h if self.session_6h is not None else self.session_24h

        # Run autoregressive steps
        for step in range(n_steps):
            # Normalize inputs before ONNX inference
            input_pl_norm, input_sfc_norm = self._normalize_input(input_pl, input_sfc)

            # Run ONNX model
            outputs = session.run(
                None,
                {
                    "input": input_pl_norm,
                    "input_surface": input_sfc_norm
                }
            )
            output_pl_norm, output_sfc_norm = outputs[0], outputs[1]

            # Denormalize outputs back to physical units
            input_pl, input_sfc = self._denormalize_output(output_pl_norm, output_sfc_norm)

        # Extract features if requested
        features = None
        if extract_features:
            if dual_aggregation:
                features = self._aggregate_features_dual(
                    input_pl[np.newaxis, ...],
                    input_sfc[np.newaxis, ...]
                )
            else:
                features = self._aggregate_features(
                    input_pl[np.newaxis, ...],
                    input_sfc[np.newaxis, ...]
                )

        return (input_pl, input_sfc), features

    # Variable ordering matching model I/O
    PRESSURE_VARS = ["z", "q", "t", "u", "v"]
    SURFACE_VARS = ["msl", "u10", "v10", "t2m"]
    PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    def decode_output_to_vars(
        self,
        state: Tuple[np.ndarray, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Decode model output state into per-variable 2D arrays.

        Args:
            state: (output_pl, output_sfc) from step_forward
                output_pl: (5, 13, H, W) -- z, q, t, u, v at 13 pressure levels
                output_sfc: (4, H, W) -- msl, u10, v10, t2m

        Returns:
            Dict mapping variable names (e.g. 'z_500', 't2m') to (H, W) arrays
        """
        output_pl, output_sfc = state
        result = {}

        # Surface variables
        for i, var in enumerate(self.SURFACE_VARS):
            result[var] = output_sfc[i]

        # Pressure level variables
        for i, var in enumerate(self.PRESSURE_VARS):
            for j, level in enumerate(self.PRESSURE_LEVELS):
                result[f"{var}_{level}"] = output_pl[i, j]

        return result

    def extract_spatial_output(
        self,
        state: Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """
        Get raw spatial output channels without aggregation.

        Args:
            state: (output_pl, output_sfc) from step_forward

        Returns:
            (C, H, W) array where C=69 (65 pressure + 4 surface)
        """
        output_pl, output_sfc = state
        # output_pl: (5, 13, H, W) -> flatten to (65, H, W)
        pl_flat = output_pl.reshape(-1, *output_pl.shape[-2:])  # (65, H, W)
        # output_sfc: (4, H, W)
        return np.concatenate([pl_flat, output_sfc], axis=0)  # (69, H, W)

    def _aggregate_features(
        self,
        output_pl: np.ndarray,
        output_sfc: np.ndarray
    ) -> np.ndarray:
        """
        Aggregate output tensors into feature vector.

        Args:
            output_pl: (1, 5, 13, 721, 1440) pressure level output
            output_sfc: (1, 4, 721, 1440) surface output

        Returns:
            (feature_dim,) feature vector
        """
        if self.feature_aggregation == "global_avg":
            # Global average pooling per variable
            # Pressure levels: (5, 13) -> flatten to 65 features
            pl_features = output_pl.mean(axis=(3, 4)).flatten()  # (65,)
            # Surface: 4 features
            sfc_features = output_sfc.mean(axis=(2, 3)).flatten()  # (4,)
            features = np.concatenate([pl_features, sfc_features])  # (69,)

        elif self.feature_aggregation == "spatial_sample":
            # Sample spatial points
            H, W = output_pl.shape[3], output_pl.shape[4]
            np.random.seed(42)
            n_samples = 100
            h_idx = np.random.randint(0, H, n_samples)
            w_idx = np.random.randint(0, W, n_samples)

            pl_sampled = output_pl[0, :, :, h_idx, w_idx]  # (5, 13, n_samples)
            sfc_sampled = output_sfc[0, :, h_idx, w_idx]   # (4, n_samples)

            features = np.concatenate([
                pl_sampled.flatten(),
                sfc_sampled.flatten()
            ])

        else:
            # Full output (very large!)
            features = np.concatenate([
                output_pl.flatten(),
                output_sfc.flatten()
            ])

        return features

    def _aggregate_features_dual(
        self,
        output_pl: np.ndarray,
        output_sfc: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute BOTH global_avg and multi_stat aggregations.

        Args:
            output_pl: (1, 5, 13, 721, 1440) pressure level output
            output_sfc: (1, 4, 721, 1440) surface output

        Returns:
            Dict with 'global_avg' and 'multi_stat' arrays
        """
        # Global average (69-dim)
        pl_global_avg = output_pl.mean(axis=(3, 4)).flatten()  # (65,)
        sfc_global_avg = output_sfc.mean(axis=(2, 3)).flatten()  # (4,)
        global_avg = np.concatenate([pl_global_avg, sfc_global_avg])  # (69,)

        # Multi-stat aggregation (69 * 6 = 414-dim)
        # Pressure level: (1, 5, 13, H, W) -> (1, 65, H, W) for stats
        pl_reshaped = output_pl.reshape(1, 65, output_pl.shape[3], output_pl.shape[4])
        # Compute stats over spatial dims
        pl_mean = pl_reshaped.mean(axis=(2, 3)).flatten()  # (65,)
        pl_std = pl_reshaped.std(axis=(2, 3)).flatten()    # (65,)
        pl_flat = pl_reshaped.reshape(1, 65, -1)  # (1, 65, H*W)
        pl_min = pl_flat.min(axis=2).flatten()     # (65,)
        pl_max = pl_flat.max(axis=2).flatten()     # (65,)
        pl_p25 = np.percentile(pl_flat, 25, axis=2).flatten()  # (65,)
        pl_p75 = np.percentile(pl_flat, 75, axis=2).flatten()  # (65,)

        # Surface: (1, 4, H, W)
        sfc_mean = output_sfc.mean(axis=(2, 3)).flatten()  # (4,)
        sfc_std = output_sfc.std(axis=(2, 3)).flatten()    # (4,)
        sfc_flat = output_sfc.reshape(1, 4, -1)  # (1, 4, H*W)
        sfc_min = sfc_flat.min(axis=2).flatten()  # (4,)
        sfc_max = sfc_flat.max(axis=2).flatten()  # (4,)
        sfc_p25 = np.percentile(sfc_flat, 25, axis=2).flatten()  # (4,)
        sfc_p75 = np.percentile(sfc_flat, 75, axis=2).flatten()  # (4,)

        # Concatenate all stats: (69 * 6 = 414)
        multi_stat = np.concatenate([
            pl_mean, pl_std, pl_min, pl_max, pl_p25, pl_p75,
            sfc_mean, sfc_std, sfc_min, sfc_max, sfc_p25, sfc_p75
        ])

        return {
            'global_avg': global_avg,
            'multi_stat': multi_stat
        }

    def extract_features_dual(
        self,
        x: Optional[np.ndarray] = None,
        x_pressure: Optional[np.ndarray] = None,
        x_surface: Optional[np.ndarray] = None,
        lead_time_hours: int = 24,
    ) -> Dict[str, np.ndarray]:
        """
        Extract BOTH global_avg and multi_stat features.

        Returns:
            Dict with 'global_avg' (69-dim) and 'multi_stat' (414-dim) arrays
        """
        # Prepare inputs (same as extract_features)
        if x is not None:
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()

            if x.ndim == 3:
                x = x[np.newaxis, ...]

            x_surface = x[:, :4, :, :]
            x_pressure = x[:, 4:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])
            batch_size = x.shape[0]
        elif x_pressure is not None:
            if isinstance(x_pressure, torch.Tensor):
                x_pressure = x_pressure.cpu().numpy()
            if isinstance(x_surface, torch.Tensor):
                x_surface = x_surface.cpu().numpy()

            if x_pressure.ndim == 5:
                batch_size = x_pressure.shape[0]
            else:
                x_pressure = x_pressure[np.newaxis, ...]
                x_surface = x_surface[np.newaxis, ...]
                batch_size = 1
        else:
            raise ValueError("Must provide either x or x_pressure and x_surface")

        # Determine rollout steps
        if lead_time_hours <= 24:
            n_steps = 1
            session = self.session_24h
        else:
            n_steps = lead_time_hours // 24
            session = self.session_24h

        # Run inference
        features_global_list = []
        features_multi_list = []

        for b in range(batch_size):
            input_pl = x_pressure[b].astype(np.float32)
            input_sfc = x_surface[b].astype(np.float32)

            for step in range(n_steps):
                # Normalize inputs before ONNX inference (CRITICAL FIX)
                input_pl_norm, input_sfc_norm = self._normalize_input(input_pl, input_sfc)

                outputs = session.run(
                    None,
                    {
                        "input": input_pl_norm,
                        "input_surface": input_sfc_norm
                    }
                )
                output_pl_norm, output_sfc_norm = outputs[0], outputs[1]

                # Denormalize outputs back to physical units
                input_pl, input_sfc = self._denormalize_output(output_pl_norm, output_sfc_norm)

            # Extract dual features from final output
            features_dict = self._aggregate_features_dual(
                output_pl[np.newaxis, ...],
                output_sfc[np.newaxis, ...]
            )
            features_global_list.append(features_dict['global_avg'])
            features_multi_list.append(features_dict['multi_stat'])

        return {
            'global_avg': np.stack(features_global_list, axis=0),
            'multi_stat': np.stack(features_multi_list, axis=0)
        }

    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features."""
        if self.feature_aggregation == "global_avg":
            # 5 vars * 13 levels + 4 surface = 69
            return self.n_pressure_vars * self.n_pressure_levels + self.n_surface_vars
        elif self.feature_aggregation == "spatial_sample":
            n_samples = 100
            return (self.n_pressure_vars * self.n_pressure_levels + self.n_surface_vars) * n_samples
        else:
            return (self.n_pressure_vars * self.n_pressure_levels + self.n_surface_vars) * self.img_size[0] * self.img_size[1]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        return {
            "model_name": "Pangu-Weather",
            "architecture": "3D Swin Transformer",
            "format": "ONNX",
            "n_pressure_vars": self.n_pressure_vars,
            "n_pressure_levels": self.n_pressure_levels,
            "n_surface_vars": self.n_surface_vars,
            "img_size": self.img_size,
            "feature_dim": self.get_feature_dim(),
            "feature_aggregation": self.feature_aggregation,
            "feature_location": "model output (ONNX intermediate layers not easily accessible)",
        }


def test_local_model(run_inference: bool = False):
    """
    Test feature extraction with local model files.

    Args:
        run_inference: If True, run actual inference (requires ~6GB RAM).
                      If False, just test model loading.
    """
    print("=" * 60)
    print("Pangu-Weather Feature Extractor Test")
    print("=" * 60)

    # Local model paths
    base_path = Path("C:/Users/User/Documents/Code Repositories/ligtas-risk/fastapi-workers/data/models/pw")

    weights_path_24h = base_path / "pangu_weather_24.onnx"
    weights_path_6h = base_path / "pangu_weather_6.onnx"

    # Check files exist
    if not weights_path_24h.exists():
        print(f"ERROR: {weights_path_24h} not found")
        return False

    print(f"\nModel files found at: {base_path}")

    # Create extractor
    print("\nInitializing feature extractor...")
    extractor = PanguWeatherFeatureExtractor(device="cpu")

    print("Loading model...")
    extractor.load_model(
        weights_path_24h=str(weights_path_24h),
        weights_path_6h=str(weights_path_6h) if weights_path_6h.exists() else None,
    )

    # Print model info
    print("\nModel Info:")
    for k, v in extractor.get_model_info().items():
        print(f"  {k}: {v}")

    if run_inference:
        # Test with full input (requires ~6GB RAM)
        print("\nCreating dummy inputs (full resolution)...")
        # Pressure levels: (batch, 5, 13, 721, 1440)
        x_pressure = np.random.randn(1, 5, 13, 721, 1440).astype(np.float32)
        # Surface: (batch, 4, 721, 1440)
        x_surface = np.random.randn(1, 4, 721, 1440).astype(np.float32)

        print(f"  x_pressure shape: {x_pressure.shape}")
        print(f"  x_surface shape: {x_surface.shape}")

        print("\nExtracting features (24h lead time)...")
        print("  NOTE: This requires ~6GB RAM and may take several minutes...")
        features = extractor.extract_features(x_pressure, x_surface, lead_time_hours=24)

        print(f"\nResults:")
        print(f"  Feature shape: {features.shape}")
        print(f"  Feature mean: {features.mean():.6f}")
        print(f"  Feature std: {features.std():.6f}")
        print(f"  Feature min: {features.min():.6f}")
        print(f"  Feature max: {features.max():.6f}")
    else:
        print("\n[SKIPPED] Inference test (requires ~6GB RAM)")
        print("  Run with run_inference=True on A100 VM to test full pipeline")
        print("  Model loading and ONNX structure verified successfully")

    print("\n" + "=" * 60)
    print("TEST PASSED!" if not run_inference else "FULL TEST PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_local_model()
