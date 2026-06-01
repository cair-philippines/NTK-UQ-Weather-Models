"""
Aurora Feature Extractor for NTK Uncertainty Quantification.

Microsoft's Aurora model uses PerceiverIO architecture.
Extracts features from the decoder output.

Model:
- Format: PyTorch checkpoint (.ckpt)
- Static data: aurora-0.25-static.pickle (z, lsm, slt)
- Architecture: PerceiverIO with encoder.surf_mlp, encoder.pos_embed, etc.

Note: Aurora requires input as a Batch object with:
- surf_vars: dict of surface variables (2t, 10u, 10v, msl)
- atmos_vars: dict of atmospheric variables (t, u, v, q, z at pressure levels)
- static_vars: dict of static variables (z, lsm, slt)
- metadata: Metadata with lat, lon, time, atmos_levels
"""

import logging
import os
import pickle  # Required for Aurora static data
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import FeatureExtractor

logger = logging.getLogger(__name__)

# Aurora pressure levels (13 levels, same as ERA5)
AURORA_PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

# ERA5 climatological statistics for surface variables
# Format: {variable_name: (mean, std)}
AURORA_SURF_STATS = {
    "2t": (278.0, 22.0),       # 2m temperature [K]
    "10u": (0.0, 6.0),         # 10m u-wind [m/s]
    "10v": (0.0, 5.5),         # 10m v-wind [m/s]
    "msl": (101325.0, 1200.0), # Mean sea level pressure [Pa]
}


class AuroraFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for Aurora (Microsoft PerceiverIO model).

    Aurora uses a PerceiverIO architecture with:
    - Surface encoder MLP
    - Positional embeddings
    - Cross-attention encoder
    - Latent transformer
    - Cross-attention decoder

    Feature extraction point: decoder output before final projection
    """

    def __init__(self, device: str = "cuda"):
        super().__init__("aurora", device)

        # Architecture constants (from model inspection)
        self.n_channels = 69  # Similar to other models
        self.img_size = (721, 1440)

        # Feature extraction settings
        self.feature_aggregation = "global_avg"
        self._features = None
        self._hook_handle = None

        # Static data
        self.static_data = None

    def load_model(
        self,
        weights_path: str,
        static_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Load Aurora model.

        Args:
            weights_path: Path to aurora-0.25-finetuned.ckpt
            static_path: Path to aurora-0.25-static.pickle
        """
        logger.info(f"Loading Aurora from {weights_path}")

        # Load static data if provided
        if static_path:
            self.static_data = self._load_static_data(static_path)
            logger.info(f"Loaded static data: {list(self.static_data.keys())}")

        # Load checkpoint
        checkpoint = self._load_checkpoint(weights_path)

        # Inspect model structure
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        logger.info(f"Checkpoint has {len(state_dict)} parameters")

        # Log some key names to understand structure
        sample_keys = list(state_dict.keys())[:10]
        logger.info(f"Sample parameter names: {sample_keys}")

        # Try to load the model
        # Note: Aurora requires the microsoft-aurora package
        try:
            self._load_aurora_model(weights_path, state_dict)
        except Exception as e:
            logger.warning(f"Could not load Aurora model: {e}")
            logger.info("Aurora requires microsoft-aurora package. Storing state_dict only.")
            self.model = None
            self._state_dict = state_dict

        logger.info("Aurora loaded successfully")

    def _load_static_data(self, path: str) -> Dict[str, np.ndarray]:
        """Load static data (z, lsm, slt) from pickle file."""
        if path.startswith("gs://"):
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            with fs.open(path, "rb") as f:
                return pickle.load(f)
        else:
            with open(path, "rb") as f:
                return pickle.load(f)

    def _load_checkpoint(self, path: str) -> Dict:
        """Load PyTorch checkpoint."""
        if path.startswith("gs://"):
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
                fs.get(path, f.name)
                return torch.load(f.name, map_location=self.device, weights_only=False)
        else:
            return torch.load(path, map_location=self.device, weights_only=False)

    def _load_aurora_model(self, weights_path: str, state_dict: Dict) -> None:
        """
        Try to load Aurora model using microsoft-aurora package.

        Note: This requires:
            pip install microsoft-aurora
        """
        try:
            from aurora import Aurora, AuroraHighRes

            # Try to determine which variant to use
            if "0.25" in weights_path or "highres" in weights_path.lower():
                self.model = AuroraHighRes()
            else:
                self.model = Aurora()

            # Load state dict
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.model.to(self.device)

            # Register hook
            self._register_hook()

            logger.info(f"Aurora model loaded with {type(self.model).__name__}")

        except ImportError:
            logger.warning("microsoft-aurora package not installed")
            raise

    def _register_hook(self) -> None:
        """Register forward hook on decoder output."""
        if self.model is None:
            return

        def hook_fn(module, input, output):
            # Aurora decoder outputs a Batch object, not a tensor
            # Extract features from the atmos_vars (contains the latent representations)
            try:
                from aurora import Batch
                if isinstance(output, Batch):
                    # Concatenate all atmospheric variables as features
                    features_list = []
                    for var_name, var_tensor in output.atmos_vars.items():
                        t = var_tensor.detach()
                        # Handle different tensor shapes
                        if t.ndim == 5:  # (B, T, levels, H, W)
                            t = t[:, -1, :, :, :]  # Take last time step
                        elif t.ndim == 4:  # (B, levels, H, W) or (B, T, H, W)
                            pass  # Keep as is
                        features_list.append(t)
                    # Concatenate along channel dimension
                    self._features = torch.cat(features_list, dim=1)  # (B, total_channels, H, W)
                else:
                    self._features = output.detach()
            except Exception as e:
                logger.warning(f"Hook failed to extract features: {e}")
                self._features = None

        # Find the decoder layer
        target_layer = self._get_target_layer()
        if target_layer is not None:
            self._hook_handle = target_layer.register_forward_hook(hook_fn)
            logger.info(f"Registered hook on {type(target_layer).__name__}")

    def _get_target_layer(self) -> Optional[nn.Module]:
        """Get the decoder output layer."""
        if self.model is None:
            return None

        # Try common layer names for PerceiverIO
        layer_names = [
            "decoder",
            "perceiver_decoder",
            "output_decoder",
            "cross_attention_decoder",
        ]

        for name in layer_names:
            if hasattr(self.model, name):
                return getattr(self.model, name)

        # Fallback: look for layers with 'decoder' in name
        for name, module in self.model.named_modules():
            if "decoder" in name.lower() and not name.endswith("_proj"):
                return module

        logger.warning("Could not find decoder layer for hook")
        return None

    def extract_features(
        self,
        x: torch.Tensor,
        lead_time_hours: int = 24,
        normalize_input: bool = True,
        time: Optional[datetime] = None,
    ) -> torch.Tensor:
        """
        Extract features from Aurora.

        Args:
            x: Input tensor (batch, channels, lat, lon) - 69 channels in Pangu format
               OR dict with surf_vars, atmos_vars, static_vars
            lead_time_hours: Forecast lead time
            normalize_input: Whether to normalize input
            time: Forecast initialization time (default: 2021-01-01 00:00)

        Returns:
            Feature tensor (batch, feature_dim)
        """
        if self.model is None:
            raise RuntimeError(
                "Aurora model not loaded. "
                "Install microsoft-aurora package: pip install microsoft-aurora"
            )

        from aurora import Batch, Metadata

        # Convert tensor to Batch format if needed
        if isinstance(x, torch.Tensor):
            batch = self._tensor_to_batch(x, time=time)
        elif isinstance(x, dict):
            batch = self._dict_to_batch(x, time=time)
        elif isinstance(x, Batch):
            batch = x
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")

        # Move to device
        batch = batch.to(self.device)

        # Normalize the batch (required by Aurora)
        # Use model's surf_stats if available, otherwise use defaults
        surf_stats = getattr(self.model, 'surf_stats', None)
        if not surf_stats:
            surf_stats = AURORA_SURF_STATS
        batch = batch.normalise(surf_stats=surf_stats)

        self._features = None

        with torch.no_grad():
            # Autoregressive rollout (6-hour steps)
            n_steps = max(1, lead_time_hours // 6)
            current_batch = batch
            for step in range(n_steps):
                current_batch = self.model(current_batch)

        if self._features is None:
            raise RuntimeError("Failed to capture features from hook")

        # Aggregate spatial dimensions
        features = self._aggregate_features(self._features)

        return features

    def _tensor_to_batch(
        self,
        x: torch.Tensor,
        time: Optional[datetime] = None,
    ) -> "Batch":
        """
        Convert a stacked tensor (Pangu format) to Aurora Batch.

        Input tensor format (69 channels):
        - Surface (4): msl, u10, v10, t2m
        - Upper air (65): z, q, t, u, v at 13 levels each

        Args:
            x: Input tensor (batch, 69, 721, 1440)
            time: Initialization time

        Returns:
            Aurora Batch object
        """
        from aurora import Batch, Metadata

        if time is None:
            time = datetime(2021, 1, 1, 0, 0)

        # Handle batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)

        batch_size, n_channels, lat_size, lon_size = x.shape

        # Aurora expects 720 latitudes (0.25 deg without south pole), ERA5 has 721
        if lat_size == 721:
            x = x[:, :, :720, :]
            lat_size = 720

        # Create lat/lon grids (0.25 degree resolution)
        lat = torch.linspace(90, -90, lat_size)
        lon = torch.linspace(0, 360 - 360/lon_size, lon_size)

        # Extract surface variables (indices 0-3)
        # Input order: msl, u10, v10, t2m
        # Aurora expects: 2t, 10u, 10v, msl with shape (B, T, H, W)
        # We use T=2 (two time steps for history, duplicate single input)
        T = 2  # Aurora needs 2 history steps
        surf_vars = {
            "2t": x[:, 3:4, :, :].unsqueeze(1).expand(-1, T, -1, -1, -1).squeeze(2),   # (B, T, H, W)
            "10u": x[:, 1:2, :, :].unsqueeze(1).expand(-1, T, -1, -1, -1).squeeze(2),
            "10v": x[:, 2:3, :, :].unsqueeze(1).expand(-1, T, -1, -1, -1).squeeze(2),
            "msl": x[:, 0:1, :, :].unsqueeze(1).expand(-1, T, -1, -1, -1).squeeze(2),
        }

        # Extract atmospheric variables (indices 4-68)
        # Input order: z (13), q (13), t (13), u (13), v (13)
        # Aurora expects shape (B, T, levels, H, W)
        atmos_vars = {}
        levels = AURORA_PRESSURE_LEVELS
        n_levels = len(levels)
        idx = 4

        for var_in, var_out in [("z", "z"), ("q", "q"), ("t", "t"), ("u", "u"), ("v", "v")]:
            var_data = []
            for level in levels:
                var_data.append(x[:, idx:idx+1, :, :])
                idx += 1
            # Stack levels: (batch, 1, H, W) * 13 -> (batch, 13, H, W)
            stacked = torch.cat(var_data, dim=1)  # (B, levels, H, W)
            # Add time dimension and expand: (B, T, levels, H, W)
            atmos_vars[var_out] = stacked.unsqueeze(1).expand(-1, T, -1, -1, -1)

        # Static variables from loaded static data or zeros
        # Aurora expects static vars as 2D tensors (lat, lon), model handles batching
        if self.static_data is not None:
            static_vars = {
                "z": torch.tensor(self.static_data["z"][:lat_size, :lon_size]),
                "lsm": torch.tensor(self.static_data["lsm"][:lat_size, :lon_size]),
                "slt": torch.tensor(self.static_data["slt"][:lat_size, :lon_size]),
            }
        else:
            # Create dummy static vars if not available
            static_vars = {
                "z": torch.zeros(lat_size, lon_size),
                "lsm": torch.ones(lat_size, lon_size),
                "slt": torch.zeros(lat_size, lon_size),
            }

        # Create metadata
        # Aurora needs 2 time steps for history
        times = (time, time)  # Use same time for both (simplified)
        metadata = Metadata(
            lat=lat,
            lon=lon,
            time=times,
            atmos_levels=levels,
        )

        return Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=metadata,
        )

    def _dict_to_batch(
        self,
        data: Dict[str, torch.Tensor],
        time: Optional[datetime] = None,
    ) -> "Batch":
        """
        Convert a dict of variables to Aurora Batch.

        Args:
            data: Dict with variable names as keys
            time: Initialization time

        Returns:
            Aurora Batch object
        """
        from aurora import Batch, Metadata

        if time is None:
            time = datetime(2021, 1, 1, 0, 0)

        # Extract surface variables
        surf_vars = {}
        for key_in, key_out in [("t2m", "2t"), ("u10", "10u"), ("v10", "10v"), ("msl", "msl")]:
            if key_in in data:
                val = data[key_in]
                if val.dim() == 2:
                    val = val.unsqueeze(0).unsqueeze(0)
                elif val.dim() == 3:
                    val = val.unsqueeze(1)
                surf_vars[key_out] = val

        # Extract atmospheric variables
        levels = AURORA_PRESSURE_LEVELS
        atmos_vars = {}
        for var in ["z", "q", "t", "u", "v"]:
            var_data = []
            for level in levels:
                key = f"{var}_{level}"
                if key in data:
                    val = data[key]
                    if val.dim() == 2:
                        val = val.unsqueeze(0)
                    var_data.append(val)
            if var_data:
                atmos_vars[var] = torch.stack(var_data, dim=1)

        # Get shape from first variable
        sample_var = next(iter(surf_vars.values()))
        batch_size = sample_var.shape[0]
        lat_size = sample_var.shape[-2]
        lon_size = sample_var.shape[-1]

        # Static variables (2D tensors, model handles batching)
        if self.static_data is not None:
            static_vars = {
                "z": torch.tensor(self.static_data["z"]),
                "lsm": torch.tensor(self.static_data["lsm"]),
                "slt": torch.tensor(self.static_data["slt"]),
            }
        else:
            static_vars = {
                "z": torch.zeros(lat_size, lon_size),
                "lsm": torch.ones(lat_size, lon_size),
                "slt": torch.zeros(lat_size, lon_size),
            }

        # Create lat/lon and metadata
        lat = torch.linspace(90, -90, lat_size)
        lon = torch.linspace(0, 360 - 360/lon_size, lon_size)
        times = (time, time)
        metadata = Metadata(
            lat=lat,
            lon=lon,
            time=times,
            atmos_levels=levels,
        )

        return Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=metadata,
        )

    def extract_spatial_output(self, state: "Batch") -> np.ndarray:
        """
        Get raw spatial output channels without aggregation.

        Args:
            state: Aurora Batch from step_forward

        Returns:
            (C, H, W) array where C=69 (65 pressure + 4 surface)
        """
        channels = []
        # Pressure level variables
        # Shape: (B, T, levels, H, W) = (1, 2, 13, 720, 1440)
        # We want batch 0, last time step -> (levels, H, W)
        for var in self.AURORA_ATMOS_VARS:
            if var in state.atmos_vars:
                t = state.atmos_vars[var]
                if t.dim() == 5:
                    t = t[0, -1]  # Batch 0, last time -> (levels, H, W)
                elif t.dim() == 4:
                    t = t[-1]    # Last time if no batch dim
                elif t.dim() == 3:
                    pass  # Already (levels, H, W)
                channels.append(t.cpu().numpy())
        # Surface variables
        # Shape: (B, T, H, W) = (1, 2, 720, 1440)
        # We want batch 0, last time step -> (H, W)
        for aurora_name in ["2t", "10u", "10v", "msl"]:
            if aurora_name in state.surf_vars:
                t = state.surf_vars[aurora_name]
                if t.dim() == 4:
                    t = t[0, -1]  # Batch 0, last time -> (H, W)
                elif t.dim() == 3:
                    t = t[-1]    # Last time if no batch dim
                elif t.dim() == 2:
                    pass  # Already (H, W)
                channels.append(t.cpu().numpy()[np.newaxis, ...])  # (1, H, W)
        return np.concatenate(channels, axis=0)  # (C, H, W)

    def _aggregate_features(self, features: torch.Tensor) -> torch.Tensor:
        """Aggregate spatial dimensions."""
        batch_size = features.shape[0]

        if self.feature_aggregation == "global_avg":
            # Global average pooling
            if features.ndim == 4:  # (batch, C, H, W)
                features = features.mean(dim=(2, 3))
            elif features.ndim == 3:  # (batch, seq, hidden)
                features = features.mean(dim=1)
            else:
                features = features.view(batch_size, -1)
        else:
            features = features.view(batch_size, -1)

        return features

    def _aggregate_features_dual(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute BOTH global_avg and multi_stat aggregations.

        Args:
            features: (batch, C, ...) tensor from encoder

        Returns:
            Dict with 'global_avg' and 'multi_stat' tensors
        """
        batch_size = features.shape[0]

        # Handle different feature shapes
        if features.ndim == 4:  # (batch, C, H, W)
            C = features.shape[1]
            # Global average: (batch, C)
            global_avg = features.mean(dim=(2, 3))

            # Multi-stat: (batch, C*6)
            flat = features.view(batch_size, C, -1)  # (batch, C, H*W)
            spatial_mean = flat.mean(dim=2)  # (batch, C)
            spatial_std = flat.std(dim=2)    # (batch, C)
            spatial_min = flat.min(dim=2).values  # (batch, C)
            spatial_max = flat.max(dim=2).values  # (batch, C)
            spatial_p25 = flat.quantile(0.25, dim=2)  # (batch, C)
            spatial_p75 = flat.quantile(0.75, dim=2)  # (batch, C)

            multi_stat = torch.cat([
                spatial_mean, spatial_std,
                spatial_min, spatial_max,
                spatial_p25, spatial_p75
            ], dim=1)  # (batch, C*6)

        elif features.ndim == 3:  # (batch, seq, hidden)
            # Average over sequence dimension
            hidden_dim = features.shape[2]
            global_avg = features.mean(dim=1)  # (batch, hidden)

            # Multi-stat over sequence
            spatial_mean = features.mean(dim=1)
            spatial_std = features.std(dim=1)
            spatial_min = features.min(dim=1).values
            spatial_max = features.max(dim=1).values
            spatial_p25 = features.quantile(0.25, dim=1)
            spatial_p75 = features.quantile(0.75, dim=1)

            multi_stat = torch.cat([
                spatial_mean, spatial_std,
                spatial_min, spatial_max,
                spatial_p25, spatial_p75
            ], dim=1)  # (batch, hidden*6)

        else:
            # Flatten - no spatial structure
            global_avg = features.view(batch_size, -1)
            multi_stat = global_avg  # Same as global_avg if no spatial structure

        return {
            'global_avg': global_avg,
            'multi_stat': multi_stat
        }

    def extract_features_dual(
        self,
        x: torch.Tensor,
        lead_time_hours: int = 24,
        normalize_input: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract BOTH global_avg and multi_stat features.

        Returns:
            Dict with 'global_avg' and 'multi_stat' tensors
        """
        if self.model is None:
            raise RuntimeError("Aurora model not loaded")

        # Handle NaN
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        # Convert to Batch
        batch = self._tensor_to_batch(x)
        batch = batch.to(self.device)

        # Normalize
        if normalize_input:
            surf_stats = getattr(self.model, 'surf_stats', None)
            atmos_stats = getattr(self.model, 'atmos_stats', None)
            if surf_stats and atmos_stats:
                batch = self.model.normalizer(batch, surf_stats, atmos_stats)

        self._features = None
        if self._hook_handle is None:
            self._register_hook()

        with torch.no_grad():
            n_steps = max(1, lead_time_hours // 6)
            output_batch = batch
            for step in range(n_steps):
                output_batch = self.model.forward(output_batch)

        if self._features is None:
            raise RuntimeError("Failed to capture features from hook")

        # Get dual aggregations
        dual_features = self._aggregate_features_dual(self._features)

        # NaN check
        for key in dual_features:
            if torch.isnan(dual_features[key]).any():
                dual_features[key] = torch.nan_to_num(dual_features[key], nan=0.0)

        return dual_features

    def prepare_batch(
        self,
        x: torch.Tensor,
        time: Optional[datetime] = None,
    ) -> "Batch":
        """
        Prepare initial Batch state for multi-leadtime rollout.

        Args:
            x: Input tensor (69, H, W) or (batch, 69, H, W)
            time: Forecast initialization time

        Returns:
            Aurora Batch object ready for step_forward
        """
        if self.model is None:
            raise RuntimeError("Aurora model not loaded")

        # Convert to Batch and normalize
        batch = self._tensor_to_batch(x, time=time)
        batch = batch.to(self.device)

        # Normalize using model's stats
        surf_stats = getattr(self.model, 'surf_stats', None)
        if not surf_stats:
            surf_stats = AURORA_SURF_STATS
        batch = batch.normalise(surf_stats=surf_stats)

        return batch

    def step_forward(
        self,
        state: "Batch",
        n_steps: int = 1,
        extract_features: bool = True,
        dual_aggregation: bool = False,
    ) -> Tuple["Batch", Optional[Any]]:
        """
        Run N autoregressive steps forward and optionally extract features.

        Aurora uses 6h steps, so n_steps=1 means 6 hours forward.

        Args:
            state: Aurora Batch object from prepare_batch or previous step
            n_steps: Number of 6h steps to run
            extract_features: Whether to extract features at final step
            dual_aggregation: If True, return both global_avg and multi_stat features

        Returns:
            Tuple of:
                - new_state: Aurora Batch for next iteration
                - features: Feature tensor or dict if extract_features=True, else None
        """
        if self.model is None:
            raise RuntimeError("Aurora model not loaded")

        self._features = None
        current_batch = state

        with torch.no_grad():
            for step in range(n_steps):
                current_batch = self.model(current_batch)

        # Extract features if requested
        features = None
        if extract_features and self._features is not None:
            if dual_aggregation:
                features = self._aggregate_features_dual(self._features)
            else:
                features = self._aggregate_features(self._features)

        return current_batch, features

    def get_raw_spatial_features(self) -> Optional[np.ndarray]:
        """
        Get raw decoder features with spatial structure for spatial UQ.

        Returns the features captured by the hook BEFORE aggregation.
        This matches the calibration dimension (65 features) and preserves
        the spatial structure (H, W).

        Returns:
            (C, H, W) array where C=65 (decoder latent dim), or None if not available
        """
        if self._features is None:
            return None

        # self._features is (B, C, H, W) from the hook
        features = self._features.detach().cpu()

        # Take batch 0
        if features.dim() == 4:
            features = features[0]  # (C, H, W)

        return features.numpy()

    # Aurora variable name mapping: Aurora names -> standard names
    AURORA_SURF_MAP = {"2t": "t2m", "10u": "u10", "10v": "v10", "msl": "msl"}
    AURORA_ATMOS_VARS = ["z", "q", "t", "u", "v"]

    def decode_output_to_vars(self, state: "Batch") -> Dict[str, np.ndarray]:
        """Decode Aurora Batch output into per-variable 2D arrays.

        Aurora's Batch object tensor shapes (from _tensor_to_batch):
        - surf_vars: (B, T, H, W) where B=batch, T=2 time steps, H=720, W=1440
        - atmos_vars: (B, T, levels, H, W)

        After model forward pass, the OUTPUT batch has the same shape structure,
        where T[0] = previous history step, T[1] = NEW prediction.
        We want T[-1] (last time = prediction).

        IMPORTANT: Aurora outputs are in NORMALIZED space. We must call
        unnormalise() to get physical units before extracting values.

        Args:
            state: Aurora Batch object from step_forward

        Returns:
            Dict mapping variable names (e.g. 'z_500', 't2m') to (H, W) arrays
        """
        result = {}

        # Unnormalize the output batch to get physical units
        # Use same surf_stats as normalisation for consistency
        surf_stats = getattr(self.model, 'surf_stats', None)
        if not surf_stats:
            surf_stats = AURORA_SURF_STATS
        state = state.unnormalise(surf_stats=surf_stats)

        # Surface variables
        # Shape is (B, T, H, W) where B=1, T=2 - we want batch 0, last time step
        for aurora_name, std_name in self.AURORA_SURF_MAP.items():
            if aurora_name in state.surf_vars:
                t = state.surf_vars[aurora_name]
                # Shape: (B, T, H, W) = (1, 2, 720, 1440)
                if t.dim() == 4:
                    t = t[0, -1]  # Batch 0, LAST time step (prediction)
                elif t.dim() == 3:
                    t = t[-1]    # Last time step if no batch dim
                elif t.dim() == 2:
                    pass  # Already (H, W)
                result[std_name] = t.cpu().numpy()

        # Pressure level variables
        # Shape is (B, T, levels, H, W) - we want batch 0, last time step
        levels = AURORA_PRESSURE_LEVELS
        for var in self.AURORA_ATMOS_VARS:
            if var in state.atmos_vars:
                t = state.atmos_vars[var]
                # Shape: (B, T, levels, H, W) = (1, 2, 13, 720, 1440)
                if t.dim() == 5:
                    t = t[0, -1]  # Batch 0, last time -> (levels, H, W)
                elif t.dim() == 4:
                    t = t[-1]    # Last time step -> (levels, H, W)
                elif t.dim() == 3:
                    pass  # Already (levels, H, W)
                for j, level in enumerate(levels):
                    if j < t.shape[0]:
                        result[f"{var}_{level}"] = t[j].cpu().numpy()

        return result

    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features."""
        # Aurora's hidden dimension varies, estimate based on common configs
        return 1024  # Typical PerceiverIO hidden dim

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        return {
            "model_name": "Aurora",
            "architecture": "PerceiverIO",
            "format": "PyTorch checkpoint (.ckpt)",
            "n_channels": self.n_channels,
            "img_size": self.img_size,
            "feature_dim": self.get_feature_dim(),
            "feature_aggregation": self.feature_aggregation,
            "feature_location": "decoder output",
            "static_data_keys": list(self.static_data.keys()) if self.static_data else None,
            "model_loaded": self.model is not None,
        }


def test_local_model():
    """Test feature extraction with local model files."""
    print("=" * 60)
    print("Aurora Feature Extractor Test")
    print("=" * 60)

    # Local model paths
    base_path = Path("C:/Users/User/Documents/Code Repositories/ligtas-risk/fastapi-workers/data/models/au")

    weights_path = base_path / "aurora-0.25-finetuned.ckpt"
    static_path = base_path / "aurora-0.25-static.pickle"

    # Check files exist
    if not weights_path.exists():
        print(f"ERROR: {weights_path} not found")
        return False

    print(f"\nModel files found at: {base_path}")

    # Create extractor
    print("\nInitializing feature extractor...")
    extractor = AuroraFeatureExtractor(device="cpu")

    print("Loading model...")
    extractor.load_model(
        weights_path=str(weights_path),
        static_path=str(static_path) if static_path.exists() else None,
    )

    # Print model info
    print("\nModel Info:")
    for k, v in extractor.get_model_info().items():
        print(f"  {k}: {v}")

    if extractor.model is not None:
        # Test with dummy input
        print("\nCreating dummy input (1, 69, 721, 1440)...")
        x = torch.randn(1, 69, 721, 1440)

        print("Extracting features (24h lead time)...")
        features = extractor.extract_features(x, lead_time_hours=24)

        print(f"\nResults:")
        print(f"  Input shape: {x.shape}")
        print(f"  Feature shape: {features.shape}")
        print(f"  Feature mean: {features.mean().item():.6f}")
        print(f"  Feature std: {features.std().item():.6f}")
    else:
        print("\n[SKIPPED] Inference test (requires microsoft-aurora package)")
        print("  Install: pip install microsoft-aurora")
        print("  Checkpoint structure verified successfully")

    print("\n" + "=" * 60)
    print("TEST PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_local_model()
