"""
FourCastNetV2 Feature Extractor for NTK Uncertainty Quantification.

Extracts features from the last SFNO block before the decoder.
Architecture: encoder (MLP) -> pos_embed -> 12 SFNO blocks -> decoder (MLP)
Feature extraction point: after SFNO blocks (embed_dim=256 features)

NOTE: weights.tar is actually a ZIP file containing a PyTorch sharded checkpoint.
"""

import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import FeatureExtractor

logger = logging.getLogger(__name__)


class FourCastNetFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for FourCastNetV2 (SFNO architecture).

    FourCastNetV2 uses Spherical Fourier Neural Operator (SFNO) blocks.
    We extract features from the final SFNO block before the decoder.

    Architecture details:
    - Input: (batch, 73, 721, 1440)
    - Encoder MLP: 73 -> 256 channels
    - Positional embedding: (1, 256, 721, 1440)
    - 12 SFNO blocks: (batch, 256, H, W) -> (batch, 256, 721, 1440)
    - Big skip: concatenate input
    - Decoder MLP: 256+73 -> 73 channels

    Feature extraction point: after last SFNO block
    - Raw features: (batch, 256, 721, 1440) = 265M dimensions (too large!)
    - With global avg pool: (batch, 256) = 256 dimensions (tractable)
    """

    def __init__(self, device: str = "cuda"):
        super().__init__("fourcastnetv2", device)
        self.means = None
        self.stds = None
        self._features = None
        self._hook_handle = None

        # Architecture constants
        self.n_channels = 73
        self.embed_dim = 256
        self.img_size = (721, 1440)
        self.num_layers = 12

        # Feature aggregation strategy (dual mode computes both)
        # IMPORTANT: "global_avg" produces identical features across samples!
        # Use "spatial_sample" or "multi_stat" to preserve sample-specific variations
        self.feature_aggregation = "multi_stat"  # "global_avg", "spatial_sample", "multi_stat" (legacy single mode)
        self.spatial_sample_size = 100  # if using spatial_sample

    def load_model(
        self,
        weights_path: str,
        means_path: Optional[str] = None,
        stds_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Load FourCastNetV2 model.

        Args:
            weights_path: Path to weights.tar (actually a ZIP file)
            means_path: Path to global_means.npy
            stds_path: Path to global_stds.npy
        """
        logger.info(f"Loading FourCastNetV2 from {weights_path}")

        # Load normalization stats
        if means_path:
            self.means = self._load_npy(means_path)
            logger.info(f"Loaded means: shape {self.means.shape}")
        if stds_path:
            self.stds = self._load_npy(stds_path)
            logger.info(f"Loaded stds: shape {self.stds.shape}")

        # Create model architecture
        self.model = self._create_model()

        # Load checkpoint weights
        self._load_weights(weights_path)

        self.model.to(self.device)
        self.model.eval()

        # Register feature extraction hook
        self._register_hook()

        logger.info("FourCastNetV2 loaded successfully")
        logger.info(f"  Embed dim: {self.embed_dim}")
        logger.info(f"  Num layers: {self.num_layers}")
        logger.info(f"  Feature aggregation: {self.feature_aggregation}")

    def _create_model(self) -> nn.Module:
        """Create FourierNeuralOperatorNet architecture."""
        from ai_models_fourcastnetv2.fourcastnetv2 import FourierNeuralOperatorNet

        return FourierNeuralOperatorNet(
            spectral_transform="sht",
            filter_type="non-linear",
            img_size=self.img_size,
            scale_factor=6,
            in_chans=self.n_channels,
            out_chans=self.n_channels,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            mlp_mode="distributed",
            mlp_ratio=2.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            num_blocks=8,
            sparsity_threshold=0.0,
            normalization_layer="instance_norm",
            hard_thresholding_fraction=1.0,
            use_complex_kernels=True,
            big_skip=True,
            compression=None,
            rank=128,
            complex_network=True,
            complex_activation="real",
            spectral_layers=3,
        )

    def _load_npy(self, path: str) -> torch.Tensor:
        """Load numpy file from local or GCS path."""
        if path.startswith("gs://"):
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            with fs.open(path, "rb") as f:
                arr = np.load(f)
        else:
            arr = np.load(path)

        # Ensure correct shape for broadcasting
        arr = arr.astype(np.float32)
        if len(arr.shape) == 4:
            arr = arr[:, :self.n_channels, ...]

        return torch.from_numpy(arr).float().to(self.device)

    def _load_weights(self, weights_path: str) -> None:
        """Load weights from ZIP file (weights.tar is actually a ZIP)."""
        if weights_path.startswith("gs://"):
            # Download from GCS first
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = Path(tmpdir) / "weights.tar"
                fs.get(weights_path, str(local_path))
                self._load_weights_from_zip(local_path)
        else:
            self._load_weights_from_zip(Path(weights_path))

    def _load_weights_from_zip(self, zip_path: Path) -> None:
        """Load weights from ZIP file containing sharded PyTorch checkpoint.

        Repacks as ZIP with 'archive/' prefix for torch.load compatibility.
        """
        logger.info(f"Extracting weights from {zip_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmpdir)
                files = zf.namelist()
                logger.info(f"Extracted {len(files)} files: {files[:3]}...")

            # The ZIP contains weights/ subdirectory with sharded checkpoint
            # Structure: weights/data.pkl, weights/data/*, weights/version
            weights_dir = tmpdir_path / "weights"

            if not weights_dir.exists():
                # Fallback: maybe directly in tmpdir
                weights_dir = tmpdir_path

            # Sharded checkpoints need to be repacked as ZIP with 'archive/' prefix
            # This works on all platforms (Windows, Linux, Mac)
            repacked_zip = tmpdir_path / "checkpoint.zip"
            logger.info("Repacking sharded checkpoint as ZIP...")

            with zipfile.ZipFile(repacked_zip, 'w') as zf_out:
                for root, dirs, files_list in os.walk(weights_dir):
                    for file in files_list:
                        file_path = Path(root) / file
                        rel_path = file_path.relative_to(weights_dir)
                        arcname = 'archive/' + str(rel_path).replace(os.sep, '/')
                        zf_out.write(file_path, arcname)

            checkpoint_path = str(repacked_zip)

            # Load checkpoint
            logger.info("Loading checkpoint...")
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False
            )

            # Extract model state dict
            weights = checkpoint["model_state"]

            # Remove module. prefix and excluded keys
            drop_vars = ["module.norm.weight", "module.norm.bias"]
            new_state_dict = {}
            for k, v in weights.items():
                if k in drop_vars:
                    continue
                # Remove "module." prefix
                name = k[7:] if k.startswith("module.") else k
                if name != "ged":  # Skip ged key
                    new_state_dict[name] = v

            # Load into model
            self.model.load_state_dict(new_state_dict)
            logger.info(f"Loaded {len(new_state_dict)} weight tensors")

    def _register_hook(self) -> None:
        """Register forward hook on last SFNO block."""
        def hook_fn(module, input, output):
            self._features = output.detach()

        # Hook the last SFNO block (before decoder)
        target_layer = self.model.blocks[-1]
        self._hook_handle = target_layer.register_forward_hook(hook_fn)
        logger.info(f"Registered hook on blocks[-1] ({type(target_layer).__name__})")

    def _get_target_layer(self) -> nn.Module:
        """Get the layer to extract features from."""
        return self.model.blocks[-1]

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization using loaded means/stds."""
        if self.means is not None and self.stds is not None:
            n_input_channels = x.shape[1]
            n_norm_channels = self.means.shape[1]

            # Move means/stds to same device as input
            means = self.means.to(x.device)
            stds = self.stds.to(x.device)

            if n_input_channels < n_norm_channels:
                # Pad with zeros for missing channels
                logger.warning(
                    f"Channel mismatch: input has {n_input_channels}, "
                    f"expected {n_norm_channels}. Padding with zeros."
                )
                padding = torch.zeros(
                    x.shape[0], n_norm_channels - n_input_channels,
                    x.shape[2], x.shape[3],
                    device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, padding], dim=1)
            elif n_input_channels > n_norm_channels:
                logger.warning(
                    f"Channel mismatch: input has {n_input_channels}, "
                    f"expected {n_norm_channels}. Truncating."
                )
                x = x[:, :n_norm_channels, :, :]

            return (x - means) / (stds + 1e-6)
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Remove normalization."""
        if self.means is not None and self.stds is not None:
            return x * self.stds + self.means
        return x

    # Official ECMWF channel ordering (from ai-models-fourcastnetv2 model.py)
    # Surface (8): 10u, 10v, 100u, 100v, 2t, sp, msl, tcwv
    # Pressure (65): ux13, vx13, zx13, tx13, rx13  (levels: 50->1000)
    FCN_SURFACE_VARS = ["u10", "v10", "u100", "v100", "t2m", "sp", "msl", "tcwv"]
    FCN_PRESSURE_VARS = ["u", "v", "z", "t", "r"]
    FCN_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    def decode_output_to_vars(self, state: torch.Tensor) -> Dict[str, np.ndarray]:
        """Decode FCNv2 output state (73 channels) into per-variable 2D arrays.

        Args:
            state: Model output tensor (batch, 73, H, W) in normalized space

        Returns:
            Dict mapping variable names (e.g. 'z_500', 't2m') to (H, W) arrays
        """
        # Denormalize to physical units
        output = self.denormalize(state)
        output = output[0].cpu().numpy()  # (73, H, W)

        result = {}
        ch = 0
        # Surface variables (first 8 channels)
        for var in self.FCN_SURFACE_VARS:
            result[var] = output[ch]
            ch += 1
        # Pressure level variables: each var at all 13 levels, then next var
        for var in self.FCN_PRESSURE_VARS:
            for level in self.FCN_PRESSURE_LEVELS:
                result[f"{var}_{level}"] = output[ch]
                ch += 1

        return result

    def step_forward(
        self,
        state: torch.Tensor,
        n_steps: int = 1,
        extract_features: bool = True,
        is_normalized: bool = True,
        dual_aggregation: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        Run N autoregressive steps forward and optionally extract features.

        This enables efficient multi-leadtime rollout by continuing from
        a previous state rather than starting from scratch.

        Args:
            state: Current model state (batch, 73, 721, 1440), normalized
            n_steps: Number of 6h steps to run
            extract_features: Whether to extract features at final step
            is_normalized: Whether input state is already normalized
            dual_aggregation: If True, return both global_avg and multi_stat features

        Returns:
            Tuple of (new_state, features) where features is:
            - None if extract_features=False
            - torch.Tensor if dual_aggregation=False
            - Dict[str, torch.Tensor] if dual_aggregation=True
        """
        if not is_normalized:
            state = self.normalize(state)

        state = state.to(self.device)
        self._features = None

        if extract_features and self._hook_handle is None:
            self._register_hook()

        with torch.no_grad():
            output = state
            for step in range(n_steps):
                output = self.model(output)

        features = None
        if extract_features and self._features is not None:
            if dual_aggregation:
                features = self._aggregate_features_dual(self._features)
                for key in features:
                    if torch.isnan(features[key]).any():
                        features[key] = torch.nan_to_num(features[key], nan=0.0)
            else:
                features = self._aggregate_features(self._features)
                if torch.isnan(features).any():
                    features = torch.nan_to_num(features, nan=0.0)

        return output, features

    def extract_features(
        self,
        x: torch.Tensor,
        lead_time_hours: int = 24,
        normalize_input: bool = True,
    ) -> torch.Tensor:
        """
        Extract features from FourCastNetV2.

        Args:
            x: Input tensor (batch, 73, 721, 1440)
            lead_time_hours: Forecast lead time (for rollout)
            normalize_input: Whether to normalize input

        Returns:
            Feature tensor (batch, feature_dim)
        """
        # Handle any NaN in input (defensive check)
        if torch.isnan(x).any():
            nan_count = torch.isnan(x).sum().item()
            logger.warning(f"Input contains {nan_count} NaN values, filling with 0")
            x = torch.nan_to_num(x, nan=0.0)

        if normalize_input:
            x = self.normalize(x)

        # Check for NaN after normalization
        if torch.isnan(x).any():
            nan_count = torch.isnan(x).sum().item()
            logger.warning(f"Normalized input contains {nan_count} NaN values, filling with 0")
            x = torch.nan_to_num(x, nan=0.0)

        x = x.to(self.device)
        self._features = None

        if self._hook_handle is None:
            self._register_hook()

        with torch.no_grad():
            # Autoregressive rollout (6-hour steps)
            n_steps = max(1, lead_time_hours // 6)
            output = x
            for step in range(n_steps):
                output = self.model(output)
                # Hook captures features at each step
                # We keep the last one

        if self._features is None:
            raise RuntimeError("Failed to capture features from hook")

        # Aggregate spatial dimensions
        features = self._aggregate_features(self._features)

        # Final NaN check on features
        if torch.isnan(features).any():
            nan_count = torch.isnan(features).sum().item()
            logger.warning(f"Extracted features contain {nan_count} NaN values, filling with 0")
            features = torch.nan_to_num(features, nan=0.0)

        return features

    def extract_features_dual(
        self,
        x: torch.Tensor,
        lead_time_hours: int = 24,
        normalize_input: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract BOTH global_avg and multi_stat features from FourCastNetV2.

        This enables dual calibration:
        - global_avg (256-dim) -> for spatial uncertainty (per-grid-point)
        - multi_stat (1536-dim) -> for scalar uncertainty (better discrimination)

        Args:
            x: Input tensor (batch, 73, 721, 1440)
            lead_time_hours: Forecast lead time (for rollout)
            normalize_input: Whether to normalize input

        Returns:
            Dict with 'global_avg' and 'multi_stat' feature tensors
        """
        # Handle any NaN in input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        if normalize_input:
            x = self.normalize(x)

        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        x = x.to(self.device)
        self._features = None

        if self._hook_handle is None:
            self._register_hook()

        with torch.no_grad():
            n_steps = max(1, lead_time_hours // 6)
            output = x
            for step in range(n_steps):
                output = self.model(output)

        if self._features is None:
            raise RuntimeError("Failed to capture features from hook")

        # Get both aggregations
        dual_features = self._aggregate_features_dual(self._features)

        # NaN check
        for key in dual_features:
            if torch.isnan(dual_features[key]).any():
                dual_features[key] = torch.nan_to_num(dual_features[key], nan=0.0)

        return dual_features

    def extract_spatial_output(self, state: torch.Tensor) -> np.ndarray:
        """
        Get raw spatial output channels without aggregation.

        Args:
            state: Model output tensor (batch, 73, H, W) in normalized space

        Returns:
            (73, H, W) array in physical units (denormalized)
        """
        output = self.denormalize(state)
        return output[0].cpu().numpy()  # (73, H, W)

    def extract_spatial_features(self) -> Optional[np.ndarray]:
        """
        Get the raw SFNO block features (before aggregation) from the last forward pass.

        Returns:
            (256, H, W) array from the last SFNO block, or None if no features captured
        """
        if self._features is None:
            return None
        return self._features[0].cpu().numpy()  # (256, H, W)

    def _aggregate_features_dual(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute BOTH global_avg and multi_stat aggregations in one pass.

        This is needed for dual calibration:
        - global_avg (256-dim) -> spatial uncertainty (matches per-grid-point dim)
        - multi_stat (1536-dim) -> scalar uncertainty (better discrimination)

        Args:
            features: (batch, embed_dim, H, W) tensor from SFNO blocks

        Returns:
            Dict with 'global_avg' and 'multi_stat' tensors
        """
        batch_size = features.shape[0]

        # Global average pooling: (batch, 256, H, W) -> (batch, 256)
        global_avg = features.mean(dim=(2, 3))

        # Multi-statistic aggregation: mean + std + percentiles
        # (batch, 256, H, W) -> (batch, 256 * 6) = (batch, 1536)
        spatial_mean = features.mean(dim=(2, 3))  # (batch, 256)
        spatial_std = features.std(dim=(2, 3))    # (batch, 256)

        # Flatten spatial dims for percentile computation
        flat = features.view(batch_size, features.shape[1], -1)  # (batch, 256, H*W)
        spatial_min = flat.min(dim=2).values     # (batch, 256)
        spatial_max = flat.max(dim=2).values     # (batch, 256)
        spatial_p25 = flat.quantile(0.25, dim=2) # (batch, 256)
        spatial_p75 = flat.quantile(0.75, dim=2) # (batch, 256)

        multi_stat = torch.cat([
            spatial_mean, spatial_std,
            spatial_min, spatial_max,
            spatial_p25, spatial_p75
        ], dim=1)  # (batch, 256*6 = 1536)

        return {
            'global_avg': global_avg,
            'multi_stat': multi_stat
        }

    def _aggregate_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate spatial dimensions for tractable NTK computation.

        Args:
            features: (batch, embed_dim, H, W) tensor from SFNO blocks

        Returns:
            (batch, feature_dim) tensor
        """
        batch_size = features.shape[0]

        if self.feature_aggregation == "global_avg":
            # Global average pooling: (batch, 256, H, W) -> (batch, 256)
            # WARNING: This produces nearly identical features across samples!
            features = features.mean(dim=(2, 3))

        elif self.feature_aggregation == "multi_stat":
            # Multi-statistic aggregation: mean + std + percentiles
            # This preserves sample-specific variations in the feature distribution
            # (batch, 256, H, W) -> (batch, 256 * 6) = (batch, 1536)
            spatial_mean = features.mean(dim=(2, 3))  # (batch, 256)
            spatial_std = features.std(dim=(2, 3))    # (batch, 256)

            # Flatten spatial dims for percentile computation
            flat = features.view(batch_size, features.shape[1], -1)  # (batch, 256, H*W)
            spatial_min = flat.min(dim=2).values     # (batch, 256)
            spatial_max = flat.max(dim=2).values     # (batch, 256)
            spatial_p25 = flat.quantile(0.25, dim=2) # (batch, 256)
            spatial_p75 = flat.quantile(0.75, dim=2) # (batch, 256)

            features = torch.cat([
                spatial_mean, spatial_std,
                spatial_min, spatial_max,
                spatial_p25, spatial_p75
            ], dim=1)  # (batch, 256*6 = 1536)

        elif self.feature_aggregation == "spatial_sample":
            # Deterministic spatial sampling for reproducibility
            H, W = features.shape[2], features.shape[3]
            n_samples = min(self.spatial_sample_size, H * W)

            # Fixed sampling pattern (same across batches)
            torch.manual_seed(42)
            flat_idx = torch.randperm(H * W)[:n_samples]
            h_idx = flat_idx // W
            w_idx = flat_idx % W

            # Sample: (batch, embed_dim, n_samples)
            sampled = features[:, :, h_idx, w_idx]
            # Flatten: (batch, embed_dim * n_samples)
            features = sampled.view(batch_size, -1)

        elif self.feature_aggregation == "full":
            # No reduction - flatten all (WARNING: 265M dimensions!)
            features = features.view(batch_size, -1)

        else:
            raise ValueError(f"Unknown aggregation: {self.feature_aggregation}")

        return features

    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features."""
        if self.feature_aggregation == "global_avg":
            return self.embed_dim  # 256
        elif self.feature_aggregation == "multi_stat":
            return self.embed_dim * 6  # 256 * 6 = 1536 (mean, std, min, max, p25, p75)
        elif self.feature_aggregation == "spatial_sample":
            return self.embed_dim * self.spatial_sample_size
        elif self.feature_aggregation == "full":
            return self.embed_dim * self.img_size[0] * self.img_size[1]
        else:
            raise ValueError(f"Unknown aggregation: {self.feature_aggregation}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        return {
            "model_name": "FourCastNetV2",
            "architecture": "SFNO (Spherical Fourier Neural Operator)",
            "n_channels": self.n_channels,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "img_size": self.img_size,
            "feature_dim": self.get_feature_dim(),
            "feature_aggregation": self.feature_aggregation,
            "feature_location": "after last SFNO block, before decoder",
        }


def test_local_model():
    """Test feature extraction with local model files."""
    print("=" * 60)
    print("FourCastNetV2 Feature Extractor Test")
    print("=" * 60)

    # Local model paths
    base_path = Path("C:/Users/User/Documents/Code Repositories/ligtas-risk/fastapi-workers/data/models/fcnv2")

    weights_path = base_path / "weights.tar"
    means_path = base_path / "global_means.npy"
    stds_path = base_path / "global_stds.npy"

    # Check files exist
    for p in [weights_path, means_path, stds_path]:
        if not p.exists():
            print(f"ERROR: {p} not found")
            return False

    print(f"\nModel files found at: {base_path}")

    # Create extractor
    print("\nInitializing feature extractor...")
    extractor = FourCastNetFeatureExtractor(device="cpu")

    print("Loading model...")
    extractor.load_model(
        weights_path=str(weights_path),
        means_path=str(means_path),
        stds_path=str(stds_path),
    )

    # Print model info
    print("\nModel Info:")
    for k, v in extractor.get_model_info().items():
        print(f"  {k}: {v}")

    # Test with dummy input
    print("\nCreating dummy input (1, 73, 721, 1440)...")
    x = torch.randn(1, 73, 721, 1440)

    print("Extracting features (24h lead time)...")
    features = extractor.extract_features(x, lead_time_hours=24)

    print(f"\nResults:")
    print(f"  Input shape: {x.shape}")
    print(f"  Feature shape: {features.shape}")
    print(f"  Feature mean: {features.mean().item():.6f}")
    print(f"  Feature std: {features.std().item():.6f}")
    print(f"  Feature min: {features.min().item():.6f}")
    print(f"  Feature max: {features.max().item():.6f}")

    print("\n" + "=" * 60)
    print("TEST PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_local_model()
