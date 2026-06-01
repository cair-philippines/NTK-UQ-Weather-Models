"""
AIFS Feature Extractor for NTK Uncertainty Quantification.

ECMWF's Artificial Intelligence Forecasting System (AIFS).
Uses Anemoi framework with SimpleRunner and CDS for ERA5 data.
Handles static fields (lsm, z, slor, sdor) separately.

Required packages:
    pip install anemoi-inference>=0.8.0 anemoi-models==0.9.7
    pip install ecmwf-opendata earthkit-data earthkit-regrid
    pip install flash-attn (requires CUDA 11.7+)
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn

from .base import FeatureExtractor

logger = logging.getLogger(__name__)


class AIFSFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for AIFS (ECMWF's operational AI model).

    Uses SimpleRunner with CDS input for historical ERA5 data.
    Extracts features from the transformer processor output.
    Handles static fields separately to avoid CDS issues.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__("aifs", device)

        # Architecture constants
        self.img_size = (721, 1440)
        self.n_pressure_levels = 13
        self.hidden_dim = 1024

        # Feature extraction settings (dual mode computes both)
        self.feature_aggregation = "multi_stat"
        self._features = None
        self._hook_handle = None
        self.cds_input = None
        self._static_fields = None  # Cache for static fields

    def load_model(self, weights_path: str, **kwargs) -> None:
        """Load AIFS model from checkpoint."""
        logger.info(f"Loading AIFS from {weights_path}")
        if weights_path.startswith("ecmwf/"):
            self._load_from_huggingface(weights_path)
        else:
            self._load_from_checkpoint(weights_path)

    def _load_from_huggingface(self, model_id: str) -> None:
        """Load model from HuggingFace using SimpleRunner."""
        try:
            from huggingface_hub import snapshot_download
            from anemoi.inference.runners.simple import SimpleRunner
            from anemoi.inference.inputs import create_input

            logger.info(f"Downloading from HuggingFace: {model_id}")
            model_path = snapshot_download(repo_id=model_id)
            logger.info(f"Downloaded to: {model_path}")

            ckpt_file = None
            for f in os.listdir(model_path):
                if f.endswith('.ckpt'):
                    ckpt_file = os.path.join(model_path, f)
                    break

            if ckpt_file is None:
                raise FileNotFoundError(f"No .ckpt file found in {model_path}")

            logger.info(f"Loading checkpoint: {ckpt_file}")

            # Use SimpleRunner which handles coupled forcings gracefully
            self.runner = SimpleRunner(ckpt_file, device=self.device)
            self.model = self.runner.model
            self.model.eval()  # Use eval() not eval_mode()

            # Create CDS input for ERA5 data
            config = {'cds': {'dataset': 'reanalysis-era5-complete'}}
            self.cds_input = create_input(self.runner, config)
            logger.info("CDS input source configured")

            # Fetch and cache static fields
            self._fetch_static_fields()

            self._register_hook()
            logger.info("AIFS loaded from HuggingFace successfully")

        except ImportError as e:
            logger.warning(f"Import error: {e}")
            logger.info("Install: pip install anemoi-inference>=0.8.0 anemoi-models==0.9.7")
            self.model = None
            self.runner = None
        except Exception as e:
            logger.warning(f"Could not load AIFS from HuggingFace: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.runner = None

    def _load_from_checkpoint(self, weights_path: str) -> None:
        """Load model from local checkpoint."""
        try:
            from anemoi.inference.runners.simple import SimpleRunner
            from anemoi.inference.inputs import create_input

            logger.info(f"Loading from checkpoint: {weights_path}")

            if weights_path.startswith("gs://"):
                import gcsfs
                import tempfile
                fs = gcsfs.GCSFileSystem()
                with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
                    fs.get(weights_path, f.name)
                    weights_path = f.name

            self.runner = SimpleRunner(weights_path, device=self.device)
            self.model = self.runner.model
            self.model.eval()  # Use eval() not eval_mode()

            config = {'cds': {'dataset': 'reanalysis-era5-complete'}}
            self.cds_input = create_input(self.runner, config)

            # Fetch and cache static fields
            self._fetch_static_fields()

            self._register_hook()
            logger.info("AIFS loaded from checkpoint successfully")

        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.runner = None

    def _fetch_static_fields(self) -> None:
        """Fetch and cache static fields from CDS."""
        import cdsapi
        import earthkit.data as ekd

        cache_dir = Path.home() / ".cache" / "ntk-uq-weather" / "aifs"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "static_fields_n320.grib"

        if cache_file.exists():
            logger.info("Loading cached static fields")
            self._static_fields = ekd.from_source("file", str(cache_file))
            return

        logger.info("Fetching static fields from CDS (one-time download)...")
        try:
            client = cdsapi.Client()
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "land_sea_mask",
                        "geopotential",
                        "slope_of_sub_gridscale_orography",
                        "standard_deviation_of_orography",
                    ],
                    "year": "2021",
                    "month": "01",
                    "day": "01",
                    "time": "00:00",
                    "grid": [0.25, 0.25],  # AIFS uses 0.25 degree grid
                    "format": "grib",
                },
                str(cache_file)
            )
            self._static_fields = ekd.from_source("file", str(cache_file))
            logger.info("Static fields cached successfully")
        except Exception as e:
            logger.warning(f"Could not fetch static fields: {e}")
            self._static_fields = None

    def _register_hook(self) -> None:
        """Register forward hook on transformer processor output."""
        if self.model is None:
            return

        def hook_fn(module, input, output):
            self._features = output.detach()

        target_layer = self._get_target_layer()
        if target_layer is not None:
            self._hook_handle = target_layer.register_forward_hook(hook_fn)
            logger.info(f"Registered hook on {type(target_layer).__name__}")

    def _get_target_layer(self) -> Optional[nn.Module]:
        """Get the transformer processor output layer."""
        if self.model is None:
            return None

        # For AIFS, the processor is inside model.model.processor
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'processor'):
            return self.model.model.processor

        layer_names = ["processor", "transformer"]
        for name in layer_names:
            if hasattr(self.model, name):
                return getattr(self.model, name)

        for name, module in self.model.named_modules():
            if "processor" in name.lower():
                return module

        logger.warning("Could not find processor layer for hook")
        return None

    def extract_features(
        self,
        x: torch.Tensor,
        lead_time_hours: int = 24,
        normalize_input: bool = True,
    ) -> torch.Tensor:
        """AIFS requires date-based input. Use extract_features_from_date() instead."""
        raise RuntimeError(
            "AIFS requires date-based input. Use extract_features_from_date(date, lead_time_hours) instead."
        )

    def prepare_state_from_date(self, date: str) -> Any:
        """
        Prepare input state from date using CDS.

        Args:
            date: ISO format date string (e.g., "2021-06-15T00:00:00")

        Returns:
            input_state: State dict for runner.run()
        """
        if self.runner is None or self.cds_input is None:
            raise RuntimeError("AIFS runner or CDS input not loaded.")

        # Parse date string
        if isinstance(date, str):
            dt = datetime.fromisoformat(date.replace('Z', '+00:00'))
        else:
            dt = date

        logger.info(f"Loading ERA5 data for {dt} via CDS...")
        input_state = self.cds_input.create_input_state(date=dt)

        # Inject static fields into input state
        if self._static_fields is not None:
            self._add_static_fields_to_state(input_state)

        return input_state

    def _move_state_to_device(self, input_state: Any) -> Any:
        """
        Recursively move all tensors in input_state to the target device.

        This fixes device mismatch errors when CDS input returns CPU tensors.
        """
        if input_state is None:
            return None

        if isinstance(input_state, torch.Tensor):
            return input_state.to(self.device)
        elif isinstance(input_state, dict):
            return {k: self._move_state_to_device(v) for k, v in input_state.items()}
        elif isinstance(input_state, (list, tuple)):
            return type(input_state)(self._move_state_to_device(item) for item in input_state)
        else:
            # Not a tensor/dict/list, return as-is (e.g., strings, ints, earthkit objects)
            return input_state

    def step_forward_multi_lt(
        self,
        input_state: Any,
        lead_times: list,
        return_predictions: bool = False,
        dual_aggregation: bool = False,
    ) -> Dict[int, Any]:
        """
        Run autoregressive rollout and capture features at multiple lead times.

        AIFS uses 6h steps. This runs the full rollout and captures features
        at each specified lead time checkpoint.

        Args:
            input_state: State from prepare_state_from_date()
            lead_times: List of lead times in hours (must be multiples of 6)
            return_predictions: If True, also return forecast predictions at each lead time
            dual_aggregation: If True, return both global_avg and multi_stat features

        Returns:
            Dict mapping lead_time -> feature tensor/dict (if return_predictions=False)
            Dict mapping lead_time -> (features, predictions) tuple (if return_predictions=True)
        """
        if self.runner is None:
            raise RuntimeError("AIFS runner not loaded.")

        # Move input_state to device to prevent device mismatch
        input_state = self._move_state_to_device(input_state)

        # Convert lead times to step indices (AIFS uses 6h steps)
        lead_time_to_step = {lt: lt // 6 for lt in lead_times}
        max_lead_time = max(lead_times)
        target_steps = set(lead_time_to_step.values())

        # Track features and predictions at each step
        step_features = {}
        step_predictions = {}
        captured_at_step = []

        def capture_hook(module, input, output):
            captured_at_step.append(output.detach().clone())

        target_layer = self._get_target_layer()
        hook_handle = None
        if target_layer is not None:
            hook_handle = target_layer.register_forward_hook(capture_hook)

        try:
            logger.info(f"Running AIFS rollout to t+{max_lead_time}h, capturing at {lead_times}h")
            step_count = 0

            for state in self.runner.run(input_state=input_state, lead_time=max_lead_time):
                step_count += 1
                # After each forward pass, check if we should capture
                if step_count in target_steps:
                    if captured_at_step:
                        if dual_aggregation:
                            features = self._aggregate_features_dual(captured_at_step[-1])
                            # Ensure all features are on the correct device
                            features = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                       for k, v in features.items()}
                        else:
                            features = self._aggregate_features(captured_at_step[-1])
                            # Ensure features are on the correct device
                            if isinstance(features, torch.Tensor):
                                features = features.to(self.device)
                        step_features[step_count] = features
                        logger.debug(f"Captured features at step {step_count}")
                    if return_predictions:
                        # Store the state for decoding predictions
                        step_predictions[step_count] = state

                    # Clear captured features list to free memory
                    if len(captured_at_step) > 1:
                        # Keep only the last one, delete older captures
                        captured_at_step = captured_at_step[-1:]

            logger.info(f"Completed {step_count} steps, captured {len(step_features)} feature sets")

        finally:
            if hook_handle is not None:
                hook_handle.remove()

            # Clear temporary storage and move everything to device to avoid residual CPU tensors
            import gc
            for item in captured_at_step:
                if isinstance(item, torch.Tensor):
                    del item
            captured_at_step.clear()

            # Clear any residual tensors from both CPU and GPU
            for features in step_features.values():
                if isinstance(features, dict):
                    for v in features.values():
                        if isinstance(v, torch.Tensor) and v.device.type == 'cpu':
                            v.to(self.device)  # Move any CPU tensors to device
                elif isinstance(features, torch.Tensor) and features.device.type == 'cpu':
                    features.to(self.device)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Map step indices back to lead times
        result = {}
        for lt in lead_times:
            step_idx = lead_time_to_step[lt]
            if return_predictions:
                features = step_features.get(step_idx)
                preds = step_predictions.get(step_idx)
                result[lt] = (features, preds)
            else:
                if step_idx in step_features:
                    result[lt] = step_features[step_idx]
                else:
                    logger.warning(f"No features captured for lead_time={lt}h (step {step_idx})")

        return result

    def extract_features_from_date(
        self,
        date: str,
        lead_time_hours: int = 24,
    ) -> torch.Tensor:
        """
        Extract features from AIFS using date-based input.

        The CDS input fetches ERA5 data including:
        - Prognostic variables (temperature, wind, humidity, etc.)
        - Static fields are added separately from cached GRIB file

        Args:
            date: ISO format date string (e.g., "2021-06-15T00:00:00")
            lead_time_hours: Forecast lead time in hours

        Returns:
            Feature tensor (1, feature_dim)
        """
        if self.runner is None or self.cds_input is None:
            raise RuntimeError("AIFS runner or CDS input not loaded.")

        self._features = None
        captured_features = []

        def capture_hook(module, input, output):
            captured_features.append(output.detach().clone())

        target_layer = self._get_target_layer()
        hook_handle = None
        if target_layer is not None:
            hook_handle = target_layer.register_forward_hook(capture_hook)

        try:
            # Parse date string
            if isinstance(date, str):
                dt = datetime.fromisoformat(date.replace('Z', '+00:00'))
            else:
                dt = date

            logger.info(f"Loading ERA5 data for {dt} via CDS...")
            input_state = self.cds_input.create_input_state(date=dt)

            # Inject static fields into input state
            if self._static_fields is not None:
                self._add_static_fields_to_state(input_state)

            logger.info(f"Running AIFS for lead_time={lead_time_hours}h")
            step_count = 0
            for state in self.runner.run(input_state=input_state, lead_time=lead_time_hours):
                step_count += 1

            logger.info(f"Completed {step_count} steps")

        finally:
            if hook_handle is not None:
                hook_handle.remove()

        if not captured_features:
            raise RuntimeError("Failed to capture features from hook during run")

        self._features = captured_features[-1]
        features = self._aggregate_features(self._features)
        return features

    def extract_features_from_date_dual(
        self,
        date: str,
        lead_time_hours: int = 24,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract BOTH global_avg and multi_stat features from AIFS.

        This enables dual calibration:
        - global_avg (1024-dim) -> for spatial uncertainty (per-grid-point)
        - multi_stat (6144-dim) -> for scalar uncertainty (better discrimination)

        Args:
            date: ISO format date string (e.g., "2021-06-15T00:00:00")
            lead_time_hours: Forecast lead time in hours

        Returns:
            Dict with 'global_avg' and 'multi_stat' feature tensors
        """
        if self.runner is None or self.cds_input is None:
            raise RuntimeError("AIFS runner or CDS input not loaded.")

        captured_features = []

        def capture_hook(module, input, output):
            captured_features.append(output.detach().clone())

        target_layer = self._get_target_layer()
        hook_handle = None
        if target_layer is not None:
            hook_handle = target_layer.register_forward_hook(capture_hook)

        try:
            if isinstance(date, str):
                dt = datetime.fromisoformat(date.replace('Z', '+00:00'))
            else:
                dt = date

            logger.info(f"Loading ERA5 data for {dt} via CDS...")
            input_state = self.cds_input.create_input_state(date=dt)

            if self._static_fields is not None:
                self._add_static_fields_to_state(input_state)

            logger.info(f"Running AIFS for lead_time={lead_time_hours}h (dual aggregation)")
            step_count = 0
            for state in self.runner.run(input_state=input_state, lead_time=lead_time_hours):
                step_count += 1

            logger.info(f"Completed {step_count} steps")

        finally:
            if hook_handle is not None:
                hook_handle.remove()

        if not captured_features:
            raise RuntimeError("Failed to capture features from hook during run")

        self._features = captured_features[-1]
        dual_features = self._aggregate_features_dual(self._features)
        return dual_features

    def _add_static_fields_to_state(self, input_state: dict) -> None:
        """Add static fields to the input state for AIFS."""
        if self._static_fields is None:
            return

        # Variable name mapping from GRIB shortName to AIFS expected names
        var_mapping = {
            "lsm": "lsm",      # land-sea mask
            "z": "z",          # geopotential at surface
            "slor": "slor",    # slope of sub-gridscale orography
            "sdor": "sdor",    # standard deviation of orography
        }

        for field in self._static_fields:
            short_name = field.metadata("shortName")
            if short_name in var_mapping:
                aifs_name = var_mapping[short_name]
                data = field.to_numpy()
                # AIFS expects (2, ngrid) for static fields (2 time steps)
                if "fields" in input_state and aifs_name not in input_state.get("fields", {}):
                    if "fields" not in input_state:
                        input_state["fields"] = {}
                    input_state["fields"][aifs_name] = np.stack([data.flatten(), data.flatten()], axis=0)
                    logger.debug(f"Added static field {aifs_name} to input state")

    def extract_spatial_features(self) -> Optional[torch.Tensor]:
        """
        Get the raw processor features (before aggregation) from the last forward pass.

        Returns:
            Raw processor output tensor, or None if no features captured.
            Shape depends on AIFS architecture (typically (1, nodes, channels)).
        """
        if self._features is None:
            return None
        return self._features

    def _aggregate_features_dual(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute BOTH global_avg and multi_stat aggregations in one pass.

        This is needed for dual calibration:
        - global_avg (1024-dim) -> spatial uncertainty (matches per-grid-point dim)
        - multi_stat (6144-dim) -> scalar uncertainty (better discrimination)

        Args:
            features: Tensor from processor (varies by architecture)

        Returns:
            Dict with 'global_avg' and 'multi_stat' tensors
        """
        # Ensure float32 for quantile() and other ops (AIFS may output bfloat16)
        if features.dtype not in (torch.float32, torch.float64):
            features = features.float()
        batch_size = features.shape[0] if features.ndim > 1 else 1

        if features.ndim == 4:  # (batch, channels, H, W)
            global_avg = features.mean(dim=(2, 3))

            spatial_mean = features.mean(dim=(2, 3))
            spatial_std = features.std(dim=(2, 3))
            flat = features.view(batch_size, features.shape[1], -1)
            spatial_min = flat.min(dim=2).values
            spatial_max = flat.max(dim=2).values
            spatial_p25 = flat.quantile(0.25, dim=2)
            spatial_p75 = flat.quantile(0.75, dim=2)
            multi_stat = torch.cat([
                spatial_mean, spatial_std, spatial_min, spatial_max, spatial_p25, spatial_p75
            ], dim=1)

        elif features.ndim == 3:  # (batch, nodes, channels) - AIFS GNN format
            global_avg = features.mean(dim=1)

            spatial_mean = features.mean(dim=1)
            spatial_std = features.std(dim=1)
            spatial_min = features.min(dim=1).values
            spatial_max = features.max(dim=1).values
            spatial_p25 = features.quantile(0.25, dim=1)
            spatial_p75 = features.quantile(0.75, dim=1)
            multi_stat = torch.cat([
                spatial_mean, spatial_std, spatial_min, spatial_max, spatial_p25, spatial_p75
            ], dim=1)

        elif features.ndim == 2:  # (nodes, channels)
            global_avg = features.mean(dim=0, keepdim=True)

            spatial_mean = features.mean(dim=0, keepdim=True)
            spatial_std = features.std(dim=0, keepdim=True)
            spatial_min = features.min(dim=0, keepdim=True).values
            spatial_max = features.max(dim=0, keepdim=True).values
            spatial_p25 = features.quantile(0.25, dim=0, keepdim=True)
            spatial_p75 = features.quantile(0.75, dim=0, keepdim=True)
            multi_stat = torch.cat([
                spatial_mean, spatial_std, spatial_min, spatial_max, spatial_p25, spatial_p75
            ], dim=1)

        else:
            global_avg = features.view(1, -1)
            multi_stat = features.view(1, -1)

        return {
            'global_avg': global_avg,
            'multi_stat': multi_stat
        }

    def _aggregate_features(self, features: torch.Tensor) -> torch.Tensor:
        """Aggregate spatial/node dimensions using selected method."""
        # Ensure float32 for quantile() and other ops (AIFS may output bfloat16)
        if features.dtype not in (torch.float32, torch.float64):
            features = features.float()
        batch_size = features.shape[0] if features.ndim > 1 else 1

        if self.feature_aggregation == "global_avg":
            if features.ndim == 4:
                features = features.mean(dim=(2, 3))
            elif features.ndim == 3:
                features = features.mean(dim=1)
            elif features.ndim == 2:
                features = features.mean(dim=0, keepdim=True)
            else:
                features = features.view(1, -1)

        elif self.feature_aggregation == "multi_stat":
            # Multi-stat aggregation: mean, std, min, max, p25, p75 per channel
            # This captures distribution info, avoiding identical features
            if features.ndim == 4:  # (batch, channels, H, W)
                spatial_mean = features.mean(dim=(2, 3))
                spatial_std = features.std(dim=(2, 3))
                flat = features.view(batch_size, features.shape[1], -1)
                spatial_min = flat.min(dim=2).values
                spatial_max = flat.max(dim=2).values
                spatial_p25 = flat.quantile(0.25, dim=2)
                spatial_p75 = flat.quantile(0.75, dim=2)
                features = torch.cat([
                    spatial_mean, spatial_std, spatial_min, spatial_max, spatial_p25, spatial_p75
                ], dim=1)
            elif features.ndim == 3:  # (batch, nodes, channels) - AIFS GNN format
                spatial_mean = features.mean(dim=1)
                spatial_std = features.std(dim=1)
                spatial_min = features.min(dim=1).values
                spatial_max = features.max(dim=1).values
                spatial_p25 = features.quantile(0.25, dim=1)
                spatial_p75 = features.quantile(0.75, dim=1)
                features = torch.cat([
                    spatial_mean, spatial_std, spatial_min, spatial_max, spatial_p25, spatial_p75
                ], dim=1)
            elif features.ndim == 2:  # (nodes, channels)
                spatial_mean = features.mean(dim=0, keepdim=True)
                spatial_std = features.std(dim=0, keepdim=True)
                spatial_min = features.min(dim=0, keepdim=True).values
                spatial_max = features.max(dim=0, keepdim=True).values
                spatial_p25 = features.quantile(0.25, dim=0, keepdim=True)
                spatial_p75 = features.quantile(0.75, dim=0, keepdim=True)
                features = torch.cat([
                    spatial_mean, spatial_std, spatial_min, spatial_max, spatial_p25, spatial_p75
                ], dim=1)
            else:
                features = features.view(1, -1)
        else:
            features = features.view(batch_size, -1)

        return features

    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features."""
        if self.feature_aggregation == "multi_stat":
            return self.hidden_dim * 6  # 1024 * 6 = 6144
        return self.hidden_dim

    def decode_output_to_vars(self, state) -> Dict[str, np.ndarray]:
        """
        Decode AIFS output state into per-variable 2D arrays.

        Args:
            state: Dict from runner.run() iterator with keys:
                   'fields' (dict of var_name -> ClonedGribField on reduced Gaussian grid),
                   'latitudes', 'longitudes' (1D arrays of grid point coords),
                   '_grib_templates_for_output' (dict of var_name -> ClonedGribField)

        Returns:
            Dict mapping variable names (e.g. 't2m', 'z_500') to (H, W) arrays on 0.25 deg grid
        """
        result = {}
        if state is None or not isinstance(state, dict):
            return result

        fields = state.get('fields', {})
        lats = state.get('latitudes')
        lons = state.get('longitudes')

        if not fields:
            return result

        if lats is None or lons is None:
            logger.warning("No lat/lon coordinates in state, cannot regrid")
            return result

        # Variable name mapping from AIFS names to our standard names
        var_mapping = {
            '2t': 't2m',
            '10u': 'u10',
            '10v': 'v10',
            'msl': 'msl',
            'sp': 'sp',
            'tcwv': 'tcwv',
        }

        # Pressure level variable prefixes (AIFS uses format like 't_500', 'z_850')
        pl_prefixes = {'t': 't', 'u': 'u', 'v': 'v', 'z': 'z', 'q': 'q'}

        # Target variables we care about
        target_vars = {'t2m', 'u10', 'v10', 'msl', 'z_500', 't_850', 'sp'}

        # Build target regular grid (0.25 degree)
        target_lats = np.linspace(90, -90, 721)
        target_lons = np.linspace(0, 359.75, 1440)
        target_lon_grid, target_lat_grid = np.meshgrid(target_lons, target_lats)

        try:
            from scipy.interpolate import griddata

            # Convert lat/lon to numpy if needed
            if hasattr(lats, 'to_numpy'):
                lats = lats.to_numpy()
            if hasattr(lons, 'to_numpy'):
                lons = lons.to_numpy()
            lats = np.asarray(lats).flatten()
            lons = np.asarray(lons).flatten()

            # Normalize longitudes to [0, 360)
            lons = lons % 360.0

            for aifs_name, field in fields.items():
                # Map variable name
                our_name = None

                # Check surface variables
                if aifs_name in var_mapping:
                    our_name = var_mapping[aifs_name]

                # Check pressure level variables (format: 't_500', 'u_850', etc.)
                elif '_' in aifs_name:
                    parts = aifs_name.split('_')
                    if len(parts) == 2:
                        var_prefix, level = parts
                        if var_prefix in pl_prefixes:
                            our_name = f"{pl_prefixes[var_prefix]}_{level}"

                if our_name is None or our_name not in target_vars:
                    continue

                try:
                    # Extract numpy data from field (ClonedGribField or similar)
                    if hasattr(field, 'to_numpy'):
                        values = field.to_numpy().flatten()
                    elif hasattr(field, 'values'):
                        values = np.asarray(field.values()).flatten()
                    elif isinstance(field, np.ndarray):
                        values = field.flatten()
                    else:
                        logger.warning(f"Cannot extract data from {aifs_name}: type={type(field)}")
                        continue

                    # Regrid using scipy griddata (linear interpolation)
                    points = np.column_stack([lons, lats])
                    regridded = griddata(
                        points, values,
                        (target_lon_grid, target_lat_grid),
                        method='linear'
                    )

                    if regridded is not None and regridded.shape == (721, 1440):
                        # Fill NaN values (edges) with nearest neighbor
                        if np.any(np.isnan(regridded)):
                            regridded_nn = griddata(
                                points, values,
                                (target_lon_grid, target_lat_grid),
                                method='nearest'
                            )
                            regridded = np.where(np.isnan(regridded), regridded_nn, regridded)
                        result[our_name] = regridded.astype(np.float32)
                        logger.debug(f"Regridded {aifs_name} -> {our_name}: {result[our_name].shape}")
                    else:
                        logger.warning(f"Unexpected regrid result for {aifs_name}: {regridded.shape if regridded is not None else 'None'}")
                except Exception as e:
                    logger.warning(f"Regrid failed for {aifs_name}: {e}")
                    continue

        except ImportError as e:
            logger.warning(f"Import error for regridding: {e}")
        except Exception as e:
            logger.warning(f"Error decoding AIFS output: {e}")
            import traceback
            traceback.print_exc()

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        return {
            "model_name": "AIFS",
            "architecture": "GNN Encoder + Transformer Processor + GNN Decoder",
            "huggingface": "ecmwf/aifs-single-1.1",
            "feature_dim": self.get_feature_dim(),
            "feature_aggregation": self.feature_aggregation,
            "model_loaded": self.model is not None,
            "cds_configured": self.cds_input is not None,
            "static_fields_cached": self._static_fields is not None,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extractor = AIFSFeatureExtractor(device="cuda")
    extractor.load_model("ecmwf/aifs-single-1.1")
    print(extractor.get_model_info())
