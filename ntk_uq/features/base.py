"""
Base Feature Extractor

Abstract base class for extracting last-layer features from AI weather models.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)


class FeatureExtractor(ABC):
    """
    Base class for model feature extraction.

    Subclasses implement model-specific loading and feature extraction.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
    ):
        """
        Initialize feature extractor.

        Args:
            model_name: Name of the model
            device: Device to run on (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._features = None
        self._hook_handle = None

    @abstractmethod
    def load_model(self, weights_path: str, **kwargs) -> None:
        """
        Load model weights.

        Args:
            weights_path: Path to model weights
            **kwargs: Model-specific arguments
        """
        pass

    @abstractmethod
    def _register_hook(self) -> None:
        """Register forward hook to capture features."""
        pass

    @abstractmethod
    def _get_target_layer(self) -> torch.nn.Module:
        """Get the layer to extract features from."""
        pass

    def extract_features(
        self,
        x: torch.Tensor,
        lead_time_hours: int = 24,
    ) -> torch.Tensor:
        """
        Extract last-layer features from model.

        Args:
            x: Input tensor (batch, channels, lat, lon)
            lead_time_hours: Forecast lead time in hours

        Returns:
            Feature tensor (batch, feature_dim)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        x = x.to(self.device)
        self._features = None

        # Register hook if not already
        if self._hook_handle is None:
            self._register_hook()

        # Forward pass to capture features
        with torch.no_grad():
            # Run model for specified lead time
            n_steps = lead_time_hours // 6  # 6-hour steps
            output = x
            for step in range(n_steps):
                output = self.model(output)

        # Get captured features
        if self._features is None:
            raise RuntimeError("Failed to capture features. Check hook registration.")

        # Flatten spatial dimensions
        batch_size = self._features.shape[0]
        features_flat = self._features.view(batch_size, -1)

        return features_flat

    def cleanup(self) -> None:
        """Remove hooks and free memory."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._features = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    @property
    def feature_dim(self) -> int:
        """Get feature dimension (after flattening)."""
        if self._features is not None:
            return self._features.numel() // self._features.shape[0]
        return None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "feature_dim": self.feature_dim,
        }
