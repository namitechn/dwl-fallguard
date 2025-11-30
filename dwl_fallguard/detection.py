import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Union


class FallDetector(nn.Module):
    """Bayesian Neural Network for uncertainty-aware fall detection"""

    def __init__(
        self,
        state_dim: int = 42,
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        n_classes: int = 2
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(self, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with Monte Carlo Dropout"""
        batch_size = x.shape[0]
        predictions = []

        self.train()  # Enable dropout for uncertainty

        for _ in range(n_samples):
            features = self.feature_extractor(x)
            logits = self.classifier(features)
            probs = F.softmax(logits, dim=-1)
            predictions.append(probs)

        predictions = torch.stack(predictions)  # [n_samples, batch, n_classes]
        mean_predictions = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)  # Variance across samples

        return mean_predictions, uncertainty

    def detect_fall(
        self,
        state: torch.Tensor,
        fall_threshold: float = 0.7,
        uncertainty_threshold: float = 0.15
    ) -> Dict[str, Union[bool, float]]:
        """Detect fall with confidence thresholding"""
        self.eval()
        with torch.no_grad():
            predictions, uncertainty = self(state.unsqueeze(0))

        fall_prob = predictions[0, 1].item()
        uncertainty_score = uncertainty[0, 1].item()
        confidence = 1.0 - uncertainty_score

        is_fall = fall_prob > fall_threshold and uncertainty_score < uncertainty_threshold

        return {
            "is_fall": bool(is_fall),
            "fall_probability": float(fall_prob),
            "uncertainty": float(uncertainty_score),
            "confidence": float(confidence)
        }