import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class DWLExtended(nn.Module):
    """Extended Denoising World Model for robust dynamics prediction"""

    def __init__(
        self,
        state_dim: int = 42,
        action_dim: int = 12,
        hidden_dim: int = 512,
        num_layers: int = 4
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Input encoding
        self.input_proj = nn.Linear(state_dim + action_dim, hidden_dim)

        # Recurrent layers for temporal modeling
        self.rnn_layers = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Prediction heads
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim)
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state and uncertainty"""
        # Encode input
        input_features = torch.cat([state, action], dim=-1)
        hidden = F.relu(self.input_proj(input_features))

        # Process through RNN layers
        for rnn_layer in self.rnn_layers:
            hidden = rnn_layer(hidden)

        # Predict dynamics
        next_state = self.dynamics_head(hidden)

        # Predict uncertainty (output is log-variance)
        logvar = self.uncertainty_head(hidden).squeeze(-1)  # [batch]
        uncertainty = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ²))

        return next_state, uncertainty

    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state_true: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute negative log-likelihood loss with uncertainty weighting.
        Returns a scalar total loss and diagnostic metrics.
        """
        next_state_pred, uncertainty = self(state, action)  # uncertainty = σ

        eps = 1e-8
        variance = uncertainty ** 2 + eps  # σ² + ε for numerical stability

        # Per-dimension squared error
        sq_err = (next_state_true - next_state_pred) ** 2  # [batch, state_dim]

        # Negative log-likelihood for heteroscedastic Gaussian:
        # NLL = 0.5 * (sq_err / σ² + log(σ²))
        nll_per_dim = 0.5 * (sq_err / variance.unsqueeze(-1) + torch.log(variance.unsqueeze(-1)))
        nll = nll_per_dim.sum(dim=-1).mean()  # scalar: mean over batch, sum over state

        # Optional: uncertainty regularization to avoid collapse
        uncertainty_reg = 0.01 * uncertainty.mean()

        total_loss = nll + uncertainty_reg

        return total_loss, {
            "nll_loss": nll.item(),
            "uncertainty_reg": uncertainty_reg.item(),
            "mean_uncertainty": uncertainty.mean().item()
        }