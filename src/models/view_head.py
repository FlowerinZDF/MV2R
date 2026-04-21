from typing import Optional

import torch
from torch import nn

from src.data.schema import FIXED_VIEWS


class MV2RViewHead(nn.Module):
    """Predict per-view logits from a shared feature representation."""

    def __init__(self, hidden_dim: int, mlp_hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.num_views = len(FIXED_VIEWS)

        if mlp_hidden_dim is None:
            self.classifier = nn.Linear(hidden_dim, self.num_views)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, self.num_views),
            )

    def forward(self, shared_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shared_feature: Tensor with shape [batch_size, hidden_dim]

        Returns:
            Tensor with shape [batch_size, num_views]
        """
        return self.classifier(shared_feature)
