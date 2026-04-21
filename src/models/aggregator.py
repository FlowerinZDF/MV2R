from typing import Literal, Optional

import torch
from torch import nn


class MV2RAggregator(nn.Module):
    """Predict overall logits from either shared features or view logits."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        input_type: Literal["shared", "view_logits"] = "shared",
        mlp_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if input_type not in {"shared", "view_logits"}:
            raise ValueError("input_type must be 'shared' or 'view_logits'")

        self.input_type = input_type

        if mlp_hidden_dim is None:
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape [batch_size, input_dim]

        Returns:
            Tensor with shape [batch_size, num_classes]
        """
        return self.classifier(x)
