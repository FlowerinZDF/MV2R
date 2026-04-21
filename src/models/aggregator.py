from typing import Literal, Optional

import torch
from torch import nn


class MV2RAggregator(nn.Module):
    """Predict overall logits from shared features, view logits, or both."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        input_type: Literal["shared", "view_logits", "shared_and_view_logits"] = "shared",
        mlp_hidden_dim: Optional[int] = None,
        view_logits_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        valid_input_types = {"shared", "view_logits", "shared_and_view_logits"}
        if input_type not in valid_input_types:
            raise ValueError(
                "input_type must be one of: 'shared', 'view_logits', 'shared_and_view_logits'"
            )

        if input_type == "shared_and_view_logits" and view_logits_dim is None:
            raise ValueError("view_logits_dim must be provided when input_type='shared_and_view_logits'")

        self.input_type = input_type
        self.view_logits_dim = view_logits_dim

        classifier_input_dim = input_dim
        if input_type == "shared_and_view_logits":
            classifier_input_dim = input_dim + int(view_logits_dim)

        if mlp_hidden_dim is None:
            self.classifier = nn.Linear(classifier_input_dim, num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor, view_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape [batch_size, input_dim]
            view_logits: Optional tensor with shape [batch_size, view_logits_dim].
                Required when input_type='shared_and_view_logits'.

        Returns:
            Tensor with shape [batch_size, num_classes]
        """
        if self.input_type == "shared_and_view_logits":
            if view_logits is None:
                raise ValueError("view_logits must be provided when input_type='shared_and_view_logits'")
            x = torch.cat([x, view_logits], dim=-1)

        return self.classifier(x)
