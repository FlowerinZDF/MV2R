import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.aggregator import MV2RAggregator
from src.models.view_head import MV2RViewHead


def main() -> None:
    batch_size = 4
    hidden_dim = 16
    num_classes = 2

    shared_feature = torch.randn(batch_size, hidden_dim)

    view_head = MV2RViewHead(hidden_dim=hidden_dim)
    view_logits = view_head(shared_feature)

    agg_from_shared = MV2RAggregator(
        input_dim=hidden_dim,
        num_classes=num_classes,
        input_type="shared",
    )
    overall_logits_from_shared = agg_from_shared(shared_feature)

    agg_from_view = MV2RAggregator(
        input_dim=view_logits.size(-1),
        num_classes=num_classes,
        input_type="view_logits",
    )
    overall_logits_from_view = agg_from_view(view_logits)

    print("shared_feature shape:", tuple(shared_feature.shape))
    print("view_logits shape:", tuple(view_logits.shape))
    print("overall_logits_from_shared shape:", tuple(overall_logits_from_shared.shape))
    print("overall_logits_from_view shape:", tuple(overall_logits_from_view.shape))


if __name__ == "__main__":
    main()
