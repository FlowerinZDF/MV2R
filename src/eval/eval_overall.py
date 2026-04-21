from typing import Dict, Union

import torch

TensorLike = Union[torch.Tensor, list]


def _to_tensor(x: TensorLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x)


def _labels_from_logits_or_preds(outputs: torch.Tensor) -> torch.Tensor:
    if outputs.ndim == 1:
        return outputs.long()
    if outputs.ndim == 2:
        if outputs.size(-1) == 1:
            return (outputs.squeeze(-1) >= 0).long()
        return outputs.argmax(dim=-1).long()
    raise ValueError(f"Unsupported output shape: {tuple(outputs.shape)}")


def evaluate_overall(outputs: TensorLike, labels: TensorLike) -> Dict[str, float]:
    """Compute overall binary classification metrics.

    Args:
        outputs: Predicted class ids [N], or logits [N, C] / [N, 1].
        labels: Gold labels [N], expected to be in {0, 1}.

    Returns:
        Dict with accuracy, precision, recall, f1, and support.
    """
    outputs_t = _to_tensor(outputs)
    labels_t = _to_tensor(labels).long().view(-1)
    preds_t = _labels_from_logits_or_preds(outputs_t).view(-1)

    if preds_t.numel() != labels_t.numel():
        raise ValueError(
            f"Prediction and label sizes must match. Got {preds_t.numel()} vs {labels_t.numel()}"
        )

    if labels_t.numel() == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": 0.0,
        }

    tp = ((preds_t == 1) & (labels_t == 1)).sum().item()
    fp = ((preds_t == 1) & (labels_t == 0)).sum().item()
    fn = ((preds_t == 0) & (labels_t == 1)).sum().item()
    correct = (preds_t == labels_t).sum().item()
    n = labels_t.numel()

    accuracy = correct / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": float(n),
    }
