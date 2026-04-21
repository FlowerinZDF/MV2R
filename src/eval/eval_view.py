from typing import Dict, Union

import torch

from src.data.schema import FIXED_VIEWS

TensorLike = Union[torch.Tensor, list]


def _to_tensor(x: TensorLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x)


def _to_binary_preds(view_outputs: torch.Tensor) -> torch.Tensor:
    if view_outputs.ndim != 2:
        raise ValueError(f"Expected view outputs with shape [N, V], got {tuple(view_outputs.shape)}")
    if view_outputs.dtype.is_floating_point:
        return (view_outputs >= 0).long()
    return view_outputs.long()


def evaluate_view(view_outputs: TensorLike, view_labels: TensorLike) -> Dict[str, object]:
    """Compute per-view and aggregate metrics while ignoring missing labels (-1)."""
    outputs_t = _to_tensor(view_outputs)
    labels_t = _to_tensor(view_labels).long()
    preds_t = _to_binary_preds(outputs_t)

    if preds_t.shape != labels_t.shape:
        raise ValueError(f"Shape mismatch: preds={tuple(preds_t.shape)} labels={tuple(labels_t.shape)}")

    if preds_t.ndim != 2:
        raise ValueError(f"Expected shape [N, V], got {tuple(preds_t.shape)}")

    n, v = preds_t.shape
    if v != len(FIXED_VIEWS):
        raise ValueError(f"Expected {len(FIXED_VIEWS)} views, got {v}")

    per_view_accuracy: Dict[str, float] = {}
    per_view_f1: Dict[str, float] = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for i, view_name in enumerate(FIXED_VIEWS):
        valid = labels_t[:, i] != -1
        if not valid.any():
            per_view_accuracy[view_name] = 0.0
            per_view_f1[view_name] = 0.0
            continue

        pv = preds_t[valid, i]
        lv = labels_t[valid, i]

        acc = (pv == lv).float().mean().item()
        tp = ((pv == 1) & (lv == 1)).sum().item()
        fp = ((pv == 1) & (lv == 0)).sum().item()
        fn = ((pv == 0) & (lv == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_view_accuracy[view_name] = float(acc)
        per_view_f1[view_name] = float(f1)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_f1 = sum(per_view_f1.values()) / len(FIXED_VIEWS) if FIXED_VIEWS else 0.0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    return {
        "per_view_accuracy": per_view_accuracy,
        "per_view_f1": per_view_f1,
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "num_samples": float(n),
    }
