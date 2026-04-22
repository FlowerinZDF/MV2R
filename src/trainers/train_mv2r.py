import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Literal, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.collator import MV2RCollator
from src.data.dataset import MV2RDataset
from src.models.aggregator import MV2RAggregator
from src.models.shared_encoder import build_shared_encoder
from src.models.view_head import MV2RViewHead
from src.eval.eval_overall import evaluate_overall
from src.eval.eval_view import evaluate_view


class MinimalMV2RModel(nn.Module):
    """Minimal end-to-end module producing shared feature, view logits, and overall logits."""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 2,
        encoder_type: str = "simple_text",
        aggregator_mode: Literal["shared", "view_logits", "shared_and_view_logits"] = "shared",
        encoder_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()
        multimodal_encoder_types = {"multimodal_ready_text", "multimodal_light", "multimodal_image_features"}
        selected_encoder_kwargs = dict(encoder_kwargs or {}) if encoder_type in multimodal_encoder_types else {}
        if encoder_type in {"multimodal_ready_text", "multimodal_light"}:
            selected_encoder_kwargs.pop("image_feature_path", None)
            selected_encoder_kwargs.pop("image_feature_key", None)
            selected_encoder_kwargs.pop("image_feature_dim", None)
        if encoder_type == "multimodal_image_features":
            selected_encoder_kwargs.pop("use_conflict_type", None)
            selected_encoder_kwargs.pop("use_image_hint", None)

        self.encoder = build_shared_encoder(
            encoder_type=encoder_type,
            hidden_dim=hidden_dim,
            **selected_encoder_kwargs,
        )
        self.view_head = MV2RViewHead(hidden_dim=hidden_dim)
        self.aggregator_mode = aggregator_mode

        aggregator_input_dim = hidden_dim if aggregator_mode != "view_logits" else self.view_head.num_views
        self.aggregator = MV2RAggregator(
            input_dim=aggregator_input_dim,
            num_classes=num_classes,
            input_type=aggregator_mode,
            view_logits_dim=self.view_head.num_views if aggregator_mode == "shared_and_view_logits" else None,
        )

    def forward(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        shared_feature = self.encoder(batch)
        view_logits = self.view_head(shared_feature)
        if self.aggregator_mode == "shared":
            overall_logits = self.aggregator(shared_feature)
        elif self.aggregator_mode == "view_logits":
            overall_logits = self.aggregator(view_logits)
        elif self.aggregator_mode == "shared_and_view_logits":
            overall_logits = self.aggregator(shared_feature, view_logits=view_logits)
        else:
            raise ValueError(f"Unsupported aggregator_mode: {self.aggregator_mode}")

        return {
            "shared_feature": shared_feature,
            "view_logits": view_logits,
            "overall_logits": overall_logits,
        }


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, object],
    device: torch.device,
    view_loss_weight: float,
) -> Dict[str, torch.Tensor]:
    overall_labels = batch["overall_labels"].to(device)
    view_labels = batch["view_labels"].to(device)

    ce_loss = nn.CrossEntropyLoss()
    overall_loss = ce_loss(outputs["overall_logits"], overall_labels)

    valid_mask = view_labels != -1
    if valid_mask.any():
        bce = nn.BCEWithLogitsLoss(reduction="none")
        raw_view_loss = bce(outputs["view_logits"], view_labels.float())
        view_loss = raw_view_loss[valid_mask].mean()
    else:
        view_loss = torch.zeros((), device=device)

    total_loss = overall_loss + view_loss_weight * view_loss
    return {
        "total_loss": total_loss,
        "overall_loss": overall_loss,
        "view_loss": view_loss,
    }


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    view_loss_weight: float,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total = 0.0
    total_overall = 0.0
    total_view = 0.0
    steps = 0

    for batch in dataloader:
        for key in ("overall_labels", "view_labels"):
            if key in batch:
                batch[key] = batch[key].to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            losses = compute_losses(outputs, batch, device=device, view_loss_weight=view_loss_weight)

            if is_train:
                optimizer.zero_grad()
                losses["total_loss"].backward()
                optimizer.step()

        total += losses["total_loss"].item()
        total_overall += losses["overall_loss"].item()
        total_view += losses["view_loss"].item()
        steps += 1

    if steps == 0:
        return {"total_loss": 0.0, "overall_loss": 0.0, "view_loss": 0.0}

    return {
        "total_loss": total / steps,
        "overall_loss": total_overall / steps,
        "view_loss": total_view / steps,
    }


def compute_eval_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model.eval()

    overall_logits = []
    overall_labels = []
    view_logits = []
    view_labels = []

    with torch.no_grad():
        for batch in dataloader:
            for key in ("overall_labels", "view_labels"):
                if key in batch:
                    batch[key] = batch[key].to(device)

            outputs = model(batch)
            overall_logits.append(outputs["overall_logits"].detach().cpu())
            overall_labels.append(batch["overall_labels"].detach().cpu())
            view_logits.append(outputs["view_logits"].detach().cpu())
            view_labels.append(batch["view_labels"].detach().cpu())

    if not overall_logits:
        return {"overall": {}, "view": {}}

    all_overall_logits = torch.cat(overall_logits, dim=0)
    all_overall_labels = torch.cat(overall_labels, dim=0)
    all_view_logits = torch.cat(view_logits, dim=0)
    all_view_labels = torch.cat(view_labels, dim=0)

    overall_metrics = evaluate_overall(all_overall_logits, all_overall_labels)
    view_metrics = evaluate_view(all_view_logits, all_view_labels)

    return {"overall": overall_metrics, "view": view_metrics}


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )
    return ckpt_path


def load_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> int:
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0))


def build_dataloader(data_path: str, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = MV2RDataset(data_path)
    collator = MV2RCollator()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def run_sanity_check(args: argparse.Namespace) -> None:
    dataloader = build_dataloader(args.data_path, args.batch_size, shuffle=False)
    batch = next(iter(dataloader))

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MinimalMV2RModel(
        hidden_dim=args.hidden_dim,
        encoder_type=args.encoder_type,
        aggregator_mode=args.aggregator_mode,
        encoder_kwargs={
            "use_evidence_text": not args.disable_evidence_text,
            "use_conflict_type": args.enable_conflict_type and not args.disable_conflict_type,
            "use_image_hint": not args.disable_image_hint,
            "image_feature_path": args.image_feature_path,
            "image_feature_key": args.image_feature_key,
            "image_feature_dim": args.image_feature_dim,
        },
    ).to(device)

    for key in ("overall_labels", "view_labels"):
        batch[key] = batch[key].to(device)

    outputs = model(batch)
    losses = compute_losses(outputs, batch, device=device, view_loss_weight=args.view_loss_weight)

    print(f"batch size: {len(batch['texts'])}")
    print(f"shared feature shape: {tuple(outputs['shared_feature'].shape)}")
    print(f"view logits shape: {tuple(outputs['view_logits'].shape)}")
    print(f"overall logits shape: {tuple(outputs['overall_logits'].shape)}")
    print(f"overall loss: {losses['overall_loss'].item():.6f}")
    print(f"view loss: {losses['view_loss'].item():.6f}")
    print(f"total loss: {losses['total_loss'].item():.6f}")


def maybe_write_tiny_example(path: Path) -> None:
    if path.exists():
        return

    tiny_records = [
        {
            "id": "sample-1",
            "image": "dummy_1.jpg",
            "text": "a person giving a speech in a park",
            "overall_label": 1,
            "view_labels": {"subject": 1, "scene": 0, "event": 1},
        },
        {
            "id": "sample-2",
            "image": "dummy_2.jpg",
            "text": "night street photo with no visible crowd",
            "overall_label": 0,
            "view_labels": {"subject": 0, "scene": 1, "event": 0, "location": 1},
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(tiny_records, f, indent=2)
    print(f"Wrote tiny example dataset to: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal MV2R trainer")
    parser.add_argument("--data-path", type=str, required=False, help="Path to train JSON/JSONL file")
    parser.add_argument("--val-data-path", type=str, default=None, help="Optional validation JSON/JSONL file")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="simple_text",
        choices=["simple_text", "multimodal_ready_text", "multimodal_light", "multimodal_image_features"],
        help="Shared encoder type",
    )
    parser.add_argument(
        "--aggregator-mode",
        type=str,
        default="shared",
        choices=["shared", "view_logits", "shared_and_view_logits"],
        help="Overall aggregation input mode",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--view-loss-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="outputs/mv2r")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (default auto)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--sanity-check", action="store_true", help="Run one forward pass and print tensor shapes/losses")
    parser.add_argument(
        "--write-tiny-example",
        type=str,
        default=None,
        help="Optional path to write a tiny JSON dataset for quick sanity-checking",
    )
    parser.add_argument(
        "--disable-evidence-text",
        action="store_true",
        help="Disable evidence_text feature path in multimodal_ready_text encoder",
    )
    parser.add_argument(
        "--enable-conflict-type",
        action="store_true",
        help="Enable conflict_type feature path in multimodal_ready_text encoder (disabled by default)",
    )
    parser.add_argument(
        "--disable-conflict-type",
        action="store_true",
        help="Deprecated compatibility flag. If set, conflict_type feature path stays disabled.",
    )
    parser.add_argument(
        "--disable-image-hint",
        action="store_true",
        help="Disable image-path presence hint in multimodal_ready_text encoder",
    )
    parser.add_argument(
        "--image-feature-path",
        type=str,
        default=None,
        help="Optional JSON file with pre-extracted image features keyed by sample id or image path",
    )
    parser.add_argument(
        "--image-feature-key",
        type=str,
        default="auto",
        choices=["auto", "sample_id", "image_path"],
        help="Key strategy used to fetch image features for multimodal_image_features encoder",
    )
    parser.add_argument(
        "--image-feature-dim",
        type=int,
        default=None,
        help="Optional override for expected image feature dimension",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.write_tiny_example:
        maybe_write_tiny_example(Path(args.write_tiny_example))

    if not args.data_path:
        raise ValueError("--data-path is required unless you only use --write-tiny-example")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.sanity_check:
        run_sanity_check(args)
        return

    train_loader = build_dataloader(args.data_path, args.batch_size, shuffle=True)
    val_loader = None
    if args.val_data_path:
        val_loader = build_dataloader(args.val_data_path, args.batch_size, shuffle=False)

    model = MinimalMV2RModel(
        hidden_dim=args.hidden_dim,
        encoder_type=args.encoder_type,
        aggregator_mode=args.aggregator_mode,
        encoder_kwargs={
            "use_evidence_text": not args.disable_evidence_text,
            "use_conflict_type": args.enable_conflict_type and not args.disable_conflict_type,
            "use_image_hint": not args.disable_image_hint,
            "image_feature_path": args.image_feature_path,
            "image_feature_key": args.image_feature_key,
            "image_feature_dim": args.image_feature_dim,
        },
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    start_epoch = 1
    if args.resume_from:
        start_epoch = load_checkpoint(Path(args.resume_from), model, optimizer, map_location=str(device)) + 1
        print(f"Resumed from {args.resume_from}; starting epoch {start_epoch}")

    output_dir = Path(args.output_dir)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            view_loss_weight=args.view_loss_weight,
        )

        print(
            f"Epoch {epoch} | "
            f"train total={train_metrics['total_loss']:.6f} "
            f"overall={train_metrics['overall_loss']:.6f} "
            f"view={train_metrics['view_loss']:.6f}"
        )

        if val_loader is not None:
            val_metrics = run_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                optimizer=None,
                view_loss_weight=args.view_loss_weight,
            )
            eval_metrics = compute_eval_metrics(model=model, dataloader=val_loader, device=device)
            print(
                f"Epoch {epoch} | "
                f"val total={val_metrics['total_loss']:.6f} "
                f"overall={val_metrics['overall_loss']:.6f} "
                f"view={val_metrics['view_loss']:.6f} "
                f"| overall_acc={eval_metrics['overall'].get('accuracy', 0.0):.4f} "
                f"overall_f1={eval_metrics['overall'].get('f1', 0.0):.4f} "
                f"view_micro_f1={eval_metrics['view'].get('micro_f1', 0.0):.4f}"
            )

        ckpt_path = save_checkpoint(output_dir, epoch, model, optimizer, args)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
