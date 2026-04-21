import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.collator import MV2RCollator
from src.data.dataset import MV2RDataset
from src.models.aggregator import MV2RAggregator
from src.models.view_head import MV2RViewHead


class SimpleTextEncoder(nn.Module):
    """Very small placeholder text encoder that outputs [B, H] features."""

    def __init__(self, hidden_dim: int, vocab_size: int = 5000) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = nn.EmbeddingBag(vocab_size, hidden_dim, mode="mean")

    def _text_to_token_ids(self, text: str) -> List[int]:
        tokens = text.lower().split()
        if not tokens:
            return [0]
        return [abs(hash(tok)) % self.vocab_size for tok in tokens]

    def forward(self, texts: List[str]) -> torch.Tensor:
        all_ids: List[int] = []
        offsets: List[int] = [0]

        for text in texts:
            ids = self._text_to_token_ids(text)
            all_ids.extend(ids)
            offsets.append(offsets[-1] + len(ids))

        input_ids = torch.tensor(all_ids, dtype=torch.long, device=self.embedding.weight.device)
        offsets_tensor = torch.tensor(offsets[:-1], dtype=torch.long, device=self.embedding.weight.device)
        return self.embedding(input_ids, offsets_tensor)


class MinimalMV2RModel(nn.Module):
    """Minimal end-to-end module producing shared feature, view logits, and overall logits."""

    def __init__(self, hidden_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.encoder = SimpleTextEncoder(hidden_dim=hidden_dim)
        self.view_head = MV2RViewHead(hidden_dim=hidden_dim)
        self.aggregator = MV2RAggregator(
            input_dim=hidden_dim,
            num_classes=num_classes,
            input_type="shared",
        )

    def forward(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        texts = batch["texts"]
        if not isinstance(texts, list):
            raise TypeError("batch['texts'] must be a list of strings")

        shared_feature = self.encoder(texts)
        view_logits = self.view_head(shared_feature)
        overall_logits = self.aggregator(shared_feature)

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
    model = MinimalMV2RModel(hidden_dim=args.hidden_dim).to(device)

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

    model = MinimalMV2RModel(hidden_dim=args.hidden_dim).to(device)
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
            print(
                f"Epoch {epoch} | "
                f"val total={val_metrics['total_loss']:.6f} "
                f"overall={val_metrics['overall_loss']:.6f} "
                f"view={val_metrics['view_loss']:.6f}"
            )

        ckpt_path = save_checkpoint(output_dir, epoch, model, optimizer, args)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
