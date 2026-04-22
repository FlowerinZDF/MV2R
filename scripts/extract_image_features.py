import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CLIP image embeddings keyed by sample id for MV2R pilot data.",
    )
    parser.add_argument(
        "--metadata",
        nargs="+",
        required=True,
        help="One or more JSON metadata files containing records with `id` and `image` fields.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path. Writes {sample_id: [float, ...]} by default.",
    )
    parser.add_argument(
        "--image-root",
        default="",
        help="Optional base directory used to resolve relative image paths.",
    )
    parser.add_argument(
        "--model-name",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP checkpoint name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Image batch size for CLIP inference.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Device for extraction, e.g. cpu or cuda. Default: auto-detect.",
    )
    parser.add_argument(
        "--feature-key",
        choices=["sample_id", "features"],
        default="sample_id",
        help="`sample_id`: plain dict output; `features`: wrap dict under {\"features\": ...}.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail instead of writing zero vectors when an image cannot be loaded.",
    )
    return parser.parse_args()


def load_metadata_records(paths: Sequence[Path]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"Metadata file must contain a JSON list: {path}")

        for idx, sample in enumerate(payload):
            if not isinstance(sample, dict):
                raise ValueError(f"Malformed record in {path} at index {idx}: expected object")
            sample_id = sample.get("id")
            image_path = sample.get("image")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Missing/invalid 'id' in {path} at index {idx}")
            if not isinstance(image_path, str) or not image_path:
                raise ValueError(f"Missing/invalid 'image' in {path} at index {idx}")
            records.append({"sample_id": sample_id, "image": image_path})
    return records


def resolve_image_path(raw_path: str, image_root: Optional[Path]) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if image_root is not None:
        return image_root / path
    return path


def batched(items: Sequence[Dict[str, str]], batch_size: int) -> Iterable[Sequence[Dict[str, str]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def extract_embeddings(
    records: Sequence[Dict[str, str]],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int,
    image_root: Optional[Path],
    strict_missing: bool,
) -> Tuple[Dict[str, List[float]], int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    model.eval()
    features: Dict[str, List[float]] = {}
    missing_count = 0
    inferred_dim: Optional[int] = None

    for chunk in tqdm(list(batched(records, batch_size)), desc="extract", unit="batch"):
        loaded_images: List[Image.Image] = []
        loaded_sample_ids: List[str] = []

        for item in chunk:
            sample_id = item["sample_id"]
            image_path = resolve_image_path(item["image"], image_root)

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                if strict_missing:
                    raise RuntimeError(f"Failed to load image for sample '{sample_id}': {image_path}") from exc
                missing_count += 1
                if inferred_dim is not None:
                    features[sample_id] = [0.0] * inferred_dim
                continue

            loaded_images.append(image)
            loaded_sample_ids.append(sample_id)

        if loaded_images:
            model_inputs = processor(images=loaded_images, return_tensors="pt", padding=True)
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            with torch.no_grad():
                image_features = model.get_image_features(**model_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            image_features = image_features.detach().cpu()
            if inferred_dim is None:
                inferred_dim = int(image_features.shape[-1])

            for sid, vec in zip(loaded_sample_ids, image_features):
                features[sid] = [float(v) for v in vec.tolist()]

        if inferred_dim is not None:
            for item in chunk:
                sid = item["sample_id"]
                if sid not in features:
                    features[sid] = [0.0] * inferred_dim

    if inferred_dim is None:
        raise RuntimeError(
            "Could not infer embedding dimension. Ensure at least one image exists or disable strict missing only when some images are present."
        )

    return features, missing_count


def main() -> None:
    args = parse_args()

    metadata_paths = [Path(p) for p in args.metadata]
    for path in metadata_paths:
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_root = Path(args.image_root) if args.image_root else None
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading metadata files: {[str(p) for p in metadata_paths]}")
    records = load_metadata_records(metadata_paths)
    print(f"Loaded {len(records)} records")

    print(f"Loading CLIP model: {args.model_name}")
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name).to(device)

    features, missing_count = extract_embeddings(
        records=records,
        model=model,
        processor=processor,
        device=device,
        batch_size=args.batch_size,
        image_root=image_root,
        strict_missing=args.strict_missing,
    )

    payload = features if args.feature_key == "sample_id" else {"features": features}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    feature_dim = len(next(iter(features.values())))
    print(f"Saved {len(features)} feature vectors (dim={feature_dim}) to: {output_path}")
    if missing_count > 0:
        print(f"Warning: {missing_count} images were missing/unreadable and were replaced with zero vectors.")


if __name__ == "__main__":
    main()
