import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.shared_encoder import build_shared_encoder


def main() -> None:
    features_path = REPO_ROOT / "outputs" / "sanity_image_features.json"
    features_path.parent.mkdir(parents=True, exist_ok=True)

    fake_features = {
        "sample-1": [0.2, -0.1, 0.7, 0.0],
        "sample-2": [0.4, 0.5, -0.3, 0.9],
        "dummy_3.jpg": [0.1, 0.1, 0.1, 0.1],
    }
    with features_path.open("w", encoding="utf-8") as f:
        json.dump(fake_features, f, indent=2)

    encoder = build_shared_encoder(
        encoder_type="multimodal_image_features",
        hidden_dim=16,
        image_feature_path=str(features_path),
        image_feature_key="auto",
        image_feature_dim=4,
        use_evidence_text=True,
    )

    batch = {
        "sample_ids": ["sample-1", "sample-2", "sample-3"],
        "image_paths": ["dummy_1.jpg", "dummy_2.jpg", "dummy_3.jpg"],
        "texts": [
            "A speaker at a rally.",
            "A quiet street scene at night.",
            "An empty park in the afternoon.",
        ],
        "evidence_text": [
            "microphone and stage",
            "street and lights",
            "trees and benches",
        ],
    }

    with torch.no_grad():
        out = encoder(batch)

    print(f"image features file: {features_path}")
    print(f"output shape: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
