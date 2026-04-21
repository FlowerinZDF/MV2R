from typing import Dict

import torch

from src.models.shared_encoder import build_shared_encoder


if __name__ == "__main__":
    encoder = build_shared_encoder("multimodal_light", hidden_dim=32)
    fake_batch: Dict[str, object] = {
        "texts": [
            "A person speaking at a rally.",
            "Empty street during the night.",
            "Crowd near a waterfront.",
        ],
        "evidence_text": [
            "A microphone stand is visible.",
            "No crowd can be seen.",
            "People near boats and a bridge.",
        ],
        "image_paths": ["img_001.jpg", "", "img_003.jpg"],
    }

    with torch.no_grad():
        output = encoder(fake_batch)

    print(f"multimodal_light shared feature shape: {tuple(output.shape)}")
