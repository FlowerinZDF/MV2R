from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from torch import nn


class MV2RSharedEncoderBase(nn.Module, ABC):
    """Base interface for MV2R shared encoders.

    Implementations should accept a batch dict produced by MV2RCollator
    and return a tensor shaped [batch_size, hidden_dim].
    """

    @abstractmethod
    def forward(self, batch: Dict[str, object]) -> torch.Tensor:
        raise NotImplementedError


class SimpleTextSharedEncoder(MV2RSharedEncoderBase):
    """Minimal text-only shared encoder used for smoke training.

    This reproduces the current placeholder behavior by:
    - reading `batch["texts"]`
    - tokenizing via lower-case whitespace split
    - hashing tokens into a fixed-size vocab
    - mean-pooling token embeddings with EmbeddingBag
    """

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

    def forward(self, batch: Dict[str, object]) -> torch.Tensor:
        texts = batch.get("texts")
        if not isinstance(texts, list):
            raise TypeError("batch['texts'] must be a list of strings")

        all_ids: List[int] = []
        offsets: List[int] = [0]

        for text in texts:
            if not isinstance(text, str):
                raise TypeError("batch['texts'] must contain strings")
            ids = self._text_to_token_ids(text)
            all_ids.extend(ids)
            offsets.append(offsets[-1] + len(ids))

        input_ids = torch.tensor(all_ids, dtype=torch.long, device=self.embedding.weight.device)
        offsets_tensor = torch.tensor(offsets[:-1], dtype=torch.long, device=self.embedding.weight.device)
        return self.embedding(input_ids, offsets_tensor)


def build_shared_encoder(encoder_type: str, hidden_dim: int, **kwargs: object) -> MV2RSharedEncoderBase:
    """Factory for shared encoder construction.

    Args:
        encoder_type: Encoder mode name (currently supports: "simple_text").
        hidden_dim: Output feature size H.
        **kwargs: Extra encoder-specific kwargs.
    """

    if encoder_type == "simple_text":
        return SimpleTextSharedEncoder(hidden_dim=hidden_dim, **kwargs)

    raise ValueError(f"Unsupported encoder_type: {encoder_type}")


if __name__ == "__main__":
    encoder = build_shared_encoder("simple_text", hidden_dim=16)
    fake_batch: Dict[str, object] = {
        "texts": [
            "A person giving a speech in a park",
            "Night street photo with no visible crowd",
        ]
    }
    with torch.no_grad():
        shared_feature = encoder(fake_batch)
    print(f"shared feature shape: {tuple(shared_feature.shape)}")
