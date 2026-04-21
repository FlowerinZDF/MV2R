import re
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

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
    """Lightweight text shared encoder with trainable token embeddings.

    Design goals:
    - keep the same public interface: `batch["texts"] -> [B, H]`
    - remain lightweight and fully self-contained in PyTorch
    - improve over hash-only placeholder by learning a token embedding table,
      sequence-aware masked mean pooling, and a small projection MLP
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int = 12000,
        token_dim: int = 128,
        max_tokens: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.token_dim = token_dim
        self.max_tokens = max_tokens

        # Index 0 is reserved for PAD/UNK fallback handling.
        self.embedding = nn.Embedding(vocab_size, token_dim, padding_idx=0)

        if token_dim == hidden_dim:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            )

        # Tokenize words + standalone punctuation without external tokenizers.
        self._token_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def _tokenize(self, text: str) -> List[str]:
        tokens = self._token_pattern.findall(text.lower())
        if not tokens:
            return ["<empty>"]
        return tokens[: self.max_tokens]

    def _token_to_id(self, token: str) -> int:
        # Stable deterministic hashing mapped to [1, vocab_size-1].
        if self.vocab_size <= 1:
            return 0
        return (abs(hash(token)) % (self.vocab_size - 1)) + 1

    def _encode_texts(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        batch_size = len(texts)
        token_ids = torch.zeros((batch_size, self.max_tokens), dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, self.max_tokens), dtype=torch.float32, device=device)

        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError("batch['texts'] must contain strings")
            token_list = self._tokenize(text)
            ids = [self._token_to_id(tok) for tok in token_list]
            length = min(len(ids), self.max_tokens)

            token_ids[idx, :length] = torch.tensor(ids[:length], dtype=torch.long, device=device)
            attention_mask[idx, :length] = 1.0

        token_emb = self.embedding(token_ids)  # [B, T, D]
        masked_sum = (token_emb * attention_mask.unsqueeze(-1)).sum(dim=1)
        denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = masked_sum / denom
        return pooled

    def forward(self, batch: Dict[str, object]) -> torch.Tensor:
        texts = batch.get("texts")
        if not isinstance(texts, list):
            raise TypeError("batch['texts'] must be a list of strings")

        pooled = self._encode_texts(texts=texts, device=self.embedding.weight.device)
        return self.projection(pooled)


class StubMultimodalSharedEncoder(MV2RSharedEncoderBase):
    """Placeholder to reserve architecture room for future multimodal encoders."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, batch: Dict[str, object]) -> torch.Tensor:
        raise NotImplementedError("Multimodal shared encoder is not implemented yet.")


def build_shared_encoder(encoder_type: str, hidden_dim: int, **kwargs: object) -> MV2RSharedEncoderBase:
    """Factory for shared encoder construction.

    Args:
        encoder_type: Encoder mode name (currently supports: "simple_text").
        hidden_dim: Output feature size H.
        **kwargs: Extra encoder-specific kwargs.
    """

    if encoder_type == "simple_text":
        return SimpleTextSharedEncoder(hidden_dim=hidden_dim, **kwargs)

    if encoder_type == "multimodal_stub":
        return StubMultimodalSharedEncoder(hidden_dim=hidden_dim)

    raise ValueError(f"Unsupported encoder_type: {encoder_type}")


if __name__ == "__main__":
    encoder = build_shared_encoder("simple_text", hidden_dim=16)
    fake_batch: Dict[str, object] = {
        "texts": [
            "A person giving a speech in a park.",
            "Night street photo with no visible crowd!",
            "",
        ]
    }

    with torch.no_grad():
        shared_feature = encoder(fake_batch)

    print(f"shared feature shape: {tuple(shared_feature.shape)}")
