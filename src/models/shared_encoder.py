import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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


class MultimodalReadyTextSharedEncoder(MV2RSharedEncoderBase):
    """Text-first encoder with multimodal-ready batch inputs.

    This mode keeps compute lightweight while preparing a richer shared feature:
    - main text path: required (`batch["texts"]`)
    - evidence text path: optional (`batch["evidence_text"]`)
    - conflict type path: optional categorical feature (`batch["conflict_types"]`)
    - image path presence hint: optional scalar from `batch["image_paths"]`
    """

    def __init__(
        self,
        hidden_dim: int,
        text_vocab_size: int = 12000,
        text_token_dim: int = 128,
        text_max_tokens: int = 64,
        evidence_vocab_size: int = 6000,
        evidence_token_dim: int = 64,
        evidence_max_tokens: int = 40,
        conflict_type_vocab_size: int = 32,
        conflict_type_dim: int = 16,
        dropout: float = 0.1,
        use_evidence_text: bool = True,
        use_conflict_type: bool = False,
        use_image_hint: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_evidence_text = use_evidence_text
        self.use_conflict_type = use_conflict_type
        self.use_image_hint = use_image_hint
        self.main_text_encoder = SimpleTextSharedEncoder(
            hidden_dim=hidden_dim,
            vocab_size=text_vocab_size,
            token_dim=text_token_dim,
            max_tokens=text_max_tokens,
            dropout=dropout,
        )
        self.evidence_text_encoder = SimpleTextSharedEncoder(
            hidden_dim=hidden_dim,
            vocab_size=evidence_vocab_size,
            token_dim=evidence_token_dim,
            max_tokens=evidence_max_tokens,
            dropout=dropout,
        )

        # 0: missing/unknown, 1..N-1: hashed conflict types.
        self.conflict_type_embedding = nn.Embedding(conflict_type_vocab_size, conflict_type_dim, padding_idx=0)
        self.conflict_projection = nn.Sequential(
            nn.Linear(conflict_type_dim + 1, hidden_dim),  # +1 for image-presence hint
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self._conflict_type_vocab_size = conflict_type_vocab_size

    def _conflict_type_to_id(self, conflict_type: object) -> int:
        if not isinstance(conflict_type, str) or not conflict_type:
            return 0
        if self._conflict_type_vocab_size <= 1:
            return 0
        return (abs(hash(conflict_type.lower())) % (self._conflict_type_vocab_size - 1)) + 1

    def _coerce_optional_texts(
        self, values: object, batch_size: int, field_name: str, fallback: str = ""
    ) -> List[str]:
        if values is None:
            return [fallback] * batch_size
        if not isinstance(values, list):
            raise TypeError(f"batch['{field_name}'] must be a list when provided")
        if len(values) != batch_size:
            raise ValueError(f"batch['{field_name}'] length must match batch['texts'] length")
        normalized: List[str] = []
        for value in values:
            normalized.append(value if isinstance(value, str) else fallback)
        return normalized

    def forward(self, batch: Dict[str, object]) -> torch.Tensor:
        texts = batch.get("texts")
        if not isinstance(texts, list):
            raise TypeError("batch['texts'] must be a list of strings")
        batch_size = len(texts)
        device = self.main_text_encoder.embedding.weight.device

        main_text_feature = self.main_text_encoder(batch)

        evidence_text_feature = torch.zeros((batch_size, self.hidden_dim), dtype=torch.float32, device=device)
        if self.use_evidence_text:
            evidence_texts = self._coerce_optional_texts(
                batch.get("evidence_text"),
                batch_size,
                field_name="evidence_text",
            )
            evidence_text_feature = self.evidence_text_encoder({"texts": evidence_texts})

        conflict_ids = torch.zeros((batch_size,), dtype=torch.long, device=device)
        if self.use_conflict_type:
            conflict_types = self._coerce_optional_texts(
                batch.get("conflict_types"),
                batch_size,
                field_name="conflict_types",
                fallback="",
            )
            conflict_ids = torch.tensor(
                [self._conflict_type_to_id(value) for value in conflict_types],
                dtype=torch.long,
                device=device,
            )
        conflict_emb = self.conflict_type_embedding(conflict_ids)

        image_presence = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        if self.use_image_hint:
            image_paths = batch.get("image_paths")
            if image_paths is not None:
                if not isinstance(image_paths, list):
                    raise TypeError("batch['image_paths'] must be a list when provided")
                if len(image_paths) != batch_size:
                    raise ValueError("batch['image_paths'] length must match batch['texts'] length")
                image_presence = torch.tensor(
                    [[1.0 if isinstance(path, str) and path.strip() else 0.0] for path in image_paths],
                    dtype=torch.float32,
                    device=device,
                )

        aux_feature = self.conflict_projection(torch.cat([conflict_emb, image_presence], dim=-1))
        fused = torch.cat([main_text_feature, evidence_text_feature, aux_feature], dim=-1)
        return self.fusion(fused)


class MultimodalLightSharedEncoder(MV2RSharedEncoderBase):
    """Lightweight multimodal-aware encoder.

    This encoder keeps text as the primary signal while introducing a tiny,
    learnable image-side path that depends only on image path presence.
    It is intended as a structure-ready bridge toward future true image feature
    integration without adding heavy vision dependencies.
    """

    def __init__(
        self,
        hidden_dim: int,
        text_vocab_size: int = 12000,
        text_token_dim: int = 128,
        text_max_tokens: int = 64,
        evidence_vocab_size: int = 6000,
        evidence_token_dim: int = 64,
        evidence_max_tokens: int = 40,
        image_state_dim: int = 8,
        dropout: float = 0.1,
        use_evidence_text: bool = True,
        use_conflict_type: bool = False,
        use_image_hint: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_evidence_text = use_evidence_text
        self.use_conflict_type = use_conflict_type
        self.use_image_hint = use_image_hint

        self.main_text_encoder = SimpleTextSharedEncoder(
            hidden_dim=hidden_dim,
            vocab_size=text_vocab_size,
            token_dim=text_token_dim,
            max_tokens=text_max_tokens,
            dropout=dropout,
        )
        self.evidence_text_encoder = SimpleTextSharedEncoder(
            hidden_dim=hidden_dim,
            vocab_size=evidence_vocab_size,
            token_dim=evidence_token_dim,
            max_tokens=evidence_max_tokens,
            dropout=dropout,
        )

        # 0 = no image path, 1 = image path present.
        self.image_state_embedding = nn.Embedding(2, image_state_dim)
        self.image_projection = nn.Sequential(
            nn.Linear(image_state_dim + 1, hidden_dim),  # +1 for raw presence scalar
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self._conflict_type_vocab_size = 32

    def _conflict_type_to_id(self, conflict_type: object) -> int:
        if not isinstance(conflict_type, str) or not conflict_type:
            return 0
        if self._conflict_type_vocab_size <= 1:
            return 0
        return (abs(hash(conflict_type.lower())) % (self._conflict_type_vocab_size - 1)) + 1

    def _coerce_optional_texts(
        self, values: object, batch_size: int, field_name: str, fallback: str = ""
    ) -> List[str]:
        if values is None:
            return [fallback] * batch_size
        if not isinstance(values, list):
            raise TypeError(f"batch['{field_name}'] must be a list when provided")
        if len(values) != batch_size:
            raise ValueError(f"batch['{field_name}'] length must match batch['texts'] length")
        normalized: List[str] = []
        for value in values:
            normalized.append(value if isinstance(value, str) else fallback)
        return normalized

    def _compute_image_feature(
        self,
        image_paths: object,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        image_presence = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        image_state_ids = torch.zeros((batch_size,), dtype=torch.long, device=device)

        if image_paths is None:
            image_state_emb = self.image_state_embedding(image_state_ids)
            return self.image_projection(torch.cat([image_state_emb, image_presence], dim=-1))

        if not isinstance(image_paths, list):
            raise TypeError("batch['image_paths'] must be a list when provided")
        if len(image_paths) != batch_size:
            raise ValueError("batch['image_paths'] length must match batch['texts'] length")

        for idx, path in enumerate(image_paths):
            has_image = isinstance(path, str) and bool(path.strip())
            if has_image:
                image_presence[idx, 0] = 1.0
                image_state_ids[idx] = 1

        image_state_emb = self.image_state_embedding(image_state_ids)
        return self.image_projection(torch.cat([image_state_emb, image_presence], dim=-1))

    def forward(self, batch: Dict[str, object]) -> torch.Tensor:
        texts = batch.get("texts")
        if not isinstance(texts, list):
            raise TypeError("batch['texts'] must be a list of strings")
        batch_size = len(texts)
        device = self.main_text_encoder.embedding.weight.device

        main_text_feature = self.main_text_encoder(batch)

        evidence_text_feature = torch.zeros((batch_size, self.hidden_dim), dtype=torch.float32, device=device)
        if self.use_evidence_text:
            evidence_texts = self._coerce_optional_texts(
                batch.get("evidence_text"),
                batch_size,
                field_name="evidence_text",
            )
            evidence_text_feature = self.evidence_text_encoder({"texts": evidence_texts})

        image_feature = self._compute_image_feature(
            image_paths=batch.get("image_paths") if self.use_image_hint else None,
            batch_size=batch_size,
            device=device,
        )
        if self.use_conflict_type:
            conflict_types = self._coerce_optional_texts(
                batch.get("conflict_types"),
                batch_size,
                field_name="conflict_types",
                fallback="",
            )
            conflict_ids = torch.tensor(
                [self._conflict_type_to_id(value) for value in conflict_types],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(-1)
            image_feature = image_feature + (0.01 * conflict_ids)

        fused = torch.cat([main_text_feature, evidence_text_feature, image_feature], dim=-1)
        return self.fusion(fused)


class MultimodalImageFeaturesSharedEncoder(MV2RSharedEncoderBase):
    """Multimodal shared encoder using pre-extracted image feature vectors.

    Data contract (no pilot JSON schema changes required):
    - `batch["texts"]`: required list[str]
    - `batch["evidence_text"]`: optional list[str] (or list[any], coerced to fallback)
    - image features are looked up from a separate JSON file, keyed by sample id
      and/or image path, using `batch["sample_ids"]` / `batch["image_paths"]`.
    """

    def __init__(
        self,
        hidden_dim: int,
        image_feature_path: Optional[str] = None,
        image_feature_key: str = "auto",
        image_feature_dim: Optional[int] = None,
        text_vocab_size: int = 12000,
        text_token_dim: int = 128,
        text_max_tokens: int = 64,
        evidence_vocab_size: int = 6000,
        evidence_token_dim: int = 64,
        evidence_max_tokens: int = 40,
        dropout: float = 0.1,
        use_evidence_text: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_evidence_text = use_evidence_text
        self.image_feature_key = image_feature_key

        self.main_text_encoder = SimpleTextSharedEncoder(
            hidden_dim=hidden_dim,
            vocab_size=text_vocab_size,
            token_dim=text_token_dim,
            max_tokens=text_max_tokens,
            dropout=dropout,
        )
        self.evidence_text_encoder = SimpleTextSharedEncoder(
            hidden_dim=hidden_dim,
            vocab_size=evidence_vocab_size,
            token_dim=evidence_token_dim,
            max_tokens=evidence_max_tokens,
            dropout=dropout,
        )

        self.image_feature_lookup: Dict[str, List[float]] = {}
        if image_feature_path:
            self.image_feature_lookup = self._load_image_feature_lookup(Path(image_feature_path))

        inferred_dim = image_feature_dim if image_feature_dim and image_feature_dim > 0 else None
        if inferred_dim is None and self.image_feature_lookup:
            first_vec = next(iter(self.image_feature_lookup.values()))
            inferred_dim = len(first_vec)
        self.image_feature_dim = inferred_dim or hidden_dim

        self.image_feature_projection = nn.Sequential(
            nn.Linear(self.image_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def _load_image_feature_lookup(self, feature_path: Path) -> Dict[str, List[float]]:
        if not feature_path.exists():
            raise FileNotFoundError(f"image_feature_path not found: {feature_path}")
        if feature_path.suffix.lower() != ".json":
            raise ValueError("Only .json feature files are supported for multimodal_image_features")

        with feature_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict) and "features" in payload:
            records = payload["features"]
        else:
            records = payload

        if not isinstance(records, dict):
            raise ValueError("Image feature JSON must be a dict or contain a dict under 'features'")

        lookup: Dict[str, List[float]] = {}
        for key, vector in records.items():
            if not isinstance(key, str):
                continue
            if not isinstance(vector, list) or not vector:
                continue
            lookup[key] = [float(v) for v in vector]

        return lookup

    def _coerce_optional_texts(
        self,
        values: object,
        batch_size: int,
        field_name: str,
        fallback: str = "",
    ) -> List[str]:
        if values is None:
            return [fallback] * batch_size
        if not isinstance(values, list):
            raise TypeError(f"batch['{field_name}'] must be a list when provided")
        if len(values) != batch_size:
            raise ValueError(f"batch['{field_name}'] length must match batch['texts'] length")
        return [value if isinstance(value, str) else fallback for value in values]

    def _resolve_feature_key(
        self,
        idx: int,
        sample_ids: Optional[List[object]],
        image_paths: Optional[List[object]],
    ) -> Optional[str]:
        sample_key = None
        image_key = None
        if sample_ids is not None and idx < len(sample_ids):
            sid = sample_ids[idx]
            if isinstance(sid, str) and sid:
                sample_key = sid
        if image_paths is not None and idx < len(image_paths):
            ipath = image_paths[idx]
            if isinstance(ipath, str) and ipath:
                image_key = ipath

        if self.image_feature_key == "sample_id":
            return sample_key
        if self.image_feature_key == "image_path":
            return image_key
        # auto mode prefers sample id, then image path.
        return sample_key or image_key

    def _encode_image_features(self, batch: Dict[str, object], batch_size: int, device: torch.device) -> torch.Tensor:
        sample_ids_obj = batch.get("sample_ids")
        image_paths_obj = batch.get("image_paths")

        sample_ids = sample_ids_obj if isinstance(sample_ids_obj, list) else None
        image_paths = image_paths_obj if isinstance(image_paths_obj, list) else None

        if sample_ids is not None and len(sample_ids) != batch_size:
            raise ValueError("batch['sample_ids'] length must match batch['texts'] length")
        if image_paths is not None and len(image_paths) != batch_size:
            raise ValueError("batch['image_paths'] length must match batch['texts'] length")

        matrix = torch.zeros((batch_size, self.image_feature_dim), dtype=torch.float32, device=device)

        for i in range(batch_size):
            key = self._resolve_feature_key(i, sample_ids=sample_ids, image_paths=image_paths)
            if not key:
                continue
            values = self.image_feature_lookup.get(key)
            if values is None:
                continue
            vector = values[: self.image_feature_dim]
            if len(vector) < self.image_feature_dim:
                vector = vector + [0.0] * (self.image_feature_dim - len(vector))
            matrix[i] = torch.tensor(vector, dtype=torch.float32, device=device)

        return self.image_feature_projection(matrix)

    def forward(self, batch: Dict[str, object]) -> torch.Tensor:
        texts = batch.get("texts")
        if not isinstance(texts, list):
            raise TypeError("batch['texts'] must be a list of strings")

        batch_size = len(texts)
        device = self.main_text_encoder.embedding.weight.device

        text_feature = self.main_text_encoder(batch)

        evidence_feature = torch.zeros((batch_size, self.hidden_dim), dtype=torch.float32, device=device)
        if self.use_evidence_text:
            evidence_texts = self._coerce_optional_texts(
                batch.get("evidence_text"),
                batch_size,
                field_name="evidence_text",
            )
            evidence_feature = self.evidence_text_encoder({"texts": evidence_texts})

        image_feature = self._encode_image_features(batch, batch_size=batch_size, device=device)
        fused = torch.cat([text_feature, evidence_feature, image_feature], dim=-1)
        return self.fusion(fused)


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

    if encoder_type == "multimodal_ready_text":
        return MultimodalReadyTextSharedEncoder(hidden_dim=hidden_dim, **kwargs)

    if encoder_type == "multimodal_light":
        return MultimodalLightSharedEncoder(hidden_dim=hidden_dim, **kwargs)

    if encoder_type == "multimodal_image_features":
        return MultimodalImageFeaturesSharedEncoder(hidden_dim=hidden_dim, **kwargs)

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

    mm_encoder = build_shared_encoder("multimodal_ready_text", hidden_dim=16)
    fake_mm_batch: Dict[str, object] = {
        "texts": [
            "A person giving a speech in a park.",
            "Night street photo with no visible crowd!",
            "",
        ],
        "image_paths": ["dummy_1.jpg", "", "dummy_3.jpg"],
        "evidence_text": [
            "Microphone and stage are visible.",
            "",
            "Street lights with no people nearby.",
        ],
        "conflict_types": ["subject_scene", "", "event_location"],
    }
    with torch.no_grad():
        mm_shared_feature = mm_encoder(fake_mm_batch)

    print(f"multimodal-ready shared feature shape: {tuple(mm_shared_feature.shape)}")

    mm_light_encoder = build_shared_encoder("multimodal_light", hidden_dim=16)
    with torch.no_grad():
        mm_light_shared_feature = mm_light_encoder(fake_mm_batch)

    print(f"multimodal-light shared feature shape: {tuple(mm_light_shared_feature.shape)}")
