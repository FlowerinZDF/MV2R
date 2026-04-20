from typing import Dict, List

import torch

from src.data.schema import FIXED_VIEWS, MV2RSample


class MV2RCollator:
    """Collate MV2RSample instances into a minimal batch dictionary."""

    def __call__(self, batch: List[MV2RSample]) -> Dict[str, object]:
        if not batch:
            raise ValueError("Cannot collate an empty batch")

        sample_ids = [s.sample_id for s in batch]
        image_paths = [s.image_path for s in batch]
        texts = [s.text for s in batch]
        overall_labels = torch.tensor([s.overall_label for s in batch], dtype=torch.long)

        dense_view_labels = []
        for sample in batch:
            dense = [sample.view_labels.get(view, -1) for view in FIXED_VIEWS]
            dense_view_labels.append(dense)

        view_labels = torch.tensor(dense_view_labels, dtype=torch.long)

        evidence_text = [s.evidence_text for s in batch]
        evidence_region = [s.evidence_region for s in batch]
        conflict_types = [s.conflict_type for s in batch]

        return {
            "sample_ids": sample_ids,
            "image_paths": image_paths,
            "texts": texts,
            "overall_labels": overall_labels,
            "view_labels": view_labels,
            "evidence_text": evidence_text,
            "evidence_region": evidence_region,
            "conflict_types": conflict_types,
        }
