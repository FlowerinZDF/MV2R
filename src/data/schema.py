from dataclasses import dataclass, field
from typing import Dict, List, Optional


FIXED_VIEWS = [
    "subject",
    "scene",
    "event",
    "time",
    "location",
    "ocr",
    "relation",
    "tampering",
]


@dataclass
class MV2RSample:
    sample_id: str
    image_path: str
    text: str
    overall_label: int
    view_labels: Dict[str, int] = field(default_factory=dict)
    evidence_text: List[str] = field(default_factory=list)
    evidence_region: List[dict] = field(default_factory=list)
    conflict_type: Optional[str] = None

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if not self.image_path:
            raise ValueError("image_path must not be empty")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if self.overall_label not in [0, 1]:
            raise ValueError("overall_label must be 0 or 1")

        for view in self.view_labels:
            if view not in FIXED_VIEWS:
                raise ValueError(f"Unknown view label: {view}")
            if self.view_labels[view] not in [0, 1]:
                raise ValueError(f"View label for {view} must be 0 or 1")

    def to_dict(self) -> dict:
        return {
            "id": self.sample_id,
            "image": self.image_path,
            "text": self.text,
            "overall_label": self.overall_label,
            "view_labels": self.view_labels,
            "evidence_text": self.evidence_text,
            "evidence_region": self.evidence_region,
            "conflict_type": self.conflict_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MV2RSample":
        sample = cls(
            sample_id=data["id"],
            image_path=data["image"],
            text=data["text"],
            overall_label=data["overall_label"],
            view_labels=data.get("view_labels", {}),
            evidence_text=data.get("evidence_text", []),
            evidence_region=data.get("evidence_region", []),
            conflict_type=data.get("conflict_type"),
        )
        sample.validate()
        return sample