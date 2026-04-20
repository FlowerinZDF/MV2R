import json
from pathlib import Path
from typing import List, Union

from src.data.schema import MV2RSample


class MV2RDataset:
    """Minimal dataset loader for MV2R JSON/JSONL files."""

    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        self.samples: List[MV2RSample] = self._load_samples()

    def _load_samples(self) -> List[MV2RSample]:
        suffix = self.data_path.suffix.lower()
        if suffix == ".json":
            return self._load_json()
        if suffix == ".jsonl":
            return self._load_jsonl()
        raise ValueError(
            f"Unsupported file type: '{self.data_path.suffix}'. Expected .json or .jsonl"
        )

    def _load_json(self) -> List[MV2RSample]:
        try:
            with self.data_path.open("r", encoding="utf-8") as f:
                records = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse JSON file '{self.data_path}': {exc}"
            ) from exc

        if not isinstance(records, list):
            raise ValueError(
                f"JSON file '{self.data_path}' must contain a top-level list of records"
            )

        samples: List[MV2RSample] = []
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Malformed record at index {idx} in '{self.data_path}': expected object, got {type(record).__name__}"
                )
            try:
                samples.append(MV2RSample.from_dict(record))
            except Exception as exc:
                raise ValueError(
                    f"Malformed record at index {idx} in '{self.data_path}': {exc}"
                ) from exc
        return samples

    def _load_jsonl(self) -> List[MV2RSample]:
        samples: List[MV2RSample] = []
        with self.data_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Failed to parse JSONL at line {line_no} in '{self.data_path}': {exc}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Malformed record at line {line_no} in '{self.data_path}': expected object, got {type(record).__name__}"
                    )
                try:
                    samples.append(MV2RSample.from_dict(record))
                except Exception as exc:
                    raise ValueError(
                        f"Malformed record at line {line_no} in '{self.data_path}': {exc}"
                    ) from exc
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MV2RSample:
        return self.samples[idx]
