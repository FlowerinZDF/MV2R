#!/usr/bin/env python3
"""Export a readable CSV annotation sheet from an MV2R Weibo JSON file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

CSV_COLUMNS = [
    "id",
    "overall_label",
    "image",
    "text",
    "subject",
    "event",
    "scene",
    "time",
]

VIEW_COLUMNS = ("subject", "event", "scene", "time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a CSV annotation sheet from an MV2R Weibo JSON list."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to input Weibo JSON file (list of samples).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to output CSV annotation sheet.",
    )
    return parser.parse_args()


def load_samples(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}.")

    for idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            raise ValueError(
                f"Expected each sample to be an object, got {type(sample).__name__} at index {idx}."
            )

    return data


def to_csv_row(sample: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": sample.get("id", ""),
        "overall_label": sample.get("overall_label", ""),
        "image": sample.get("image", ""),
        "text": sample.get("text", ""),
        "subject": "",
        "event": "",
        "scene": "",
        "time": "",
    }

    view_labels = sample.get("view_labels")
    if isinstance(view_labels, dict):
        for view in VIEW_COLUMNS:
            if view in view_labels:
                row[view] = view_labels[view]

    return row


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    samples = load_samples(args.input_json)
    rows = (to_csv_row(sample) for sample in samples)
    write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
