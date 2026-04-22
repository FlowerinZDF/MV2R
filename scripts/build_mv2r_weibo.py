#!/usr/bin/env python3
"""Convert raw Weibo fake-news files into MV2R pilot JSON files."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


class ParseStats:
    def __init__(self) -> None:
        self.discard_empty_text = 0
        self.discard_missing_image = 0
        self.raw_records_by_file: Dict[str, int] = {}



def clean_text(text: str) -> str:
    text = URL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def split_image_candidates(raw: str) -> List[str]:
    chunks = re.split(r"[\s,;|]+", raw.strip())
    return [c for c in chunks if c and c.lower() != "null"]



def looks_like_image_token(token: str) -> bool:
    token_l = token.lower()
    if any(token_l.endswith(ext) for ext in IMAGE_EXTS):
        return True
    # also allow bare image ids that are usually numeric/alphanumeric
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{4,}", token))



def parse_weibo_file(path: Path) -> Tuple[List[Tuple[str, str, List[str]]], int]:
    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    record_count = len(raw_lines) // 3
    items: List[Tuple[str, str, List[str]]] = []

    for i in range(0, record_count * 3, 3):
        metadata_line = raw_lines[i].strip()
        image_line = raw_lines[i + 1].strip()
        text_line = raw_lines[i + 2].strip()

        metadata_parts = metadata_line.split("|")
        post_id = metadata_parts[0].strip() if metadata_parts else ""

        image_candidates = split_image_candidates(image_line)
        text = text_line

        items.append((post_id, text, image_candidates))

    return items, record_count



def build_image_index(weibo_root: Path) -> Dict[str, str]:
    image_index: Dict[str, str] = {}

    for subdir in ("rumor_images", "nonrumor_images"):
        folder = weibo_root / subdir
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if not p.is_file():
                continue
            key_full = p.name.lower()
            key_stem = p.stem.lower()
            rel = str((Path(subdir) / p.name).as_posix())
            image_index.setdefault(key_full, rel)
            image_index.setdefault(key_stem, rel)

    return image_index



def resolve_image(candidates: Iterable[str], image_index: Dict[str, str]) -> Optional[str]:
    for raw in candidates:
        c = raw.strip().strip('"\'')
        if not c:
            continue

        if not looks_like_image_token(c):
            continue

        key = Path(c).name.lower()
        stem_key = Path(c).stem.lower()

        if key in image_index:
            return image_index[key]
        if stem_key in image_index:
            return image_index[stem_key]

    return None



def to_mv2r_record(post_id: str, text: str, image: str, label: int) -> Dict[str, object]:
    return {
        "id": post_id,
        "image": image,
        "text": text,
        "overall_label": label,
        "view_labels": {},
        "evidence_text": [],
        "evidence_region": [],
        "conflict_type": None,
    }



def build_split(
    rumor_path: Path,
    nonrumor_path: Path,
    image_index: Dict[str, str],
    stats: ParseStats,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []

    for path, label in ((rumor_path, 1), (nonrumor_path, 0)):
        samples, raw_count = parse_weibo_file(path)
        stats.raw_records_by_file[str(path)] = raw_count

        for post_id, raw_text, candidates in samples:
            text = clean_text(raw_text)
            if not text:
                stats.discard_empty_text += 1
                continue

            image = resolve_image(candidates, image_index)
            if image is None:
                stats.discard_missing_image += 1
                continue

            records.append(to_mv2r_record(post_id, text, image, label))

    return records



def balanced_subsample(records: List[Dict[str, object]], max_n: Optional[int], rng: random.Random) -> List[Dict[str, object]]:
    if max_n is None or len(records) <= max_n:
        rng.shuffle(records)
        return records

    by_label = {0: [], 1: []}
    for row in records:
        by_label[int(row["overall_label"])].append(row)

    for label in by_label:
        rng.shuffle(by_label[label])

    take = {0: max_n // 2, 1: max_n // 2}
    remainder = max_n - (take[0] + take[1])

    if remainder:
        order = sorted((0, 1), key=lambda lbl: len(by_label[lbl]), reverse=True)
        for lbl in order:
            if remainder == 0:
                break
            take[lbl] += 1
            remainder -= 1

    selected = by_label[0][: take[0]] + by_label[1][: take[1]]

    # backfill if one class had too few samples
    if len(selected) < max_n:
        leftovers = by_label[0][take[0] :] + by_label[1][take[1] :]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_n - len(selected)])

    rng.shuffle(selected)
    return selected



def write_json(path: Path, data: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def class_dist(records: List[Dict[str, object]]) -> Dict[int, int]:
    c = Counter(int(r["overall_label"]) for r in records)
    return {0: c.get(0, 0), 1: c.get(1, 0)}



def build_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    weibo_root = Path(args.weibo_root)
    tweets_root = weibo_root / "tweets"

    image_index = build_image_index(weibo_root)
    stats = ParseStats()

    train_records = build_split(
        tweets_root / "train_rumor.txt",
        tweets_root / "train_nonrumor.txt",
        image_index,
        stats,
    )
    val_records = build_split(
        tweets_root / "test_rumor.txt",
        tweets_root / "test_nonrumor.txt",
        image_index,
        stats,
    )

    train_records = balanced_subsample(train_records, args.max_train, rng)
    val_records = balanced_subsample(val_records, args.max_val, rng)

    output_dir = Path(args.output_dir)
    train_out = output_dir / "pilot_real_weibo_train.json"
    val_out = output_dir / "pilot_real_weibo_val.json"

    write_json(train_out, train_records)
    write_json(val_out, val_records)

    train_dist = class_dist(train_records)
    val_dist = class_dist(val_records)

    print(f"Wrote train: {len(train_records)} -> {train_out}")
    print(f"Wrote val:   {len(val_records)} -> {val_out}")
    print(f"Train class distribution (0=nonrumor, 1=rumor): {train_dist}")
    print(f"Val class distribution (0=nonrumor, 1=rumor): {val_dist}")
    print(
        "Discarded samples: "
        f"empty_text={stats.discard_empty_text}, missing_image={stats.discard_missing_image}"
    )
    for path, raw_count in sorted(stats.raw_records_by_file.items()):
        print(f"Raw 3-line records parsed from {path}: {raw_count}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw Weibo fake-news data into MV2R JSON files."
    )
    parser.add_argument(
        "--weibo-root",
        required=True,
        help="Path to Weibo root containing tweets/, rumor_images/, nonrumor_images/.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/weibo_mv2r",
        help="Output directory for pilot_real_weibo_train.json and pilot_real_weibo_val.json.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional max number of train samples (attempts class balance).",
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="Optional max number of val samples (attempts class balance).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling/shuffling.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
