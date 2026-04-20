import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.collator import MV2RCollator
from src.data.dataset import MV2RDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check for MV2R data pipeline")
    parser.add_argument("data_path", type=str, help="Path to JSON/JSONL data file")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of samples to include in sanity-check batch",
    )
    args = parser.parse_args()

    dataset = MV2RDataset(args.data_path)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("Dataset is empty; nothing to collate.")
        return

    print("One sample:")
    print(dataset[0].to_dict())

    batch_size = min(args.batch_size, len(dataset))
    samples = [dataset[i] for i in range(batch_size)]
    collator = MV2RCollator()
    batch = collator(samples)

    print("Batch keys:", list(batch.keys()))
    print("Tensor shapes:")
    print("  overall_labels:", tuple(batch["overall_labels"].shape))
    print("  view_labels:", tuple(batch["view_labels"].shape))


if __name__ == "__main__":
    main()
