import argparse
import json
from pathlib import Path
from typing import List, Tuple

REQUIRED_FILES = [
    "Comprehensive_decision_result.json",
    "Feature_synthesizer_result.json",
    "Quantitative_reasoning_result.json",
]


def load_processed_samples(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("processed_samples.json must contain a JSON array.")

    invalid = [x for x in data if not isinstance(x, str)]
    if invalid:
        raise ValueError("processed_samples.json must contain only string sample names.")

    return data


def validate_sample(output_dir: Path, sample_name: str) -> Tuple[bool, List[str]]:
    sample_dir = output_dir / sample_name
    missing = []

    if not sample_dir.is_dir():
        return False, ["<sample directory missing>"]

    for required_name in REQUIRED_FILES:
        if not (sample_dir / required_name).is_file():
            missing.append(required_name)

    return len(missing) == 0, missing


def filter_processed_samples(processed: List[str], output_dir: Path) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    valid_samples: List[str] = []
    removed_samples: List[Tuple[str, List[str]]] = []

    for sample_name in processed:
        ok, missing = validate_sample(output_dir, sample_name)
        if ok:
            valid_samples.append(sample_name)
        else:
            removed_samples.append((sample_name, missing))

    return valid_samples, removed_samples


def save_processed_samples(path: Path, samples: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate processed_samples.json against output folders and required result files."
        )
    )
    parser.add_argument(
        "--base-dir",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Base directory containing processed_samples.json and output/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without writing changes.",
    )

    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    processed_file = base_dir / "processed_samples.json"
    output_dir = base_dir / "output"

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    processed = load_processed_samples(processed_file)
    valid_samples, removed_samples = filter_processed_samples(processed, output_dir)

    print(f"Total entries in processed_samples.json: {len(processed)}")
    print(f"Valid samples (kept): {len(valid_samples)}")
    print(f"Invalid samples (removed): {len(removed_samples)}")

    if removed_samples:
        print("\nRemoved sample details:")
        for sample_name, missing in removed_samples:
            print(f"- {sample_name}: missing {', '.join(missing)}")

    if args.dry_run:
        print("\nDry run mode: no changes written.")
        return

    if len(valid_samples) != len(processed):
        save_processed_samples(processed_file, valid_samples)
        print(f"\nUpdated file: {processed_file}")
    else:
        print("\nNo changes needed.")


if __name__ == "__main__":
    main()
