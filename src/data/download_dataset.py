from pathlib import Path
import shutil

import kagglehub

from src.config import RAW_DATA_DIR


def download_dataset(output_dir: Path = RAW_DATA_DIR) -> None:
    """Download GTSRB only when the local dataset is incomplete."""
    train_dir = output_dir / "Train"
    test_dir = output_dir / "Test"
    test_csv = output_dir / "Test.csv"

    if train_dir.is_dir() and test_dir.is_dir() and test_csv.is_file():
        print("Dataset already exists. Skipping download.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download(
        "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign",
        output_dir=str(output_dir),
    )
    print(f"Dataset downloaded to {path}")

    completion_dir = output_dir / ".complete"
    if completion_dir.is_dir():
        shutil.rmtree(completion_dir)
        print("Removed .complete directory")


if __name__ == "__main__":
    download_dataset()
