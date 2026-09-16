from pathlib import Path
import shutil

import pandas as pd

from src.config import PROCESSED_TEST_DIR, TEST_CSV, RAW_DATA_DIR


def prepare_test_dataset(
    input_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_TEST_DIR,
    test_csv: Path = TEST_CSV,
) -> None:
    """Copy test images into class folders for Keras evaluation."""
    existing_images = list(output_dir.rglob("*.png")) if output_dir.is_dir() else []
    rows = pd.read_csv(test_csv)

    if len(existing_images) == len(rows):
        print("Test dataset is already organized. Skipping organization.")
        return

    for _, row in rows.iterrows():
        class_dir = output_dir / str(row["ClassId"])
        class_dir.mkdir(parents=True, exist_ok=True)
        source = input_dir / row["Path"]
        destination = class_dir / Path(row["Path"]).name
        if not destination.exists():
            shutil.copy2(source, destination)

    print(f"Organized test dataset at {output_dir}")


if __name__ == "__main__":
    prepare_test_dataset()
