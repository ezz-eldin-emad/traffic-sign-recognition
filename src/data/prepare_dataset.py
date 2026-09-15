from pathlib import Path
import shutil
import pandas as pd

output_dir = Path("./data/raw/")
processed_test_dir = Path("./data/processed/Test")
test_csv = output_dir / "Test.csv"

# Organize test images only if needed
if processed_test_dir.is_dir():
    organized_images = list(processed_test_dir.rglob("*.png"))
else:
    organized_images = []

if len(organized_images) == len(pd.read_csv(test_csv)):
    print("Test dataset is already organized. Skipping organization.")
else:
    test_data = pd.read_csv(test_csv)

    for _, row in test_data.iterrows():
        class_dir = processed_test_dir / str(row["ClassId"])
        class_dir.mkdir(parents=True, exist_ok=True)

        source = output_dir / row["Path"]
        destination = class_dir / Path(row["Path"]).name

        if not destination.exists():
            shutil.copy2(source, destination)

    print(f"Organized test dataset at {processed_test_dir}")