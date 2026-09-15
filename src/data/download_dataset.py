from pathlib import Path
import shutil
import kagglehub

output_dir = Path("./data/raw/")

train_dir = output_dir / "Train"
test_dir = output_dir / "Test"
test_csv = output_dir / "Test.csv"


# Download the dataset only if it does not already exist
if train_dir.is_dir() and test_dir.is_dir() and test_csv.is_file():
    print("Dataset already exists. Skipping download.")
else:
    output_dir.mkdir(parents=True, exist_ok=True)

    path = kagglehub.dataset_download(
        "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign",
        output_dir=str(output_dir),
    )

    print(f"Dataset downloaded to {path}")

# Remove the .complete directory automatically
completion_dir = output_dir / ".complete"

if completion_dir.is_dir():
    shutil.rmtree(completion_dir)
    print("Removed .complete directory")