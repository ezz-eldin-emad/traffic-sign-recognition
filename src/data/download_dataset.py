from pathlib import Path
import shutil
import kagglehub

output_dir = Path("./data/raw/")

# Download the dataset
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