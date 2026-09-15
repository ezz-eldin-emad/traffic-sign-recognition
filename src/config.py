from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_DIR = RAW_DATA_DIR / "Train"
RAW_TEST_DIR = RAW_DATA_DIR / "Test"
TEST_CSV = RAW_DATA_DIR / "Test.csv"
PROCESSED_TEST_DIR = PROCESSED_DATA_DIR / "Test"

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"
SEED = 42

# Dataset configuration
IMG_SIZE_MOBILENET = (224, 224)
IMG_SIZE_CUSTOM_CNN = (128, 128)
NUM_CLASSES = 43
BATCH_SIZE = 32

# Training configuration
CNN_EPOCHS = 100
HEAD_EPOCHS = 20
FINE_TUNE_EPOCHS = 15

LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-4
