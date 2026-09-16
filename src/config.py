from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_DIR = RAW_DATA_DIR / "Train"
RAW_TEST_DIR = RAW_DATA_DIR / "Test"
TEST_CSV = RAW_DATA_DIR / "Test.csv"
PROCESSED_TEST_DIR = PROCESSED_DATA_DIR / "Test"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

SEED = 42
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128
NUM_CLASSES = 43
IMG_SIZE_CUSTOM_CNN = (128, 128)
IMG_SIZE_MOBILENET = (224, 224)

CNN_EPOCHS = 100
HEAD_EPOCHS = 20
FINE_TUNE_EPOCHS = 15
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-4


# image_dataset_from_directory sorts numeric folder names lexicographically.
MODEL_CLASS_IDS = tuple(
    int(class_id)
    for class_id in (
        "0", "1", "10", "11", "12", "13", "14", "15", "16", "17",
        "18", "19", "2", "20", "21", "22", "23", "24", "25", "26",
        "27", "28", "29", "3", "30", "31", "32", "33", "34", "35",
        "36", "37", "38", "39", "4", "40", "41", "42", "5", "6",
        "7", "8", "9",
    )
)


CLASS_NAMES = (
    "Speed limit 20 km/h",
    "Speed limit 30 km/h",
    "Speed limit 50 km/h",
    "Speed limit 60 km/h",
    "Speed limit 70 km/h",
    "Speed limit 80 km/h",
    "End of speed limit 80 km/h",
    "Speed limit 100 km/h",
    "Speed limit 120 km/h",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
)


# One source of truth for model files, display names, and input sizes.
MODEL_REGISTRY = {
    "custom_cnn": {
        "display_name": "Custom CNN",
        "type": "cnn",
        "path": MODEL_DIR / "cnn_model_customized.keras",
        "image_size": IMG_SIZE_CUSTOM_CNN,
    },
    "mobilenet_v3_small_gtsrb": {
        "display_name": "MobileNetV3-Small",
        "type": "cnn",
        "path": MODEL_DIR / "mobilenet_v3_small_gtsrb.keras",
        "image_size": IMG_SIZE_MOBILENET,
    },
    "custom_cnn_float16_tflite": {
        "display_name": "Custom CNN (Float16 TFLite)",
        "type": "tflite",
        "path": MODEL_DIR / "cnn_model_customized_float16.tflite",
        "image_size": IMG_SIZE_CUSTOM_CNN,
        "supports_gradcam": False,
    },
    "custom_cnn_int8_tflite": {
        "display_name": "Custom CNN (Int8 TFLite)",
        "type": "tflite",
        "path": MODEL_DIR / "cnn_model_customized_int8.tflite",
        "image_size": IMG_SIZE_CUSTOM_CNN,
        "supports_gradcam": False,
    },
    "svm": {
        "display_name": "SVM",
        "type": "ml",
        "path": MODEL_DIR / "svm_model.joblib",
        "image_size": None,
    },
    "random_forest": {
        "display_name": "Random Forest",
        "type": "ml",
        "path": MODEL_DIR / "random_forest_model.joblib",
        "image_size": None,
    },
}

DEEP_LEARNING_MODELS = (
    "custom_cnn",
    "mobilenet_v3_small_gtsrb",
    "custom_cnn_float16_tflite",
    "custom_cnn_int8_tflite",
)
