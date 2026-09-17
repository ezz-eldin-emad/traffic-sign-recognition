# Intelligent Traffic Sign Recognition System

An end-to-end traffic-sign image-classification system built with the German
Traffic Sign Recognition Benchmark (GTSRB). The project combines classical
machine learning, convolutional neural networks, transfer learning,
TensorFlow Lite deployment, model evaluation, robustness analysis, and a
Streamlit interface for interactive predictions.

## Live application

[Open the Intelligent Traffic Sign Recognition System](https://intelligent-traffic-sign-recognition-system.streamlit.app)

## Project highlights

- Classification across 43 German traffic-sign classes.
- Comparison of classical ML and deep-learning models.
- Top-1, Top-5, macro F1, latency, throughput, and model-size evaluation.
- Robustness testing under blur, illumination, and perspective transformations.
- Grad-CAM explanations for supported Keras models.
- Float16 and Int8 TensorFlow Lite deployment artifacts.
- Interactive single-image inference through Streamlit.

## Models

| Model | Method | Saved artifact | Grad-CAM |
| --- | --- | --- | --- |
| Custom CNN | Keras convolutional neural network | `models/cnn_model_customized.keras` | Yes |
| MobileNetV3-Small | Transfer learning and fine-tuning | `models/mobilenet_v3_small_gtsrb.keras` | Yes |
| HOG + Linear SVM | HOG features with scaling and `LinearSVC` | `models/svm_model.joblib` | No |
| Random Forest | Equalized grayscale pixel features | `models/random_forest_model.joblib` | No |
| Custom CNN Float16 | TensorFlow Lite deployment model | `models/cnn_model_customized_float16.tflite` | No |
| Custom CNN Int8 | Quantized TensorFlow Lite deployment model | `models/cnn_model_customized_int8.tflite` | No |

The model registry in `src/config.py` is the single source of truth for model
names, display labels, artifact paths, input sizes, and inference types.

## Streamlit application

The application contains two pages:

- **Experiments**: upload a `.jpg`, `.jpeg`, or `.png` image, select one or
  more available models, compare predictions, inspect Top-3 results, and view
  Grad-CAM visualizations.
- **Reports**: browse saved comparison metrics, confusion matrices,
  classification reports, robustness results, and latency benchmarks.

Run the application locally from the repository root:

```bash
uv run streamlit run app/streamlit_app.py
```

The application automatically loads the model artifacts available in
`models/`.

## Repository structure

```text
traffic-sign-recognition/
├── app/
│   ├── app_pages/              # Streamlit pages and shared UI/inference helpers
│   ├── requirements.txt        # Streamlit Community Cloud dependencies
│   └── streamlit_app.py        # Streamlit entry point
├── models/                     # Keras, TensorFlow Lite, and Joblib artifacts
├── notebooks/                  # EDA, classical ML, CNN, and transfer-learning work
├── reports/                    # Evaluation outputs and model comparison data
├── src/
│   ├── data/                   # Dataset download, preparation, and loaders
│   ├── evaluation/             # Metrics, robustness, latency, and evaluation
│   ├── inference/              # Model loading and preprocessing
│   ├── models/                 # CNN and classical ML model definitions
│   ├── quantization/           # TensorFlow Lite quantization workflow
│   ├── training/               # Shared model-training entry point
│   ├── utils/                  # Metrics and artifact utilities
│   └── xai/                    # Grad-CAM and target-layer utilities
├── tests/                      # Unit and application smoke tests
├── pyproject.toml              # Project metadata and dependency definitions
├── requirements.txt            # Exported development dependencies
└── uv.lock                    # Reproducible uv dependency lockfile
```

## Requirements

- Python `>=3.10,<3.13`
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A local copy of the GTSRB dataset for training and evaluation
- TensorFlow CPU or CUDA support, depending on the selected environment

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/ezz-eldin-emad/traffic-sign-recognition.git
cd traffic-sign-recognition
```

Create the environment and install the CPU dependencies:

```bash
uv venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv sync --extra cpu
```

For a CUDA-enabled environment:

```bash
uv sync --extra cuda
```

## Dataset preparation

Download the GTSRB dataset into `data/raw`:

```bash
uv run python src/data/download_dataset.py
```

Prepare the class-directory test set used by the Keras evaluation pipeline:

```bash
uv run python src/data/prepare_dataset.py
```

The project uses:

- `data/raw/Train` for deep-learning training.
- `data/raw/Test.csv` and the related images for classical ML evaluation.
- `data/processed/Test` for directory-based Keras evaluation.

## Training

Train the custom CNN:

```bash
uv run python -m src.training.train --model custom_cnn
```

Train MobileNetV3-Small with transfer learning and fine-tuning:

```bash
uv run python -m src.training.train --model mobilenet_v3_small_gtsrb
```

Train the classical baselines:

```bash
uv run python -m src.training.train --model svm
uv run python -m src.training.train --model random_forest
```

Training artifacts are saved in `models/`.

## Evaluation

Evaluate one model:

```bash
uv run python -m src.evaluation.evaluate_models --model custom_cnn
```

Available model names are:

```text
custom_cnn
mobilenet_v3_small_gtsrb
custom_cnn_float16_tflite
custom_cnn_int8_tflite
svm
random_forest
```

Evaluate all registered models and update the consolidated comparison:

```bash
uv run python -m src.evaluation.evaluate_models --model all
```

Evaluation outputs are written to `reports/<model-name>/`, including:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `robustness.csv`
- `robustness_summary.json`
- `latency.json`
- Saved true and predicted labels

The combined model comparison is saved as:

```text
reports/model_comparison.csv
```

## Quantization

The custom CNN can be converted to TensorFlow Lite Float16 and Int8 artifacts
using the quantization workflow in:

```text
src/quantization/cnn_quantization_customized.py
```

The generated deployment files are stored in `models/` and are supported by
the Streamlit Experiments page.

## Tests

Run the complete test suite:

```bash
uv run python -m unittest discover -s tests -v
```

The tests cover preprocessing, robustness transformations, Grad-CAM behavior,
TensorFlow Lite inference, classical ML pipelines, and Streamlit UI helpers.

## Team

- Ezz El-Din Emad Ali Aref
- Ali Ahmed Ali Mohamed Ahmed
- Ammar Mohamed Hassan Ahmed
- Omar Osama Abdelgawad Hussein
- Omar Khaled Mohamed El-Garwany
- Omar Alaa Salah Abdel Aziz Hussein
