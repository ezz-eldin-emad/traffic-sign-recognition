# Intelligent Traffic Sign Recognition System

## Overview

This project is an image-classification system for recognizing German traffic
signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset. It
combines classical machine-learning baselines and deep-learning models with
model comparison, robustness evaluation, explainability, and a Streamlit
application for single-image predictions.

## Team members

- Ezz El-Din Emad Ali Aref
- Ali Ahmed Ali Mohamed Ahmed
- Ammar Mohamed Hassan Ahmed
- Omar Osama Abdelgawad Hussein
- Omar Khaled Mohamed El-Garwany
- Omar Alaa Salah Abdel Aziz

## Project scope

The repository supports the following workflows:

- Training classical ML and deep-learning traffic-sign classifiers.
- Comparing models using test accuracy, Top-5 accuracy, macro F1, model size, and latency.
- Evaluating robustness under blur, illumination, and perspective transformations.
- Generating Grad-CAM visualizations for supported Keras models.
- Using Float16 and Int8 TensorFlow Lite artifacts for the custom CNN.
- Uploading a traffic-sign image in Streamlit and displaying the Top-3 predictions.

## Models

| Model | Approach | Artifact | Grad-CAM |
| --- | --- | --- | --- |
| Custom CNN | Keras convolutional network | `models/cnn_model_customized.keras` | Yes |
| MobileNetV3-Small | Transfer learning with fine-tuning | `models/mobilenet_v3_small_gtsrb.keras` | Yes |
| HOG + Linear SVM | Equalized grayscale images, HOG features, scaling, and `LinearSVC` | `models/svm_model.joblib` | No |
| Random Forest | Equalized grayscale pixel features and `RandomForestClassifier` | `models/random_forest_model.joblib` | No |
| Custom CNN Float16 | TensorFlow Lite deployment artifact | `models/cnn_model_customized_float16.tflite` | No |
| Custom CNN Int8 | TensorFlow Lite deployment artifact | `models/cnn_model_customized_int8.tflite` | No |

The classical ML preprocessing follows `Final_Project_ML.ipynb`. The shared
training code saves each classical model as a Joblib pipeline, including the
preprocessing steps required during inference.

## Repository structure

The repository structure follows the preferred organization for this project
and the requirements of its implementation.

```text
traffic-sign-recognition/
├── app/
│   ├── app_pages/              # Streamlit pages and shared application logic
│   └── streamlit_app.py        # Streamlit entry point
├── models/                     # Saved Keras, TensorFlow Lite, and Joblib artifacts
├── notebooks/                  # EDA, preprocessing, CNN, and transfer-learning notebooks
├── reports/                    # Per-model evaluation outputs and comparison data
├── src/
│   ├── data/                   # Dataset download, preparation, and data loaders
│   ├── evaluation/             # Metrics, latency, robustness, and evaluation commands
│   ├── inference/              # Model loading and input preprocessing
│   ├── models/                 # Deep-learning builders and classical ML pipelines
│   ├── quantization/           # TensorFlow Lite quantization utilities
│   ├── training/               # Shared training entry point
│   ├── utils/                  # Metrics, artifacts, and model metadata helpers
│   └── xai/                    # Grad-CAM and target-layer utilities
├── tests/                      # Unit and smoke tests
├── pyproject.toml              # Project metadata and dependencies
└── README.md
```

## Requirements

- Python `>=3.10,<3.13`
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A local GTSRB dataset for training and evaluation
- Either the CPU or CUDA TensorFlow extra

The main dependencies are NumPy, pandas, scikit-learn, scikit-image, OpenCV,
Joblib, TensorFlow, Streamlit, matplotlib, seaborn, and KaggleHub. The complete
dependency list is defined in `pyproject.toml`.

## Installation and dataset preparation

Clone the repository:

```bash
git clone https://github.com/ezz-eldin-emad/traffic-sign-recognition.git
cd traffic-sign-recognition
```

Create the virtual environment and install the CPU dependencies:

```bash
uv venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
uv sync --extra cpu
```

For a supported CUDA environment, use:

```bash
uv sync --extra cuda
```

Download the dataset into `data/raw` and prepare the directory-based test
dataset used by the Keras loaders:

```bash
uv run python src/data/download_dataset.py
uv run python src/data/prepare_dataset.py
```

The classical ML workflow reads `data/raw/Train.csv` and
`data/raw/Test.csv`. The deep-learning workflow uses class directories under
`data/raw/Train` and `data/processed/Test`.

## Training

Train a model through the shared entry point:

```bash
uv run python -m src.training.train --model custom_cnn
uv run python -m src.training.train --model mobilenet_v3_small_gtsrb
uv run python -m src.training.train --model svm
uv run python -m src.training.train --model random_forest
```

The training commands save artifacts under `models/`. The classical commands
produce:

```text
models/svm_model.joblib
models/random_forest_model.joblib
```

The Float16 and Int8 files are TensorFlow Lite deployment artifacts for the
custom CNN. They are used when the corresponding files are available.

## Evaluation

Evaluation is a post-training workflow and does not train models. Evaluate one
registered model with:

```bash
uv run python -m src.evaluation.evaluate_models --model custom_cnn
uv run python -m src.evaluation.evaluate_models --model mobilenet_v3_small_gtsrb
uv run python -m src.evaluation.evaluate_models --model custom_cnn_float16_tflite
uv run python -m src.evaluation.evaluate_models --model custom_cnn_int8_tflite
uv run python -m src.evaluation.evaluate_models --model svm
uv run python -m src.evaluation.evaluate_models --model random_forest
```

Evaluate all registered models with:

```bash
uv run python -m src.evaluation.evaluate_models --model all
```

If an artifact is missing, the evaluation command reports that model as
skipped. For available artifacts, evaluation produces:

- Top-1 and Top-5 accuracy
- Macro F1 and a classification report
- A confusion matrix image and NumPy matrix
- Robustness results for blur, illumination, and perspective conditions
- Latency and throughput measurements
- Model-size and model-complexity metadata
- Saved true and predicted labels

Per-model files are written to `reports/<model-name>/`. The consolidated
comparison is written to `reports/model_comparison.csv`.

## Streamlit application

Start the application with:

```bash
uv run streamlit run app/streamlit_app.py
```

The application includes:

- An Experiments page for `.jpg`, `.jpeg`, and `.png` uploads.
- Predictions from every saved model that is available in `models/`.
- Top-3 predictions for each model.
- Probability confidence for probabilistic models and decision margin for the SVM.
- Grad-CAM heatmaps and overlays for supported Keras models.
- A Reports page for viewing saved evaluation results.

The Experiments page loads only artifacts that are present in `models/`. Add
the missing model files after training to include them in the comparison.

## Tests

Run the repository test suite with:

```bash
uv run python -m unittest discover -s tests -v
```

The tests cover core preprocessing, robustness transformations, saved
deep-learning model behavior, TensorFlow Lite output handling, and the
classical pipeline structure.

## Limitations and quality checks

- The dataset domain is limited to German traffic signs from GTSRB. Results
  should not be assumed to transfer to other countries, sign systems, cameras,
  or road conditions.
- The current application performs single-image classification. It is not a
  complete traffic-sign detection, tracking, or autonomous-driving system.
- Reported metrics depend on the dataset, model artifacts, and runtime
  configuration used for the evaluation.
- Missing model artifacts must be generated or supplied before their models
  can be evaluated or displayed in the application.
- The project is intended for education, experimentation, and research.
- The system has no road-safety certification and must not be used as an
  independent road-safety, driving, or vehicle-control decision system.
