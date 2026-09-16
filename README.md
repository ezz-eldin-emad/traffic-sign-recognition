# Intelligent Traffic Sign Recognition System

## Project Name

Intelligent Traffic Sign Recognition System (GTSRB)

## Team Members

* Ezz El-Din Emad Ali Aref
* Ali Ahmed Ali Mohamed Ahmed
* Ammar Mohamed Hassan Ahmed
* Omar Osama Abdelgawad Hussein
* Omar Khaled Mohamed El-Garwany
* Omar Alaa Salah Abdel Aziz

## Project Idea

A system that recognizes German traffic signs using the GTSRB dataset through:

1. A classical Machine Learning baseline (HOG/Color/Texture features + SVM / Random Forest).
2. A Deep Learning model (Custom CNN + Transfer Learning using architectures such as ResNet, MobileNet, or EfficientNet).
3. A performance comparison between ML and DL.
4. Explainable AI using Grad-CAM to visualize the model's decisions.
5. Robustness testing under different conditions, including Blur, Illumination, and Perspective transformations.
6. A Streamlit application for uploading an image (or using a camera where supported) and displaying the top 3 predictions.

## Getting Started

### Clone the Repository
Clone the project from GitHub:

```bash
git clone https://github.com/ezz-eldin-emad/traffic-sign-recognition.git
```
Navigate to the project directory:

```bash
cd traffic-sign-recognition
```
After cloning the repository, follow the setup instructions below to install the dependencies and prepare the project environment.

### Setup and Running Instructions
* [Install](https://docs.astral.sh/uv/getting-started/installation/) uv (if not already installed) 

```bash
# 1) Create a virtual environment
uv venv

# 2) Activate the environment
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# 3) Install the dependencies

# For CPU-only systems:
uv sync --extra cpu

# For systems with a supported Cuda GPU:
uv sync --extra cuda

# 4) Download the dataset (GTSRB) when it is not already in data/raw
uv run python src/data/download_dataset.py
```

Train a model through the shared entry point:

```bash
uv run python -m src.training.train --model custom_cnn
uv run python -m src.training.train --model mobilenet_v3_small_gtsrb
```

Run the Streamlit application

```bash
uv run streamlit run app/streamlit_app.py
```

Evaluation (post-training only)

```bash
uv run python -m src.evaluation.evaluate_models --model random_forest
uv run python -m src.evaluation.evaluate_models --model svm
uv run python -m src.evaluation.evaluate_models --model custom_cnn
uv run python -m src.evaluation.evaluate_models --model custom_cnn_float16_tflite
uv run python -m src.evaluation.evaluate_models --model custom_cnn_int8_tflite
uv run python -m src.evaluation.evaluate_models --model mobilenet_v3_small_gtsrb

# Evaluate all registered models. Missing ML artifacts are skipped.
uv run python -m src.evaluation.evaluate_models --model all

```

The evaluation command also writes model footprint, latency, and robustness
results for blur, illumination, and perspective transformations. It does not
train models. The notebooks are separate artifacts and are not required to run
the application or evaluation.


## Libraries / Requirements

Refer to the `pyproject.toml` file for the complete list of dependencies.

The main libraries include:

* `numpy`
* `pandas`
* `scikit-learn`
* `tensorflow` 
* `streamlit`
* `kagglehub`

## Repository Structure

Refer to the **Repository Structure** section in the attached project guide (`PROJECT_GUIDE.md`).
