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

# 4) Download the dataset (GTSRB) and place it in data/raw
uv run src/data/download_dataset.py
```

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

## Current Status

This scaffold is an initial project structure with no actual implementation yet. Each file contains `TODOs` that clearly specify the required tasks for each step, according to the official TechTrek requirements (**Project 8 + Universal Requirements**).
