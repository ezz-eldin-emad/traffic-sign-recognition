import matplotlib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.config import (
    CLASS_NAMES,
    DEEP_LEARNING_MODELS,
    MODEL_CLASS_IDS,
    MODEL_REGISTRY,
)
from src.inference.model_loader import load_model_by_name
from src.inference.preprocessing import preprocess_image
from src.xai.gradcam import make_gradcam_heatmap
from src.xai.targets import get_target_layer


@st.cache_resource(show_spinner=False)
def get_model(model_name: str, model_path: str) -> tf.keras.Model:
    """Cache a saved model and invalidate it when its file timestamp changes."""
    del model_path
    return load_model_by_name(model_name)


def available_models() -> list[str]:
    return [
        name
        for name in DEEP_LEARNING_MODELS
        if MODEL_REGISTRY[name]["path"].is_file()
    ]


def create_gradcam_images(
    original_image: Image.Image,
    heatmap: np.ndarray,
) -> tuple[Image.Image, Image.Image]:
    base = original_image.convert("RGB")
    resized_heatmap = Image.fromarray(np.uint8(heatmap * 255), mode="L").resize(
        base.size,
        Image.Resampling.BILINEAR,
    )
    normalized = np.asarray(resized_heatmap, dtype=np.float32) / 255.0
    color = np.uint8(matplotlib.colormaps["jet"](normalized)[..., :3] * 255)
    heatmap_image = Image.fromarray(color, mode="RGB")
    return heatmap_image, Image.blend(base, heatmap_image, alpha=0.4)


def run_prediction(model_name: str, image: Image.Image) -> dict:
    config = MODEL_REGISTRY[model_name]
    model_path = config["path"]
    model = get_model(model_name, f"{model_path}:{model_path.stat().st_mtime_ns}")
    tensor = preprocess_image(image, config["image_size"])
    probabilities = model(tensor, training=False).numpy()[0]
    top_indices = np.argsort(probabilities)[-3:][::-1]
    predicted_index = int(top_indices[0])
    class_id = MODEL_CLASS_IDS[predicted_index]

    heatmap = make_gradcam_heatmap(
        tensor,
        model,
        get_target_layer(model),
        class_index=predicted_index,
    )
    heatmap_image, overlay = create_gradcam_images(image, heatmap)
    top3 = [
        {
            "Class": CLASS_NAMES[MODEL_CLASS_IDS[int(index)]],
            "Confidence": f"{float(probabilities[index]):.2%}",
        }
        for index in top_indices
    ]
    return {
        "display_name": config["display_name"],
        "label": CLASS_NAMES[class_id],
        "confidence": float(probabilities[predicted_index]),
        "top3": top3,
        "heatmap": heatmap_image,
        "overlay": overlay,
    }
