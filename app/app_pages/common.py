import numpy as np
import streamlit as st
from PIL import Image

from src.config import (
    CLASS_NAMES,
    MODEL_CLASS_IDS,
    MODEL_REGISTRY,
)


DISPLAY_IMAGE_SIZE = 360


def fit_image_to_box(
    image: Image.Image,
    size: int = DISPLAY_IMAGE_SIZE,
    background: tuple[int, int, int] = (245, 245, 245),
) -> Image.Image:
    """Fit an image into a fixed square without stretching it."""
    fitted = image.convert("RGB")
    fitted.thumbnail((size, size), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (size, size), background)
    position = (
        (size - fitted.width) // 2,
        (size - fitted.height) // 2,
    )
    canvas.paste(fitted, position)
    return canvas


@st.cache_resource(show_spinner=False)
def get_model(model_name: str, model_path: str):
    """Cache a saved model and invalidate it when its file timestamp changes."""
    # Load the ML backend only when a prediction is requested. This keeps
    # navigation and the Reports page bootable on deployment environments
    # where TensorFlow is unavailable or fails to initialize.
    from src.inference.model_loader import load_model_by_name

    del model_path
    return load_model_by_name(model_name)


def available_models() -> list[str]:
    return [
        name
        for name in MODEL_REGISTRY
        if MODEL_REGISTRY[name]["path"].is_file()
    ]


def create_gradcam_images(
    original_image: Image.Image,
    heatmap: np.ndarray,
) -> tuple[Image.Image, Image.Image]:
    base = original_image.convert("RGB")
    import matplotlib

    resized_heatmap = Image.fromarray(np.uint8(heatmap * 255), mode="L").resize(
        base.size,
        Image.Resampling.BILINEAR,
    )
    normalized = np.asarray(resized_heatmap, dtype=np.float32) / 255.0
    color = np.uint8(matplotlib.colormaps["jet"](normalized)[..., :3] * 255)
    heatmap_image = Image.fromarray(color, mode="RGB")
    return heatmap_image, Image.blend(base, heatmap_image, alpha=0.4)


def run_prediction(model_name: str, image: Image.Image) -> dict:
    from src.inference.model_loader import (
        load_classical_threshold,
        load_model_by_name,
    )

    config = MODEL_REGISTRY[model_name]
    model_path = config["path"]
    model = get_model(model_name, f"{model_path}:{model_path.stat().st_mtime_ns}")

    if config["type"] == "ml":
        # The notebook's cv2 pipeline consumes BGR images.
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)[..., ::-1][None, ...]
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(array)[0]
            confidence_type = "probability"
        else:
            scores = model.decision_function(array)[0]
            confidence_type = "margin"

        threshold = load_classical_threshold(model_name)
        class_labels = getattr(model, "classes_", None)
        if class_labels is None and hasattr(model, "named_steps"):
            class_labels = model.named_steps["clf"].classes_
        class_labels = np.asarray(class_labels)
        top_indices = np.argsort(scores)[-3:][::-1]
        top_class_ids = class_labels[top_indices]
        raw_predicted_class_id = int(top_class_ids[0])
        predicted_class_id = raw_predicted_class_id
        score_label = (
            "Confidence" if confidence_type == "probability" else "Decision margin"
        )
        top3 = []
        for index, class_id in zip(top_indices, top_class_ids):
            value = float(scores[index])
            top3.append(
                {
                    "Class": CLASS_NAMES[int(class_id)],
                    score_label: (
                        f"{value:.2%}"
                        if confidence_type == "probability"
                        else f"{value:.3f}"
                    ),
                }
            )

        if confidence_type == "probability":
            confidence = float(scores[top_indices[0]])
        else:
            ordered_scores = np.sort(scores)
            confidence = float(ordered_scores[-1] - ordered_scores[-2])
        accepted = confidence >= threshold
        return {
            "display_name": config["display_name"],
            "label": CLASS_NAMES[predicted_class_id] if accepted else "UNCERTAIN",
            "raw_label": CLASS_NAMES[raw_predicted_class_id],
            "confidence": confidence,
            "threshold": threshold,
            "accepted": accepted,
            "confidence_label": score_label,
            "confidence_display": (
                f"{confidence:.2%}"
                if confidence_type == "probability"
                else f"{confidence:.3f}"
            ),
            "top3": top3,
            "heatmap": None,
            "overlay": None,
            "gradcam_available": False,
            "gradcam_message": (
                "Grad-CAM is unavailable for this classical ML pipeline. "
                "The prediction uses its saved feature pipeline."
            ),
        }

    from src.inference.preprocessing import preprocess_image

    tensor = preprocess_image(image, config["image_size"])
    if config["type"] == "tflite":
        probabilities = model.predict(tensor.numpy())[0]
    else:
        probabilities = model(tensor, training=False).numpy()[0]
    top_indices = np.argsort(probabilities)[-3:][::-1]
    predicted_index = int(top_indices[0])
    class_id = MODEL_CLASS_IDS[predicted_index]

    heatmap_image = None
    overlay = None
    if config.get("supports_gradcam", config["type"] == "cnn"):
        from src.xai.gradcam import make_gradcam_heatmap
        from src.xai.targets import get_target_layer

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
        "confidence_label": "Confidence",
        "confidence_display": f"{float(probabilities[predicted_index]):.2%}",
        "top3": top3,
        "heatmap": heatmap_image,
        "overlay": overlay,
        "gradcam_available": heatmap_image is not None,
        "gradcam_message": (
            "Grad-CAM is unavailable for this TensorFlow Lite deployment "
            "artifact. Its prediction is generated by the quantized model."
        ),
    }
