"""Reusable rendering and execution helpers for the Experiments page."""

import pandas as pd
import streamlit as st

from src.config import MODEL_REGISTRY
from app.app_pages.common import DISPLAY_IMAGE_SIZE, fit_image_to_box


def model_display_name(model_name: str) -> str:
    return MODEL_REGISTRY[model_name]["display_name"]


def execute_predictions(model_names: list[str], image, predictor, progress=None) -> dict:
    """Run selected models independently and retain failures in the result."""
    results = {}
    for model_name in model_names:
        display_name = model_display_name(model_name)
        if progress is not None:
            progress(f"Running {display_name}...")
        try:
            results[model_name] = {
                "status": "success",
                "result": predictor(model_name, image),
            }
            if progress is not None:
                progress(f"Completed {display_name}.")
        except Exception as exc:
            results[model_name] = {
                "status": "error",
                "error": f"{display_name} failed: {exc}",
            }
            if progress is not None:
                progress(f"{display_name} failed.")
    return results


def render_model_summary(model_name: str, entry: dict) -> None:
    """Render a compact result card without detailed prediction outputs."""
    config = MODEL_REGISTRY[model_name]
    with st.container(border=True):
        st.subheader(config["display_name"])
        if entry["status"] == "error":
            st.error("Prediction failed")
            st.caption(entry["error"])
            return

        st.success("Completed")
        result = entry["result"]
        metric_columns = st.columns(2)
        with metric_columns[0]:
            st.metric("Prediction", result["label"])
        with metric_columns[1]:
            st.metric(result["confidence_label"], result["confidence_display"])
        if not result.get("accepted", True):
            st.warning(
                f"Below calibrated threshold ({result['threshold']:.3f}); "
                "prediction marked UNCERTAIN."
            )


def render_model_details(model_name: str, entry: dict) -> None:
    """Render detailed output for the selected model only."""
    st.subheader(f"Details: {model_display_name(model_name)}")
    if entry["status"] == "error":
        st.error(entry["error"])
        return

    result = entry["result"]
    if not result.get("accepted", True):
        st.warning(
            f"The raw prediction is **{result['raw_label']}**, but its "
            f"{result['confidence_label'].lower()} ({result['confidence_display']}) "
            f"is below the calibrated threshold ({result['threshold']:.3f})."
        )
    st.dataframe(
        pd.DataFrame(result["top3"]),
        hide_index=True,
        width="stretch",
    )

    if result["gradcam_available"]:
        view_columns = st.columns(2)
        with view_columns[0]:
            st.image(
                fit_image_to_box(result["heatmap"]),
                caption="Grad-CAM heatmap",
                width=DISPLAY_IMAGE_SIZE,
            )
        with view_columns[1]:
            st.image(
                fit_image_to_box(result["overlay"]),
                caption="Grad-CAM overlay",
                width=DISPLAY_IMAGE_SIZE,
            )
    else:
        st.info(result["gradcam_message"])
