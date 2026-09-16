import hashlib

import streamlit as st
from PIL import Image

from src.config import MODEL_REGISTRY
from app.app_pages.common import (
    DISPLAY_IMAGE_SIZE,
    available_models,
    fit_image_to_box,
    run_prediction,
)
from app.app_pages.experiment_ui import (
    execute_predictions,
    model_display_name,
    render_model_details,
    render_model_summary,
)


RESULTS_STATE_KEY = "experiment_results"
IMAGE_KEY_STATE_KEY = "experiment_image_key"
DETAIL_MODEL_STATE_KEY = "experiment_detail_model"


def get_image_key(uploaded_file) -> str:
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


st.title("Experiments")
st.caption("Compare selected traffic-sign recognition models on one uploaded image.")

available = available_models()
missing = [model_name for model_name in MODEL_REGISTRY if model_name not in available]

if missing:
    missing_names = ", ".join(model_display_name(name) for name in missing)
    st.info(f"Missing model artifacts: {missing_names}")

with st.container(border=True):
    with st.form("experiment_form"):
        upload_column, model_column = st.columns(2)
        with upload_column:
            uploaded_file = st.file_uploader(
                "Upload a traffic-sign image",
                type=["jpg", "jpeg", "png"],
            )
        with model_column:
            selected_models = st.multiselect(
                "Models to run",
                options=available,
                default=available,
                format_func=model_display_name,
                help="Choose one model or compare several models at once.",
            )
        submitted = st.form_submit_button(
            "Run selected models",
            type="primary",
        )

if uploaded_file is None:
    st.info("Upload an image to begin an experiment.")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()

with st.container(horizontal_alignment="center"):
    st.image(
        fit_image_to_box(image),
        caption=f"Uploaded image ({image.width} × {image.height}px)",
        width=DISPLAY_IMAGE_SIZE,
    )

current_image_key = get_image_key(uploaded_file)
if st.session_state.get(IMAGE_KEY_STATE_KEY) != current_image_key:
    st.session_state[IMAGE_KEY_STATE_KEY] = current_image_key
    st.session_state[RESULTS_STATE_KEY] = {}

if submitted:
    if not selected_models:
        st.warning("Select at least one available model before running the experiment.")
        st.stop()

    with st.status(
        f"Running {len(selected_models)} selected model(s)...",
        expanded=True,
    ) as status:
        results = execute_predictions(
            selected_models,
            image,
            run_prediction,
            progress=status.write,
        )
        status.update(
            label=f"Finished {len(selected_models)} model(s)",
            state="complete",
            expanded=False,
        )

    st.session_state[RESULTS_STATE_KEY] = results
    st.session_state[DETAIL_MODEL_STATE_KEY] = next(
        (
            model_name
            for model_name, entry in results.items()
            if entry["status"] == "success"
        ),
        selected_models[0],
    )

results = st.session_state.get(RESULTS_STATE_KEY, {})
if not results:
    st.info("Press Run selected models to generate predictions.")
    st.stop()

st.subheader("Model summary")
st.caption("Select a model below to inspect its Top-3 predictions and visual explanations.")

result_model_names = list(results)
for start in range(0, len(result_model_names), 3):
    summary_columns = st.columns(min(3, len(result_model_names) - start))
    for column, model_name in zip(
        summary_columns,
        result_model_names[start : start + 3],
    ):
        with column:
            render_model_summary(model_name, results[model_name])

if st.session_state.get(DETAIL_MODEL_STATE_KEY) not in result_model_names:
    st.session_state[DETAIL_MODEL_STATE_KEY] = result_model_names[0]

detail_model = st.selectbox(
    "Model details",
    options=result_model_names,
    format_func=model_display_name,
    key=DETAIL_MODEL_STATE_KEY,
)
render_model_details(detail_model, results[detail_model])
