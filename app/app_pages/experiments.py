import pandas as pd
import streamlit as st
from PIL import Image

from src.config import MODEL_REGISTRY
from app.app_pages.common import available_models, run_prediction


st.title("Experiments")
st.caption("Upload one image and compare the saved deep-learning models.")

with st.form("experiment_form"):
    uploaded_file = st.file_uploader(
        "Upload a traffic-sign image",
        type=["jpg", "jpeg", "png"],
    )
    submitted = st.form_submit_button("Run models", type="primary")

if uploaded_file is None:
    st.info("Upload an image to begin an experiment.")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()

st.image(image, caption="Uploaded image", width="stretch")
if not submitted:
    st.info("Press Run models to generate predictions and Grad-CAM views.")
    st.stop()

models = available_models()
if not models:
    st.error("No saved deep-learning model was found in models/.")
    st.stop()

result_columns = st.columns(len(models))
for column, model_name in zip(result_columns, models):
    with column:
        display_name = MODEL_REGISTRY[model_name]["display_name"]
        with st.container(border=True):
            st.subheader(display_name)
            try:
                with st.spinner(f"Running {display_name}..."):
                    result = run_prediction(model_name, image)
            except Exception as exc:
                st.error(f"{display_name} failed: {exc}")
                continue

            st.metric("Prediction", result["label"])
            st.metric("Confidence", f"{result['confidence']:.2%}")
            st.dataframe(
                pd.DataFrame(result["top3"]),
                hide_index=True,
                width="stretch",
            )
            st.image(result["heatmap"], caption="Grad-CAM heatmap", width="stretch")
            st.image(result["overlay"], caption="Grad-CAM overlay", width="stretch")
