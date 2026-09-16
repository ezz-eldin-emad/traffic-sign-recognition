import json

import pandas as pd
import streamlit as st
from PIL import Image

from src.config import COMPARISON_PATH, MODEL_REGISTRY, REPORTS_DIR
from app.app_pages.common import fit_image_to_box


@st.cache_data
def load_comparison(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return pd.read_csv(path)


@st.cache_data
def load_json(path: str, modified_ns: int) -> dict | None:
    del modified_ns
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def file_json(path):
    return load_json(str(path), path.stat().st_mtime_ns) if path.is_file() else None


def render_model_cards(comparison: pd.DataFrame) -> None:
    model_names = list(MODEL_REGISTRY)
    for start in range(0, len(model_names), 3):
        columns = st.columns(3)
        for column, model_name in zip(columns, model_names[start : start + 3]):
            config = MODEL_REGISTRY[model_name]
            with column:
                with st.container(border=True):
                    st.subheader(config["display_name"])
                    row = (
                        comparison.loc[comparison["model"] == model_name]
                        if not comparison.empty
                        else pd.DataFrame()
                    )
                    if row.empty:
                        st.info("No report yet")
                        continue

                    values = row.iloc[0]
                    st.metric("Top-1", f"{float(values['top1_accuracy']):.2%}")
                    st.metric("Macro F1", f"{float(values['macro_f1']):.2%}")


def render_comparison(comparison: pd.DataFrame) -> None:
    columns = [
        "display_name",
        "model_type",
        "model_size",
        "top1_accuracy",
        "top5_accuracy",
        "macro_f1",
        "latency_p95_ms",
    ]
    display = comparison[[column for column in columns if column in comparison]].copy()
    for column in ("top1_accuracy", "top5_accuracy", "macro_f1"):
        if column in display:
            display[column] = display[column].map(
                lambda value: f"{float(value):.2%}" if pd.notna(value) else "—"
            )
    if "latency_p95_ms" in display:
        display["latency_p95_ms"] = display["latency_p95_ms"].map(
            lambda value: f"{float(value):.2f} ms" if pd.notna(value) else "—"
        )
    st.subheader("Comparison")
    st.dataframe(display, hide_index=True, width="stretch")


def render_details(model_name: str) -> None:
    report_dir = REPORTS_DIR / model_name
    metrics = file_json(report_dir / "metrics.json")
    if metrics is None:
        st.info("This model has no saved report yet.")
        return

    metric_columns = st.columns(4)
    metric_items = (
        ("Top-1", "top1_accuracy"),
        ("Top-5", "top5_accuracy"),
        ("Macro F1", "macro_f1"),
        ("P95 latency", "latency_p95_ms"),
    )
    for column, (label, key) in zip(metric_columns, metric_items):
        with column:
            value = metrics.get(key)
            if value is None:
                st.metric(label, "—")
            elif "latency" in key:
                st.metric(label, f"{float(value):.2f} ms")
            else:
                st.metric(label, f"{float(value):.2%}")

    confusion_path = report_dir / "confusion_matrix.png"
    if confusion_path.is_file():
        confusion_image = fit_image_to_box(
            Image.open(confusion_path),
            size=650,
            background=(255, 255, 255),
        )
        with st.container(horizontal_alignment="center"):
            st.image(confusion_image, caption="Confusion matrix", width=650)

    report_path = report_dir / "classification_report.txt"
    if report_path.is_file():
        with st.expander("Classification report"):
            st.code(report_path.read_text(encoding="utf-8"))

    robustness_path = report_dir / "robustness.csv"
    if robustness_path.is_file():
        with st.expander("Robustness results"):
            st.dataframe(pd.read_csv(robustness_path), hide_index=True, width="stretch")

    latency = file_json(report_dir / "latency.json")
    if latency is not None:
        with st.expander("Latency benchmark"):
            st.json(latency)


st.title("Model reports")
st.caption("Results are read from the saved files in reports/.")

if COMPARISON_PATH.is_file():
    comparison = load_comparison(
        str(COMPARISON_PATH),
        COMPARISON_PATH.stat().st_mtime_ns,
    )
else:
    comparison = pd.DataFrame()

render_model_cards(comparison)
if comparison.empty:
    st.warning("model_comparison.csv is not available yet.")
    st.stop()

render_comparison(comparison)
st.subheader("Model details")
model_name = st.selectbox(
    "Choose a model",
    list(MODEL_REGISTRY),
    format_func=lambda name: MODEL_REGISTRY[name]["display_name"],
)
render_details(model_name)
