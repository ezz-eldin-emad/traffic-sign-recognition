import base64
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


st.set_page_config(
    page_title="Traffic sign recognition",
    page_icon=":material/traffic:",
    layout="wide",
)


BACKGROUND_IMAGE = PROJECT_ROOT / "app" / "city-vehicle-map-road-traffic-transport-powerpoint-background.avif"


def apply_app_background() -> None:
    """Add the local project image as a readable, theme-aware app background."""
    if not BACKGROUND_IMAGE.is_file():
        return

    encoded_image = base64.b64encode(BACKGROUND_IMAGE.read_bytes()).decode("ascii")
    theme_type = getattr(st.context.theme, "type", "dark")
    if theme_type == "light":
        overlay = "rgba(244, 247, 251, 0.88), rgba(244, 247, 251, 0.94)"
        panel = "rgba(255, 255, 255, 0.90)"
    else:
        overlay = "rgba(10, 18, 32, 0.78), rgba(10, 18, 32, 0.90)"
        panel = "rgba(16, 24, 39, 0.88)"

    st.html(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: linear-gradient({overlay}),
                url("data:image/avif;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        [data-testid="stMainBlockContainer"] {{
            max-width: 1120px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}
        [data-testid="stSidebar"] {{
            background: {panel};
        }}
        @media (max-width: 640px) {{
            [data-testid="stMainBlockContainer"] {{
                padding: 1rem 0.75rem 1.5rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_javascript=False,
    )


apply_app_background()

pages = {
    "Recognition": [
        st.Page(
            "app_pages/experiments.py",
            title="Experiments",
            icon=":material/science:",
        ),
        st.Page(
            "app_pages/reports.py",
            title="Reports",
            icon=":material/analytics:",
        ),
    ],
}

navigation = st.navigation(pages, position="top")
navigation.run()
