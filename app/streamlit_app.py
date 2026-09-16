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
