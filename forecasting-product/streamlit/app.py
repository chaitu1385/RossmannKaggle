"""
Forecasting Product — Streamlit Dashboard

Main entry point.  Run with:
    streamlit run forecasting-product/streamlit/app.py

Or via Docker Compose:
    docker compose up
"""

import sys
from pathlib import Path

# Ensure the forecasting-product package is importable.
_PLATFORM_ROOT = Path(__file__).resolve().parent.parent
if str(_PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_ROOT))

import streamlit as st

# ---------------------------------------------------------------------------
#  Apply SWD Plotly theme globally (before any page renders charts)
# ---------------------------------------------------------------------------
try:
    from src.visualization.plotly_theme import apply_swd_plotly_theme
    apply_swd_plotly_theme()
except ImportError:
    pass  # plotly not installed — charts will use Plotly defaults

# ---------------------------------------------------------------------------
#  Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Forecasting Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
#  Explicit page registration (replaces file-based auto-discovery)
# ---------------------------------------------------------------------------
_PAGES_DIR = Path(__file__).resolve().parent / "_pages"

pages = st.navigation(
    [
        st.Page("home.py", title="Home", icon="📈", default=True),
        st.Page(
            _PAGES_DIR / "1_Data_Onboarding.py",
            title="Data Onboarding",
            icon="📊",
        ),
        st.Page(
            _PAGES_DIR / "2_Series_Explorer.py",
            title="Series Explorer",
            icon="🔍",
        ),
        st.Page(
            _PAGES_DIR / "3_SKU_Transitions.py",
            title="SKU Transitions",
            icon="🔄",
        ),
        st.Page(
            _PAGES_DIR / "4_Hierarchy_Manager.py",
            title="Hierarchy Manager",
            icon="🌳",
        ),
        st.Page(
            _PAGES_DIR / "5_Backtest_Results.py",
            title="Backtest Results",
            icon="🏆",
        ),
        st.Page(
            _PAGES_DIR / "6_Forecast_Viewer.py",
            title="Forecast Viewer",
            icon="🔮",
        ),
        st.Page(
            _PAGES_DIR / "7_Platform_Health.py",
            title="Platform Health",
            icon="🏥",
        ),
        st.Page(
            _PAGES_DIR / "8_SOP_Meeting.py",
            title="S&OP Meeting Prep",
            icon="📋",
        ),
    ]
)

pages.run()
