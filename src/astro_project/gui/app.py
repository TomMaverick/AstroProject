import sys
from pathlib import Path

# Ensure `src/` is in the Python path if `streamlit run src/astro_project/gui/app.py` is directly started
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st

from astro_project.gui.components import render_sidebar_controls
from astro_project.gui.styles import DARK_MODE_CSS
from astro_project.gui.tabs_inspector import render_tab_fits_inspector
from astro_project.gui.tabs_pipeline import render_tab_pipeline_grid_search


def main():
    st.set_page_config(page_title="AstroProject", page_icon="*", layout="wide")
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
    device, nav_selection = render_sidebar_controls()

    if nav_selection == "FITS-Inspector & Photometry":
        render_tab_fits_inspector(device)
    else:
        render_tab_pipeline_grid_search(device)


if __name__ == "__main__":
    main()
