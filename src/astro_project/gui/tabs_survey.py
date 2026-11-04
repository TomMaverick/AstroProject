import pandas as pd
import streamlit as st
import torch
from astropy.time import Time

from astro_project.core.hardware import resolve_device
from astro_project.pipeline.data_fetcher import (
    SURVEY_TELESCOPE_PRESETS,
    generate_or_fetch_survey_fields,
)
from astro_project.pipeline.survey_engine import run_automated_survey_batch


def render_survey_tab(device: str | torch.device):
    """Renders the Automated Sky Survey tab with telescope presets, ledger deduplication, and 3-tier storage."""
    st.markdown("## \U0001f6f0\ufe0f Automated Sky Survey & Tiered Discovery Engine")
    st.caption(
        "Scans multi-field telescope archives on the GPU, filters cosmic rays via temporal coherence gating, "
        "deduplicates via 64-bit Blake2b SQLite ledger, and auto-purges empty staging fields."
    )

    data_source = st.radio(
        "Data Source",
        options=["Synthetic Benchmark (Mock Data)", "Multi-Teleskop Archive (Live)"],
        horizontal=True,
        key="survey_data_source"
    )

    preset_keys = list(SURVEY_TELESCOPE_PRESETS.keys())

    if data_source == "Multi-Teleskop Archive (Live)":
        st.markdown("### Archive Query Parameters \u2604\ufe0f")
        date_input_method = st.radio(
            "Date Input Method",
            options=["Calendar \U0001f4c5", "MJD \u23f1\ufe0f"],
            horizontal=True,
            key="date_input_method"
        )

        if date_input_method == "Calendar \U0001f4c5":
            col_ztf1, col_ztf2 = st.columns(2)
            with col_ztf1:
                ztf_start_date = st.date_input("Start Date", value=pd.Timestamp.now().date(), key="ztf_start_date")
            with col_ztf2:
                ztf_end_date = st.date_input("End Date", value=pd.Timestamp.now().date() + pd.Timedelta(days=1), key="ztf_end_date")
            ztf_mjd_start = Time(str(ztf_start_date)).mjd
            ztf_mjd_end = Time(str(ztf_end_date)).mjd
        else:
            col_ztf3, col_ztf4 = st.columns(2)
            with col_ztf3:
                ztf_mjd_start = st.number_input("MJD Start", value=58200.0, key="ztf_mjd_start")
            with col_ztf4:
                ztf_mjd_end = st.number_input("MJD End", value=58200.5, key="ztf_mjd_end")

        n_fields = 1
    else:
        ztf_ra = ztf_dec = ztf_mjd_start = ztf_mjd_end = None

    with st.expander("Teleskop-Leitfaden & Empfehlungen \U0001f52d"):
        st.markdown(
            "- **ZTF Palomar 48-inch**: Weitfeld-Survey, hohe Kadenz (High-Cadence). Ideal fuer die Entdeckung von Near-Earth Asteroids (NEAs) und schnellen Transienten.\n"
            "- **Pan-STARRS PS1**: Hohe Aufloesung (High-Resolution), tiefe Aufnahmen. Perfekt fuer praezise Astrometrie und schwache Objekte im aeusseren Sonnensystem.\n"
            "- **Catalina CSS Schmidt**: Klassisches Weitfeld mit groeberer Pixelskala. Bewaehrt fuer grossflaechige Himmelsdurchmusterungen."
        )

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        telescope = st.selectbox(
            "Survey Telescope Preset",
            options=preset_keys,
            format_func=lambda k: f"{k} ({SURVEY_TELESCOPE_PRESETS[k]['label']})",
            key="survey_telescope_select",
        )
        if data_source == "Synthetic Benchmark (Mock Data)":
            n_fields = st.number_input(
                "Number of Fields (n_fields)",
                min_value=1,
                max_value=50,
                value=5,
                step=1,
                key="survey_n_fields",
            )

    with col_cfg2:
        min_keep_score = st.slider(
            "Min Keep Score (Silver Tier Threshold)",
            min_value=0.0,
            max_value=100.0,
            value=45.0,
            step=1.0,
            key="survey_min_keep_score",
        )
        stack_detection_sigma = st.slider(
            "Stack Detection Threshold (\u03c3)",
            min_value=2.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            key="survey_stack_sigma",
        )

    with col_cfg3:
        preset_info = SURVEY_TELESCOPE_PRESETS[telescope]
        st.markdown(
            f"**Pixel Scale:** `{preset_info['pixel_scale_arcsec']}` arcsec/px  \n"
            f"**Cadence (\u0394t):** `{preset_info['default_dt_hours']}` h  \n"
            f"**MPC Code:** `{preset_info['obs_code']}`"
        )
        auto_purge = st.checkbox(
            "Auto-Purge Empty & Raw Staging FITS",
            value=True,
            key="survey_auto_purge",
        )

    with st.form(key="survey_form"):
        submitted = st.form_submit_button(label="\U0001f680 Start Automated Archive Survey", type="primary", use_container_width=True)

        if submitted:
            with st.spinner("Running Archive Scan... \u231b"):
                dev = resolve_device(st.session_state.get("selected_dev", "cpu"))
                batches = generate_or_fetch_survey_fields(telescope=telescope, n_fields=2, active_field_indices=[1], device=dev)
                summary = run_automated_survey_batch(field_batches=batches, device=dev)
            st.success(f"Scan complete \u2705 Gold: {summary['gold_saved']} | Silver: {summary['silver_saved']} | Purged: {summary['empty_purged']}")
