import pandas as pd
import streamlit as st
import torch

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

    preset_keys = list(SURVEY_TELESCOPE_PRESETS.keys())

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        telescope = st.selectbox(
            "Survey Telescope Preset",
            options=preset_keys,
            format_func=lambda k: f"{k} ({SURVEY_TELESCOPE_PRESETS[k]['label']})",
            key="survey_telescope_select",
        )
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

    if st.button(
        "\U0001f680 Start Automated Archive Survey",
        type="primary",
        use_container_width=True,
        key="start_automated_survey_btn",
    ):
        with st.spinner("Staging survey fields & running GPU velocity search..."):
            active_indices = [i for i in (1, 3) if i < int(n_fields)]
            if not active_indices and int(n_fields) > 0:
                active_indices = [0]

            field_batches = generate_or_fetch_survey_fields(
                base_data_dir="data",
                telescope=telescope,
                n_fields=int(n_fields),
                active_field_indices=active_indices,
                device=device,
            )

            results = run_automated_survey_batch(
                field_batches=field_batches,
                base_data_dir="data",
                device=device,
                stack_detection_sigma=float(stack_detection_sigma),
                min_keep_score=float(min_keep_score),
                auto_purge=bool(auto_purge),
            )
            st.session_state["survey_last_results"] = results

    results = st.session_state.get("survey_last_results")
    if results is not None:
        st.markdown("---")
        st.markdown("### \U0001f4ca Survey Batch Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scanned Fields", int(results.get("scanned", 0)))
        m2.metric("Empty/Purged", int(results.get("empty_purged", 0)))
        m3.metric("\u2b50 Gold Candidates", int(results.get("gold_saved", 0)))
        m4.metric("\U0001f948 Silver Candidates", int(results.get("silver_saved", 0)))

        candidates_list = results.get("candidates_list", [])
        if candidates_list:
            st.markdown("### \U0001f3af Discovered Candidates")
            display_cols = ["field_id", "tier", "score", "coherence", "peak"]
            df_cands = pd.DataFrame(candidates_list)
            available_cols = [c for c in display_cols if c in df_cands.columns]
            st.dataframe(df_cands[available_cols], use_container_width=True)
        else:
            st.info("No candidates above the keep score threshold in this batch.")
