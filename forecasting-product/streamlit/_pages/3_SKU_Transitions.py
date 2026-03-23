"""
Page 3 — SKU Transitions

Run the SKU mapping pipeline to discover product transitions, manage planner
overrides, and visualize transition ramp shapes.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_PLATFORM_ROOT = Path(__file__).resolve().parent.parent.parent
_STREAMLIT_DIR = Path(__file__).resolve().parent.parent
if str(_PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_ROOT))
if str(_STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_DIR))

import polars as pl

from utils import COLORS, format_number, format_pct, polars_to_pandas

st.title("SKU Transitions")
st.markdown(
    "Discover product transitions using the SKU mapping pipeline, manage "
    "planner overrides, and preview ramp shapes for demand transfer."
)

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
_CONFIDENCE_COLORS = {
    "High": COLORS["success"],
    "Medium": COLORS["warning"],
    "Low": COLORS["danger"],
    "Very Low": COLORS["neutral"],
}


def _generate_ramp_curve(
    shape: str, n_periods: int = 13, proportion: float = 1.0,
) -> pl.DataFrame:
    """Generate a synthetic ramp curve for visualization."""
    import numpy as np

    periods = list(range(n_periods))
    t = np.linspace(0, 1, n_periods)

    if shape == "step":
        values = [proportion] * n_periods
    elif shape == "exponential":
        values = (proportion * (np.exp(3 * t) - 1) / (np.exp(3) - 1)).tolist()
    else:  # linear
        values = (proportion * t).tolist()

    return pl.DataFrame({
        "period": periods,
        "proportion": values,
    })


# =========================================================================== #
#  Section 1: SKU Mapping Pipeline
# =========================================================================== #
st.divider()
st.header("SKU Mapping Pipeline")

# Sidebar controls
with st.sidebar:
    st.subheader("Pipeline Parameters")
    launch_window_days = st.slider(
        "Launch window (days)",
        min_value=90,
        max_value=365,
        value=180,
        help="Maximum days between old-SKU discontinuation and new-SKU launch.",
    )
    min_base_similarity = st.slider(
        "Min base similarity",
        min_value=0.50,
        max_value=1.0,
        value=0.70,
        step=0.05,
        help="Minimum naming similarity score (0-1) for candidate pairs.",
    )
    min_confidence = st.selectbox(
        "Min confidence",
        options=["Low", "Medium", "High"],
        index=0,
        help="Drop candidates below this confidence level.",
    )

# File uploads
col_master, col_phase = st.columns([2, 1])

with col_master:
    product_master_file = st.file_uploader(
        "Upload product master CSV",
        type=["csv"],
        key="product_master_upload",
        help="CSV with product attributes (sku_id, description, category, launch_date, etc.).",
    )

with col_phase:
    phase = st.radio(
        "Pipeline Phase",
        ["Phase 1 (Attribute + Naming)", "Phase 2 (Full)"],
        help=(
            "Phase 1 uses attribute matching and naming conventions. "
            "Phase 2 adds curve fitting and temporal comovement (requires sales history)."
        ),
    )

# Optional sales history for Phase 2
sales_df = None
if phase == "Phase 2 (Full)":
    sales_file = st.file_uploader(
        "Upload sales history CSV (optional for Phase 2)",
        type=["csv"],
        key="sales_history_upload",
        help="Weekly sales history with columns: sku_id, week, quantity.",
    )
    if sales_file is not None:
        try:
            sales_df = pl.read_csv(
                sales_file.getvalue(), try_parse_dates=True,
            )
            st.success(
                f"Sales history loaded: {sales_df.shape[0]:,} rows x "
                f"{sales_df.shape[1]} columns."
            )
        except Exception as exc:
            st.error(f"Failed to parse sales history CSV: {exc}")

# Run pipeline
if product_master_file is not None:
    try:
        product_master = pl.read_csv(
            product_master_file.getvalue(), try_parse_dates=True,
        )
        st.success(
            f"Product master loaded: {product_master.shape[0]:,} rows x "
            f"{product_master.shape[1]} columns."
        )
    except Exception as exc:
        st.error(
            f"Failed to parse product master CSV: {exc}\n\n"
            "Ensure the file is a valid CSV with headers in the first row."
        )
        product_master = None

    if product_master is not None and st.button("Run Mapping", type="primary"):
        with st.status("Running SKU mapping pipeline...", expanded=True) as status:
            try:
                from src.sku_mapping.pipeline import (
                    build_phase1_pipeline,
                    build_phase2_pipeline,
                )

                if phase == "Phase 1 (Attribute + Naming)":
                    st.write("Building Phase 1 pipeline (attribute + naming)...")
                    pipeline = build_phase1_pipeline(
                        launch_window_days=launch_window_days,
                        min_base_similarity=min_base_similarity,
                        min_confidence=min_confidence,
                    )
                else:
                    st.write("Building Phase 2 pipeline (full)...")
                    pipeline = build_phase2_pipeline(
                        sales_df=sales_df,
                        launch_window_days=launch_window_days,
                        min_base_similarity=min_base_similarity,
                        min_confidence=min_confidence,
                    )

                st.write(
                    f"Processing {product_master.shape[0]:,} products with "
                    f"launch window = {launch_window_days} days..."
                )
                results = pipeline.run(product_master)

                st.session_state["sku_mapping_results"] = results
                status.update(label="Mapping complete!", state="complete")

                st.success(f"Discovered {len(results):,} SKU transition mappings.")

            except Exception as exc:
                status.update(label="Pipeline failed", state="error")
                st.error(
                    f"SKU mapping pipeline failed: {exc}\n\n"
                    "Common causes: missing required columns (sku_id, description, "
                    "category, launch_date), or incompatible data format."
                )

# Display results
results_df = st.session_state.get("sku_mapping_results")
if results_df is not None and not results_df.is_empty():
    st.subheader("Mapping Results")

    # Summary metrics
    n_mappings = len(results_df)
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Mappings", f"{n_mappings:,}")

    if "confidence" in results_df.columns:
        confidence_counts = results_df["confidence"].value_counts()
        high_count = confidence_counts.filter(
            pl.col("confidence") == "High"
        )
        high_n = int(high_count["count"][0]) if len(high_count) > 0 else 0
        col_m2.metric("High Confidence", f"{high_n:,}")
    if "score" in results_df.columns:
        avg_score = results_df["score"].mean()
        col_m3.metric("Avg Score", format_number(avg_score))

    # Results table
    st.dataframe(polars_to_pandas(results_df), use_container_width=True)

    # Confidence distribution pie chart
    if "confidence" in results_df.columns:
        st.subheader("Confidence Distribution")
        conf_counts = results_df["confidence"].value_counts()
        conf_pd = polars_to_pandas(conf_counts)

        color_map = {
            k: v for k, v in _CONFIDENCE_COLORS.items()
            if k in conf_pd["confidence"].values
        }
        fig_pie = px.pie(
            conf_pd,
            names="confidence",
            values="count",
            color="confidence",
            color_discrete_map=color_map,
            hole=0.4,
        )
        fig_pie.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Download button
    csv_data = polars_to_pandas(results_df).to_csv(index=False)
    st.download_button(
        label="Download Results CSV",
        data=csv_data,
        file_name="sku_mapping_results.csv",
        mime="text/csv",
    )

elif results_df is not None and results_df.is_empty():
    st.info(
        "Pipeline returned no mappings. Try lowering the minimum confidence "
        "or increasing the launch window."
    )

# =========================================================================== #
#  Section 2: Planner Override Management
# =========================================================================== #
st.divider()
st.header("Planner Overrides")

store = None
try:
    from src.overrides.store import get_override_store

    store = get_override_store()

    # Add Override form
    with st.form("add_override_form"):
        st.subheader("Add Override")
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            ovr_old_sku = st.text_input(
                "Old SKU",
                help="The discontinued SKU identifier.",
            )
            ovr_new_sku = st.text_input(
                "New SKU",
                help="The replacement SKU identifier.",
            )
            ovr_proportion = st.slider(
                "Proportion",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
                help="Fraction of demand to transfer (0.0 to 1.0).",
            )
            ovr_scenario = st.selectbox(
                "Scenario",
                options=["Scenario A", "Scenario B", "Scenario C", "manual"],
                index=3,
                help="Planning scenario for this override.",
            )

        with col_f2:
            ovr_ramp_shape = st.selectbox(
                "Ramp Shape",
                options=["linear", "step", "exponential"],
                help="Shape of the demand ramp-up curve.",
            )
            ovr_effective_date = st.date_input(
                "Effective Date",
                value=date.today(),
                help="Date when this override takes effect.",
            )
            ovr_notes = st.text_area(
                "Notes",
                help="Optional notes or justification for this override.",
            )

        submitted = st.form_submit_button("Add Override", type="primary")

        if submitted:
            if not ovr_old_sku or not ovr_new_sku:
                st.error("Both Old SKU and New SKU are required.")
            else:
                try:
                    override_id = store.add_override(
                        old_sku=ovr_old_sku,
                        new_sku=ovr_new_sku,
                        proportion=ovr_proportion,
                        scenario=ovr_scenario,
                        ramp_shape=ovr_ramp_shape,
                        effective_date=str(ovr_effective_date),
                        notes=ovr_notes if ovr_notes else None,
                    )
                    st.success(f"Override created: **{override_id}**")
                except Exception as exc:
                    st.error(f"Failed to create override: {exc}")

    # Current overrides table
    st.subheader("Current Overrides")
    try:
        overrides_df = store.get_all()
        if overrides_df is not None and not overrides_df.is_empty():
            st.dataframe(polars_to_pandas(overrides_df), use_container_width=True)

            # Delete override
            st.subheader("Delete Override")
            override_ids = overrides_df["override_id"].to_list()
            selected_override = st.selectbox(
                "Select override to delete",
                options=override_ids,
                help="Choose the override ID to remove.",
            )
            if st.button("Delete Selected Override", type="secondary"):
                try:
                    deleted = store.delete_override(selected_override)
                    if deleted:
                        st.success(f"Override **{selected_override}** deleted.")
                        st.rerun()
                    else:
                        st.warning(
                            f"Override **{selected_override}** not found or "
                            f"already deleted."
                        )
                except Exception as exc:
                    st.error(f"Failed to delete override: {exc}")
        else:
            st.info("No overrides recorded yet. Use the form above to add one.")
    except Exception as exc:
        st.error(f"Failed to load overrides: {exc}")

except ImportError as exc:
    st.warning(
        f"Override store module not available: {exc}. "
        "Install required dependencies to enable override management."
    )
except Exception as exc:
    st.error(f"Failed to initialize override store: {exc}")
finally:
    if store is not None:
        try:
            store.close()
        except Exception:
            pass

# =========================================================================== #
#  Section 3: Transition Visualization
# =========================================================================== #
st.divider()
st.header("Transition Visualization")

has_results = (
    st.session_state.get("sku_mapping_results") is not None
    and not st.session_state["sku_mapping_results"].is_empty()
)
actuals_df = st.session_state.get("actuals_df")
has_actuals = actuals_df is not None

if not has_results and not has_actuals:
    st.info(
        "Run the SKU mapping pipeline above or load actuals data via the "
        "Data Onboarding page to enable transition visualizations."
    )
else:
    # Pair selector
    if has_results:
        mapping_df = st.session_state["sku_mapping_results"]
        if "old_sku" in mapping_df.columns and "new_sku" in mapping_df.columns:
            pair_labels = [
                f"{row['old_sku']} -> {row['new_sku']}"
                for row in mapping_df.select("old_sku", "new_sku").to_dicts()
            ]
            selected_pair = st.selectbox(
                "Select SKU pair",
                options=pair_labels,
                help="Choose a transition pair to visualize.",
            )
            if selected_pair:
                old_sku, new_sku = selected_pair.split(" -> ", 1)
            else:
                old_sku, new_sku = None, None
        else:
            old_sku = st.text_input("Old SKU", key="viz_old_sku")
            new_sku = st.text_input("New SKU", key="viz_new_sku")
    else:
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            old_sku = st.text_input("Old SKU", key="viz_old_sku")
        with col_v2:
            new_sku = st.text_input("New SKU", key="viz_new_sku")

    # Side-by-side actuals chart
    if has_actuals and old_sku and new_sku:
        st.subheader("Historical Sales Comparison")
        try:
            # Detect SKU column name
            sku_col = None
            for candidate in ["sku_id", "product_id", "sku", "item_id"]:
                if candidate in actuals_df.columns:
                    sku_col = candidate
                    break

            # Detect time column name
            time_col = None
            for candidate in ["week", "date", "ds"]:
                if candidate in actuals_df.columns:
                    time_col = candidate
                    break

            # Detect target column name
            target_col = None
            for candidate in ["quantity", "sales", "demand", "y"]:
                if candidate in actuals_df.columns:
                    target_col = candidate
                    break

            if sku_col and time_col and target_col:
                old_data = actuals_df.filter(pl.col(sku_col) == old_sku)
                new_data = actuals_df.filter(pl.col(sku_col) == new_sku)

                if old_data.is_empty() and new_data.is_empty():
                    st.warning(
                        f"No actuals data found for either '{old_sku}' or "
                        f"'{new_sku}' in column '{sku_col}'."
                    )
                else:
                    fig = go.Figure()

                    if not old_data.is_empty():
                        old_pd = polars_to_pandas(
                            old_data.select(time_col, target_col).sort(time_col)
                        )
                        fig.add_trace(go.Scatter(
                            x=old_pd[time_col],
                            y=old_pd[target_col],
                            mode="lines+markers",
                            name=f"Old: {old_sku}",
                            line=dict(color=COLORS["danger"]),
                        ))

                    if not new_data.is_empty():
                        new_pd = polars_to_pandas(
                            new_data.select(time_col, target_col).sort(time_col)
                        )
                        fig.add_trace(go.Scatter(
                            x=new_pd[time_col],
                            y=new_pd[target_col],
                            mode="lines+markers",
                            name=f"New: {new_sku}",
                            line=dict(color=COLORS["success"]),
                        ))

                    fig.update_layout(
                        title=f"Sales: {old_sku} vs {new_sku}",
                        xaxis_title="Date",
                        yaxis_title=target_col.title(),
                        height=400,
                        margin=dict(t=40, b=40, l=40, r=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                missing = []
                if not sku_col:
                    missing.append("SKU identifier (sku_id, product_id)")
                if not time_col:
                    missing.append("time column (week, date, ds)")
                if not target_col:
                    missing.append("target column (quantity, sales, demand)")
                st.warning(
                    f"Cannot plot actuals — missing columns: {', '.join(missing)}."
                )
        except Exception as exc:
            st.error(f"Failed to generate actuals chart: {exc}")

    # Ramp shape preview
    st.subheader("Ramp Shape Preview")
    st.markdown(
        "Preview how demand transfers from the old SKU to the new SKU over time."
    )

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        preview_shape = st.selectbox(
            "Ramp shape",
            options=["linear", "step", "exponential"],
            key="ramp_preview_shape",
        )
    with col_r2:
        preview_periods = st.slider(
            "Transition periods",
            min_value=1,
            max_value=52,
            value=13,
            key="ramp_preview_periods",
        )
    with col_r3:
        preview_proportion = st.slider(
            "Target proportion",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            key="ramp_preview_proportion",
        )

    try:
        ramp_df = _generate_ramp_curve(
            shape=preview_shape,
            n_periods=preview_periods,
            proportion=preview_proportion,
        )
        ramp_pd = polars_to_pandas(ramp_df)

        fig_ramp = go.Figure()
        fig_ramp.add_trace(go.Scatter(
            x=ramp_pd["period"],
            y=ramp_pd["proportion"],
            mode="lines+markers",
            name="New SKU share",
            line=dict(color=COLORS["primary"], width=3),
            fill="tozeroy",
            fillcolor=f"rgba(67, 97, 238, 0.15)",
        ))
        fig_ramp.add_trace(go.Scatter(
            x=ramp_pd["period"],
            y=[preview_proportion - v for v in ramp_pd["proportion"]],
            mode="lines+markers",
            name="Old SKU share",
            line=dict(color=COLORS["neutral"], width=2, dash="dash"),
        ))
        fig_ramp.update_layout(
            title=f"Ramp Shape: {preview_shape.title()}",
            xaxis_title="Period",
            yaxis_title="Demand Proportion",
            yaxis=dict(range=[0, max(preview_proportion * 1.1, 0.1)]),
            height=350,
            margin=dict(t=40, b=40, l=40, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_ramp, use_container_width=True)
    except Exception as exc:
        st.error(f"Failed to generate ramp preview: {exc}")
