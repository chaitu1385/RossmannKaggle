"""
Page 4 — Hierarchy Manager

Visualize hierarchy structure, run reconciliation, compare before/after.
Supports detected hierarchies from Data Onboarding or user-defined hierarchy
configuration.
"""

import sys
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

from src.config.schema import HierarchyConfig, ReconciliationConfig
from src.hierarchy.aggregator import HierarchyAggregator
from src.hierarchy.reconciler import Reconciler
from src.hierarchy.tree import HierarchyTree
from utils import COLORS, format_number, polars_to_pandas

st.set_page_config(page_title="Hierarchy Manager", page_icon="🌳", layout="wide")
st.title("Hierarchy Manager")
st.markdown(
    "Visualize hierarchy structure, explore aggregations across levels, "
    "and run forecast reconciliation to ensure coherent predictions."
)

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _detect_time_column(df: pl.DataFrame) -> str | None:
    """Try to find the time column in a DataFrame."""
    for col in df.columns:
        if col.lower() in ("week", "date", "ds", "time", "period", "month"):
            return col
        if df[col].dtype in (pl.Date, pl.Datetime):
            return col
    return None


def _detect_value_column(df: pl.DataFrame) -> str | None:
    """Try to find the primary numeric value column."""
    for col in df.columns:
        if col.lower() in ("quantity", "sales", "demand", "value", "target", "y"):
            if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64,
                                  pl.UInt32, pl.UInt64, pl.Int16, pl.Int8):
                return col
    # Fallback: first numeric column that is not obviously an ID
    for col in df.columns:
        if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64,
                              pl.UInt32, pl.UInt64):
            if not col.lower().endswith("_id") and col.lower() != "id":
                return col
    return None


def _build_sunburst_data(tree: HierarchyTree) -> pl.DataFrame:
    """Build a DataFrame suitable for px.sunburst from the hierarchy tree."""
    ids = []
    labels = []
    parents = []
    values = []

    # Add root
    root_id = tree.name
    ids.append(root_id)
    labels.append(tree.name)
    parents.append("")
    values.append(len(tree.get_leaves()))

    for level in tree.levels:
        for node in tree.get_nodes(level):
            node_id = f"{level}/{node.key}"
            ids.append(node_id)
            labels.append(node.key)

            # Determine parent id
            if node.parent is not None and node.parent.key == "__root__":
                parents.append(root_id)
            elif node.parent is not None:
                parents.append(f"{node.parent.level}/{node.parent.key}")
            else:
                parents.append(root_id)

            # Value = number of leaves under this node
            leaf_count = len(node.leaf_descendants())
            values.append(max(leaf_count, 1))

    return pl.DataFrame({
        "id": ids,
        "label": labels,
        "parent": parents,
        "value": values,
    })



# --------------------------------------------------------------------------- #
#  Section 1: Data & Hierarchy Setup
# --------------------------------------------------------------------------- #
st.divider()
st.header("Data & Hierarchy Setup")

actuals_df = st.session_state.get("actuals_df")
analysis_report = st.session_state.get("analysis_report")

if actuals_df is None:
    st.warning(
        "No data loaded. Go to **Data Onboarding** to upload data first, "
        "or the hierarchy manager will not be able to build a tree."
    )
    st.stop()

st.success(
    f"Using loaded data: {actuals_df.shape[0]:,} rows x {actuals_df.shape[1]} columns."
)

# Detect or define hierarchy
hierarchy_source = "manual"
detected_hierarchies = []

if analysis_report is not None:
    try:
        hier_info = analysis_report.hierarchy
        if hier_info and hier_info.hierarchies:
            detected_hierarchies = hier_info.hierarchies
            hierarchy_source = "detected"
    except (AttributeError, TypeError):
        pass

tree = None

if detected_hierarchies:
    st.info(
        f"Found {len(detected_hierarchies)} detected "
        f"{'hierarchy' if len(detected_hierarchies) == 1 else 'hierarchies'} "
        f"from Data Onboarding."
    )

    hierarchy_options = [h.name for h in detected_hierarchies]
    selected_name = st.selectbox(
        "Select hierarchy",
        options=hierarchy_options,
        help="Choose a detected hierarchy to visualize and manage.",
    )

    selected_hier = next(h for h in detected_hierarchies if h.name == selected_name)
    st.markdown(f"**Levels**: {' -> '.join(selected_hier.levels)}")

    # Build HierarchyConfig from detected hierarchy
    config = HierarchyConfig(
        name=selected_hier.name,
        levels=list(selected_hier.levels),
        id_column=selected_hier.levels[-1] if selected_hier.levels else "",
    )

    try:
        tree = HierarchyTree(config, actuals_df)
        st.success(f"Hierarchy tree built: {tree}")
    except Exception as exc:
        st.error(
            f"Failed to build hierarchy tree: {exc}\n\n"
            "The hierarchy columns may not be present in the loaded data. "
            "Try defining a hierarchy manually below."
        )
        tree = None

if tree is None:
    st.subheader("Define Hierarchy Manually")
    st.markdown(
        "Specify the hierarchy levels as column names from your data, "
        "ordered from broadest (top) to most granular (bottom)."
    )

    available_cols = [
        c for c in actuals_df.columns
        if actuals_df[c].dtype in (pl.Utf8, pl.Categorical)
        or (actuals_df[c].dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)
            and actuals_df[c].n_unique() < actuals_df.shape[0] * 0.5)
    ]

    # Filter out likely time/value columns
    time_col_guess = _detect_time_column(actuals_df)
    value_col_guess = _detect_value_column(actuals_df)
    available_cols = [
        c for c in available_cols
        if c != time_col_guess and c != value_col_guess
    ]

    if not available_cols:
        st.warning(
            "No suitable categorical or ID columns found in the data. "
            "Hierarchy requires columns with discrete values (e.g. region, "
            "category, store_id)."
        )
        st.stop()

    st.caption(f"Available columns: {', '.join(available_cols)}")

    levels_input = st.text_input(
        "Level columns (comma-separated, top to bottom)",
        value=", ".join(available_cols[:3]) if len(available_cols) >= 2 else "",
        help="Example: region, store_id — where region is the broader level.",
    )

    hier_name = st.text_input(
        "Hierarchy name",
        value="custom",
        help="A descriptive name for this hierarchy dimension.",
    )

    if levels_input.strip():
        levels = [l.strip() for l in levels_input.split(",") if l.strip()]

        if len(levels) < 2:
            st.warning("At least two levels are required to form a hierarchy.")
            st.stop()

        # Validate columns exist
        missing = [l for l in levels if l not in actuals_df.columns]
        if missing:
            st.error(
                f"Columns not found in data: {', '.join(missing)}. "
                f"Available: {', '.join(actuals_df.columns)}"
            )
            st.stop()

        config = HierarchyConfig(
            name=hier_name,
            levels=levels,
            id_column=levels[-1],
        )

        try:
            tree = HierarchyTree(config, actuals_df)
            st.success(f"Hierarchy tree built: {tree}")
        except Exception as exc:
            st.error(
                f"Failed to build hierarchy tree: {exc}\n\n"
                "Check that the level columns form a valid parent-child "
                "relationship (each child belongs to exactly one parent)."
            )
            st.stop()
    else:
        st.info("Enter level column names above to build a hierarchy tree.")
        st.stop()

if tree is None:
    st.stop()

# Store tree in session for potential reuse
st.session_state["hierarchy_tree"] = tree

# --------------------------------------------------------------------------- #
#  Section 2: Hierarchy Tree Visualization
# --------------------------------------------------------------------------- #
st.divider()
st.header("Hierarchy Tree")

# Tree stats
col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Levels", len(tree.levels))
col_s2.metric(
    "Total Leaves",
    format_number(len(tree.get_leaves()), decimals=0),
)

nodes_per_level = {level: len(tree.get_nodes(level)) for level in tree.levels}
total_nodes = sum(nodes_per_level.values())
col_s3.metric("Total Nodes", format_number(total_nodes, decimals=0))

# Nodes per level breakdown
st.markdown("**Nodes per level:**")
level_cols = st.columns(len(tree.levels))
for i, level in enumerate(tree.levels):
    level_cols[i].metric(level, nodes_per_level[level])

# Sunburst chart
try:
    sunburst_df = _build_sunburst_data(tree)
    sunburst_pd = polars_to_pandas(sunburst_df)

    fig_sunburst = px.sunburst(
        sunburst_pd,
        ids="id",
        labels="label",
        parents="parent",
        values="value",
        color_discrete_sequence=[
            COLORS["primary"], COLORS["secondary"], COLORS["accent"],
            COLORS["success"], COLORS["warning"], COLORS["neutral"],
        ],
    )
    fig_sunburst.update_layout(
        height=500,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_sunburst, use_container_width=True)
except Exception as exc:
    st.warning(
        f"Could not render sunburst chart: {exc}\n\n"
        "Falling back to text-based tree display."
    )

    # Fallback: indented tree display
    for level in tree.levels:
        indent = tree.levels.index(level)
        with st.expander(f"{'  ' * indent}Level: {level} ({nodes_per_level[level]} nodes)"):
            nodes = tree.get_nodes(level)
            display_nodes = nodes[:50]
            node_keys = [n.key for n in display_nodes]
            st.markdown(", ".join(f"`{k}`" for k in node_keys))
            if len(nodes) > 50:
                st.caption(f"...and {len(nodes) - 50} more nodes")

# Summing matrix preview
with st.expander("Summing Matrix (S)"):
    try:
        s_matrix = tree.summing_matrix()
        st.markdown(
            f"Matrix shape: **{s_matrix.shape[0]}** rows (all nodes) x "
            f"**{s_matrix.shape[1] - 2}** columns (leaf nodes)"
        )
        # Show a sample if the matrix is large
        if s_matrix.shape[0] > 50 or s_matrix.shape[1] > 20:
            st.caption("Showing first 50 rows and 20 columns.")
            display_cols = s_matrix.columns[:20]
            st.dataframe(
                polars_to_pandas(s_matrix.select(display_cols).head(50)),
                use_container_width=True,
            )
        else:
            st.dataframe(polars_to_pandas(s_matrix), use_container_width=True)
    except Exception as exc:
        st.error(f"Could not compute summing matrix: {exc}")

# --------------------------------------------------------------------------- #
#  Section 3: Aggregation Explorer
# --------------------------------------------------------------------------- #
st.divider()
st.header("Aggregation Explorer")

time_col = _detect_time_column(actuals_df)
value_col = _detect_value_column(actuals_df)

if time_col is None:
    st.warning("Could not detect a time column in the data. Aggregation requires a time column.")
elif value_col is None:
    st.warning("Could not detect a numeric value column in the data.")
else:
    st.caption(f"Using time column: `{time_col}` | value column: `{value_col}`")

    # Exclude leaf level from target options (already at that level)
    non_leaf_levels = tree.levels[:-1]

    if not non_leaf_levels:
        st.info("Only one hierarchy level — no aggregation possible.")
    else:
        target_level = st.selectbox(
            "Aggregate to level",
            options=non_leaf_levels,
            help="Select a hierarchy level to aggregate leaf-level data up to.",
        )

        top_n = st.slider(
            "Top N nodes to display",
            min_value=1,
            max_value=min(20, len(tree.get_nodes(target_level))),
            value=min(5, len(tree.get_nodes(target_level))),
        )

        try:
            aggregator = HierarchyAggregator(tree)
            agg_df = aggregator.aggregate_to(
                df=actuals_df,
                target_level=target_level,
                value_columns=[value_col],
                time_column=time_col,
                agg="sum",
            )

            if agg_df.is_empty():
                st.warning("Aggregation returned no results. Check that leaf-level data matches the hierarchy.")
            else:
                # Find top N nodes by total value
                totals = (
                    agg_df.group_by(target_level)
                    .agg(pl.col(value_col).sum().alias("_total"))
                    .sort("_total", descending=True)
                    .head(top_n)
                )
                top_keys = totals[target_level].to_list()

                plot_df = agg_df.filter(pl.col(target_level).is_in(top_keys))
                plot_pd = polars_to_pandas(plot_df)

                fig_agg = px.line(
                    plot_pd,
                    x=time_col,
                    y=value_col,
                    color=target_level,
                    title=f"Aggregated {value_col} by {target_level} (Top {top_n})",
                    color_discrete_sequence=[
                        COLORS["primary"], COLORS["accent"], COLORS["success"],
                        COLORS["warning"], COLORS["secondary"], COLORS["danger"],
                        COLORS["neutral"],
                    ],
                )
                fig_agg.update_layout(
                    height=400,
                    margin=dict(t=40, b=40, l=40, r=20),
                    xaxis_title=time_col,
                    yaxis_title=value_col,
                )
                st.plotly_chart(fig_agg, use_container_width=True)

                with st.expander("Aggregated data"):
                    st.dataframe(
                        polars_to_pandas(agg_df.sort(time_col)),
                        use_container_width=True,
                    )

        except Exception as exc:
            st.error(
                f"Aggregation failed: {exc}\n\n"
                "This can happen when the leaf-level ID column does not match "
                "the hierarchy tree's expected column. Verify that "
                f"`{tree.config.id_column}` exists in your data."
            )

# --------------------------------------------------------------------------- #
#  Section 4: Reconciliation
# --------------------------------------------------------------------------- #
st.divider()
st.header("Forecast Reconciliation")

forecast_df = st.session_state.get("forecast_df")

if forecast_df is None:
    st.info(
        "No forecast data available. Run a forecast from the **Data Onboarding** "
        "page or the **Forecast Viewer** page first, then return here to reconcile."
    )
else:
    st.success(
        f"Forecast data loaded: {forecast_df.shape[0]:,} rows x "
        f"{forecast_df.shape[1]} columns."
    )

    # Detect forecast columns
    forecast_time_col = _detect_time_column(forecast_df)
    forecast_value_col = None
    for col in forecast_df.columns:
        if col.lower() in ("forecast", "prediction", "yhat", "forecast_value"):
            forecast_value_col = col
            break
    if forecast_value_col is None:
        forecast_value_col = _detect_value_column(forecast_df)

    # Detect ID column in forecast
    leaf_col = tree.config.id_column
    forecast_id_col = None
    if leaf_col in forecast_df.columns:
        forecast_id_col = leaf_col
    else:
        # Try to find a matching column
        for col in forecast_df.columns:
            if col == tree.leaf_level or col.lower() == tree.leaf_level.lower():
                forecast_id_col = col
                break

    if forecast_time_col is None or forecast_value_col is None:
        st.warning(
            "Could not detect time or value columns in the forecast data. "
            f"Columns available: {', '.join(forecast_df.columns)}"
        )
    elif forecast_id_col is None:
        st.warning(
            f"Could not find hierarchy ID column `{leaf_col}` in the forecast data. "
            f"Columns available: {', '.join(forecast_df.columns)}"
        )
    else:
        st.caption(
            f"Forecast columns — time: `{forecast_time_col}` | "
            f"value: `{forecast_value_col}` | ID: `{forecast_id_col}`"
        )

        method = st.selectbox(
            "Reconciliation method",
            options=["bottom_up", "top_down", "middle_out", "ols", "wls", "mint"],
            help=(
                "**bottom_up**: Leaf forecasts are authoritative; parents = sum of children.\n\n"
                "**top_down**: Top-level forecast disaggregated by historical proportions.\n\n"
                "**middle_out**: Mid-level forecast, disaggregated down and aggregated up.\n\n"
                "**ols**: Optimal least squares reconciliation.\n\n"
                "**wls**: Weighted least squares with structural weights.\n\n"
                "**mint**: Minimum trace reconciliation (most sophisticated)."
            ),
        )

        run_reconciliation = st.button("Run Reconciliation", type="primary")

        if run_reconciliation:
            with st.status("Running reconciliation...", expanded=True) as status:
                try:
                    st.write(f"Building reconciler with method: **{method}**...")

                    recon_config = ReconciliationConfig(
                        method=method,
                        enabled=True,
                    )

                    reconciler = Reconciler(
                        trees={tree.name: tree},
                        config=recon_config,
                    )

                    # Prepare forecasts dict keyed by level
                    forecasts_dict = {
                        tree.leaf_level: forecast_df.rename(
                            {forecast_id_col: tree.leaf_level}
                        ) if forecast_id_col != tree.leaf_level else forecast_df,
                    }

                    st.write("Reconciling forecasts...")
                    reconciled_df = reconciler.reconcile(
                        forecasts=forecasts_dict,
                        actuals=actuals_df if method in ("top_down", "middle_out") else None,
                        value_columns=[forecast_value_col],
                        time_column=forecast_time_col,
                    )

                    st.session_state["reconciled_forecast"] = reconciled_df

                    status.update(label="Reconciliation complete!", state="complete")

                    # Before/After comparison
                    st.subheader("Before vs. After Reconciliation")

                    original_total = forecast_df[forecast_value_col].sum()
                    reconciled_total = reconciled_df[forecast_value_col].sum()

                    col_b, col_a, col_d = st.columns(3)
                    col_b.metric(
                        "Original Total",
                        format_number(original_total),
                    )
                    col_a.metric(
                        "Reconciled Total",
                        format_number(reconciled_total),
                    )
                    diff_pct = (
                        (reconciled_total - original_total) / original_total * 100
                        if original_total != 0 else 0.0
                    )
                    col_d.metric(
                        "Change",
                        f"{diff_pct:+.2f}%",
                    )

                    # Time series comparison
                    try:
                        orig_by_time = (
                            forecast_df.group_by(forecast_time_col)
                            .agg(pl.col(forecast_value_col).sum().alias("Original"))
                            .sort(forecast_time_col)
                        )
                        recon_by_time = (
                            reconciled_df.group_by(forecast_time_col)
                            .agg(pl.col(forecast_value_col).sum().alias("Reconciled"))
                            .sort(forecast_time_col)
                        )
                        comparison = orig_by_time.join(
                            recon_by_time, on=forecast_time_col, how="outer"
                        )

                        comp_pd = polars_to_pandas(comparison)

                        fig_compare = go.Figure()
                        fig_compare.add_trace(go.Scatter(
                            x=comp_pd[forecast_time_col],
                            y=comp_pd["Original"],
                            mode="lines",
                            name="Original",
                            line=dict(color=COLORS["neutral"], dash="dash"),
                        ))
                        fig_compare.add_trace(go.Scatter(
                            x=comp_pd[forecast_time_col],
                            y=comp_pd["Reconciled"],
                            mode="lines",
                            name="Reconciled",
                            line=dict(color=COLORS["primary"]),
                        ))
                        fig_compare.update_layout(
                            title="Total Forecast: Original vs Reconciled",
                            xaxis_title=forecast_time_col,
                            yaxis_title=forecast_value_col,
                            height=350,
                            margin=dict(t=40, b=40, l=40, r=20),
                        )
                        st.plotly_chart(fig_compare, use_container_width=True)
                    except Exception:
                        st.caption("Could not generate comparison chart.")

                    # Reconciled data preview
                    with st.expander("Reconciled forecast data"):
                        st.dataframe(
                            polars_to_pandas(reconciled_df.head(500)),
                            use_container_width=True,
                        )

                    st.success(
                        f"Reconciled forecast stored in session "
                        f"({reconciled_df.shape[0]:,} rows). "
                        f"Method: **{method}**."
                    )

                except Exception as exc:
                    status.update(label="Reconciliation failed", state="error")
                    st.error(
                        f"Reconciliation failed: {exc}\n\n"
                        "Common causes:\n"
                        "- **top_down / middle_out**: Requires historical actuals in session.\n"
                        "- **Column mismatch**: The forecast ID column must match the "
                        f"hierarchy leaf level (`{tree.leaf_level}`).\n"
                        "- **Missing data**: Forecasts must cover all leaf nodes in the hierarchy."
                    )

        # Show previous reconciliation result if available
        elif st.session_state.get("reconciled_forecast") is not None:
            reconciled_df = st.session_state["reconciled_forecast"]
            st.info(
                f"Previous reconciliation result available: "
                f"{reconciled_df.shape[0]:,} rows. "
                f"Click **Run Reconciliation** to re-run with a different method."
            )
            with st.expander("Previous reconciled forecast"):
                st.dataframe(
                    polars_to_pandas(reconciled_df.head(500)),
                    use_container_width=True,
                )
