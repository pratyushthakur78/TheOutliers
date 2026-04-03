"""
The Outliers — Multi-File Data Joiner and Analytics Dashboard.
"""

from __future__ import annotations

import io
import os
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths / branding
# ---------------------------------------------------------------------------
HACKATHON_DIR = r"D:\OneDrive - Biz2X Platform Pvt Ltd\Documents\Hackathon"
NOTEBOOK_OUTPUT_PATH = os.path.join(HACKATHON_DIR, "data_analysis.ipynb")

BRAND_NAME = "The Outliers"
ACCENT = "#FFB347"


# ---------------------------------------------------------------------------
# Notebook generation (kept from previous build)
# ---------------------------------------------------------------------------
def build_notebook_cells() -> list:
    cells = [
        new_markdown_cell(
            """# Data Analysis Notebook\n\nGenerated automatically from the Streamlit app."""
        ),
        new_code_cell(
            """import pandas as pd\nimport numpy as np\n\n# Load your data\n# df = pd.read_csv('your_file.csv')\n\n# Basic profiling\n# print(df.shape)\n# print(df.dtypes)\n# print(df.isna().sum())\n"""
        ),
    ]
    return cells


def save_notebook_to_path(path: str) -> tuple[bool, str]:
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        nb = new_notebook(
            metadata={
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11"},
            }
        )
        nb["cells"] = build_notebook_cells()
        with io.open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        return True, path
    except OSError as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# UI styling
# ---------------------------------------------------------------------------
def inject_theme() -> None:
    st.markdown(
        f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif !important; }}
  .stApp {{ background: #f5f7fb !important; color: #1f2937; }}
  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(255,179,71,0.18), rgba(255,179,71,0.07));
    border-right: 2px solid rgba(255,179,71,0.45);
  }}
  [data-testid="stHeader"] {{ background: #ffffff !important; border-bottom: 1px solid #e5e7eb; }}
  .block-card {{
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    padding: 0.9rem;
    margin-bottom: 0.8rem;
  }}
  .section-title {{
    color: {ACCENT};
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
  }}
  .minor-title {{
    color: #111827;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
  }}
  .stButton > button[kind="primary"] {{
    background: {ACCENT} !important;
    color: #1f2937 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
  }}
  .stButton > button[kind="primary"]:hover {{ background: #e59a3a !important; }}
  div[data-testid="stExpander"] {{ border: 1px solid #e5e7eb; border-radius: 8px; background: #fafafa; }}
</style>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data engineering helpers (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def parse_uploaded_file(file_name: str, file_bytes: bytes) -> dict[str, pd.DataFrame]:
    """Parse one uploaded file into table(s). Excel may return multiple sheets."""
    lower = file_name.lower()

    if lower.endswith(".csv"):
        return {file_name: pd.read_csv(io.BytesIO(file_bytes))}

    if lower.endswith(".xlsx"):
        xl = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
        return {f"{file_name}::{sheet}": xl.parse(sheet) for sheet in xl.sheet_names}

    if lower.endswith(".xls"):
        xl = pd.ExcelFile(io.BytesIO(file_bytes), engine="xlrd")
        return {f"{file_name}::{sheet}": xl.parse(sheet) for sheet in xl.sheet_names}

    raise ValueError("Unsupported file format. Please upload CSV or Excel files.")


@st.cache_data(show_spinner=False)
def perform_cached_join(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    join_key: str,
    join_type: str,
) -> pd.DataFrame:
    """
    Join two DataFrames on a shared key.

    Join logic:
    - Validates that the selected key exists in both dataframes.
    - Uses pandas merge with user-selected join type.
    - Adds suffixes to prevent column name collisions.
    """
    if join_key not in left_df.columns:
        raise KeyError(f"Join key '{join_key}' not found in left dataframe.")
    if join_key not in right_df.columns:
        raise KeyError(f"Join key '{join_key}' not found in right dataframe.")

    merged = pd.merge(
        left_df,
        right_df,
        on=join_key,
        how=join_type,
        suffixes=("_left", "_right"),
    )
    return merged


@st.cache_data(show_spinner=False)
def build_column_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-column statistics table with datatype-aware metrics.
    Non-applicable metrics are left as NaN.
    """
    total_records = len(df)
    percentile_points = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
    percentile_cols = ["5%", "10%", "20%", "30%", "50%", "70%", "80%", "90%", "95%"]
    rows: list[dict[str, Any]] = []

    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        non_missing_count = int(s.notna().sum())
        is_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        is_datetime = pd.api.types.is_datetime64_any_dtype(s)

        row: dict[str, Any] = {
            "column": col,
            "dtype": str(s.dtype),
            "total_count": total_records,
            "missing_count": missing_count,
            "non_missing_count": non_missing_count,
            "0_count": np.nan,
            "5%": np.nan,
            "10%": np.nan,
            "20%": np.nan,
            "30%": np.nan,
            "50%": np.nan,
            "70%": np.nan,
            "80%": np.nan,
            "90%": np.nan,
            "95%": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "total_unique_count": int(s.nunique(dropna=True)),
        }

        if is_numeric:
            numeric_s = pd.to_numeric(s, errors="coerce")
            row["0_count"] = int((numeric_s == 0).sum())
            row["mean"] = float(numeric_s.mean()) if numeric_s.notna().any() else np.nan
            row["min"] = float(numeric_s.min()) if numeric_s.notna().any() else np.nan
            row["max"] = float(numeric_s.max()) if numeric_s.notna().any() else np.nan
            q = numeric_s.quantile(percentile_points)
            for p_col, p_val in zip(percentile_cols, q.values):
                row[p_col] = float(p_val) if pd.notna(p_val) else np.nan
        elif is_datetime:
            dt_s = pd.to_datetime(s, errors="coerce")
            row["min"] = dt_s.min()
            row["max"] = dt_s.max()

        rows.append(row)

    out = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    if "data_registry" not in st.session_state:
        st.session_state.data_registry: dict[str, pd.DataFrame] = {}
    if "file_signatures" not in st.session_state:
        st.session_state.file_signatures: set[str] = set()
    if "joined_df" not in st.session_state:
        st.session_state.joined_df: pd.DataFrame | None = None
    if "join_summary" not in st.session_state:
        st.session_state.join_summary: str = ""


# ---------------------------------------------------------------------------
# Sidebar control plane
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    st.sidebar.markdown(f"### {BRAND_NAME}")
    st.sidebar.caption("Multi-File Data Joiner")

    uploads = st.sidebar.file_uploader(
        "Upload CSV / Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if uploads:
        for file_obj in uploads:
            raw = file_obj.getvalue()
            signature = f"{file_obj.name}:{file_obj.size}:{hash(raw)}"
            if signature in st.session_state.file_signatures:
                continue
            try:
                parsed_tables = parse_uploaded_file(file_obj.name, raw)
                st.session_state.data_registry.update(parsed_tables)
                st.session_state.file_signatures.add(signature)
                st.toast(f"Loaded: {file_obj.name}")
            except Exception as exc:
                st.sidebar.error(f"{file_obj.name}: {exc}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Registry**")

    if not st.session_state.data_registry:
        st.sidebar.info("No files uploaded yet.")
    else:
        for table_name in st.session_state.data_registry.keys():
            st.sidebar.caption(f"- {table_name}")

    if st.sidebar.button("Clear Registry", use_container_width=True):
        st.session_state.data_registry = {}
        st.session_state.file_signatures = set()
        st.session_state.joined_df = None
        st.session_state.join_summary = ""
        st.sidebar.success("Registry cleared.")


# ---------------------------------------------------------------------------
# Tab 1: Data Preview
# ---------------------------------------------------------------------------
def render_preview_tab() -> None:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)

    if not st.session_state.data_registry:
        st.info("Upload files from the sidebar to preview data.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for name, df in st.session_state.data_registry.items():
        with st.expander(f"{name}", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{len(df):,}")
            c2.metric("Columns", df.shape[1])
            c3.metric("Null Cells", f"{int(df.isna().sum().sum()):,}")

            dtypes_df = pd.DataFrame(
                {
                    "column": df.columns,
                    "dtype": [str(t) for t in df.dtypes],
                    "total_records": [len(df)] * len(df.columns),
                    "null_count": df.isna().sum().values,
                    "non_null_count": df.notna().sum().values,
                }
            )
            if len(df) > 0:
                dtypes_df["fill_rate_%"] = (
                    (dtypes_df["non_null_count"] / dtypes_df["total_records"]) * 100
                ).round(2)
                dtypes_df["missing_rate_%"] = (
                    (dtypes_df["null_count"] / dtypes_df["total_records"]) * 100
                ).round(2)
            else:
                dtypes_df["fill_rate_%"] = 0.0
                dtypes_df["missing_rate_%"] = 0.0
            left_col, right_col = st.columns(2)
            chart_key_base = name.replace(" ", "_").replace(":", "_").replace(".", "_")

            with left_col:
                st.markdown('<div class="minor-title">Schema & Null Summary</div>', unsafe_allow_html=True)
                st.dataframe(dtypes_df, use_container_width=True, height=480)

            with right_col:
                st.markdown('<div class="minor-title">Quick Graph Builder</div>', unsafe_allow_html=True)

                all_cols = list(df.columns)
                numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
                graph_options = ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie", "Bar Count"]

                graph_type = st.selectbox(
                    "Graph",
                    graph_options,
                    key=f"prev_graph_{chart_key_base}",
                )

                x_axis = st.selectbox(
                    "X Axis",
                    all_cols,
                    key=f"prev_x_{chart_key_base}",
                )
                y_axis = st.selectbox(
                    "Y Axis",
                    ["(auto/count)"] + numeric_cols,
                    key=f"prev_y_{chart_key_base}",
                )
                y_col = None if y_axis == "(auto/count)" else y_axis

                preview_df = df.head(1000).copy()
                fig = None

                if preview_df.empty:
                    st.info("Selected column has no non-null values to chart.")
                else:
                    if graph_type == "Bar":
                        if y_col:
                            fig = px.bar(preview_df, x=x_axis, y=y_col, color_discrete_sequence=[ACCENT])
                        else:
                            vc = preview_df[x_axis].astype(str).value_counts().head(25)
                            fig = px.bar(
                                x=vc.index,
                                y=vc.values,
                                labels={"x": x_axis, "y": "count"},
                                color=vc.values,
                                color_continuous_scale="Oranges",
                            )
                    elif graph_type == "Line":
                        if y_col:
                            fig = px.line(preview_df, x=x_axis, y=y_col, markers=True)
                        else:
                            st.info("Select a numeric Y Axis for Line chart.")
                    elif graph_type == "Scatter":
                        if y_col:
                            fig = px.scatter(preview_df, x=x_axis, y=y_col)
                        else:
                            st.info("Select a numeric Y Axis for Scatter chart.")
                    elif graph_type == "Histogram":
                        fig = px.histogram(preview_df, x=x_axis, color_discrete_sequence=[ACCENT])
                    elif graph_type == "Box":
                        target = y_col or x_axis
                        if target in numeric_cols:
                            fig = px.box(preview_df, y=target, color_discrete_sequence=[ACCENT])
                        else:
                            st.info("Box chart requires a numeric axis.")
                    elif graph_type == "Bar Count":
                        vc = preview_df[x_axis].astype(str).value_counts().head(25)
                        fig = px.bar(
                            x=vc.index,
                            y=vc.values,
                            labels={"x": x_axis, "y": "count"},
                            color=vc.values,
                            color_continuous_scale="Oranges",
                        )
                    elif graph_type == "Pie":
                        if y_col:
                            fig = px.pie(preview_df, names=x_axis, values=y_col, color_discrete_sequence=px.colors.sequential.Oranges)
                        else:
                            vc = preview_df[x_axis].astype(str).value_counts().head(20)
                            fig = px.pie(values=vc.values, names=vc.index, color_discrete_sequence=px.colors.sequential.Oranges)

                    if fig is not None:
                        fig.update_layout(
                            paper_bgcolor="rgba(255,255,255,0)",
                            plot_bgcolor="rgba(255,255,255,0.75)",
                            font=dict(family="Inter", color="#1f2937"),
                            margin=dict(t=30, l=20, r=20, b=20),
                            height=320,
                        )
                        st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="block-card">', unsafe_allow_html=True)
            st.markdown('<div class="minor-title">Raw Data Sample</div>', unsafe_allow_html=True)
            rows_to_show = int(
                st.number_input(
                    "Rows to display",
                    min_value=1,
                    max_value=max(1, len(df)),
                    value=min(25, len(df)),
                    step=1,
                    key=f"raw_rows_{chart_key_base}",
                    help="Type the exact number of rows to print.",
                )
            )
            st.dataframe(df.head(rows_to_show), use_container_width=True, height=220)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="block-card">', unsafe_allow_html=True)
            st.markdown('<div class="minor-title">Column Statistics</div>', unsafe_allow_html=True)
            stats_df = build_column_statistics(df)
            numeric_cols = [
                "mean",
                "min",
                "max",
                "5%",
                "10%",
                "20%",
                "30%",
                "50%",
                "70%",
                "80%",
                "90%",
                "95%",
            ]
            for c in numeric_cols:
                if c in stats_df.columns:
                    stats_df[c] = pd.to_numeric(stats_df[c], errors="coerce").round(4)
            st.dataframe(stats_df, use_container_width=True, height=320)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 2: Join Builder
# ---------------------------------------------------------------------------
def render_join_builder_tab() -> None:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Join Builder</div>', unsafe_allow_html=True)

    tables = st.session_state.data_registry
    if len(tables) < 2:
        st.warning("Upload at least two dataframes to perform joins.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    table_names = list(tables.keys())
    left_name = st.selectbox("Left DataFrame", table_names, key="left_df")
    right_options = [n for n in table_names if n != left_name]
    right_name = st.selectbox("Right DataFrame", right_options, key="right_df")

    left_df = tables[left_name]
    right_df = tables[right_name]

    common_cols = sorted(set(left_df.columns).intersection(right_df.columns))
    if not common_cols:
        st.error("No common columns found between selected dataframes.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    join_key = st.selectbox("Join Key (must exist in both)", common_cols, key="join_key")
    join_type = st.selectbox("Join Type", ["inner", "left", "right", "outer"], format_func=str.title)

    if st.button("Perform Join", type="primary"):
        try:
            merged = perform_cached_join(left_df, right_df, join_key, join_type)
            st.session_state.joined_df = merged
            st.session_state.join_summary = (
                f"Join completed: {join_type.upper()} on '{join_key}' | "
                f"Result shape: {merged.shape[0]:,} x {merged.shape[1]}"
            )
            st.success(st.session_state.join_summary)
            st.toast("Join successful.")
        except Exception as exc:
            st.error(f"Join failed: {exc}")

    if st.session_state.joined_df is not None:
        st.markdown('<div class="minor-title">Joined Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.joined_df.head(200), use_container_width=True, height=280)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 3: Analytics
# ---------------------------------------------------------------------------
def render_analytics_tab() -> None:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analytics</div>', unsafe_allow_html=True)

    joined_df = st.session_state.joined_df
    if joined_df is None or joined_df.empty:
        st.info("Perform a join in 'Join Builder' to unlock analytics.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    rows_to_use = int(
        st.number_input(
            "Rows for analysis",
            min_value=1,
            max_value=max(1, len(joined_df)),
            value=min(500, len(joined_df)),
            step=1,
            help="Controls sample size used in charts.",
        )
    )
    plot_df = joined_df.head(rows_to_use).copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{len(joined_df):,}")
    k2.metric("Columns", joined_df.shape[1])
    k3.metric("Null Cells", f"{int(joined_df.isna().sum().sum()):,}")
    k4.metric("Duplicates", f"{int(joined_df.duplicated().sum()):,}")

    all_cols = list(plot_df.columns)
    numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(plot_df[c])]

    chart_type = st.selectbox(
        "Chart Type",
        ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie", "Heatmap"],
    )

    fig: go.Figure | None = None

    if chart_type == "Heatmap":
        if len(numeric_cols) < 2:
            st.warning("Heatmap requires at least 2 numeric columns.")
        else:
            corr = plot_df[numeric_cols].corr(numeric_only=True)
            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Oranges",
                title="Correlation Heatmap",
            )
    else:
        x_col = st.selectbox("X Axis", all_cols, key="x_col")
        y_col = st.selectbox("Y Axis", [""] + all_cols, key="y_col")
        color_col = st.selectbox("Legend / Color", [""] + all_cols, key="color_col")

        y_val = y_col or None
        color_val = color_col or None

        if chart_type == "Bar":
            if not y_val:
                st.warning("Select Y Axis for Bar chart.")
            else:
                fig = px.bar(plot_df, x=x_col, y=y_val, color=color_val)
        elif chart_type == "Line":
            if not y_val:
                st.warning("Select Y Axis for Line chart.")
            else:
                fig = px.line(plot_df, x=x_col, y=y_val, color=color_val, markers=True)
        elif chart_type == "Scatter":
            if not y_val:
                st.warning("Select Y Axis for Scatter chart.")
            else:
                fig = px.scatter(plot_df, x=x_col, y=y_val, color=color_val)
        elif chart_type == "Histogram":
            fig = px.histogram(plot_df, x=x_col, color=color_val)
        elif chart_type == "Box":
            if not y_val:
                st.warning("Select Y Axis for Box chart.")
            else:
                fig = px.box(plot_df, x=x_col, y=y_val, color=color_val)
        elif chart_type == "Pie":
            if not y_val:
                vc = plot_df[x_col].astype(str).value_counts().head(30)
                fig = px.pie(values=vc.values, names=vc.index)
            else:
                fig = px.pie(plot_df, names=x_col, values=y_val, color=color_val)

    if fig is not None:
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Inter", color="#1f2937"),
            margin=dict(t=50, l=30, r=20, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="minor-title">Joined Data Table</div>', unsafe_allow_html=True)
    st.dataframe(plot_df, use_container_width=True, height=280)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# App entry
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title=f"{BRAND_NAME} — Multi-File Joiner",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    init_state()
    render_sidebar()

    ok_nb, nb_msg = save_notebook_to_path(NOTEBOOK_OUTPUT_PATH)

    st.markdown(f"## {BRAND_NAME} — Multi-File Data Joiner & Analytics Dashboard")
    st.caption("Professional workspace for loading, joining, and exploring data.")
    if ok_nb:
        st.caption(f"Notebook exported: `{nb_msg}`")
    else:
        st.caption(f"Notebook export warning: `{nb_msg}`")

    tab_preview, tab_join, tab_analytics = st.tabs(
        ["Data Preview", "Join Builder", "Analytics"]
    )

    with tab_preview:
        render_preview_tab()

    with tab_join:
        render_join_builder_tab()

    with tab_analytics:
        render_analytics_tab()


if __name__ == "__main__":
    main()
