"""
The Outliers — Multi-File Data Joiner and Analytics Dashboard.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import html
import pickle
import time
import urllib.error
import urllib.request
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from modules import data_processing as dp
from modules import synthetic_engine as se

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    SDV_AVAILABLE = True
except Exception:
    SDV_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths / branding
# ---------------------------------------------------------------------------
if os.name == "nt":
    DEFAULT_HACKATHON_DIR = r"D:\OneDrive - Biz2X Platform Pvt Ltd\Documents\Hackathon"
else:
    # Azure App Service (Linux) writable app directory
    DEFAULT_HACKATHON_DIR = "/home/site/wwwroot"

HACKATHON_DIR = os.getenv("HACKATHON_DIR", DEFAULT_HACKATHON_DIR)
NOTEBOOK_OUTPUT_PATH = os.path.join(HACKATHON_DIR, "data_analysis.ipynb")
SESSION_SNAPSHOT_PATH = os.path.join(HACKATHON_DIR, ".streamlit_session_snapshot.pkl")
SESSION_SNAPSHOT_TTL_SECONDS = 900
ENABLE_SESSION_SNAPSHOT = False

BRAND_NAME = "The Outliers"
ACCENT = "#FFB347"
REFERENCE_HACKATHON_DIR = os.path.join(HACKATHON_DIR, "Hackathon")

# Optional: load exact ML validation helpers from reference Hackathon folder.
REF_ML_EVAL_AVAILABLE = False
ref_guess_target_column = None
ref_infer_problem_kind = None
ref_default_algorithm = None
ref_ml_fidelity_rating_extended = None
ref_permutation_importance_df = None
ref_train_on_synthetic_eval_on_real = None
ref_train_predict_metrics_train_test = None
if os.path.isdir(REFERENCE_HACKATHON_DIR):
    if REFERENCE_HACKATHON_DIR not in sys.path:
        sys.path.insert(0, REFERENCE_HACKATHON_DIR)
    try:
        from foundry_ml_eval import (
            guess_target_column as ref_guess_target_column,
            infer_problem_kind as ref_infer_problem_kind,
            default_algorithm as ref_default_algorithm,
            ml_fidelity_rating_extended as ref_ml_fidelity_rating_extended,
            permutation_importance_df as ref_permutation_importance_df,
            train_on_synthetic_eval_on_real as ref_train_on_synthetic_eval_on_real,
            train_predict_metrics_train_test as ref_train_predict_metrics_train_test,
        )
        REF_ML_EVAL_AVAILABLE = True
    except Exception:
        REF_ML_EVAL_AVAILABLE = False


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
  p, li, label, [data-testid="stMarkdownContainer"] {{ font-size: 0.92rem; }}
  .stApp {{ background: #f5f7fb !important; color: #1f2937; }}
  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #fffdf8 0%, #fff8ee 45%, #fffdf9 100%);
    border-right: 1px solid #f3dfc6;
  }}
  [data-testid="stSidebar"] > div:first-child {{
    padding-top: 1rem;
  }}
  [data-testid="stHeader"] {{ background: #ffffff !important; border-bottom: 1px solid #e5e7eb; }}
  .sidebar-hero {{
    background: linear-gradient(135deg, #fff7ea 0%, #fff2de 65%, #ffefd4 100%);
    border: 1px solid #f1d4ad;
    border-radius: 12px;
    padding: 0.85rem 0.8rem 0.7rem 0.8rem;
    margin-bottom: 0.92rem;
    box-shadow: 0 6px 18px rgba(179, 118, 42, 0.08);
  }}
  .sidebar-kicker {{
    font-size: 1.42rem;
    font-weight: 800;
    color: #1f2937;
    line-height: 1.2;
    margin-bottom: 0.12rem;
  }}
  .sidebar-subtitle {{
    margin-top: 0;
    color: #9ca3af;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }}
  .arch-main-title {{
    font-size: 1.42rem;
    font-weight: 800;
    color: #1f2937;
    line-height: 1.2;
  }}
  .arch-subtitle {{
    margin-top: 0.04rem;
    color: #9ca3af;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }}
  .arch-strip {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.55rem;
    margin: 0.18rem 0 0.8rem 0;
  }}
  .arch-node {{
    border: 1px solid #efd8b5;
    background: linear-gradient(160deg, #fffaf2 0%, #fff3e0 100%);
    border-radius: 12px;
    padding: 0.5rem 0.62rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.8), 0 4px 10px rgba(245,158,11,0.08);
  }}
  .arch-node-title {{
    font-size: 0.98rem;
    font-weight: 800;
    color: #1f2937;
    line-height: 1.2;
  }}
  .arch-node-sub {{
    margin-top: 0.08rem;
    color: #9ca3af;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }}
  .sidebar-card {{
    background: #ffffff;
    border: 1px solid #efdfc8;
    border-radius: 12px;
    padding: 0.72rem 0.68rem 0.56rem 0.68rem;
    margin-bottom: 0.82rem;
    box-shadow: 0 4px 12px rgba(17, 24, 39, 0.04);
  }}
  .sidebar-card-title {{
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9a3412;
    margin-bottom: 0.5rem;
  }}
  .sidebar-helper {{
    color: #4b5563;
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 0.38rem;
  }}
  [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
    border: 1.2px dashed #f1c27d !important;
    background: #fffaf2 !important;
    border-radius: 10px !important;
  }}
  [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {{
    border-color: #e59a3a !important;
    background: #fff4e5 !important;
  }}
  [data-testid="stSidebar"] div[data-testid="stExpander"] {{
    border: 1px solid rgba(245, 166, 35, 0.32);
    border-radius: 10px;
    background: linear-gradient(150deg, rgba(255,255,255,0.95) 0%, rgba(255,245,227,0.92) 55%, rgba(255,236,206,0.95) 100%);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.74), 0 8px 18px rgba(245, 158, 11, 0.1);
    margin-bottom: 0.38rem;
    overflow: hidden;
  }}
  [data-testid="stSidebar"] div[data-testid="stExpander"] summary {{
    font-size: 0.82rem;
    font-weight: 700;
    color: #1f2937;
  }}
  [data-testid="stSidebar"] .stButton {{
    margin-top: 0.42rem;
  }}
  .sidebar-reg-item {{
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(245, 166, 35, 0.35);
    background: linear-gradient(145deg, rgba(255,255,255,0.95) 0%, rgba(255,243,223,0.92) 56%, rgba(255,236,206,0.96) 100%);
    border-radius: 10px;
    padding: 0.45rem 0.52rem;
    margin-bottom: 0.36rem;
    color: #1f2937;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    word-break: break-word;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.75), 0 8px 18px rgba(245, 158, 11, 0.12);
  }}
  .sidebar-reg-item::before {{
    content: "";
    position: absolute;
    top: 0;
    left: -125%;
    width: 52%;
    height: 100%;
    transform: skewX(-18deg);
    background: linear-gradient(90deg, rgba(255,255,255,0.0), rgba(255,255,255,0.52), rgba(255,255,255,0.0));
    animation: registryShine 4.4s ease-in-out infinite;
    pointer-events: none;
  }}
  .sidebar-empty {{
    border: 1px dashed #edd7b8;
    background: #fffaf2;
    color: #6b7280;
    border-radius: 8px;
    padding: 0.45rem 0.55rem;
    font-size: 0.76rem;
  }}
  .columns-liquid-box {{
    position: relative;
    max-height: 150px;
    overflow-y: auto;
    overflow-x: hidden;
    border-radius: 12px;
    border: 1px solid rgba(245, 166, 35, 0.42);
    background:
      linear-gradient(155deg, rgba(255,255,255,0.94) 0%, rgba(255,246,229,0.92) 52%, rgba(255,235,200,0.95) 100%);
    padding: 0.4rem 0.42rem 0.28rem 0.42rem;
    box-shadow:
      inset 0 1px 0 rgba(255,255,255,0.76),
      inset 0 -1px 0 rgba(255,255,255,0.46),
      0 10px 22px rgba(245, 158, 11, 0.16);
    backdrop-filter: blur(2px);
  }}
  .columns-liquid-head {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.45rem;
    margin-bottom: 0.3rem;
    padding: 0 0.08rem;
    color: #7c4a11;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }}
  .columns-liquid-count {{
    border: 1px solid rgba(214, 134, 18, 0.38);
    border-radius: 999px;
    padding: 0.05rem 0.42rem;
    background: rgba(255, 255, 255, 0.74);
    color: #9a3412;
    font-size: 0.65rem;
    font-weight: 700;
  }}
  .columns-liquid-box::before {{
    content: "";
    position: absolute;
    top: 0;
    left: -120%;
    width: 60%;
    height: 100%;
    transform: skewX(-16deg);
    background: linear-gradient(90deg, rgba(255,255,255,0.0), rgba(255,255,255,0.45), rgba(255,255,255,0.0));
    animation: liquidShine 3.8s ease-in-out infinite;
    pointer-events: none;
  }}
  .columns-chip-grid {{
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 0.24rem;
  }}
  .column-chip {{
    border: 1px solid #f1dfc9;
    background: rgba(255, 255, 255, 0.84);
    border-radius: 7px;
    padding: 0.24rem 0.45rem;
    color: #374151;
    font-size: 0.76rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .columns-liquid-box::-webkit-scrollbar {{
    width: 8px;
  }}
  .columns-liquid-box::-webkit-scrollbar-track {{
    background: rgba(255, 255, 255, 0.4);
    border-radius: 999px;
  }}
  .columns-liquid-box::-webkit-scrollbar-thumb {{
    background: linear-gradient(180deg, rgba(245, 158, 11, 0.66), rgba(217, 119, 6, 0.72));
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.7);
  }}
  .columns-liquid-box::-webkit-scrollbar-thumb:hover {{
    background: linear-gradient(180deg, rgba(234, 135, 8, 0.76), rgba(194, 92, 0, 0.84));
  }}
  @keyframes liquidShine {{
    0% {{ left: -120%; opacity: 0; }}
    14% {{ opacity: 1; }}
    52% {{ left: 145%; opacity: 0.95; }}
    100% {{ left: 145%; opacity: 0; }}
  }}
  @keyframes registryShine {{
    0% {{ left: -125%; opacity: 0; }}
    12% {{ opacity: 1; }}
    48% {{ left: 145%; opacity: 0.9; }}
    100% {{ left: 145%; opacity: 0; }}
  }}
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
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
  }}
  .minor-title {{
    color: #111827;
    font-size: 0.98rem;
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
  [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    position: relative;
    overflow: hidden;
    border-radius: 12px !important;
    border: 1px solid rgba(245, 166, 35, 0.5) !important;
    background: linear-gradient(145deg, rgba(255, 224, 170, 0.95) 0%, rgba(255, 193, 102, 0.94) 100%) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.68), 0 8px 16px rgba(245, 158, 11, 0.2) !important;
    color: #1f2937 !important;
    font-weight: 700 !important;
  }}
  [data-testid="stSidebar"] .stButton > button[kind="primary"]::before {{
    content: "";
    position: absolute;
    top: 0;
    left: -130%;
    width: 56%;
    height: 100%;
    transform: skewX(-16deg);
    background: linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,0.52), rgba(255,255,255,0));
    animation: sidebarCtaShine 3.7s ease-in-out infinite;
    pointer-events: none;
  }}
  @keyframes sidebarCtaShine {{
    0% {{ left: -130%; opacity: 0; }}
    14% {{ opacity: 1; }}
    52% {{ left: 145%; opacity: 0.95; }}
    100% {{ left: 145%; opacity: 0; }}
  }}
  .stTabs [data-baseweb="tab-list"] {{
    gap: 0.4rem;
    background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,250,243,0.92));
    border: 1px solid #eedec8;
    border-radius: 999px;
    padding: 0.34rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.75), 0 6px 14px rgba(17,24,39,0.05);
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 999px !important;
    border: 1px solid transparent !important;
    background: rgba(255,255,255,0.65) !important;
    color: #4b5563 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    min-height: 2.2rem !important;
    padding: 0.2rem 0.95rem !important;
    transition: all 0.18s ease;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
    border-color: #f2d2a6 !important;
    background: rgba(255,247,235,0.95) !important;
    color: #1f2937 !important;
  }}
  .stTabs [aria-selected="true"] {{
    background: linear-gradient(145deg, #ffd9a1 0%, #ffbe66 85%) !important;
    color: #1f2937 !important;
    border-color: #f2b357 !important;
    box-shadow: 0 8px 18px rgba(245, 158, 11, 0.28), inset 0 1px 0 rgba(255,255,255,0.65);
  }}
  .stTabs [data-baseweb="tab-highlight"] {{
    background: transparent !important;
    height: 0 !important;
  }}
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

    if lower.endswith(".json"):
        data = json.loads(file_bytes.decode("utf-8-sig"))
        if isinstance(data, list):
            return {file_name: pd.json_normalize(data)}
        if isinstance(data, dict):
            return {file_name: pd.json_normalize([data])}
        raise ValueError("Unsupported JSON structure. Use object or array of objects.")

    if lower.endswith(".xlsx"):
        xl = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
        return {f"{file_name}::{sheet}": xl.parse(sheet) for sheet in xl.sheet_names}

    if lower.endswith(".xls"):
        xl = pd.ExcelFile(io.BytesIO(file_bytes), engine="xlrd")
        return {f"{file_name}::{sheet}": xl.parse(sheet) for sheet in xl.sheet_names}

    raise ValueError("Unsupported file format. Please upload CSV, Excel, or JSON files.")


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


def detect_pii_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect likely PII columns using names and lightweight pattern checks."""
    pii_name_tokens = {
        "name", "email", "phone", "mobile", "contact", "address",
        "pan", "aadhaar", "ssn", "passport", "dob"
    }
    email_re = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
    phone_re = re.compile(r"^\+?[\d\-\s\(\)]{10,}$")
    rows: list[dict[str, Any]] = []

    for c in df.columns:
        s = df[c]
        c_lower = str(c).lower()
        by_name = any(tok in c_lower for tok in pii_name_tokens)
        sample = s.dropna().astype(str).head(150)
        email_hits = sum(1 for v in sample if email_re.match(v.strip()))
        phone_hits = sum(
            1 for v in sample
            if phone_re.match(v.strip()) and sum(ch.isdigit() for ch in v) >= 10
        )
        sample_n = max(len(sample), 1)
        by_pattern = (email_hits / sample_n) >= 0.35 or (phone_hits / sample_n) >= 0.35
        pii_flag = by_name or by_pattern
        reason = []
        if by_name:
            reason.append("name/token")
        if by_pattern:
            reason.append("pattern")
        rows.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "pii_detected": pii_flag,
                "reason": ", ".join(reason) if reason else "",
            }
        )
    return pd.DataFrame(rows)


def _is_binary_indicator(series: pd.Series) -> bool:
    """True when series is effectively a binary 0/1 or boolean indicator."""
    s = series.dropna()
    if s.empty:
        return False
    if pd.api.types.is_bool_dtype(series):
        return True
    s_num = pd.to_numeric(s, errors="coerce").dropna()
    if s_num.empty:
        return False
    uniques = set(np.unique(s_num.values))
    return uniques.issubset({0, 1})


@st.cache_data(show_spinner=False)
def apply_privacy_transform(
    df: pd.DataFrame,
    pii_columns: tuple[str, ...],
    scrub_mode: str,
    k_anonymity: int,
    noise_level: float,
) -> pd.DataFrame:
    """Apply simple anonymization/scrubbing and optional k-anonymity/noise."""
    out = df.copy()

    for col in pii_columns:
        if col not in out.columns:
            continue
        if scrub_mode == "mask":
            out[col] = out[col].astype(str).where(out[col].isna(), "REDACTED")
        elif scrub_mode == "drop":
            out = out.drop(columns=[col])

    # Basic k-anonymity approximation for categorical columns
    cat_cols = [c for c in out.columns if not pd.api.types.is_numeric_dtype(out[c])]
    if k_anonymity > 1:
        for c in cat_cols:
            vc = out[c].astype(str).value_counts(dropna=False)
            rare = vc[vc < k_anonymity].index
            out[c] = out[c].astype(str).where(~out[c].astype(str).isin(rare), "__OTHER__")

    # Optional numeric noise
    if noise_level > 0:
        num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        for c in num_cols:
            # Preserve binary indicator columns as strict 0/1.
            if _is_binary_indicator(out[c]):
                continue
            std = float(pd.to_numeric(out[c], errors="coerce").std() or 0.0)
            if std <= 0:
                continue
            noise = np.random.normal(0, std * noise_level, size=len(out))
            out[c] = pd.to_numeric(out[c], errors="coerce") + noise
    return out


@st.cache_data(show_spinner=False)
def generate_bootstrap_synthetic(seed_df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
    """Diffusion-style fallback using bootstrap + light perturbation."""
    if seed_df.empty:
        return seed_df.copy()
    out = seed_df.sample(n=target_rows, replace=True, random_state=42).reset_index(drop=True)
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for c in num_cols:
        # Keep binary indicator fields unchanged (e.g., default_flag_12m).
        if _is_binary_indicator(seed_df[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").round().clip(0, 1)
            continue
        std = float(pd.to_numeric(out[c], errors="coerce").std() or 0.0)
        if std > 0:
            out[c] = pd.to_numeric(out[c], errors="coerce") + np.random.normal(0, std * 0.02, size=len(out))
    return out


def generate_sdv_synthetic(seed_df: pd.DataFrame, target_rows: int, model_type: str) -> pd.DataFrame:
    """Generate synthetic data with CTGAN/TVAE if SDV is available."""
    if not SDV_AVAILABLE:
        raise RuntimeError("SDV not installed. Use fallback model or add sdv to requirements.")
    if seed_df.empty:
        return seed_df.copy()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(seed_df)
    if model_type == "CTGAN":
        synth = CTGANSynthesizer(metadata)
    else:
        synth = TVAESynthesizer(metadata)
    synth.fit(seed_df)
    out = synth.sample(num_rows=target_rows)
    return out


def _js_divergence_from_hist(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    low = min(np.nanmin(a), np.nanmin(b))
    high = max(np.nanmax(a), np.nanmax(b))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return np.nan
    p_hist, edges = np.histogram(a, bins=bins, range=(low, high), density=True)
    q_hist, _ = np.histogram(b, bins=edges, density=True)
    p = p_hist.astype(float) + 1e-12
    q = q_hist.astype(float) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


@st.cache_data(show_spinner=False)
def compute_fidelity_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return stage-4 scores: JS divergence, correlation similarity, utility checks."""
    numeric_cols = [
        c for c in real_df.columns
        if c in synth_df.columns and pd.api.types.is_numeric_dtype(real_df[c]) and pd.api.types.is_numeric_dtype(synth_df[c])
    ]
    js_rows = []
    for c in numeric_cols:
        r = pd.to_numeric(real_df[c], errors="coerce").dropna().values
        s = pd.to_numeric(synth_df[c], errors="coerce").dropna().values
        js_rows.append({"column": c, "js_divergence": _js_divergence_from_hist(r, s)})
    js_df = pd.DataFrame(js_rows)

    corr_similarity = np.nan
    if len(numeric_cols) >= 2:
        rc = real_df[numeric_cols].corr(numeric_only=True).fillna(0.0)
        sc = synth_df[numeric_cols].corr(numeric_only=True).fillna(0.0)
        common = [c for c in rc.columns if c in sc.columns]
        if len(common) >= 2:
            diff = (rc.loc[common, common] - sc.loc[common, common]).abs().values
            corr_similarity = float(1.0 - np.nanmean(diff))

    null_rate_diff = float((real_df.isna().mean() - synth_df.isna().mean()).abs().mean())
    uniq_ratio_real = (real_df.nunique(dropna=True) / max(len(real_df), 1)).mean()
    uniq_ratio_synth = (synth_df.nunique(dropna=True) / max(len(synth_df), 1)).mean()
    utility_df = pd.DataFrame(
        [
            {"metric": "correlation_similarity", "value": corr_similarity},
            {"metric": "avg_null_rate_diff", "value": null_rate_diff},
            {"metric": "unique_ratio_real", "value": float(uniq_ratio_real)},
            {"metric": "unique_ratio_synth", "value": float(uniq_ratio_synth)},
        ]
    )
    return js_df, utility_df


def _dtype_family_compatible(seed_s: pd.Series, synth_s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(seed_s) and pd.api.types.is_numeric_dtype(synth_s):
        return True
    if pd.api.types.is_bool_dtype(seed_s) and pd.api.types.is_bool_dtype(synth_s):
        return True
    if pd.api.types.is_datetime64_any_dtype(seed_s) and pd.api.types.is_datetime64_any_dtype(synth_s):
        return True
    if seed_s.dtype.kind in "OSV" and synth_s.dtype.kind in "OSV":
        return True
    return str(seed_s.dtype) == str(synth_s.dtype)


def _to_num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _median_rel_error(resid: pd.Series, denom: pd.Series) -> float | None:
    m = resid.notna() & denom.notna()
    if m.sum() < 8:
        return None
    d = denom[m].abs().replace(0, np.nan)
    rel = (resid[m].abs() / (d + 1e-12)).dropna()
    if rel.empty:
        return None
    return float(rel.median())


def _discover_additive_balance_patterns(
    seed: pd.DataFrame,
    cols: list[str],
    *,
    rel_threshold: float = 0.10,
    max_cols: int = 14,
) -> list[dict[str, Any]]:
    cols = [c for c in cols if c in seed.columns][:max_cols]
    patterns: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    def try_pattern(a: str, b: str, c: str, tag: str) -> None:
        if len({a, b, c}) < 3:
            return
        key = (tag, tuple(sorted([a, b])), c) if tag == "sum" else (tag, a, b, c)
        if key in seen:
            return
        sa, sb, sc = _to_num(seed, a), _to_num(seed, b), _to_num(seed, c)
        pred = sa + sb if tag == "sum" else sa - sb
        m = pred.notna() & sc.notna()
        if m.sum() < 12:
            return
        mer = _median_rel_error(pred - sc, sc)
        if mer is None or mer > rel_threshold:
            return
        seen.add(key)
        patterns.append(
            {
                "kind": tag,
                "a": a,
                "b": b,
                "c": c,
                "seed_median_rel_err": mer,
                "label": f"{a} {'+' if tag == 'sum' else '-'} {b} ~= {c}",
            }
        )

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            for k in range(len(cols)):
                if k in (i, j):
                    continue
                c = cols[k]
                try_pattern(a, b, c, "sum")
                try_pattern(a, b, c, "diff")
    return patterns[:20]


def _validate_patterns_on_synth(
    synth: pd.DataFrame,
    patterns: list[dict[str, Any]],
    *,
    degrade_factor: float = 4.0,
    slack: float = 0.07,
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    ok: list[str] = []
    for p in patterns:
        a, b, c = p["a"], p["b"], p["c"]
        if not all(x in synth.columns for x in (a, b, c)):
            issues.append(f"{p.get('label', '?')}: missing columns")
            continue
        sa, sb, sc = _to_num(synth, a), _to_num(synth, b), _to_num(synth, c)
        pred = sa + sb if p["kind"] == "sum" else sa - sb
        mer = _median_rel_error(pred - sc, sc)
        seed_m = p["seed_median_rel_err"]
        if mer is None:
            issues.append(f"{p.get('label')}: could not score synthetic")
            continue
        if mer > seed_m * degrade_factor + slack:
            issues.append(f"{p.get('label')}: seed {seed_m:.2%} -> synth {mer:.2%} (weakened)")
        else:
            ok.append(f"{p.get('label')}: synth {mer:.2%} (seed {seed_m:.2%})")
    return issues, ok


def validate_synthetic_dataset(seed: pd.DataFrame, synth: pd.DataFrame, pk_cols: list[str]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    common = [c for c in seed.columns if c in synth.columns]
    primary = pk_cols[0] if pk_cols else None

    if primary and primary in synth.columns:
        dup = int(synth[primary].duplicated().sum())
        checks.append(
            {
                "id": "pk_unique",
                "name": "Primary key uniqueness",
                "pass": dup == 0,
                "detail": f"Duplicate key rows: {dup:,}",
            }
        )

    dtype_issues = [
        f"{c}: {seed[c].dtype} vs {synth[c].dtype}"
        for c in common
        if not _dtype_family_compatible(seed[c], synth[c])
    ]
    checks.append(
        {
            "id": "dtype_match",
            "name": "Datatype consistency",
            "pass": len(dtype_issues) == 0,
            "detail": "; ".join(dtype_issues[:10]) if dtype_issues else f"{len(common)} overlapping column(s) keep compatible dtype families.",
        }
    )

    cat_issues: list[str] = []
    for c in common:
        sc, sy = seed[c], synth[c]
        if pd.api.types.is_numeric_dtype(sc) or pd.api.types.is_datetime64_any_dtype(sc):
            continue
        nu = int(sc.nunique(dropna=True))
        if nu > 100 or nu < 1:
            continue
        novel = set(sy.dropna().astype(str).unique()) - set(sc.dropna().astype(str).unique())
        if novel:
            sample = next(iter(novel))
            cat_issues.append(f"{c}: {len(novel)} novel value(s), e.g. {sample[:48]!r}")
    checks.append(
        {
            "id": "categorical_vocab",
            "name": "Categorical/text levels",
            "pass": len(cat_issues) == 0,
            "detail": "; ".join(cat_issues[:6]) if cat_issues else "Low-cardinality text columns stay within seed vocabulary.",
        }
    )

    num_common = [
        c for c in common
        if pd.api.types.is_numeric_dtype(seed[c]) and pd.api.types.is_numeric_dtype(synth[c]) and c != primary
    ]
    patterns = _discover_additive_balance_patterns(seed, num_common)
    add_issues, add_ok = _validate_patterns_on_synth(synth, patterns)
    checks.append(
        {
            "id": "patterns_additive",
            "name": "Inferred additive/difference patterns",
            "pass": len(add_issues) == 0,
            "detail": "; ".join(add_issues[:8]) if add_issues else (" | ".join(add_ok[:8]) if add_ok else "No stable additive patterns inferred from seed."),
        }
    )

    definitive = [c for c in checks if c.get("pass") is False]
    return {
        "checks": checks,
        "overall_pass": len(definitive) == 0,
        "summary": f"{len(definitive)} failed" if definitive else "All automated checks passed",
        "patterns_discovered": patterns,
    }


def render_guardrails_table(report: dict[str, Any]) -> None:
    rows = report.get("checks") or []
    if not rows:
        st.caption("No Guardrails checks available.")
        return
    tbl = pd.DataFrame(
        [
            {
                "Check": r.get("name", ""),
                "Result": "Pass" if r.get("pass") is True else ("Fail" if r.get("pass") is False else "N/A"),
                "Detail": (r.get("detail") or "")[:900],
            }
            for r in rows
        ]
    )
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    overall = report.get("overall_pass")
    summary = report.get("summary", "")
    if overall is True:
        st.success(f"Guardrails: {summary}")
    elif overall is False:
        st.warning(f"Guardrails: {summary}")
    else:
        st.info(summary or "Guardrails partial.")


def render_seed_synth_corr(seed_df: pd.DataFrame, synth_df: pd.DataFrame, key_prefix: str) -> None:
    num_seed = seed_df.select_dtypes(include=[np.number])
    num_syn = synth_df.select_dtypes(include=[np.number])
    common = [c for c in num_seed.columns if c in num_syn.columns]
    if len(common) < 2:
        st.caption("Need at least two overlapping numeric columns for correlation matrices.")
        return
    cr_seed = num_seed[common].corr(numeric_only=True).round(4)
    cr_syn = num_syn[common].corr(numeric_only=True).round(4)
    st.markdown("**Numeric correlations - seed vs synthetic**")
    c1, c2 = st.columns(2)
    with c1:
        fig_seed = px.imshow(cr_seed, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Seed correlations")
        fig_seed.update_layout(height=max(360, 28 * len(common)))
        st.plotly_chart(fig_seed, use_container_width=True, key=f"{key_prefix}_seed")
    with c2:
        fig_syn = px.imshow(cr_syn, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Synthetic correlations")
        fig_syn.update_layout(height=max(360, 28 * len(common)))
        st.plotly_chart(fig_syn, use_container_width=True, key=f"{key_prefix}_syn")


SYN_CRITIC_PROFILES: dict[str, dict[str, Any]] = {
    "standard": {"title": "Standard", "pass_threshold": 3.5, "mean_js_fail": 0.12, "w_dist": 0.45, "w_corr": 0.35, "w_struct": 0.20},
    "strict": {"title": "Strict (statistical twin)", "pass_threshold": 4.0, "mean_js_fail": 0.08, "w_dist": 0.45, "w_corr": 0.35, "w_struct": 0.20},
    "exploratory": {"title": "Exploratory / demo", "pass_threshold": 2.8, "mean_js_fail": 0.18, "w_dist": 0.45, "w_corr": 0.35, "w_struct": 0.20},
    "structure_first": {"title": "Structure-first", "pass_threshold": 3.2, "mean_js_fail": 0.15, "w_dist": 0.25, "w_corr": 0.45, "w_struct": 0.30},
}


def _has_azure_generation_context() -> bool:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = (os.getenv("AZURE_OPENAI_API_KEY", "") or os.getenv("AZURE_API_KEY", "")).strip()
    return bool(endpoint and api_key)


def _auto_choose_generation_model(seed_df: pd.DataFrame) -> tuple[str, str]:
    rows, cols = seed_df.shape
    obj_cols = int(len(seed_df.select_dtypes(exclude=[np.number]).columns))
    complexity = cols + (obj_cols * 1.5)
    if SDV_AVAILABLE:
        if rows < 200:
            return "TVAE", "Small seed dataset: TVAE typically stabilizes faster."
        return "CTGAN (GAN)", "Moderate/high complexity schema: CTGAN is preferred."
    if _has_azure_generation_context() and complexity >= 70:
        return "AI Astra", "High schema complexity and Azure context detected."
    return "Diffusion-style bootstrap", "SDV unavailable: using robust bootstrap fallback."


def _synthetic_fidelity_score(
    js_df: pd.DataFrame,
    utility_df: pd.DataFrame,
    profile_id: str,
) -> float:
    prof = SYN_CRITIC_PROFILES.get(profile_id, SYN_CRITIC_PROFILES["standard"])
    mean_js = float(pd.to_numeric(js_df.get("js_divergence", pd.Series(dtype=float)), errors="coerce").dropna().mean() or 0.0)
    corr_series = utility_df.loc[utility_df["metric"] == "correlation_similarity", "value"]
    null_diff_series = utility_df.loc[utility_df["metric"] == "avg_null_rate_diff", "value"]
    uniq_real_series = utility_df.loc[utility_df["metric"] == "unique_ratio_real", "value"]
    uniq_syn_series = utility_df.loc[utility_df["metric"] == "unique_ratio_synth", "value"]
    corr_similarity = float(corr_series.iloc[0]) if not corr_series.empty and pd.notna(corr_series.iloc[0]) else 0.60
    null_diff = float(null_diff_series.iloc[0]) if not null_diff_series.empty and pd.notna(null_diff_series.iloc[0]) else 0.30
    uniq_gap = 0.0
    if (not uniq_real_series.empty and pd.notna(uniq_real_series.iloc[0])) and (not uniq_syn_series.empty and pd.notna(uniq_syn_series.iloc[0])):
        uniq_gap = abs(float(uniq_real_series.iloc[0]) - float(uniq_syn_series.iloc[0]))

    dist_score = max(0.0, min(5.0, 5.0 * (1.0 - min(1.0, mean_js / max(float(prof["mean_js_fail"]), 1e-9)))))
    corr_score = max(0.0, min(5.0, 5.0 * max(0.0, min(1.0, corr_similarity))))
    struct_raw = 1.0 - min(1.0, (null_diff + uniq_gap) / 0.80)
    struct_score = max(0.0, min(5.0, 5.0 * struct_raw))

    return round(
        float(prof["w_dist"]) * dist_score
        + float(prof["w_corr"]) * corr_score
        + float(prof["w_struct"]) * struct_score,
        2,
    )


def export_dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Export dataframe with key Lens sheets into an Excel workbook."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary_df = pd.DataFrame(
            [
                {"metric": "rows", "value": len(df)},
                {"metric": "columns", "value": len(df.columns)},
                {"metric": "numeric_columns", "value": int(len(df.select_dtypes(include=[np.number]).columns))},
                {"metric": "categorical_columns", "value": int(len(df.select_dtypes(exclude=[np.number]).columns))},
                {"metric": "duplicate_rows", "value": int(df.duplicated().sum())},
                {"metric": "null_cells", "value": int(df.isna().sum().sum())},
            ]
        )
        summary_df.to_excel(writer, sheet_name="00_Summary", index=False)

        schema_df = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(t) for t in df.dtypes],
                "non_null_count": df.notna().sum().values,
                "null_count": df.isna().sum().values,
                "unique_count": df.nunique(dropna=True).values,
            }
        )
        schema_df.to_excel(writer, sheet_name="01_Schema", index=False)

        stats_df = build_column_statistics(df)
        stats_df.to_excel(writer, sheet_name="02_Column_Stats", index=False)
        df.head(100).to_excel(writer, sheet_name="03_Head_100", index=False)
    return buf.getvalue()


def generate_with_azure_llm(
    seed_df: pd.DataFrame,
    target_rows: int,
    custom_instruction: str = "",
) -> pd.DataFrame:
    """
    Optional Azure AI Foundry / Azure OpenAI based generation.
    Uses env vars:
      AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT,
      AZURE_OPENAI_API_VERSION (optional; default 2024-02-15-preview)
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "") or os.getenv("AZURE_API_KEY", "")
    deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
        or os.getenv("MODEL_DEPLOYMENT_NAME", "").strip()
    )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    agent_id = os.getenv("AZURE_EXISTING_AGENT_ID", "").strip()
    if not endpoint or not api_key:
        raise RuntimeError("Azure LLM credentials missing in environment variables.")

    # Support both endpoint forms:
    # - https://<name>.openai.azure.com
    # - https://<name>.openai.azure.com/openai/v1
    if endpoint.endswith("/openai/v1"):
        endpoint_base = endpoint[: -len("/openai/v1")]
        endpoint_v1 = endpoint
    elif endpoint.endswith("/openai/v1/"):
        endpoint_base = endpoint[: -len("/openai/v1/")]
        endpoint_v1 = endpoint.rstrip("/")
    else:
        endpoint_base = endpoint
        endpoint_v1 = f"{endpoint_base}/openai/v1"

    # Keep token-safe batch size and scale by bootstrap for very large targets.
    llm_rows = min(target_rows, 2500)
    schema = {c: str(seed_df[c].dtype) for c in seed_df.columns}
    sample_rows = seed_df.head(20).fillna("").to_dict(orient="records")
    prompt = (
        "Generate synthetic tabular JSON records that preserve schema and distributions. "
        "Return ONLY JSON array. "
        f"Rows required: {llm_rows}. "
        f"Schema: {schema}. "
        f"Example rows: {sample_rows}."
    )
    if custom_instruction.strip():
        prompt += f" Additional user instruction: {custom_instruction.strip()}."
    def _extract_json_content(text: str) -> str:
        cleaned = text.strip()
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        # If model returns wrapper text, try to isolate JSON array
        if "[" in cleaned and "]" in cleaned:
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start != -1 and end != -1 and end > start:
                return cleaned[start : end + 1]
        return cleaned

    def _post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "api-key": api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode("utf-8")
            except Exception:
                raw = str(e)
            raise RuntimeError(f"Azure API HTTP {e.code}: {raw}") from e

    def _deployment_generation_content() -> str:
        if not deployment:
            raise RuntimeError(
                "Agent call failed and no AZURE_OPENAI_DEPLOYMENT fallback is set."
            )
        # Try Responses API first (many latest models only support this),
        # then fallback to chat completions for older deployments.
        responses_url = f"{endpoint_v1}/responses"
        responses_body = {
            "model": deployment,
            "input": [
                {"role": "system", "content": "You are a synthetic data generator. Output strict JSON array only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
        }
        try:
            payload = _post_json(responses_url, responses_body)
            content = payload.get("output_text", "")
            if not content:
                output = payload.get("output", [])
                if output and isinstance(output, list):
                    parts = []
                    for item in output:
                        for c in item.get("content", []) if isinstance(item, dict) else []:
                            t = c.get("text")
                            if t:
                                parts.append(t)
                    content = "\n".join(parts).strip()
            if content:
                return content
        except Exception:
            pass

        chat_body = {
            "messages": [
                {"role": "system", "content": "You are a synthetic data generator. Output strict JSON array only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 3000,
        }
        chat_url = (
            f"{endpoint_base}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={api_version}"
        )
        payload = _post_json(chat_url, chat_body)
        return payload["choices"][0]["message"]["content"]

    # Route A: Agent Reference (Azure AI Foundry Agent) via Responses API
    if agent_id:
        if ":" in agent_id:
            agent_name, agent_version = agent_id.split(":", 1)
        else:
            agent_name, agent_version = agent_id, "1"

        responses_url = f"{endpoint_v1}/responses"

        base_input = {"input": [{"role": "user", "content": prompt}]}
        # Azure surfaces may expect either top-level `agent_reference` or SDK-style `extra_body`.
        body_top_level = {
            **base_input,
            "agent_reference": {
                "name": agent_name,
                "version": agent_version,
                "type": "agent_reference",
            },
        }
        body_extra = {
            **base_input,
            "extra_body": {
                "agent_reference": {
                    "name": agent_name,
                    "version": agent_version,
                    "type": "agent_reference",
                }
            },
        }
        # Some Foundry setups also require a model/deployment in agent calls.
        if deployment:
            body_top_level["model"] = deployment
            body_extra["model"] = deployment

        candidate_bodies = [body_top_level, body_extra]

        payload = None
        last_error: Exception | None = None
        for body in candidate_bodies:
            try:
                payload = _post_json(responses_url, body)
                break
            except Exception as e:
                last_error = e
                continue
        if payload is None:
            # Fallback to deployment mode if configured
            if deployment:
                content = _deployment_generation_content()
                content = _extract_json_content(content)
                records = json.loads(content)
                llm_df = pd.DataFrame(records)
                if len(llm_df) < target_rows and not llm_df.empty:
                    extra = llm_df.sample(target_rows - len(llm_df), replace=True, random_state=42)
                    llm_df = pd.concat([llm_df, extra], ignore_index=True)
                return llm_df.head(target_rows)
            raise RuntimeError(
                "Agent Reference call failed. Set AZURE_OPENAI_DEPLOYMENT (or AZURE_OPENAI_DEPLOYMENT_NAME) "
                "to enable automatic fallback. "
                f"Endpoint: '{responses_url}'. Last error: {last_error}"
            )

        content = payload.get("output_text", "")
        if not content:
            # Fallback parse for possible structured output formats
            output = payload.get("output", [])
            if output and isinstance(output, list):
                parts = []
                for item in output:
                    for c in item.get("content", []) if isinstance(item, dict) else []:
                        t = c.get("text")
                        if t:
                            parts.append(t)
                content = "\n".join(parts).strip()
        if not content:
            raise RuntimeError("Agent response did not include output_text.")
    else:
        # Route B: Azure OpenAI deployment chat completion
        if not deployment:
            raise RuntimeError(
                "Set AZURE_EXISTING_AGENT_ID (agent mode) or AZURE_OPENAI_DEPLOYMENT (fallback mode)."
            )
        content = _deployment_generation_content()

    content = _extract_json_content(content)
    records = json.loads(content)
    llm_df = pd.DataFrame(records)
    # Expand to requested size with bootstrap if needed
    if len(llm_df) < target_rows and not llm_df.empty:
        extra = llm_df.sample(target_rows - len(llm_df), replace=True, random_state=42)
        llm_df = pd.concat([llm_df, extra], ignore_index=True)
    return llm_df.head(target_rows)


@st.cache_data(show_spinner=False)
def align_synthetic_to_seed_distribution(seed_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align synthetic output closer to seed distributions.
    - Numeric: rank-based quantile mapping to seed values.
    - Non-numeric: resample from seed frequency distribution.
    """
    out = synthetic_df.copy()
    if out.empty or seed_df.empty:
        return out

    rng = np.random.default_rng(42)
    common_cols = [c for c in out.columns if c in seed_df.columns]

    for col in common_cols:
        seed_s = seed_df[col]
        out_s = out[col]

        if pd.api.types.is_numeric_dtype(seed_s):
            if _is_binary_indicator(seed_s):
                # Enforce strict binary values based on seed prevalence.
                p_one = float(pd.to_numeric(seed_s, errors="coerce").fillna(0).clip(0, 1).mean())
                sampled = rng.choice([0, 1], size=len(out), replace=True, p=[1 - p_one, p_one])
                out[col] = pd.Series(sampled, index=out.index)
                continue
            seed_num = pd.to_numeric(seed_s, errors="coerce").dropna()
            out_num = pd.to_numeric(out_s, errors="coerce")
            if seed_num.empty:
                continue
            valid_mask = out_num.notna()
            if valid_mask.any():
                ranks = out_num[valid_mask].rank(method="average", pct=True).clip(0.0, 1.0)
                mapped = np.quantile(seed_num.values, ranks.values)
                out.loc[valid_mask, col] = mapped
        elif pd.api.types.is_datetime64_any_dtype(seed_s):
            seed_dt = pd.to_datetime(seed_s, errors="coerce").dropna()
            if not seed_dt.empty:
                sampled = rng.choice(seed_dt.values, size=len(out), replace=True)
                out[col] = pd.to_datetime(sampled, errors="coerce")
        else:
            seed_cat = seed_s.astype(str).fillna("NA_TOKEN")
            vc = seed_cat.value_counts(dropna=False)
            if not vc.empty:
                probs = (vc / vc.sum()).values
                sampled = rng.choice(vc.index.values, size=len(out), replace=True, p=probs)
                out[col] = pd.Series(sampled).replace("NA_TOKEN", np.nan)

    return out


@st.cache_data(show_spinner=False)
def add_skew_for_stress_testing(df: pd.DataFrame, skew_strength: float) -> pd.DataFrame:
    """Create intentionally skewed/noisy data for stress-testing pipelines/models."""
    out = df.copy()
    if out.empty:
        return out

    rng = np.random.default_rng(42)
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for c in num_cols:
        if _is_binary_indicator(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").round().clip(0, 1)
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        valid_mask = s.notna()
        if not valid_mask.any():
            continue

        s_valid = s[valid_mask]
        min_val = float(s_valid.min())
        shifted = (s_valid - min_val) + 1e-6
        power = 1.0 + float(skew_strength)
        skewed = np.power(shifted, power) + min_val

        # Add sparse outliers to mimic heavy-tailed behaviour.
        outlier_mask = rng.random(len(skewed)) < (0.01 + 0.03 * skew_strength)
        if outlier_mask.any():
            skewed[outlier_mask] = skewed[outlier_mask] * (1.5 + (2.5 * skew_strength))

        s.loc[valid_mask] = skewed
        out[c] = s

    # Inject small random missingness bump for robustness checks.
    miss_rate = 0.005 + (0.02 * skew_strength)
    for c in out.columns:
        m = rng.random(len(out)) < miss_rate
        out.loc[m, c] = np.nan

    return out


def parse_seed_upload(uploaded: Any) -> pd.DataFrame:
    """Parse CSV/JSON upload for synthetic tab seed ingestion."""
    name = uploaded.name.lower()
    raw = uploaded.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    if name.endswith(".json"):
        data = json.loads(raw.decode("utf-8-sig"))
        if isinstance(data, list):
            return pd.json_normalize(data)
        if isinstance(data, dict):
            return pd.json_normalize([data])
    raise ValueError("Unsupported seed format. Use CSV or JSON.")


def _normalize_azure_endpoint(endpoint: str) -> tuple[str, str]:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/openai/v1"):
        base = endpoint[: -len("/openai/v1")]
        v1 = endpoint
    elif endpoint.endswith("/openai/v1/"):
        base = endpoint[: -len("/openai/v1/")]
        v1 = endpoint.rstrip("/")
    else:
        base = endpoint
        v1 = f"{base}/openai/v1"
    return base, v1


def _extract_json_array_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if "[" in cleaned and "]" in cleaned:
        s = cleaned.find("[")
        e = cleaned.rfind("]")
        if s != -1 and e != -1 and e > s:
            return cleaned[s : e + 1]
    return cleaned


def _parse_llm_tabular_output(text: str) -> pd.DataFrame:
    content = _extract_json_array_text(text)
    try:
        obj = json.loads(content)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.json_normalize([obj])
    except Exception:
        pass
    # Fallback: treat as CSV-like response
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception as e:
        raise RuntimeError(f"Could not parse tabular output from model. Raw starts with: {text[:180]}") from e


def generate_tabular_from_prompt(user_prompt: str, target_rows: int) -> pd.DataFrame:
    """
    Generate tabular data from free-text requirement using Azure Agent / deployment.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "") or os.getenv("AZURE_API_KEY", "")
    deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
        or os.getenv("MODEL_DEPLOYMENT_NAME", "").strip()
    )
    agent_id = os.getenv("AZURE_EXISTING_AGENT_ID", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    if not endpoint or not api_key:
        raise RuntimeError("Missing Azure endpoint/API key in environment variables.")
    endpoint_base, endpoint_v1 = _normalize_azure_endpoint(endpoint)

    generation_prompt = (
        "Generate realistic synthetic tabular data.\n"
        f"Business request: {user_prompt}\n"
        f"Rows required: {target_rows}\n"
        "Output strict JSON array only. No markdown, no explanation.\n"
        "Ensure coherent schema, realistic distributions, and valid cross-column consistency."
    )

    def _post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "api-key": api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode("utf-8")
            except Exception:
                raw = str(e)
            raise RuntimeError(f"Azure API HTTP {e.code}: {raw}") from e

    def _text_from_responses_payload(payload: dict[str, Any]) -> str:
        txt = payload.get("output_text", "")
        if txt:
            return txt
        output = payload.get("output", [])
        if output and isinstance(output, list):
            parts = []
            for item in output:
                for c in item.get("content", []) if isinstance(item, dict) else []:
                    t = c.get("text")
                    if t:
                        parts.append(t)
            return "\n".join(parts).strip()
        return ""

    def _deployment_text() -> str:
        if not deployment:
            raise RuntimeError("AZURE_OPENAI_DEPLOYMENT is required for deployment fallback.")
        # responses first
        try:
            payload = _post_json(
                f"{endpoint_v1}/responses",
                {
                    "model": deployment,
                    "input": [{"role": "user", "content": generation_prompt}],
                    "temperature": 0.5,
                },
            )
            txt = _text_from_responses_payload(payload)
            if txt:
                return txt
        except Exception:
            pass

        # chat completions fallback
        payload = _post_json(
            f"{endpoint_base}/openai/deployments/{deployment}/chat/completions?api-version={api_version}",
            {
                "messages": [
                    {"role": "system", "content": "Output strict JSON array only."},
                    {"role": "user", "content": generation_prompt},
                ],
                "temperature": 0.5,
                "max_tokens": 3500,
            },
        )
        return payload["choices"][0]["message"]["content"]

    model_text = ""
    if agent_id:
        if ":" in agent_id:
            agent_name, agent_version = agent_id.split(":", 1)
        else:
            agent_name, agent_version = agent_id, "1"
        reqs = [
            {
                "input": [{"role": "user", "content": generation_prompt}],
                "agent_reference": {"name": agent_name, "version": agent_version, "type": "agent_reference"},
                **({"model": deployment} if deployment else {}),
            },
            {
                "input": [{"role": "user", "content": generation_prompt}],
                "extra_body": {
                    "agent_reference": {"name": agent_name, "version": agent_version, "type": "agent_reference"}
                },
                **({"model": deployment} if deployment else {}),
            },
        ]
        for body in reqs:
            try:
                payload = _post_json(f"{endpoint_v1}/responses", body)
                model_text = _text_from_responses_payload(payload)
                if model_text:
                    break
            except Exception:
                continue
        if not model_text:
            model_text = _deployment_text()
    else:
        model_text = _deployment_text()

    out = _parse_llm_tabular_output(model_text)
    if out.empty:
        raise RuntimeError("Model returned empty dataset.")
    if len(out) < target_rows:
        extra = out.sample(target_rows - len(out), replace=True, random_state=42)
        out = pd.concat([out, extra], ignore_index=True)
    return out.head(target_rows)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    if "_snapshot_restore_attempted" not in st.session_state:
        restored = False
        if ENABLE_SESSION_SNAPSHOT:
            restored = restore_session_snapshot(SESSION_SNAPSHOT_TTL_SECONDS)
        st.session_state["_snapshot_restore_attempted"] = True
        st.session_state["_snapshot_restored"] = restored

    if "data_registry" not in st.session_state:
        st.session_state.data_registry: dict[str, pd.DataFrame] = {}
    if "file_signatures" not in st.session_state:
        st.session_state.file_signatures: set[str] = set()
    if "joined_df" not in st.session_state:
        st.session_state.joined_df: pd.DataFrame | None = None
    if "join_summary" not in st.session_state:
        st.session_state.join_summary: str = ""
    if "synthetic_df" not in st.session_state:
        st.session_state.synthetic_df: pd.DataFrame | None = None
    if "bot_generated_df" not in st.session_state:
        st.session_state.bot_generated_df: pd.DataFrame | None = None
    if "jump_to_ai_astra" not in st.session_state:
        st.session_state.jump_to_ai_astra = False
    if "jump_to_architect" not in st.session_state:
        st.session_state.jump_to_architect = False
    if "architect_input_mode" not in st.session_state:
        st.session_state.architect_input_mode = "Seed Data"
    if "gateway_input_mode" not in st.session_state:
        st.session_state.gateway_input_mode = "Both"
    if "gateway_nl_prompt" not in st.session_state:
        st.session_state.gateway_nl_prompt = (
            "Generate credit-risk DPD data with customer_id, loan_id, dpd_bucket, "
            "exposure, region, segment, delinquency_date."
        )
    if "jump_to_lens" not in st.session_state:
        st.session_state.jump_to_lens = False
    if "architect_generated_df" not in st.session_state:
        st.session_state.architect_generated_df: pd.DataFrame | None = None
    if "architect_seed_label" not in st.session_state:
        st.session_state.architect_seed_label: str = ""
    if "critic_synth_df" not in st.session_state:
        st.session_state.critic_synth_df: pd.DataFrame | None = None
    if "critic_js_df" not in st.session_state:
        st.session_state.critic_js_df: pd.DataFrame | None = None
    if "critic_utility_df" not in st.session_state:
        st.session_state.critic_utility_df: pd.DataFrame | None = None
    if "syn_last_guardrails" not in st.session_state:
        st.session_state.syn_last_guardrails: dict[str, Any] | None = None
    if "mv_bundle" not in st.session_state:
        st.session_state.mv_bundle: dict[str, Any] | None = None
    if "upload_widget_nonce" not in st.session_state:
        st.session_state.upload_widget_nonce = 0


def restore_session_snapshot(max_age_seconds: int) -> bool:
    """Restore key app state if a recent snapshot is available."""
    if not os.path.exists(SESSION_SNAPSHOT_PATH):
        return False
    try:
        with io.open(SESSION_SNAPSHOT_PATH, "rb") as f:
            payload = pickle.load(f)
        saved_at = float(payload.get("saved_at", 0.0))
        if (time.time() - saved_at) > max_age_seconds:
            return False
        state = payload.get("state", {})
        if not isinstance(state, dict):
            return False
        for key, value in state.items():
            st.session_state[key] = value
        return True
    except Exception:
        return False


def save_session_snapshot() -> None:
    """Persist key app state so brief idle reconnects can recover context."""
    if not ENABLE_SESSION_SNAPSHOT:
        return
    keys_to_save = [
        "data_registry",
        "file_signatures",
        "joined_df",
        "join_summary",
        "synthetic_df",
        "bot_generated_df",
    ]
    state = {k: st.session_state.get(k) for k in keys_to_save if k in st.session_state}
    payload = {"saved_at": time.time(), "state": state}
    try:
        parent = os.path.dirname(SESSION_SNAPSHOT_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{SESSION_SNAPSHOT_PATH}.tmp"
        with io.open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, SESSION_SNAPSHOT_PATH)
    except Exception:
        # Snapshot is best-effort and should never block app usage.
        return


def delete_session_snapshot_file() -> None:
    """Remove persisted snapshot so refresh starts from a clean state."""
    try:
        if os.path.exists(SESSION_SNAPSHOT_PATH):
            os.remove(SESSION_SNAPSHOT_PATH)
    except Exception:
        return


def clear_loaded_app_state() -> None:
    """Clear all user-loaded/generated datasets and derived analysis state."""
    st.session_state.data_registry = {}
    st.session_state.file_signatures = set()
    st.session_state.joined_df = None
    st.session_state.join_summary = ""
    st.session_state.synthetic_df = None
    st.session_state.bot_generated_df = None
    st.session_state.architect_generated_df = None
    st.session_state.architect_seed_label = ""
    st.session_state.critic_synth_df = None
    st.session_state.critic_js_df = None
    st.session_state.critic_utility_df = None
    st.session_state.syn_last_guardrails = None
    st.session_state.mv_bundle = None
    st.session_state.jump_to_lens = False
    st.session_state.jump_to_ai_astra = False
    st.session_state.jump_to_architect = False
    st.session_state.architect_input_mode = "Seed Data"
    st.session_state.gateway_input_mode = "Both"
    st.session_state.gateway_nl_prompt = (
        "Generate credit-risk DPD data with customer_id, loan_id, dpd_bucket, "
        "exposure, region, segment, delinquency_date."
    )
    st.session_state.upload_widget_nonce = int(st.session_state.get("upload_widget_nonce", 0)) + 1
    delete_session_snapshot_file()


# ---------------------------------------------------------------------------
# Sidebar control plane
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-hero">
            <div class="sidebar-kicker">Gateway</div>
            <div class="sidebar-subtitle">Input</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-card-title">Input Model</div>', unsafe_allow_html=True)
    gateway_mode = st.sidebar.radio(
        "Input Model",
        ["Seed Data", "Natural Language", "Both"],
        key="gateway_input_mode",
        label_visibility="collapsed",
    )
    nl_disabled = gateway_mode == "Seed Data"
    st.sidebar.text_area(
        "Natural Language Prompt",
        key="gateway_nl_prompt",
        height=110,
        disabled=nl_disabled,
        help="Architect uses this prompt directly in Natural Language mode.",
    )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="sidebar-card-title">Upload CSV / Excel / JSON files</div>',
        unsafe_allow_html=True,
    )
    uploads = st.sidebar.file_uploader(
        "Upload CSV / Excel / JSON files",
        type=["csv", "xlsx", "xls", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        disabled=(gateway_mode == "Natural Language"),
        key=f"sidebar_uploads_{st.session_state.upload_widget_nonce}",
    )
    st.sidebar.markdown(
        '<div class="sidebar-helper">200MB per file • CSV, XLSX, XLS, JSON</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    if st.sidebar.button("✨ Generate via AI Astra", key="sidebar_jump_ai_astra", type="primary", use_container_width=True):
        st.session_state.jump_to_architect = True
        st.session_state.gateway_input_mode = "Natural Language"
        st.session_state.architect_input_mode = "Natural Language"
        st.rerun()

    if uploads:
        for file_obj in uploads:
            raw = file_obj.getvalue()
            signature = f"{file_obj.name}:{file_obj.size}:{hash(raw)}"
            if signature in st.session_state.file_signatures:
                continue
            try:
                parsed_tables = dp.parse_uploaded_file(file_obj.name, raw)
                st.session_state.data_registry.update(parsed_tables)
                st.session_state.file_signatures.add(signature)
                if parsed_tables:
                    first_loaded = next(iter(parsed_tables.keys()))
                    st.session_state.lens_source = f"Registry::{first_loaded}"
                st.session_state.jump_to_lens = True
                st.toast(f"Loaded: {file_obj.name}")
            except Exception as exc:
                st.sidebar.error(f"{file_obj.name}: {exc}")

    st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-card-title">Data Registry</div>', unsafe_allow_html=True)

    if not st.session_state.data_registry:
        st.sidebar.markdown(
            '<div class="sidebar-empty">No files uploaded yet.</div>',
            unsafe_allow_html=True,
        )
    else:
        for table_name, table_df in st.session_state.data_registry.items():
            with st.sidebar.expander(f"{table_name}", expanded=False):
                column_names = [str(c) for c in table_df.columns]
                if not column_names:
                    st.markdown(
                        '<div class="sidebar-empty">No columns found in this file.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    dynamic_height = max(96, min(172, 88 + len(column_names) * 9))
                    chips_html = "".join(
                        f'<div class="column-chip">{html.escape(col_name)}</div>'
                        for col_name in column_names
                    )
                    st.markdown(
                        (
                            f'<div class="columns-liquid-box" style="max-height:{dynamic_height}px;">'
                            f'<div class="columns-liquid-head"><span>Columns</span><span class="columns-liquid-count">{len(column_names)}</span></div>'
                            f'<div class="columns-chip-grid">{chips_html}</div>'
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )

    if st.sidebar.button("Clear Registry", use_container_width=True):
        clear_loaded_app_state()
        st.sidebar.success("Registry cleared.")
        st.rerun()
    st.sidebar.markdown("</div>", unsafe_allow_html=True)


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
            key="ana_rows_for_analysis",
            help="Controls sample size used in charts.",
        )
    )
    plot_df = joined_df.head(rows_to_use).copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{len(joined_df):,}")
    k2.metric("Columns", joined_df.shape[1])
    k3.metric("Null Cells", f"{int(joined_df.isna().sum().sum()):,}")
    k4.metric("Duplicates", f"{int(joined_df.duplicated().sum()):,}")

    dtypes_df = pd.DataFrame(
        {
            "column": joined_df.columns,
            "dtype": [str(t) for t in joined_df.dtypes],
            "total_records": [len(joined_df)] * len(joined_df.columns),
            "null_count": joined_df.isna().sum().values,
            "non_null_count": joined_df.notna().sum().values,
        }
    )
    if len(joined_df) > 0:
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
    chart_key_base = "analytics_joined"

    with left_col:
        st.markdown('<div class="minor-title">Schema & Null Summary</div>', unsafe_allow_html=True)
        st.dataframe(dtypes_df, use_container_width=True, height=480)

    with right_col:
        st.markdown('<div class="minor-title">Quick Graph Builder</div>', unsafe_allow_html=True)

        all_cols = list(plot_df.columns)
        numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(plot_df[c])]
        graph_options = ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie", "Bar Count", "Heatmap"]

        graph_type = st.selectbox(
            "Graph",
            graph_options,
            key=f"ana_graph_{chart_key_base}",
        )

        x_axis = st.selectbox(
            "X Axis",
            all_cols,
            key=f"ana_x_{chart_key_base}",
        )
        y_axis = st.selectbox(
            "Y Axis",
            ["(auto/count)"] + numeric_cols,
            key=f"ana_y_{chart_key_base}",
        )
        y_col = None if y_axis == "(auto/count)" else y_axis

        fig: go.Figure | None = None
        if plot_df.empty:
            st.info("Joined data is empty.")
        else:
            if graph_type == "Heatmap":
                if len(numeric_cols) < 2:
                    st.info("Heatmap requires at least 2 numeric columns.")
                else:
                    corr = plot_df[numeric_cols].corr(numeric_only=True)
                    fig = px.imshow(
                        corr,
                        text_auto=".2f",
                        aspect="auto",
                        color_continuous_scale="Oranges",
                        title="Correlation Heatmap",
                    )
            elif graph_type == "Bar":
                if y_col:
                    fig = px.bar(plot_df, x=x_axis, y=y_col, color_discrete_sequence=[ACCENT])
                else:
                    vc = plot_df[x_axis].astype(str).value_counts().head(25)
                    fig = px.bar(
                        x=vc.index,
                        y=vc.values,
                        labels={"x": x_axis, "y": "count"},
                        color=vc.values,
                        color_continuous_scale="Oranges",
                    )
            elif graph_type == "Line":
                if y_col:
                    fig = px.line(plot_df, x=x_axis, y=y_col, markers=True)
                else:
                    st.info("Select a numeric Y Axis for Line chart.")
            elif graph_type == "Scatter":
                if y_col:
                    fig = px.scatter(plot_df, x=x_axis, y=y_col)
                else:
                    st.info("Select a numeric Y Axis for Scatter chart.")
            elif graph_type == "Histogram":
                fig = px.histogram(plot_df, x=x_axis, color_discrete_sequence=[ACCENT])
            elif graph_type == "Box":
                target = y_col or x_axis
                if target in numeric_cols:
                    fig = px.box(plot_df, y=target, color_discrete_sequence=[ACCENT])
                else:
                    st.info("Box chart requires a numeric axis.")
            elif graph_type == "Bar Count":
                vc = plot_df[x_axis].astype(str).value_counts().head(25)
                fig = px.bar(
                    x=vc.index,
                    y=vc.values,
                    labels={"x": x_axis, "y": "count"},
                    color=vc.values,
                    color_continuous_scale="Oranges",
                )
            elif graph_type == "Pie":
                if y_col:
                    fig = px.pie(plot_df, names=x_axis, values=y_col, color_discrete_sequence=px.colors.sequential.Oranges)
                else:
                    vc = plot_df[x_axis].astype(str).value_counts().head(20)
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
            max_value=max(1, len(joined_df)),
            value=min(25, len(joined_df)),
            step=1,
            key="ana_raw_rows",
            help="Type the exact number of rows to print.",
        )
    )
    st.dataframe(joined_df.head(rows_to_show), use_container_width=True, height=220)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="minor-title">Column Statistics</div>', unsafe_allow_html=True)
    stats_df = build_column_statistics(joined_df)
    numeric_stats_cols = [
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
    for c in numeric_stats_cols:
        if c in stats_df.columns:
            stats_df[c] = pd.to_numeric(stats_df[c], errors="coerce").round(4)
    st.dataframe(stats_df, use_container_width=True, height=320)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_synthetic_generator_tab() -> None:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Synthetic Data Generator</div>', unsafe_allow_html=True)

    st.markdown(
        """```text
1) Privacy & masking (seed lens) -> 2) Configuration
3) Generate synthetic data + critic -> 4) Self-Service Download
```"""
    )

    # Seed data source
    source_options = [f"Registry::{k}" for k in st.session_state.data_registry.keys()]
    if st.session_state.joined_df is not None and not st.session_state.joined_df.empty:
        source_options.insert(0, "Joined dataset")
    if not source_options:
        st.info("Upload seed data in Gateway (left pane) to run Architect with Seed Data.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    seed_source = st.selectbox("Seed source", source_options, key="syn_seed_source")
    seed_df: pd.DataFrame | None = None

    if seed_source == "Joined dataset":
        seed_df = st.session_state.joined_df.copy()
    else:
        table_name = seed_source.split("Registry::", 1)[1]
        seed_df = st.session_state.data_registry.get(table_name)

    if seed_df is None or seed_df.empty:
        st.info("Choose a seed source to start synthetic generation.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("### Privacy & masking - seed lens")
    pii_df = dp.detect_pii_columns(seed_df)
    pii_candidates = pii_df.loc[pii_df["pii_detected"] == True, "column"].astype(str).tolist()
    with st.expander("Privacy & masking - seed lens", expanded=False):
        selected_pii = st.multiselect(
            "Columns to mask/drop",
            options=list(seed_df.columns),
            default=pii_candidates,
            key="syn_privacy_cols",
        )
        p1, p2, p3 = st.columns(3)
        scrub_mode = p1.selectbox("PII mode", ["mask", "drop"], key="syn_privacy_mode")
        k_anon = int(
            p2.number_input(
                "k-anonymity",
                min_value=1,
                max_value=50,
                value=3,
                step=1,
                key="syn_privacy_k",
            )
        )
        noise_level = float(
            p3.slider(
                "Numeric noise",
                0.0,
                0.5,
                0.02,
                0.01,
                key="syn_privacy_noise",
            )
        )
        private_preview = se.apply_privacy_transform(
            seed_df,
            tuple(selected_pii),
            scrub_mode,
            k_anon,
            noise_level,
        )
        st.caption("Preview (masked seed lens)")
        st.dataframe(private_preview.head(8), use_container_width=True, height=210)
    selected_pii = st.session_state.get("syn_privacy_cols", pii_candidates)
    scrub_mode = st.session_state.get("syn_privacy_mode", "mask")
    k_anon = int(st.session_state.get("syn_privacy_k", 3))
    noise_level = float(st.session_state.get("syn_privacy_noise", 0.02))
    private_df = se.apply_privacy_transform(
        seed_df,
        tuple(selected_pii),
        scrub_mode,
        k_anon,
        noise_level,
    )
    st.caption(f"Post-privacy shape: {private_df.shape[0]:,} x {private_df.shape[1]}")

    st.markdown("### Configuration")
    method_selection = st.radio(
        "Method selection",
        ["Architect decides (recommended)", "I will choose the method"],
        horizontal=True,
        key="syn_method_selection",
    )
    g1, g2 = st.columns(2)
    target_rows = int(
        g2.number_input(
            "Target synthetic rows",
            min_value=max(10, len(private_df)),
            max_value=1_000_000,
            value=min(max(10, len(private_df) * 10), 100_000),
            step=100,
            key="syn_target_rows",
        )
    )
    model_choice = "Diffusion-style bootstrap"
    auto_reason = ""
    if method_selection == "Architect decides (recommended)":
        model_choice, auto_reason = _auto_choose_generation_model(private_df)
        g1.markdown(f"**Generation model:** `{model_choice}`")
        g1.caption(auto_reason)
    else:
        model_choice = g1.selectbox(
            "Generation model",
            [
                "CTGAN (GAN)",
                "TVAE",
                "Diffusion-style bootstrap",
                "AI Astra",
                "Bootstrap (sample rows with replacement)",
                "Subsample (without replacement)",
                "Tile (repeat full table)",
                "Tile + shuffle",
                "Synthetic - perturb",
                "Synthetic - independent",
                "GAN (Architect style)",
                "Diffusion (Architect style)",
            ],
            key="syn_model_choice",
        )

    user_instruction = ""
    profile_mode = "Normalized (seed-aligned)"
    skew_strength = 0.35
    if model_choice == "AI Astra":
        user_instruction = st.text_area(
            "Custom generation instruction (optional)",
            value="",
            height=90,
            key="syn_user_instruction",
            help="Add business-specific rules, edge cases, or extra fields to guide generation.",
        )
        p_mode1, p_mode2 = st.columns(2)
        profile_mode = p_mode1.selectbox(
            "Distribution profile",
            ["Normalized (seed-aligned)", "Skewed (stress-test)"],
            key="syn_distribution_profile",
            help="Choose whether output follows seed distribution closely or is intentionally skewed/noisy.",
        )
        if profile_mode == "Skewed (stress-test)":
            skew_strength = float(
                p_mode2.slider(
                    "Skew intensity",
                    min_value=0.10,
                    max_value=1.00,
                    value=0.35,
                    step=0.05,
                    key="syn_skew_strength",
                )
            )

    st.markdown("#### Critic controls")
    r1, r2, r3 = st.columns(3)
    critic_profile = r1.selectbox(
        "Critic scoring profile",
        options=list(SYN_CRITIC_PROFILES.keys()),
        format_func=lambda k: SYN_CRITIC_PROFILES[k]["title"],
        key="syn_critic_profile",
    )
    stop_score = float(
        r2.number_input(
            "Stop when fidelity score >= (1-5)",
            min_value=1.0,
            max_value=5.0,
            value=float(SYN_CRITIC_PROFILES[critic_profile]["pass_threshold"]),
            step=0.1,
            key="syn_stop_score",
        )
    )
    max_attempts = int(
        r3.number_input(
            "Max generation attempts",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            key="syn_max_attempts",
        )
    )

    generate_btn = st.button(
        "Generate Synthetic Data and Critic",
        type="primary",
        key="syn_generate_btn",
    )
    synthetic_df: pd.DataFrame | None = st.session_state.get("synthetic_df")

    if generate_btn:
        try:
            with st.spinner("Generating synthetic data..."):
                best_df: pd.DataFrame | None = None
                best_score = -1.0
                best_js_df: pd.DataFrame | None = None
                best_utility_df: pd.DataFrame | None = None
                attempts_used = 0

                for attempt in range(max_attempts):
                    attempts_used = attempt + 1
                    np.random.seed(42 + attempt)
                    if model_choice == "CTGAN (GAN)":
                        if SDV_AVAILABLE:
                            candidate = se.generate_sdv_synthetic(private_df, target_rows, "CTGAN")
                        else:
                            candidate = se.generate_bootstrap_synthetic(private_df, target_rows)
                    elif model_choice == "TVAE":
                        if SDV_AVAILABLE:
                            candidate = se.generate_sdv_synthetic(private_df, target_rows, "TVAE")
                        else:
                            candidate = se.generate_bootstrap_synthetic(private_df, target_rows)
                    elif model_choice == "AI Astra":
                        candidate = generate_with_azure_llm(
                            private_df,
                            target_rows,
                            custom_instruction=user_instruction,
                        )
                    elif model_choice == "Bootstrap (sample rows with replacement)":
                        candidate = private_df.sample(n=target_rows, replace=True, random_state=42 + attempt).reset_index(drop=True)
                    elif model_choice == "Subsample (without replacement)":
                        if target_rows <= len(private_df):
                            candidate = private_df.sample(n=target_rows, replace=False, random_state=42 + attempt).reset_index(drop=True)
                        else:
                            candidate = private_df.sample(n=target_rows, replace=True, random_state=42 + attempt).reset_index(drop=True)
                    elif model_choice == "Tile (repeat full table)":
                        if len(private_df) == 0:
                            candidate = private_df.copy()
                        else:
                            reps = int(np.ceil(target_rows / len(private_df)))
                            candidate = pd.concat([private_df] * reps, ignore_index=True).head(target_rows).reset_index(drop=True)
                    elif model_choice == "Tile + shuffle":
                        if len(private_df) == 0:
                            candidate = private_df.copy()
                        else:
                            reps = int(np.ceil(target_rows / len(private_df)))
                            tiled = pd.concat([private_df] * reps, ignore_index=True)
                            candidate = tiled.sample(frac=1.0, random_state=42 + attempt).head(target_rows).reset_index(drop=True)
                    elif model_choice == "Synthetic - perturb":
                        candidate = se.generate_bootstrap_synthetic(private_df, target_rows)
                    elif model_choice == "Synthetic - independent":
                        synth_cols: dict[str, Any] = {}
                        for col in private_df.columns:
                            s = private_df[col]
                            if pd.api.types.is_numeric_dtype(s):
                                mu = float(pd.to_numeric(s, errors="coerce").mean() or 0.0)
                                sd = float(pd.to_numeric(s, errors="coerce").std() or 1.0)
                                synth_cols[col] = np.random.normal(mu, max(sd, 1e-9), size=target_rows)
                            else:
                                vals = s.dropna().astype(str)
                                if vals.empty:
                                    synth_cols[col] = [np.nan] * target_rows
                                else:
                                    synth_cols[col] = np.random.choice(vals.values, size=target_rows, replace=True)
                        candidate = pd.DataFrame(synth_cols)
                    elif model_choice == "GAN (Architect style)":
                        if SDV_AVAILABLE:
                            candidate = se.generate_sdv_synthetic(private_df, target_rows, "CTGAN")
                        else:
                            candidate = se.generate_bootstrap_synthetic(private_df, target_rows)
                    elif model_choice == "Diffusion (Architect style)":
                        candidate = se.generate_bootstrap_synthetic(private_df, target_rows)
                    else:
                        candidate = se.generate_bootstrap_synthetic(private_df, target_rows)

                    if profile_mode == "Normalized (seed-aligned)":
                        candidate = se.align_synthetic_to_seed_distribution(private_df, candidate)
                    else:
                        candidate = se.add_skew_for_stress_testing(candidate, skew_strength)

                    js_try, utility_try = se.compute_fidelity_metrics(private_df, candidate)
                    score_try = _synthetic_fidelity_score(js_try, utility_try, critic_profile)
                    if score_try > best_score:
                        best_score = score_try
                        best_df = candidate
                        best_js_df = js_try
                        best_utility_df = utility_try
                    if score_try >= stop_score:
                        break

                if best_df is None:
                    raise RuntimeError("No synthetic output generated.")
                synthetic_df = best_df
                guardrails_report = se.validate_synthetic_dataset(private_df, synthetic_df, [])
                st.session_state["syn_last_score"] = best_score
                st.session_state["syn_last_attempts"] = attempts_used
                st.session_state["syn_last_profile"] = critic_profile
                st.session_state["syn_last_js_df"] = best_js_df
                st.session_state["syn_last_utility_df"] = best_utility_df
                st.session_state["syn_last_guardrails"] = guardrails_report
            st.session_state.synthetic_df = synthetic_df
            st.success(f"Synthetic dataset generated: {len(synthetic_df):,} rows")
            st.caption(
                f"Best critic score: {st.session_state.get('syn_last_score', 0):.2f}/5 "
                f"| Attempts used: {st.session_state.get('syn_last_attempts', 1)} "
                f"| Profile: {SYN_CRITIC_PROFILES[critic_profile]['title']}"
            )
            if user_instruction.strip():
                st.caption("Custom instruction captured for generation context.")
            st.toast("Synthetic generation and critic scoring complete.")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

    synthetic_df = st.session_state.get("synthetic_df")
    if synthetic_df is not None and not synthetic_df.empty:
        st.markdown("### Critic Results")
        js_df = st.session_state.get("syn_last_js_df")
        utility_df = st.session_state.get("syn_last_utility_df")
        guardrails_report = st.session_state.get("syn_last_guardrails") or {}
        if not isinstance(js_df, pd.DataFrame) or not isinstance(utility_df, pd.DataFrame):
            js_df, utility_df = se.compute_fidelity_metrics(private_df, synthetic_df)
        latest_profile = st.session_state.get("syn_last_profile", "standard")
        latest_score = float(st.session_state.get("syn_last_score", _synthetic_fidelity_score(js_df, utility_df, latest_profile)))
        profile_meta = SYN_CRITIC_PROFILES.get(latest_profile, SYN_CRITIC_PROFILES["standard"])
        if latest_score >= float(profile_meta["pass_threshold"]):
            st.success(
                f"Critic verdict: PASS | score {latest_score:.2f}/5 "
                f"(threshold {profile_meta['pass_threshold']}/5)"
            )
        else:
            st.warning(
                f"Critic verdict: REVIEW | score {latest_score:.2f}/5 "
                f"(threshold {profile_meta['pass_threshold']}/5)"
            )
        st.markdown("**Guardrails**")
        se.render_guardrails_table(guardrails_report)
        se.render_seed_synth_corr(private_df, synthetic_df, "syn_gen_corr")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**JS Divergence (numeric columns)**")
            st.dataframe(js_df, use_container_width=True, height=220)
        with s2:
            st.markdown("**Correlation / Utility checks**")
            st.dataframe(utility_df, use_container_width=True, height=220)

        c1, c2 = st.columns((1.1, 0.9))
        with c1:
            st.markdown("**Schema (synthetic)**")
            schema_df = pd.DataFrame(
                {
                    "column": synthetic_df.columns,
                    "dtype": [str(t) for t in synthetic_df.dtypes],
                    "non_null_count": synthetic_df.notna().sum().values,
                    "null_count": synthetic_df.isna().sum().values,
                    "unique_count": synthetic_df.nunique(dropna=True).values,
                }
            )
            if len(synthetic_df) > 0:
                schema_df["missing_rate_%"] = (
                    (schema_df["null_count"] / len(synthetic_df)) * 100
                ).round(2)
            else:
                schema_df["missing_rate_%"] = 0.0
            st.dataframe(schema_df, use_container_width=True, height=320)
        with c2:
            dtype_counts = synthetic_df.dtypes.astype(str).value_counts()
            fig_dtype = px.pie(
                values=dtype_counts.values,
                names=dtype_counts.index,
                title="Types",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_dtype.update_layout(height=320, margin=dict(t=40, l=20, r=20, b=20))
            st.plotly_chart(fig_dtype, use_container_width=True, key="syn_critic_dtype")

        st.markdown("**Column statistics (synthetic)**")
        stats_df = dp.build_column_statistics(synthetic_df)
        st.dataframe(stats_df, use_container_width=True, height=340)

        st.markdown("**Preview**")
        preview_n = int(
            st.number_input(
                "Rows to preview (critic)",
                min_value=1,
                max_value=max(1, len(synthetic_df)),
                value=min(25, len(synthetic_df)),
                step=1,
                key="syn_critic_preview_rows",
            )
        )
        p1, p2 = st.columns(2)
        with p1:
            st.caption("Head")
            st.dataframe(synthetic_df.head(preview_n), use_container_width=True, height=230)
        with p2:
            st.caption("Tail")
            st.dataframe(synthetic_df.tail(min(preview_n, len(synthetic_df))), use_container_width=True, height=230)

        st.markdown("### Self-Service Download")
        preview_rows = int(
            st.number_input(
                "Rows to preview (synthetic)",
                min_value=1,
                max_value=max(1, len(synthetic_df)),
                value=min(30, len(synthetic_df)),
                step=1,
                key="syn_preview_rows",
            )
        )
        st.dataframe(synthetic_df.head(preview_rows), use_container_width=True, height=240)
        csv_bytes = synthetic_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download synthetic dataset (CSV)",
            data=csv_bytes,
            file_name="synthetic_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download critic report (JSON)",
            data=json.dumps(
                {
                    "score": latest_score,
                    "profile": latest_profile,
                    "js": js_df.to_dict(orient="records"),
                    "utility": utility_df.to_dict(orient="records"),
                    "guardrails": guardrails_report,
                },
                indent=2,
                default=str,
            ).encode("utf-8"),
            file_name="synthetic_critic_report.json",
            mime="application/json",
            use_container_width=True,
            key="syn_critic_json_dl",
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_data_bot_tab() -> None:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Astra</div>', unsafe_allow_html=True)
    st.caption("Prompt is taken from Gateway input mode.")
    request_text = str(st.session_state.get("gateway_nl_prompt", "")).strip()
    if not request_text:
        st.warning("Add a Natural Language prompt in Gateway to generate data.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    st.markdown(f"**Gateway prompt:** {request_text}")
    target_rows = int(
        st.number_input(
            "Rows to generate",
            min_value=10,
            max_value=1_000_000,
            value=1000,
            step=100,
            key="bot_target_rows",
        )
    )
    generate_btn = st.button("Generate", type="primary", key="bot_generate_btn")

    if generate_btn:
        try:
            with st.spinner("Bot is generating tabular data..."):
                df = generate_tabular_from_prompt(request_text, target_rows)
            st.session_state.bot_generated_df = df
            st.success(f"Generated {len(df):,} rows and {df.shape[1]} columns.")
            st.toast("AI Astra generation complete.")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

    bot_df = st.session_state.get("bot_generated_df")
    if bot_df is not None and not bot_df.empty:
        st.markdown('<div class="minor-title">Generated Dataset</div>', unsafe_allow_html=True)
        preview_rows = int(
            st.number_input(
                "Rows to preview",
                min_value=1,
                max_value=max(1, len(bot_df)),
                value=min(25, len(bot_df)),
                step=1,
                key="bot_preview_rows",
            )
        )
        st.dataframe(bot_df.head(preview_rows), use_container_width=True, height=240)
        st.download_button(
            "Download generated data (CSV)",
            data=bot_df.to_csv(index=False).encode("utf-8"),
            file_name="bot_generated_data.csv",
            mime="text/csv",
            use_container_width=True,
            key="bot_download_csv",
        )

        # Full quality/report sections (same style as preview/analytics)
        dtypes_df = pd.DataFrame(
            {
                "column": bot_df.columns,
                "dtype": [str(t) for t in bot_df.dtypes],
                "total_records": [len(bot_df)] * len(bot_df.columns),
                "null_count": bot_df.isna().sum().values,
                "non_null_count": bot_df.notna().sum().values,
            }
        )
        if len(bot_df) > 0:
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
        chart_key_base = "bot_generated"
        with left_col:
            st.markdown('<div class="minor-title">Schema & Null Summary</div>', unsafe_allow_html=True)
            st.dataframe(dtypes_df, use_container_width=True, height=480)

        with right_col:
            st.markdown('<div class="minor-title">Quick Graph Builder</div>', unsafe_allow_html=True)
            all_cols = list(bot_df.columns)
            numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(bot_df[c])]
            graph_options = ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie", "Bar Count"]
            graph_type = st.selectbox("Graph", graph_options, key=f"bot_graph_{chart_key_base}")
            x_axis = st.selectbox("X Axis", all_cols, key=f"bot_x_{chart_key_base}")
            y_axis = st.selectbox(
                "Y Axis", ["(auto/count)"] + numeric_cols, key=f"bot_y_{chart_key_base}"
            )
            y_col = None if y_axis == "(auto/count)" else y_axis
            graph_df = bot_df.head(1000).copy()
            fig = None
            if graph_type == "Bar":
                if y_col:
                    fig = px.bar(graph_df, x=x_axis, y=y_col, color_discrete_sequence=[ACCENT])
                else:
                    vc = graph_df[x_axis].astype(str).value_counts().head(25)
                    fig = px.bar(
                        x=vc.index, y=vc.values, labels={"x": x_axis, "y": "count"},
                        color=vc.values, color_continuous_scale="Oranges"
                    )
            elif graph_type == "Line":
                if y_col:
                    fig = px.line(graph_df, x=x_axis, y=y_col, markers=True)
                else:
                    st.info("Select a numeric Y Axis for Line chart.")
            elif graph_type == "Scatter":
                if y_col:
                    fig = px.scatter(graph_df, x=x_axis, y=y_col)
                else:
                    st.info("Select a numeric Y Axis for Scatter chart.")
            elif graph_type == "Histogram":
                fig = px.histogram(graph_df, x=x_axis, color_discrete_sequence=[ACCENT])
            elif graph_type == "Box":
                target = y_col or x_axis
                if target in numeric_cols:
                    fig = px.box(graph_df, y=target, color_discrete_sequence=[ACCENT])
                else:
                    st.info("Box chart requires a numeric axis.")
            elif graph_type == "Bar Count":
                vc = graph_df[x_axis].astype(str).value_counts().head(25)
                fig = px.bar(
                    x=vc.index, y=vc.values, labels={"x": x_axis, "y": "count"},
                    color=vc.values, color_continuous_scale="Oranges"
                )
            elif graph_type == "Pie":
                if y_col:
                    fig = px.pie(graph_df, names=x_axis, values=y_col, color_discrete_sequence=px.colors.sequential.Oranges)
                else:
                    vc = graph_df[x_axis].astype(str).value_counts().head(20)
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
                max_value=max(1, len(bot_df)),
                value=min(25, len(bot_df)),
                step=1,
                key="bot_raw_rows",
            )
        )
        st.dataframe(bot_df.head(rows_to_show), use_container_width=True, height=220)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.markdown('<div class="minor-title">Column Statistics</div>', unsafe_allow_html=True)
        stats_df = dp.build_column_statistics(bot_df)
        numeric_cols_stats = ["mean", "min", "max", "5%", "10%", "20%", "30%", "50%", "70%", "80%", "90%", "95%"]
        for c in numeric_cols_stats:
            if c in stats_df.columns:
                stats_df[c] = pd.to_numeric(stats_df[c], errors="coerce").round(4)
        st.dataframe(stats_df, use_container_width=True, height=320)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _advanced_dataset_options() -> dict[str, pd.DataFrame]:
    options: dict[str, pd.DataFrame] = {}
    if st.session_state.get("joined_df") is not None and not st.session_state["joined_df"].empty:
        options["Joined dataset"] = st.session_state["joined_df"]
    for name, df in st.session_state.data_registry.items():
        options[f"Registry::{name}"] = df
    if st.session_state.get("synthetic_df") is not None and not st.session_state["synthetic_df"].empty:
        options["Synthetic Generator output"] = st.session_state["synthetic_df"]
    if st.session_state.get("architect_generated_df") is not None and not st.session_state["architect_generated_df"].empty:
        options["Architect output"] = st.session_state["architect_generated_df"]
    if st.session_state.get("bot_generated_df") is not None and not st.session_state["bot_generated_df"].empty:
        options["AI Astra output"] = st.session_state["bot_generated_df"]
    return options


def render_lens_tab() -> None:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Lens</div>', unsafe_allow_html=True)

    options = _advanced_dataset_options()
    if not options:
        st.info("Upload or generate at least one dataset to use Lens.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    source = st.selectbox("Lens source", list(options.keys()), key="lens_source")
    df = options[source]
    is_synthetic = source in ("Synthetic Generator output", "Architect output", "AI Astra output")

    if is_synthetic:
        st.markdown("#### Synthetic - at a glance")
    else:
        st.markdown("#### Seed - at a glance")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Numeric", int(len(df.select_dtypes(include=[np.number]).columns)))
    m4.metric("Datetime (inferred)", int(len(df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns)))
    m5.metric("Duplicates", f"{int(df.duplicated().sum()):,}")
    n_cat = int(len(df.select_dtypes(exclude=[np.number]).columns))
    miss_cells = int(df.isna().sum().sum())
    mem_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
    st.caption(f"Text / boolean columns: **{n_cat}** - Missing cells: **{miss_cells:,}** - ~**{mem_mb:.2f} MB** in memory")

    schema_df = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "non_null_count": df.notna().sum().values,
            "null_count": df.isna().sum().values,
            "unique_count": df.nunique(dropna=True).values,
        }
    )
    if len(df) > 0:
        schema_df["missing_rate_%"] = ((schema_df["null_count"] / len(df)) * 100).round(2)
    else:
        schema_df["missing_rate_%"] = 0.0
    col_a, col_b = st.columns((1.15, 0.85))
    with col_a:
        st.markdown("**Schema**")
        st.dataframe(schema_df, use_container_width=True, height=320)
    with col_b:
        dtype_counts = df.dtypes.astype(str).value_counts()
        fig_dtype = px.pie(
            values=dtype_counts.values,
            names=dtype_counts.index,
            title="Types",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_dtype.update_layout(height=320, margin=dict(t=40, l=20, r=20, b=20))
        st.plotly_chart(fig_dtype, use_container_width=True, key=f"lens_dtype_{source}")

    with st.expander("All column names", expanded=False):
        column_names = [str(c) for c in df.columns]
        if not column_names:
            st.markdown(
                '<div class="sidebar-empty">No columns found.</div>',
                unsafe_allow_html=True,
            )
        else:
            dynamic_height = max(96, min(172, 88 + len(column_names) * 9))
            chips_html = "".join(
                f'<div class="column-chip">{html.escape(col_name)}</div>'
                for col_name in column_names
            )
            st.markdown(
                (
                    f'<div class="columns-liquid-box" style="max-height:{dynamic_height}px;">'
                    f'<div class="columns-liquid-head"><span>Columns</span><span class="columns-liquid-count">{len(column_names)}</span></div>'
                    f'<div class="columns-chip-grid">{chips_html}</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    st.markdown('<div class="minor-title">Explore & Export</div>', unsafe_allow_html=True)
    lens_key = re.sub(r"[^A-Za-z0-9_]+", "_", source)[:50]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    ex1, ex2 = st.columns(2)
    with ex1:
        st.markdown("**Numerical**")
        if num_cols:
            ncol = st.selectbox("Numeric column", num_cols, key=f"lens_num_col_{lens_key}")
            fig_num = px.histogram(
                df,
                x=ncol,
                marginal="box",
                color_discrete_sequence=[ACCENT],
                title=f"Distribution — {ncol}",
            )
            fig_num.update_layout(height=320, margin=dict(t=40, l=20, r=20, b=20))
            st.plotly_chart(fig_num, use_container_width=True, key=f"lens_num_fig_{lens_key}")
        else:
            st.info("No numeric columns found.")
    with ex2:
        st.markdown("**Categorical**")
        if cat_cols:
            ccol = st.selectbox("Categorical column", cat_cols, key=f"lens_cat_col_{lens_key}")
            top_cat = df[ccol].astype(str).value_counts().head(15).reset_index()
            top_cat.columns = ["value", "count"]
            fig_cat = px.bar(
                top_cat,
                x="count",
                y="value",
                orientation="h",
                color="count",
                color_continuous_scale="Oranges",
                title=f"Top values — {ccol}",
            )
            fig_cat.update_layout(height=320, margin=dict(t=40, l=20, r=20, b=20))
            st.plotly_chart(fig_cat, use_container_width=True, key=f"lens_cat_fig_{lens_key}")
        else:
            st.info("No categorical columns found.")

    with st.expander("Full column statistics", expanded=False):
        stats_df = dp.build_column_statistics(df)
        st.dataframe(stats_df, use_container_width=True, height=420)

    st.markdown("**Potentially sensitive columns**")
    pii_df = dp.detect_pii_columns(df)
    pii_cols = pii_df.loc[pii_df["pii_detected"] == True, "column"].astype(str).tolist()
    if not pii_cols:
        st.caption(
            "No columns matched address, name, phone, PAN, Aadhaar, or identifier checks "
            "(column names and sample values)."
        )
    else:
        st.dataframe(pd.DataFrame({"Column": pii_cols}), use_container_width=True, hide_index=True)

    st.markdown("**Preview**")
    npeek = st.slider("Rows (head / tail)", 5, 80, 15, key=f"lens_npeek_{lens_key}")
    p1, p2 = st.columns(2)
    with p1:
        st.caption("Head")
        st.dataframe(df.head(npeek), use_container_width=True, height=230)
    with p2:
        st.caption("Tail")
        st.dataframe(df.tail(min(npeek, len(df))), use_container_width=True, height=230)

    export_base = re.sub(r"[^A-Za-z0-9._-]+", "_", source)[:36]
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Download Lens data (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{export_base}_lens_export.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"lens_csv_{lens_key}",
        )
    with d2:
        st.download_button(
            "Download Lens insights (Excel)",
            data=dp.export_dataframe_to_excel_bytes(df),
            file_name=f"{export_base}_lens_insights.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"lens_xlsx_{lens_key}",
        )
    with d3:
        if st.button("Use as Architect seed", use_container_width=True, key=f"lens_use_seed_{lens_key}"):
            st.session_state.architect_seed_source = source
            st.success("Selected source set for Architect.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_architect_tab() -> None:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Architect</div>', unsafe_allow_html=True)
    st.caption("Mask sensitive fields, choose model strategy, and generate validated synthetic output.")

    options = _advanced_dataset_options()
    seed_only = {k: v for k, v in options.items() if k not in ("Architect output", "AI Astra output")}
    if not seed_only:
        st.info("Upload/join a dataset first to run Architect.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    seed_label = st.selectbox("Seed source", list(seed_only.keys()), key="architect_seed_source")
    seed_df = seed_only[seed_label]

    pii_df = dp.detect_pii_columns(seed_df)
    default_pii = pii_df.loc[pii_df["pii_detected"] == True, "column"].astype(str).tolist()
    st.markdown('<div class="minor-title">Privacy & Masking</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    selected_pii = st.multiselect(
        "Columns to mask/drop",
        options=list(seed_df.columns),
        default=default_pii,
        key="architect_pii_cols",
    )
    scrub_mode = p1.selectbox("PII mode", ["mask", "drop"], key="architect_scrub_mode")
    k_anon = int(p2.number_input("k-anonymity", min_value=1, max_value=50, value=3, step=1, key="architect_k"))
    noise_level = float(p3.slider("Numeric noise", 0.0, 0.5, 0.02, 0.01, key="architect_noise"))
    private_df = se.apply_privacy_transform(seed_df, tuple(selected_pii), scrub_mode, k_anon, noise_level)

    st.markdown('<div class="minor-title">The Foundry</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    model_choice = g1.selectbox(
        "Method",
        ["CTGAN (GAN)", "TVAE", "Diffusion-style bootstrap", "AI Astra"],
        key="architect_model",
    )
    target_rows = int(
        g2.number_input(
            "Target rows",
            min_value=max(10, len(private_df)),
            max_value=1_000_000,
            value=min(max(10, len(private_df) * 5), 100_000),
            step=100,
            key="architect_rows",
        )
    )
    seed_rand = int(g3.number_input("Random seed", min_value=1, max_value=999_999, value=42, step=1, key="architect_seed"))
    np.random.seed(seed_rand)

    user_instruction = ""
    if model_choice == "AI Astra":
        user_instruction = st.text_area(
            "Custom generation instruction (optional)",
            value="",
            height=85,
            key="architect_instruction",
        )

    if st.button("Run Architect", type="primary", use_container_width=True, key="architect_run"):
        try:
            if model_choice == "CTGAN (GAN)":
                out = se.generate_sdv_synthetic(private_df, target_rows, "CTGAN") if SDV_AVAILABLE else se.generate_bootstrap_synthetic(private_df, target_rows)
            elif model_choice == "TVAE":
                out = se.generate_sdv_synthetic(private_df, target_rows, "TVAE") if SDV_AVAILABLE else se.generate_bootstrap_synthetic(private_df, target_rows)
            elif model_choice == "AI Astra":
                out = generate_with_azure_llm(private_df, target_rows, custom_instruction=user_instruction)
            else:
                out = se.generate_bootstrap_synthetic(private_df, target_rows)

            out = se.align_synthetic_to_seed_distribution(private_df, out)
            st.session_state.architect_generated_df = out
            st.session_state.architect_seed_label = seed_label
            st.success(f"Architect generated {len(out):,} rows.")
        except Exception as exc:
            st.error(f"Architect generation failed: {exc}")

    out = st.session_state.get("architect_generated_df")
    if out is not None and not out.empty and st.session_state.get("architect_seed_label") == seed_label:
        st.markdown('<div class="minor-title">Latest Generation</div>', unsafe_allow_html=True)
        js_df, utility_df = se.compute_fidelity_metrics(private_df, out)
        f1, f2 = st.columns(2)
        with f1:
            st.markdown("**JS Divergence (numeric)**")
            st.dataframe(js_df, use_container_width=True, height=210)
        with f2:
            st.markdown("**Utility checks**")
            st.dataframe(utility_df, use_container_width=True, height=210)
        st.dataframe(out.head(30), use_container_width=True, height=240)
        st.download_button(
            "Download Architect output (CSV)",
            out.to_csv(index=False).encode("utf-8"),
            "architect_synthetic.csv",
            "text/csv",
            use_container_width=True,
            key="architect_download_csv",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_critic_tab() -> None:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Critic</div>', unsafe_allow_html=True)
    st.caption("Upload an external synthetic CSV and score it against your selected seed dataset.")

    options = _advanced_dataset_options()
    seed_only = {k: v for k, v in options.items() if k not in ("Architect output", "AI Astra output")}
    if not seed_only:
        st.info("Upload/join a seed dataset first for Critic.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    seed_label = st.selectbox("Seed source for critic", list(seed_only.keys()), key="critic_seed_source")
    seed_df = seed_only[seed_label]
    up = st.file_uploader("Upload synthetic CSV for Critic", type=["csv"], key="critic_upload_csv")

    if st.button("Run Critic", type="primary", use_container_width=True, key="critic_run_btn_main"):
        if up is None:
            st.warning("Please upload a synthetic CSV first.")
        else:
            try:
                synth_df = pd.read_csv(io.BytesIO(up.getvalue()), low_memory=False)
                js_df, utility_df = se.compute_fidelity_metrics(seed_df, synth_df)
                guardrails_report = se.validate_synthetic_dataset(seed_df, synth_df, [])
                st.session_state.critic_synth_df = synth_df
                st.session_state.critic_js_df = js_df
                st.session_state.critic_utility_df = utility_df
                st.session_state["critic_guardrails_report"] = guardrails_report
            except Exception as exc:
                st.error(f"Critic failed: {exc}")

    synth_df = st.session_state.get("critic_synth_df")
    js_df = st.session_state.get("critic_js_df")
    utility_df = st.session_state.get("critic_utility_df")
    guardrails_report = st.session_state.get("critic_guardrails_report") or {}
    if synth_df is not None and js_df is not None and utility_df is not None:
        avg_js = float(pd.to_numeric(js_df.get("js_divergence", pd.Series(dtype=float)), errors="coerce").dropna().mean() or 0.0)
        corr_sim = utility_df.loc[utility_df["metric"] == "correlation_similarity", "value"]
        corr_val = float(corr_sim.iloc[0]) if not corr_sim.empty and pd.notna(corr_sim.iloc[0]) else np.nan
        if (avg_js <= 0.12) and (np.isnan(corr_val) or corr_val >= 0.70):
            st.success(f"Critic verdict: PASS | avg JS={avg_js:.4f}" + (f" | corr similarity={corr_val:.3f}" if not np.isnan(corr_val) else ""))
        else:
            st.warning(f"Critic verdict: REVIEW | avg JS={avg_js:.4f}" + (f" | corr similarity={corr_val:.3f}" if not np.isnan(corr_val) else ""))

        st.markdown("**Guardrails**")
        se.render_guardrails_table(guardrails_report)
        se.render_seed_synth_corr(seed_df, synth_df, "critic_corr")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Per-column divergence**")
            st.dataframe(js_df, use_container_width=True, height=220)
        with c2:
            st.markdown("**Utility metrics**")
            st.dataframe(utility_df, use_container_width=True, height=220)

        c1, c2 = st.columns((1.1, 0.9))
        with c1:
            st.markdown("**Schema (synthetic)**")
            schema_df = pd.DataFrame(
                {
                    "column": synth_df.columns,
                    "dtype": [str(t) for t in synth_df.dtypes],
                    "non_null_count": synth_df.notna().sum().values,
                    "null_count": synth_df.isna().sum().values,
                    "unique_count": synth_df.nunique(dropna=True).values,
                }
            )
            if len(synth_df) > 0:
                schema_df["missing_rate_%"] = ((schema_df["null_count"] / len(synth_df)) * 100).round(2)
            else:
                schema_df["missing_rate_%"] = 0.0
            st.dataframe(schema_df, use_container_width=True, height=320)
        with c2:
            dtype_counts = synth_df.dtypes.astype(str).value_counts()
            fig_dtype = px.pie(
                values=dtype_counts.values,
                names=dtype_counts.index,
                title="Types",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_dtype.update_layout(height=320, margin=dict(t=40, l=20, r=20, b=20))
            st.plotly_chart(fig_dtype, use_container_width=True, key="critic_dtype")

        st.markdown("**Column statistics (synthetic)**")
        stats_df = dp.build_column_statistics(synth_df)
        st.dataframe(stats_df, use_container_width=True, height=340)

        st.markdown("**Preview**")
        preview_n = int(
            st.number_input(
                "Rows to preview (critic upload)",
                min_value=1,
                max_value=max(1, len(synth_df)),
                value=min(25, len(synth_df)),
                step=1,
                key="critic_preview_rows",
            )
        )
        p1, p2 = st.columns(2)
        with p1:
            st.caption("Head")
            st.dataframe(synth_df.head(preview_n), use_container_width=True, height=230)
        with p2:
            st.caption("Tail")
            st.dataframe(synth_df.tail(min(preview_n, len(synth_df))), use_container_width=True, height=230)

        st.download_button(
            "Download Critic upload report (JSON)",
            data=json.dumps(
                {
                    "avg_js": avg_js,
                    "corr_similarity": None if np.isnan(corr_val) else corr_val,
                    "js": js_df.to_dict(orient="records"),
                    "utility": utility_df.to_dict(orient="records"),
                    "guardrails": guardrails_report,
                },
                indent=2,
                default=str,
            ).encode("utf-8"),
            file_name="critic_upload_report.json",
            mime="application/json",
            use_container_width=True,
            key="critic_json_dl",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_model_validation_sandbox_tab() -> None:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Validation Sandbox</div>', unsafe_allow_html=True)
    st.caption(
        "Baseline model only - train on real vs synthetic and compare on same real holdout "
        "(reference flow from Hackathon folder)."
    )

    options = _advanced_dataset_options()
    seed_only = {k: v for k, v in options.items() if k not in ("Architect output", "AI Astra output")}
    # Keep all left-pane uploaded datasets visible in synthetic selector as requested.
    synth_only = dict(options)
    if not seed_only or not synth_only:
        st.info("Need both a seed dataset and a dataset to evaluate in sandbox validation.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    seed_label = st.selectbox("Seed dataset", list(seed_only.keys()), key="mv_seed_src")
    synth_label = st.selectbox("Synthetic dataset", list(synth_only.keys()), key="mv_synth_src")
    real_df = seed_only[seed_label].copy()
    synth_df = synth_only[synth_label].copy()
    common_cols = [c for c in real_df.columns if c in synth_df.columns]
    if len(common_cols) < 2:
        st.warning("Need at least two overlapping columns between seed and synthetic datasets.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    real_df = real_df[common_cols]
    synth_df = synth_df[common_cols]

    if not REF_ML_EVAL_AVAILABLE:
        st.error(
            f"Reference ML sandbox helpers are unavailable. Ensure this folder exists: {REFERENCE_HACKATHON_DIR}"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    target_mode = st.radio(
        "Target column",
        ["Auto-detect", "Choose manually", "No target - skip supervised ML"],
        horizontal=True,
        key="mv_target_mode",
    )
    target = None
    if target_mode == "Auto-detect":
        target = ref_guess_target_column(real_df, set())
        if target:
            st.success(f"Detected candidate target: `{target}`")
        else:
            st.warning("No clear target column detected. Choose manually.")
    elif target_mode == "Choose manually":
        target = st.selectbox("Target", common_cols, key="mv_target_manual")
    else:
        st.info("Supervised ML skipped.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not target:
        st.markdown("</div>", unsafe_allow_html=True)
        return
    feature_cols = [c for c in common_cols if c != target]
    if not feature_cols:
        st.warning("Need at least one feature column besides target.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    y_ser = real_df[target]
    problem = ref_infer_problem_kind(y_ser)
    algo_opts = (
        ["hgb_classifier", "rf_classifier", "logreg"]
        if problem != "regression"
        else ["hgb_regressor", "rf_regressor", "ridge"]
    )
    default_algo = ref_default_algorithm(problem)
    algo_idx = algo_opts.index(default_algo) if default_algo in algo_opts else 0
    test_frac = st.slider("Holdout fraction (real data)", 0.1, 0.4, 0.25, 0.05, key="mv_test_frac")
    algo_choice = st.selectbox("Algorithm", algo_opts, index=algo_idx, key="mv_algo")

    if st.button("Run baseline validation", type="primary", use_container_width=True, key="mv_run"):
        try:
            from sklearn.model_selection import train_test_split
        except Exception:
            st.error(
                "scikit-learn is required for Model Validation Sandbox. "
                "Install it with: pip install scikit-learn"
            )
            return

        def _safe_stratify_target(y_series: pd.Series, is_regression_problem: bool) -> pd.Series | None:
            """Use stratify only when every class has at least two rows."""
            if is_regression_problem:
                return None
            counts = y_series.astype(str).value_counts(dropna=False)
            if counts.empty or int(counts.min()) < 2:
                return None
            return y_series

        X = real_df[feature_cols].copy()
        y = real_df[target]
        m = y.notna()
        X, y = X.loc[m], y.loc[m]
        strat = _safe_stratify_target(y, problem == "regression")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=test_frac,
            random_state=42,
            stratify=strat,
        )
        try:
            m_test, m_train, pipe_real, le = ref_train_predict_metrics_train_test(
                X_tr, y_tr, X_te, y_te, problem, algo_choice
            )
        except Exception as e:
            st.error(f"Real-data model failed: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if target not in synth_df.columns:
            st.error(f"Synthetic dataset does not contain target column `{target}`.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        X_syn = synth_df[feature_cols].copy()
        y_syn = synth_df[target]
        ms = y_syn.notna()
        X_syn, y_syn = X_syn.loc[ms], y_syn.loc[ms]
        if X_syn.empty:
            st.error("Synthetic dataset has no usable rows after dropping missing target values.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        strat_s = _safe_stratify_target(y_syn, problem == "regression")
        X_str, _x, y_str, _y = train_test_split(
            X_syn,
            y_syn,
            test_size=test_frac,
            random_state=42,
            stratify=strat_s,
        )
        if le is not None:
            allowed_classes = set(le.classes_)
            ok = y_str.astype(str).isin(allowed_classes)
            X_str, y_str = X_str.loc[ok], y_str.loc[ok]
            if X_str.empty:
                # Fallback: use all overlapping synthetic rows if split subset is empty.
                ok_full = y_syn.astype(str).isin(allowed_classes)
                X_str, y_str = X_syn.loc[ok_full], y_syn.loc[ok_full]
            if X_str.empty:
                st.error(
                    "Synthetic-to-real evaluation failed: no overlapping target classes between "
                    "selected synthetic and real seed datasets."
                )
                st.markdown("</div>", unsafe_allow_html=True)
                return
            if y_str.astype(str).nunique(dropna=True) < 2:
                st.error(
                    "Synthetic-to-real evaluation failed: synthetic training subset has only one "
                    "target class after alignment with real classes."
                )
                st.markdown("</div>", unsafe_allow_html=True)
                return
        if len(X_str) < 2:
            st.error("Synthetic-to-real evaluation failed: too few synthetic rows to train a model.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        try:
            m_syn, _pipe_syn = ref_train_on_synthetic_eval_on_real(
                X_str, y_str, X_te, y_te, problem, algo_choice, le
            )
        except Exception as e:
            st.error(f"Synthetic-to-real evaluation failed: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        keys = ("r2", "mae", "rmse") if problem == "regression" else ("roc_auc", "f1", "precision", "recall", "accuracy", "gini", "ks")
        rating = ref_ml_fidelity_rating_extended(
            m_train, m_test, m_syn, keys=tuple(k for k in keys if k in m_test and k in m_syn)
        )
        if problem != "regression" and le is not None:
            y_perm = pd.Series(le.transform(y_tr.astype(str)), index=X_tr.index)
        else:
            y_perm = pd.to_numeric(y_tr, errors="coerce").fillna(0)
        imp = ref_permutation_importance_df(pipe_real, X_tr, y_perm)
        st.session_state.mv_bundle = {
            "seed_label": seed_label,
            "synth_label": synth_label,
            "target": target,
            "problem": problem,
            "real_train_metrics": m_train,
            "real_test_metrics": m_test,
            "syn_on_real_test_metrics": m_syn,
            "rating": rating,
            "importance": imp,
        }
        st.rerun()

    b = st.session_state.get("mv_bundle")
    if b and b.get("seed_label") == seed_label and b.get("synth_label") == synth_label:
        st.divider()
        st.subheader("Results")
        rating = b.get("rating") or {}
        score = float(rating.get("rating_out_of_5") or 0)
        st.metric("Final alignment score (0-5)", f"{score:.2f}")
        m_tr = b.get("real_train_metrics") or {}
        m_te = b.get("real_test_metrics") or {}
        m_sy = b.get("syn_on_real_test_metrics") or {}
        problem = str(b.get("problem") or "binary")

        if problem == "regression":
            rows = [
                {"Metric": "R2", "Train": m_tr.get("r2"), "Test": m_te.get("r2"), "Synthetic": m_sy.get("r2")},
                {"Metric": "MAE", "Train": m_tr.get("mae"), "Test": m_te.get("mae"), "Synthetic": m_sy.get("mae")},
                {"Metric": "RMSE", "Train": m_tr.get("rmse"), "Test": m_te.get("rmse"), "Synthetic": m_sy.get("rmse")},
            ]
        else:
            rows = [
                {"Metric": "KS", "Train": m_tr.get("ks"), "Test": m_te.get("ks"), "Synthetic": m_sy.get("ks")},
                {"Metric": "Gini", "Train": m_tr.get("gini"), "Test": m_te.get("gini"), "Synthetic": m_sy.get("gini")},
                {"Metric": "ROC AUC", "Train": m_tr.get("roc_auc"), "Test": m_te.get("roc_auc"), "Synthetic": m_sy.get("roc_auc")},
                {"Metric": "Precision", "Train": m_tr.get("precision"), "Test": m_te.get("precision"), "Synthetic": m_sy.get("precision")},
                {"Metric": "Recall", "Train": m_tr.get("recall"), "Test": m_te.get("recall"), "Synthetic": m_sy.get("recall")},
                {"Metric": "Accuracy", "Train": m_tr.get("accuracy"), "Test": m_te.get("accuracy"), "Synthetic": m_sy.get("accuracy")},
                {"Metric": "F1", "Train": m_tr.get("f1"), "Test": m_te.get("f1"), "Synthetic": m_sy.get("f1")},
            ]
        score_df = pd.DataFrame(rows)
        st.dataframe(score_df, use_container_width=True, hide_index=True, height=260)
        fig = px.bar(
            score_df.melt(id_vars="Metric", value_vars=["Train", "Test", "Synthetic"], var_name="Split", value_name="Value"),
            x="Metric",
            y="Value",
            color="Split",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Metrics: real train vs real test vs synthetic-trained on real holdout",
        )
        fig.update_layout(height=400, margin=dict(t=40, l=20, r=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        for line in rating.get("details") or []:
            st.caption(line)
        imp = b.get("importance")
        if isinstance(imp, pd.DataFrame) and not imp.empty:
            with st.expander("Permutation importance (real train)", expanded=False):
                st.dataframe(imp, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_architect_unified_tab() -> None:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Architect</div>', unsafe_allow_html=True)
    gateway_mode = st.session_state.get("gateway_input_mode", "Both")
    has_gateway_seed = bool(st.session_state.get("data_registry"))

    if gateway_mode == "Seed Data":
        mode = "Seed Data"
    elif gateway_mode == "Natural Language":
        mode = "Natural Language"
    else:
        if (not has_gateway_seed) and st.session_state.get("architect_input_mode") == "Seed Data":
            st.session_state.architect_input_mode = "Natural Language"
        mode = st.radio(
            "Type of input",
            ["Seed Data", "Natural Language"],
            horizontal=True,
            key="architect_input_mode",
        )

    st.caption(f"Gateway mode: **{gateway_mode}**")
    st.markdown("</div>", unsafe_allow_html=True)
    if mode == "Seed Data":
        if not has_gateway_seed and (st.session_state.get("joined_df") is None or st.session_state.get("joined_df").empty):
            st.info("Gateway is set to Seed Data, but no seed dataset is uploaded yet.")
            return
        render_synthetic_generator_tab()
    else:
        render_data_bot_tab()


def render_artifact_panel() -> None:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div>
            <div class="arch-main-title">Artifact</div>
            <div class="arch-subtitle">Output</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    final_df: pd.DataFrame | None = None
    final_label = ""
    if st.session_state.get("synthetic_df") is not None and not st.session_state["synthetic_df"].empty:
        final_df = st.session_state["synthetic_df"]
        final_label = "Architect seed-based output"
    elif st.session_state.get("bot_generated_df") is not None and not st.session_state["bot_generated_df"].empty:
        final_df = st.session_state["bot_generated_df"]
        final_label = "Architect natural-language output"
    elif st.session_state.get("architect_generated_df") is not None and not st.session_state["architect_generated_df"].empty:
        final_df = st.session_state["architect_generated_df"]
        final_label = "Architect legacy output"

    if final_df is not None:
        st.caption(f"Final dataset source: **{final_label}**")
        st.download_button(
            "Export final dataset (CSV)",
            data=final_df.to_csv(index=False).encode("utf-8"),
            file_name="artifact_final_dataset.csv",
            mime="text/csv",
            use_container_width=True,
            key="artifact_final_csv",
        )
    else:
        st.caption("Generate a dataset in Architect to enable final export.")

    critic_pack = {
        "guardrails": st.session_state.get("critic_guardrails_report"),
        "js_metrics": (
            st.session_state["critic_js_df"].to_dict(orient="records")
            if isinstance(st.session_state.get("critic_js_df"), pd.DataFrame)
            else None
        ),
        "utility_metrics": (
            st.session_state["critic_utility_df"].to_dict(orient="records")
            if isinstance(st.session_state.get("critic_utility_df"), pd.DataFrame)
            else None
        ),
    }
    mv_pack = st.session_state.get("mv_bundle")
    final_report = {
        "artifact": "Synthetic Data Foundry",
        "final_dataset_source": final_label or None,
        "critic_report": critic_pack,
        "validation_sandbox_report": mv_pack,
    }
    st.download_button(
        "Export final report (JSON)",
        data=json.dumps(final_report, indent=2, default=str).encode("utf-8"),
        file_name="artifact_final_report.json",
        mime="application/json",
        use_container_width=True,
        key="artifact_final_report",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# App entry
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Synthetic Data Foundry",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    init_state()
    if st.session_state.get("_snapshot_restored"):
        st.caption("Session recovered after idle (within last 15 minutes).")
        st.session_state["_snapshot_restored"] = False
    render_sidebar()

    ok_nb, nb_msg = save_notebook_to_path(NOTEBOOK_OUTPUT_PATH)

    st.markdown(
        """
        <div style="padding: 0.25rem 0 0.9rem 0;">
            <div class="arch-main-title" style="font-size: 1.42rem;">
                Synthetic Data Foundry
            </div>
            <div class="arch-subtitle">Core Engine</div>
            <div style="margin-top: 0.25rem; color: #6b7280; font-size: 0.94rem;">
                End-to-end: profile your seed, generate synthetic data with the Architect, auto-validate with the critic, and export.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="arch-strip">
            <div class="arch-node">
                <div class="arch-node-title">Gateway</div>
                <div class="arch-node-sub">Input</div>
            </div>
            <div class="arch-node">
                <div class="arch-node-title">Synthetic Data Foundry</div>
                <div class="arch-node-sub">Core Engine</div>
            </div>
            <div class="arch-node">
                <div class="arch-node-title">Artifact</div>
                <div class="arch-node-sub">Output</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_tabs = [
        "Lens",
        "Architect",
        "Critic",
        "Model Validation Sandbox",
    ]
    if st.session_state.get("jump_to_lens"):
        tab_labels = [
            "Lens",
            "Architect",
            "Critic",
            "Model Validation Sandbox",
        ]
    elif st.session_state.get("jump_to_architect"):
        tab_labels = [
            "Architect",
            "Lens",
            "Critic",
            "Model Validation Sandbox",
        ]
    else:
        tab_labels = default_tabs

    tab_renderer = {
        "Lens": render_lens_tab,
        "Architect": render_architect_unified_tab,
        "Critic": render_critic_tab,
        "Model Validation Sandbox": render_model_validation_sandbox_tab,
    }
    core_col, artifact_col = st.columns([3.2, 1.2], gap="large")
    with core_col:
        tabs = st.tabs(tab_labels)
        for tab_obj, label in zip(tabs, tab_labels):
            with tab_obj:
                tab_renderer[label]()
    with artifact_col:
        render_artifact_panel()
    if st.session_state.get("jump_to_lens"):
        st.session_state.jump_to_lens = False
    if st.session_state.get("jump_to_architect"):
        st.session_state.jump_to_architect = False

    save_session_snapshot()


if __name__ == "__main__":
    main()
