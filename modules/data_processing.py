from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd
import re
import streamlit as st


@st.cache_data(show_spinner=False)
def parse_uploaded_file(file_name: str, file_bytes: bytes) -> dict[str, pd.DataFrame]:
    lower = file_name.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
        return {file_name: df}
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        out = {}
        for sheet in xls.sheet_names:
            out[f"{file_name}::{sheet}"] = pd.read_excel(xls, sheet_name=sheet)
        return out
    if lower.endswith(".json"):
        data = json.loads(file_bytes.decode("utf-8-sig"))
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            df = pd.json_normalize([data])
        else:
            raise ValueError("Unsupported JSON structure.")
        return {file_name: df}
    raise ValueError("Unsupported file format. Use CSV, XLSX, XLS, or JSON.")


def build_column_statistics(df: pd.DataFrame) -> pd.DataFrame:
    percentile_points = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
    percentile_cols = ["5%", "10%", "20%", "30%", "50%", "70%", "80%", "90%", "95%"]
    rows: list[dict[str, Any]] = []

    for c in df.columns:
        s = df[c]
        is_numeric = pd.api.types.is_numeric_dtype(s)
        is_datetime = pd.api.types.is_datetime64_any_dtype(s)
        non_missing_count = int(s.notna().sum())

        row: dict[str, Any] = {
            "column_name": c,
            "dtype": str(s.dtype),
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

    return pd.DataFrame(rows)


def detect_pii_columns(df: pd.DataFrame) -> pd.DataFrame:
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


@st.cache_data(show_spinner=False)
def export_dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="data", index=False)
        stats_df = build_column_statistics(df)
        stats_df.to_excel(writer, sheet_name="column_statistics", index=False)
    output.seek(0)
    return output.getvalue()


def parse_seed_upload(uploaded: Any) -> pd.DataFrame:
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

