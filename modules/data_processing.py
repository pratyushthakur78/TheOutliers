from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd
import re
import streamlit as st

try:
    from presidio_analyzer import AnalyzerEngine

    _analyzer = AnalyzerEngine()
except Exception:
    _analyzer = None


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


_PII_PRESIDIO_TYPES = frozenset(
    {
        "PERSON",
        "PHONE_NUMBER",
        "LOCATION",
        "EMAIL_ADDRESS",
        "IN_PAN",
        "IN_AADHAAR",
    }
)
_PII_MIN_SCORE = 0.35
_RE_PAN_VALUE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_RE_AADHAAR_GROUPS = re.compile(r"\b(\d{4})\s+(\d{4})\s+(\d{4})\b")
_RE_PHONE_IN_VALUE = re.compile(r"\b(?:\+?91[\s-]?)?[6-9]\d{9}\b")
_RE_LIKELY_PERSON_NAME = re.compile(r"\b[A-Za-z]{2,}(?:\s+[A-Za-z]{2,})+\b")
_KIND_PRIORITY = {"name": 0, "value": 1, "sample": 2}
_RE_PII_SIGNAL = re.compile(
    r"^(PAN|Aadhaar|Name|Address|Phone|Email|Identifier)\s+\((name|value|sample)\)$",
    re.I,
)


def _is_identifier_column_name(col: str) -> bool:
    t = str(col).lower().strip()
    return t in ("id", "uuid", "guid") or t.endswith("_id") or t.endswith("_key") or "customer_id" in t


def _phone_related_column_name(col: str) -> bool:
    t = str(col).lower()
    return any(x in t for x in ("phone", "mobile", "tel", "cell", "contact", "whatsapp"))


def _name_semantic_column(col: str) -> bool:
    t = str(col).lower()
    tok = {x for x in re.split(r"[^a-z0-9]+", t) if x}
    if any(
        x in tok
        for x in ("firstname", "lastname", "fullname", "customername", "fname", "lname", "employername", "mothername", "fathername")
    ):
        return True
    if ("employer" in tok and "name" in tok) or ("employee" in tok and "name" in tok):
        return True
    if t.endswith("_name"):
        stem = t[:-5].strip("_")
        if stem in ("file", "column", "field", "variable", "table", "schema", "class", "index", "key", "object", "record", "user"):
            return False
        return True
    if _phone_related_column_name(col):
        return False
    return False


def _address_tenure_or_count_in_name(col: str) -> bool:
    t = str(col).lower()
    if not re.search(r"address|addr|city|street|location", t):
        return False
    return bool(
        re.search(
            r"(months?|years?|weeks?|days?|since|tenure|duration|length|time|stay|how_?long|num|n_|count|in_current|at_address)",
            t,
        )
    )


def _postal_or_pin_column_name(col: str) -> bool:
    t = str(col).lower()
    return any(x in t for x in ("pin", "postal", "zip", "zipcode", "pincode"))


def _address_semantic_column(col: str) -> bool:
    if _address_tenure_or_count_in_name(col):
        return False
    t = str(col).lower()
    tok = {x for x in re.split(r"[^a-z0-9]+", t) if x}
    if any(x in tok for x in ("address", "addr", "street", "city", "state", "zip", "postal", "pincode", "location")):
        return True
    if _postal_or_pin_column_name(col):
        return True
    return False


def _is_numeric_like_column(df: pd.DataFrame, col: str) -> bool:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return True
    s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    t = pd.to_numeric(s2, errors="coerce")
    return bool(len(s)) and t.notna().sum() / max(len(s), 1) > 0.85


def _sample_looks_like_id_codes(df: pd.DataFrame, col: str) -> bool:
    s = df[col].head(40).dropna().astype(str).str.strip()
    if s.empty:
        return False
    mostly_digits = s.str.match(r"^[\d\s\-\.]+$", na=False).mean() > 0.7
    short_tokens = (s.str.len() <= 24).mean() > 0.85
    return mostly_digits or (short_tokens and _is_identifier_column_name(str(col)))


def _values_suggest_free_text_address(blob: str) -> bool:
    if len(blob.strip()) < 8:
        return False
    if re.search(r"\d{1,6}\s+[A-Za-z]{3,}", blob):
        return True
    if re.search(r"(street|st\.|road|avenue|lane|nagar|district|pin\s*[-:]?\s*\d)", blob, re.I):
        return True
    return False


def _column_pii_hint(df: pd.DataFrame, col: str) -> list[str]:
    t = str(col).lower().strip()
    tokens = [x for x in re.split(r"[^a-z0-9]+", t) if x]
    tok = set(tokens)
    hints: list[str] = []
    if "pan" in tok or re.search(r"(^|_)pan($|_)", t):
        hints.append("PAN (name)")
    if any(x in tok for x in ("aadhaar", "aadhar", "uidai")) or "aadhaar" in t or "aadhar" in t:
        hints.append("Aadhaar (name)")
    if any(x in tok for x in ("phone", "mobile", "cell", "tel")):
        hints.append("Phone (name)")
    if "email" in t or "e_mail" in t or "e-mail" in t:
        hints.append("Email (name)")
    addr_kw = any(x in tok for x in ("address", "addr", "street", "city", "zip", "postal", "state", "pincode", "location"))
    if addr_kw:
        if _address_tenure_or_count_in_name(col):
            pass
        elif _postal_or_pin_column_name(col):
            hints.append("Address (name)")
        elif tok.intersection({"city", "state"}):
            hints.append("Address (name)")
        elif _is_numeric_like_column(df, col):
            blob_pre = " ".join(df[col].head(25).astype(str).str.strip().dropna().tolist())
            if _values_suggest_free_text_address(blob_pre):
                hints.append("Address (name)")
        else:
            blob_pre = " ".join(df[col].head(25).astype(str).str.strip().dropna().tolist())
            if _values_suggest_free_text_address(blob_pre) or len(blob_pre) > 12:
                hints.append("Address (name)")
    if any(x in tok for x in ("name", "firstname", "lastname", "fname", "lname", "fullname", "customername")):
        hints.append("Name (name)")
    if _is_identifier_column_name(t):
        hints.append("Identifier (name)")
    return hints


def _aadhaar_pattern_in_text(sample_text: str) -> bool:
    for m in _RE_AADHAAR_GROUPS.finditer(sample_text):
        a, b, c = m.group(1), m.group(2), m.group(3)
        if a == b == c:
            continue
        return True
    return bool(re.search(r"\b\d{12}\b", sample_text))


def _value_pattern_hints(sample_text: str) -> list[str]:
    h: list[str] = []
    if _RE_PAN_VALUE.search(sample_text):
        h.append("PAN (value)")
    if _aadhaar_pattern_in_text(sample_text):
        h.append("Aadhaar (value)")
    if _RE_PHONE_IN_VALUE.search(sample_text):
        h.append("Phone (value)")
    return h


def _presidio_pii_hints(df: pd.DataFrame, col: str) -> list[str]:
    if _analyzer is None:
        return []
    col_s = str(col)
    sample = df[col].head(25).astype(str).str.strip()
    blob = " ".join(sample.dropna().tolist())
    if not blob.strip():
        return []
    numeric_like = _is_numeric_like_column(df, col)
    id_name = _is_identifier_column_name(col_s)
    phone_name_ok = _phone_related_column_name(col_s)
    name_col = _name_semantic_column(col_s)
    addr_col = _address_semantic_column(col_s)
    try:
        seen: dict[str, None] = {}
        for r in _analyzer.analyze(text=blob[:50000], language="en"):
            et = getattr(r, "entity_type", None)
            sc = float(getattr(r, "score", 0) or 0)
            if et not in _PII_PRESIDIO_TYPES or sc < _PII_MIN_SCORE:
                continue
            if et == "PHONE_NUMBER" and numeric_like and not phone_name_ok:
                continue
            if et == "LOCATION" and name_col:
                continue
            if et == "PERSON" and addr_col:
                continue
            if et == "PHONE_NUMBER" and (name_col or addr_col) and not phone_name_ok:
                continue
            if et == "PERSON" and id_name and not _RE_LIKELY_PERSON_NAME.search(blob):
                continue
            if et == "PERSON" and id_name and _sample_looks_like_id_codes(df, col):
                continue
            if et == "PERSON" and numeric_like and not _RE_LIKELY_PERSON_NAME.search(blob):
                continue
            if et in ("LOCATION", "EMAIL_ADDRESS", "IN_PAN", "IN_AADHAAR") and numeric_like:
                continue
            label = {
                "PERSON": "Name (sample)",
                "PHONE_NUMBER": "Phone (sample)",
                "LOCATION": "Address (sample)",
                "EMAIL_ADDRESS": "Email (sample)",
                "IN_PAN": "PAN (sample)",
                "IN_AADHAAR": "Aadhaar (sample)",
            }.get(et, et)
            seen.setdefault(str(label), None)
        return list(seen.keys())
    except Exception:
        return []


def _dedupe_pii_signals(parts: list[str]) -> list[str]:
    best: dict[str, tuple[int, str]] = {}
    for p in parts:
        m = _RE_PII_SIGNAL.match(p.strip())
        if not m:
            continue
        entity = m.group(1).lower()
        kind = m.group(2).lower()
        pr = _KIND_PRIORITY.get(kind, 9)
        if entity not in best or pr < best[entity][0]:
            best[entity] = (pr, p)
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        m = _RE_PII_SIGNAL.match(p.strip())
        if not m:
            continue
        entity = m.group(1).lower()
        win = best[entity][1]
        if p != win or entity in seen:
            continue
        out.append(win)
        seen.add(entity)
    return out


def detect_pii_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in df.columns:
        col_s = str(c)
        sample = df[c].head(25).astype(str).str.strip()
        blob = " ".join(sample.dropna().tolist())
        parts: list[str] = []
        parts.extend(_column_pii_hint(df, c))
        parts.extend(_value_pattern_hints(blob))
        parts.extend(_presidio_pii_hints(df, c))
        parts = _dedupe_pii_signals(parts)
        rows.append(
            {
                "column": c,
                "dtype": str(df[c].dtype),
                "pii_detected": bool(parts),
                "reason": " · ".join(parts),
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

