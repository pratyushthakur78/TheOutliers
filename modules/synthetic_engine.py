from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    SDV_AVAILABLE = True
except Exception:
    SDV_AVAILABLE = False


def is_binary_indicator(series: pd.Series) -> bool:
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
    out = df.copy()
    for col in pii_columns:
        if col not in out.columns:
            continue
        if scrub_mode == "mask":
            out[col] = out[col].astype(str).where(out[col].isna(), "REDACTED")
        elif scrub_mode == "drop":
            out = out.drop(columns=[col])

    cat_cols = [c for c in out.columns if not pd.api.types.is_numeric_dtype(out[c])]
    if k_anonymity > 1:
        for c in cat_cols:
            vc = out[c].astype(str).value_counts(dropna=False)
            rare = vc[vc < k_anonymity].index
            out[c] = out[c].astype(str).where(~out[c].astype(str).isin(rare), "__OTHER__")

    if noise_level > 0:
        num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        for c in num_cols:
            if is_binary_indicator(out[c]):
                continue
            std = float(pd.to_numeric(out[c], errors="coerce").std() or 0.0)
            if std <= 0:
                continue
            out[c] = pd.to_numeric(out[c], errors="coerce") + np.random.normal(0, std * noise_level, size=len(out))
    return out


@st.cache_data(show_spinner=False)
def generate_bootstrap_synthetic(seed_df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
    if seed_df.empty:
        return seed_df.copy()
    out = seed_df.sample(n=target_rows, replace=True, random_state=42).reset_index(drop=True)
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for c in num_cols:
        if is_binary_indicator(seed_df[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").round().clip(0, 1)
            continue
        std = float(pd.to_numeric(out[c], errors="coerce").std() or 0.0)
        if std > 0:
            out[c] = pd.to_numeric(out[c], errors="coerce") + np.random.normal(0, std * 0.02, size=len(out))
    return out


def generate_sdv_synthetic(seed_df: pd.DataFrame, target_rows: int, model_type: str) -> pd.DataFrame:
    if not SDV_AVAILABLE:
        raise RuntimeError("SDV not installed. Use fallback model or add sdv to requirements.")
    if seed_df.empty:
        return seed_df.copy()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(seed_df)
    synth = CTGANSynthesizer(metadata) if model_type == "CTGAN" else TVAESynthesizer(metadata)
    synth.fit(seed_df)
    return synth.sample(num_rows=target_rows)


def _js_divergence_from_hist(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    low = min(np.nanmin(a), np.nanmin(b))
    high = max(np.nanmax(a), np.nanmax(b))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return np.nan
    p_hist, edges = np.histogram(a, bins=bins, range=(low, high), density=True)
    q_hist, _ = np.histogram(b, bins=edges, density=True)
    p = (p_hist.astype(float) + 1e-12)
    q = (q_hist.astype(float) + 1e-12)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


@st.cache_data(show_spinner=False)
def compute_fidelity_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            corr_similarity = float(1.0 - np.nanmean((rc.loc[common, common] - sc.loc[common, common]).abs().values))

    utility_df = pd.DataFrame(
        [
            {"metric": "correlation_similarity", "value": corr_similarity},
            {"metric": "avg_null_rate_diff", "value": float((real_df.isna().mean() - synth_df.isna().mean()).abs().mean())},
            {"metric": "unique_ratio_real", "value": float((real_df.nunique(dropna=True) / max(len(real_df), 1)).mean())},
            {"metric": "unique_ratio_synth", "value": float((synth_df.nunique(dropna=True) / max(len(synth_df), 1)).mean())},
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


def validate_synthetic_dataset(seed: pd.DataFrame, synth: pd.DataFrame, pk_cols: list[str]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    common = [c for c in seed.columns if c in synth.columns]
    primary = pk_cols[0] if pk_cols else None
    if primary and primary in synth.columns:
        dup = int(synth[primary].duplicated().sum())
        checks.append({"id": "pk_unique", "name": "Primary key uniqueness", "pass": dup == 0, "detail": f"Duplicate key rows: {dup:,}"})

    dtype_issues = [f"{c}: {seed[c].dtype} vs {synth[c].dtype}" for c in common if not _dtype_family_compatible(seed[c], synth[c])]
    checks.append(
        {
            "id": "dtype_match",
            "name": "Datatype consistency",
            "pass": len(dtype_issues) == 0,
            "detail": "; ".join(dtype_issues[:10]) if dtype_issues else f"{len(common)} overlapping column(s) keep compatible dtype families.",
        }
    )
    definitive = [c for c in checks if c.get("pass") is False]
    return {"checks": checks, "overall_pass": len(definitive) == 0, "summary": f"{len(definitive)} failed" if definitive else "All automated checks passed"}


def render_guardrails_table(report: dict[str, Any]) -> None:
    rows = report.get("checks") or []
    if not rows:
        st.caption("No Guardrails checks available.")
        return
    tbl = pd.DataFrame([{"Check": r.get("name", ""), "Result": "Pass" if r.get("pass") is True else ("Fail" if r.get("pass") is False else "N/A"), "Detail": (r.get("detail") or "")[:900]} for r in rows])
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
        st.plotly_chart(fig_seed, use_container_width=True, key=f"{key_prefix}_seed")
    with c2:
        fig_syn = px.imshow(cr_syn, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Synthetic correlations")
        st.plotly_chart(fig_syn, use_container_width=True, key=f"{key_prefix}_synth")


@st.cache_data(show_spinner=False)
def align_synthetic_to_seed_distribution(seed_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    out = synthetic_df.copy()
    if out.empty or seed_df.empty:
        return out
    rng = np.random.default_rng(42)
    common_cols = [c for c in out.columns if c in seed_df.columns]
    for col in common_cols:
        seed_s = seed_df[col]
        out_s = out[col]
        if pd.api.types.is_numeric_dtype(seed_s):
            if is_binary_indicator(seed_s):
                p_one = float(pd.to_numeric(seed_s, errors="coerce").fillna(0).clip(0, 1).mean())
                out[col] = pd.Series(rng.choice([0, 1], size=len(out), replace=True, p=[1 - p_one, p_one]), index=out.index)
                continue
            seed_num = pd.to_numeric(seed_s, errors="coerce").dropna()
            out_num = pd.to_numeric(out_s, errors="coerce")
            if seed_num.empty:
                continue
            valid_mask = out_num.notna()
            if valid_mask.any():
                ranks = out_num[valid_mask].rank(method="average", pct=True).clip(0.0, 1.0)
                out.loc[valid_mask, col] = np.quantile(seed_num.values, ranks.values)
        elif pd.api.types.is_datetime64_any_dtype(seed_s):
            seed_dt = pd.to_datetime(seed_s, errors="coerce").dropna()
            if not seed_dt.empty:
                out[col] = pd.to_datetime(rng.choice(seed_dt.values, size=len(out), replace=True), errors="coerce")
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
    out = df.copy()
    if out.empty:
        return out
    rng = np.random.default_rng(42)
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for c in num_cols:
        if is_binary_indicator(out[c]):
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
        outlier_mask = rng.random(len(skewed)) < (0.01 + 0.03 * skew_strength)
        if outlier_mask.any():
            skewed[outlier_mask] = skewed[outlier_mask] * (1.5 + (2.5 * skew_strength))
        s.loc[valid_mask] = skewed
        out[c] = s
    miss_rate = 0.005 + (0.02 * skew_strength)
    for c in out.columns:
        m = rng.random(len(out)) < miss_rate
        out.loc[m, c] = np.nan
    return out

