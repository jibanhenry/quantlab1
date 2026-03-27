# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _safe_log_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s > 0)
    return np.log1p(s)


def add_valuation_features(df: pd.DataFrame, cfg: Optional[Dict] = None) -> pd.DataFrame:
    """Create cross-sectional and rolling valuation features from pb/ps."""
    out = df.copy()
    sort_cols = [c for c in ["code", "date"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    rolling_window = int((cfg or {}).get("valuation", {}).get("rolling_window", 120))

    out["pb_log"] = _safe_log_series(out.get("pb_mrq"))
    out["ps_log"] = _safe_log_series(out.get("ps_ttm"))

    for raw_col, rank_col in (("pb_mrq", "pb_cs_rank"), ("ps_ttm", "ps_cs_rank")):
        if raw_col in out.columns:
            out[rank_col] = (
                out.groupby("date")[raw_col]
                .rank(method="average", pct=True, na_option="keep")
                .astype(float)
            )
        else:
            out[rank_col] = np.nan

    if "code" in out.columns:
        for raw_col, rank_col in (("pb_mrq", "pb_rolling_rank"), ("ps_ttm", "ps_rolling_rank")):
            if raw_col in out.columns:
                out[rank_col] = (
                    out.groupby("code")[raw_col]
                    .transform(lambda s: s.rolling(rolling_window, min_periods=20).rank(pct=True))
                    .astype(float)
                )
            else:
                out[rank_col] = np.nan
    else:
        out["pb_rolling_rank"] = np.nan
        out["ps_rolling_rank"] = np.nan

    pb_w = float((cfg or {}).get("valuation", {}).get("pb_weight", 0.5))
    ps_w = float((cfg or {}).get("valuation", {}).get("ps_weight", 0.5))
    denom = pb_w + ps_w
    if denom <= 0:
        pb_w = ps_w = 0.5
        denom = 1.0
    pb_w /= denom
    ps_w /= denom

    pb_component = 1.0 - out["pb_cs_rank"]
    ps_component = 1.0 - out["ps_cs_rank"]
    value_score = (pb_component * pb_w) + (ps_component * ps_w)
    out["value_score"] = value_score.where(value_score.notna(), np.nan)

    return out


def summarize_valuation_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ("pb_mrq", "ps_ttm"):
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        non_null = int(s.notna().sum())
        positive = int((s > 0).sum())
        rows.append(
            {
                "field": col,
                "rows": int(len(df)),
                "missing_rate": float(s.isna().mean()) if len(df) else np.nan,
                "non_positive_rate": float((s <= 0).mean()) if len(df) else np.nan,
                "positive_rate": float(positive / len(df)) if len(df) else np.nan,
                "p01": float(s.quantile(0.01)) if non_null else np.nan,
                "p50": float(s.quantile(0.5)) if non_null else np.nan,
                "p99": float(s.quantile(0.99)) if non_null else np.nan,
            }
        )
    return pd.DataFrame(rows)
