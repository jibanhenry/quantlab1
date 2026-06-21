# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

from .config import load_config, merge_config
from .factor_data import load_external_factor_data, merge_external_data
from .io_utils import load_market_csv_multi
from .market_state import build_index_state_from_panel
from .portfolio import _generate_windows, compute_indicator_panel


FACTOR_GROUPS: Dict[str, List[str]] = {
    "momentum": ["ret_20d", "ret_60d", "ret_120d", "ret_20_120", "breakout20_atr"],
    "reversal": ["rev_1d", "rev_5d", "pullback20"],
    "lowvol": ["low_vol_20d", "low_vol_60d", "low_atr", "low_downside_vol_20d", "low_gap", "mdd60"],
    "value": ["bp", "ep", "sp", "value_score"],
    "quality": ["roe", "gross_margin", "net_margin", "ocf_to_profit", "low_debt_to_assets", "price_quality"],
    "growth": ["revenue_yoy", "net_profit_yoy", "roe_delta"],
    "liquidity": ["amount_20d", "turnover_20d", "low_amihud_20d", "rvol20"],
}

COMPOSITE_WEIGHTS = {
    "momentum_score": 0.25,
    "quality_score_v2": 0.25,
    "value_score_v2": 0.20,
    "lowvol_score": 0.15,
    "liquidity_score_v2": 0.10,
    "reversal_score": 0.05,
}

STRATEGY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "quality_value_momentum_lowvol": COMPOSITE_WEIGHTS,
    "equal_multifactor": {
        "momentum_score": 1 / 6,
        "quality_score_v2": 1 / 6,
        "value_score_v2": 1 / 6,
        "lowvol_score": 1 / 6,
        "liquidity_score_v2": 1 / 6,
        "reversal_score": 1 / 6,
    },
    "pure_momentum": {"momentum_score": 1.0},
    "pure_lowvol": {"lowvol_score": 1.0},
    "pure_value": {"value_score_v2": 1.0},
    "legacy_multifactor": {"legacy_multifactor_score": 1.0},
}


@dataclass
class FactorPosition:
    code: str
    entry_signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_score: float
    target_weight: float
    units: float
    allocation: float
    days_held: int = 0
    peak_close: float = np.nan


def _safe_numeric(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _winsorize_by_date(s: pd.Series) -> pd.Series:
    def _clip(x: pd.Series) -> pd.Series:
        if x.notna().sum() < 10:
            return x
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        return x.clip(lo, hi)

    return s.groupby(level=0, group_keys=False).apply(_clip)


def _rank_pct_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    values = pd.to_numeric(df[col], errors="coerce")
    return values.groupby(df["date"]).rank(pct=True, method="average")


def _mean_existing(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    use = [c for c in cols if c in df.columns]
    if not use:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[use].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)


def _to_rank_score(panel: pd.DataFrame, raw_col: str, score_col: Optional[str] = None) -> pd.DataFrame:
    score_col = score_col or f"{raw_col}_rank"
    panel[score_col] = _rank_pct_by_date(panel, raw_col) * 100.0
    return panel


def build_classic_factor_panel(
    df_ind: pd.DataFrame,
    external: Optional[pd.DataFrame] = None,
    min_amount_20d: float = 20_000_000.0,
) -> pd.DataFrame:
    out = merge_external_data(df_ind, external).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["code"] = out["code"].astype(str).str.split(".").str[-1].str.zfill(6)
    out = out.sort_values(["code", "date"]).reset_index(drop=True)

    g = out.groupby("code", group_keys=False)
    close = _safe_numeric(out, "close")
    open_px = _safe_numeric(out, "open")
    amount = _safe_numeric(out, "amount")
    turnover = _safe_numeric(out, "turnover")
    pct_chg = _safe_numeric(out, "pct_chg") / 100.0
    ret1 = g["close"].pct_change(fill_method=None)

    out["ret_1d"] = ret1
    out["ret_5d"] = g["close"].pct_change(5, fill_method=None)
    out["ret_20d"] = g["close"].pct_change(20, fill_method=None)
    out["ret_60d"] = g["close"].pct_change(60, fill_method=None)
    out["ret_120d"] = g["close"].pct_change(120, fill_method=None)
    out["ret_20_120"] = out["ret_120d"] - out["ret_20d"]
    out["rev_1d"] = -out["ret_1d"]
    out["rev_5d"] = -out["ret_5d"]
    out["pullback20"] = -_safe_numeric(out, "dist_high20")

    out["vol_20d"] = ret1.groupby(out["code"]).rolling(20, min_periods=20).std().reset_index(level=0, drop=True)
    out["vol_60d"] = ret1.groupby(out["code"]).rolling(60, min_periods=40).std().reset_index(level=0, drop=True)
    down_ret = ret1.where(ret1 < 0.0, 0.0)
    out["downside_vol_20d"] = down_ret.groupby(out["code"]).rolling(20, min_periods=20).std().reset_index(level=0, drop=True)
    out["low_vol_20d"] = -out["vol_20d"]
    out["low_vol_60d"] = -out["vol_60d"]
    out["low_atr"] = -_safe_numeric(out, "atr_pct")
    out["low_downside_vol_20d"] = -out["downside_vol_20d"]
    out["low_gap"] = -_safe_numeric(out, "gap_atr14")

    pb = _safe_numeric(out, "pb_mrq").where(_safe_numeric(out, "pb_mrq") > 0)
    ps = _safe_numeric(out, "ps_ttm").where(_safe_numeric(out, "ps_ttm") > 0)
    pe = _safe_numeric(out, "pe_dynamic").where(_safe_numeric(out, "pe_dynamic") > 0)
    out["bp"] = 1.0 / pb
    out["sp"] = 1.0 / ps
    out["ep"] = 1.0 / pe

    for col in ["roe", "gross_margin", "net_margin", "ocf_to_profit", "debt_to_assets", "revenue_yoy", "net_profit_yoy"]:
        if col not in out.columns:
            out[col] = np.nan
    out["low_debt_to_assets"] = -_safe_numeric(out, "debt_to_assets")
    out["roe_delta"] = _safe_numeric(out, "roe").groupby(out["code"]).diff(252)
    out["price_quality"] = (
        _safe_numeric(out, "clv").fillna(0.0) * 0.40
        + _safe_numeric(out, "body_ratio").fillna(0.0) * 0.25
        + _safe_numeric(out, "rvol20").clip(0, 4).fillna(1.0) * 0.10
        + (-_safe_numeric(out, "gap_atr14").clip(0, 3).fillna(0.0)) * 0.15
        + (-_safe_numeric(out, "tr_pct").clip(0, 0.2).fillna(0.0)) * 0.10
    )

    out["amount_20d"] = amount.groupby(out["code"]).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    out["turnover_20d"] = turnover.groupby(out["code"]).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    out["amihud_20d"] = (ret1.abs() / amount.replace(0, np.nan)).groupby(out["code"]).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    out["low_amihud_20d"] = -out["amihud_20d"]

    out["list_age_days"] = g.cumcount() + 1
    out["is_limit_up_like"] = pct_chg >= 0.095
    out["is_limit_down_like"] = pct_chg <= -0.095
    out["tradable"] = (
        out["list_age_days"].ge(120)
        & out["amount_20d"].ge(float(min_amount_20d))
        & open_px.notna()
        & close.notna()
        & ~out["is_limit_up_like"].fillna(False)
        & ~out["is_limit_down_like"].fillna(False)
    )

    rank_cols = sorted({c for cols in FACTOR_GROUPS.values() for c in cols if c in out.columns})
    for col in rank_cols:
        _to_rank_score(out, col, f"{col}_score")

    out["momentum_score"] = _mean_existing(out, [f"{c}_score" for c in FACTOR_GROUPS["momentum"]])
    out["reversal_score"] = _mean_existing(out, [f"{c}_score" for c in FACTOR_GROUPS["reversal"]])
    out["lowvol_score"] = _mean_existing(out, [f"{c}_score" for c in FACTOR_GROUPS["lowvol"]])
    out["value_score_v2"] = _mean_existing(out, [f"{c}_score" for c in FACTOR_GROUPS["value"]])
    out["quality_score_v2"] = _mean_existing(out, [f"{c}_score" for c in FACTOR_GROUPS["quality"]])
    out["growth_score"] = _mean_existing(out, [f"{c}_score" for c in FACTOR_GROUPS["growth"]])
    out["liquidity_score_v2"] = _mean_existing(out, [f"{c}_score" for c in FACTOR_GROUPS["liquidity"]])

    if "multifactor_score" in out.columns:
        out["legacy_multifactor_score"] = out["multifactor_score"]
    else:
        out["legacy_multifactor_score"] = np.nan

    for name, weights in STRATEGY_WEIGHTS.items():
        weighted = pd.Series(0.0, index=out.index, dtype=float)
        active = pd.Series(0.0, index=out.index, dtype=float)
        for col, weight in weights.items():
            if col not in out.columns:
                continue
            vals = pd.to_numeric(out[col], errors="coerce")
            weighted += vals.fillna(0.0) * float(weight)
            active += vals.notna().astype(float) * float(weight)
        out[f"score_{name}"] = weighted / active.replace(0.0, np.nan)

    return out


def build_lowvol_factor_panel(
    df_ind: pd.DataFrame,
    min_amount_20d: float = 20_000_000.0,
) -> pd.DataFrame:
    out = df_ind.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["code"] = out["code"].astype(str).str.split(".").str[-1].str.zfill(6)
    out = out.sort_values(["code", "date"]).reset_index(drop=True)

    g = out.groupby("code", group_keys=False)
    close = _safe_numeric(out, "close")
    open_px = _safe_numeric(out, "open")
    amount = _safe_numeric(out, "amount")
    turnover = _safe_numeric(out, "turnover")
    pct_chg = _safe_numeric(out, "pct_chg") / 100.0
    ret1 = g["close"].pct_change(fill_method=None)
    down_ret = ret1.where(ret1 < 0.0, 0.0)

    out["vol_20d"] = ret1.groupby(out["code"]).rolling(20, min_periods=20).std().reset_index(level=0, drop=True)
    out["vol_60d"] = ret1.groupby(out["code"]).rolling(60, min_periods=40).std().reset_index(level=0, drop=True)
    out["downside_vol_20d"] = down_ret.groupby(out["code"]).rolling(20, min_periods=20).std().reset_index(level=0, drop=True)
    out["low_vol_20d"] = -out["vol_20d"]
    out["low_vol_60d"] = -out["vol_60d"]
    out["low_atr"] = -_safe_numeric(out, "atr_pct")
    out["low_downside_vol_20d"] = -out["downside_vol_20d"]
    out["low_gap"] = -_safe_numeric(out, "gap_atr14")
    out["amount_20d"] = amount.groupby(out["code"]).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    out["turnover_20d"] = turnover.groupby(out["code"]).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    out["list_age_days"] = g.cumcount() + 1
    out["is_limit_up_like"] = pct_chg >= 0.095
    out["is_limit_down_like"] = pct_chg <= -0.095
    out["tradable"] = (
        out["list_age_days"].ge(120)
        & out["amount_20d"].ge(float(min_amount_20d))
        & open_px.notna()
        & close.notna()
        & ~out["is_limit_up_like"].fillna(False)
        & ~out["is_limit_down_like"].fillna(False)
        & ~out["code"].astype(str).str.startswith("399")
    )

    lowvol_cols = ["low_vol_20d", "low_vol_60d", "low_atr", "low_downside_vol_20d", "low_gap", "mdd60"]
    for col in lowvol_cols:
        if col not in out.columns:
            out[col] = np.nan
        _to_rank_score(out, col, f"{col}_score")
    out["lowvol_score"] = _mean_existing(out, [f"{c}_score" for c in lowvol_cols])
    out["score_pure_lowvol"] = out["lowvol_score"]
    return out


def add_forward_returns(panel: pd.DataFrame, horizons: Iterable[int] = (1, 5, 10, 20)) -> pd.DataFrame:
    out = panel.sort_values(["code", "date"]).copy()
    g = out.groupby("code", group_keys=False)
    for h in horizons:
        out[f"fwd_ret_{h}d"] = g["close"].shift(-h) / out["close"] - 1.0
    return out


def factor_coverage_report(panel: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in factor_cols:
        vals = pd.to_numeric(panel.get(col), errors="coerce")
        rows.append(
            {
                "factor": col,
                "rows": int(len(panel)),
                "non_null": int(vals.notna().sum()),
                "coverage": float(vals.notna().mean()) if len(panel) else np.nan,
                "tradable_non_null": int((vals.notna() & panel["tradable"].fillna(False)).sum()) if "tradable" in panel.columns else np.nan,
                "avg_amount_20d": float(pd.to_numeric(panel.get("amount_20d"), errors="coerce").mean()),
                "avg_turnover_20d": float(pd.to_numeric(panel.get("turnover_20d"), errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def factor_ic_report(panel: pd.DataFrame, factor_cols: List[str], horizons: Iterable[int] = (1, 5, 10, 20)) -> pd.DataFrame:
    rows = []
    work = panel[panel.get("tradable", True).fillna(False)].copy()
    date_groups = [(dt, sub) for dt, sub in work.groupby("date", sort=False)]
    for factor in factor_cols:
        for h in horizons:
            ret_col = f"fwd_ret_{h}d"
            daily_rows = []
            if ret_col not in work.columns:
                rows.append({"factor": factor, "horizon": h, "days": 0, "ic_mean": np.nan, "rank_ic_mean": np.nan, "rank_ic_ir": np.nan})
                continue
            for dt, sub in date_groups:
                x = pd.to_numeric(sub[factor], errors="coerce")
                y = pd.to_numeric(sub[ret_col], errors="coerce")
                m = x.notna() & y.notna()
                if int(m.sum()) < 10:
                    continue
                xr = x[m].rank(method="average")
                yr = y[m].rank(method="average")
                daily_rows.append(
                    {
                        "date": dt,
                        "ic": float(x[m].corr(y[m], method="pearson")),
                        "rank_ic": float(xr.corr(yr, method="pearson")),
                        "n": int(m.sum()),
                    }
                )
            daily = pd.DataFrame(daily_rows)
            if daily.empty:
                rows.append({"factor": factor, "horizon": h, "days": 0, "ic_mean": np.nan, "rank_ic_mean": np.nan, "rank_ic_ir": np.nan})
                continue
            rank_std = float(daily["rank_ic"].std(ddof=1))
            rows.append(
                {
                    "factor": factor,
                    "horizon": h,
                    "days": int(len(daily)),
                    "avg_n": float(daily["n"].mean()),
                    "ic_mean": float(daily["ic"].mean()),
                    "rank_ic_mean": float(daily["rank_ic"].mean()),
                    "rank_ic_ir": float(daily["rank_ic"].mean() / rank_std * math.sqrt(252)) if rank_std > 0 else np.nan,
                    "rank_ic_positive_rate": float((daily["rank_ic"] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def factor_layer_return_report(panel: pd.DataFrame, factor_cols: List[str], horizon: int = 20, bins: int = 5) -> pd.DataFrame:
    rows = []
    work = panel[panel.get("tradable", True).fillna(False)].copy()
    ret_col = f"fwd_ret_{horizon}d"
    if ret_col not in work.columns:
        return pd.DataFrame(columns=["factor", "horizon", "layer", "mean_return", "days"])
    date_groups = [(dt, sub) for dt, sub in work.groupby("date", sort=False)]
    for factor in factor_cols:
        for dt, sub in date_groups:
            x = pd.to_numeric(sub[factor], errors="coerce")
            y = pd.to_numeric(sub[ret_col], errors="coerce")
            m = x.notna() & y.notna()
            if int(m.sum()) < bins * 3:
                continue
            try:
                layer = pd.qcut(x[m].rank(method="first"), bins, labels=False) + 1
            except Exception:
                continue
            tmp = pd.DataFrame({"layer": layer.astype(int), "ret": y[m].values})
            grp = tmp.groupby("layer")["ret"].mean()
            for layer_id, value in grp.items():
                rows.append({"date": dt, "factor": factor, "horizon": horizon, "layer": int(layer_id), "mean_return": float(value)})
    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(columns=["factor", "horizon", "layer", "mean_return", "days"])
    out = raw.groupby(["factor", "horizon", "layer"])["mean_return"].agg(["mean", "count"]).reset_index()
    out = out.rename(columns={"mean": "mean_return", "count": "days"})
    spread_rows = []
    for factor, sub in out.groupby("factor"):
        top = sub[sub["layer"] == bins]["mean_return"]
        bottom = sub[sub["layer"] == 1]["mean_return"]
        if top.empty or bottom.empty:
            continue
        spread_rows.append({"factor": factor, "horizon": horizon, "layer": "top_minus_bottom", "mean_return": float(top.iloc[0] - bottom.iloc[0]), "days": int(sub["days"].max())})
    return pd.concat([out, pd.DataFrame(spread_rows)], ignore_index=True)


def factor_stability_report(panel: pd.DataFrame, factor_cols: List[str], horizon: int = 20) -> pd.DataFrame:
    rows = []
    work = panel[panel.get("tradable", True).fillna(False)].copy()
    ret_col = f"fwd_ret_{horizon}d"
    if ret_col not in work.columns:
        return pd.DataFrame(columns=["factor", "bucket_type", "bucket", "n", "rank_ic"])
    date_meta = work[["date"]].drop_duplicates().copy()
    date_meta["year"] = pd.to_datetime(date_meta["date"], errors="coerce").dt.year
    if "market_state_index" in work.columns:
        regimes = work[["date", "market_state_index"]].drop_duplicates("date", keep="last")
        date_meta = date_meta.merge(regimes, on="date", how="left")
    date_groups = [(dt, sub) for dt, sub in work.groupby("date", sort=False)]
    for factor in factor_cols:
        daily_rows = []
        for dt, sub in date_groups:
            x = pd.to_numeric(sub[factor], errors="coerce")
            y = pd.to_numeric(sub[ret_col], errors="coerce")
            m = x.notna() & y.notna()
            if int(m.sum()) < 10:
                continue
            xr = x[m].rank(method="average")
            yr = y[m].rank(method="average")
            daily_rows.append({"date": dt, "n": int(m.sum()), "rank_ic": float(xr.corr(yr))})
        daily = pd.DataFrame(daily_rows)
        if daily.empty:
            continue
        daily = daily.merge(date_meta, on="date", how="left")
        for bucket_type, bucket_col in [("year", "year"), ("market_state", "market_state_index")]:
            if bucket_col not in daily.columns:
                continue
            for bucket, sub in daily.groupby(bucket_col, dropna=False):
                std = float(sub["rank_ic"].std(ddof=1))
                rows.append(
                    {
                        "factor": factor,
                        "bucket_type": bucket_type,
                        "bucket": bucket,
                        "n": int(sub["n"].sum()),
                        "days": int(len(sub)),
                        "rank_ic": float(sub["rank_ic"].mean()),
                        "rank_ic_ir": float(sub["rank_ic"].mean() / std * math.sqrt(252)) if std > 0 else np.nan,
                        "positive_rate": float((sub["rank_ic"] > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _score_candidates(day_df: pd.DataFrame, score_col: str, min_score: float) -> pd.DataFrame:
    out = day_df[day_df["tradable"].fillna(False)].copy()
    score = pd.to_numeric(out[score_col], errors="coerce")
    out = out[score.notna() & (score >= min_score)].copy()
    out["strategy_score"] = pd.to_numeric(out[score_col], errors="coerce")
    return out.sort_values(["strategy_score", "amount_20d"], ascending=[False, False])


def simulate_factor_portfolio(
    panel: pd.DataFrame,
    score_col: str,
    top_n: int = 3,
    min_score: float = 58.0,
    min_hold_days: int = 15,
    max_hold_days: int = 9999,
    exit_rank_mult: float = 8.0,
    turnover_buffer: float = 15.0,
    base_total_exposure: float = 0.45,
    max_position_weight: float = 0.26,
    cost_bp: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = panel.sort_values(["date", "code"]).copy()
    work["next_open"] = work.groupby("code")["open"].shift(-1)
    work["next_date"] = work.groupby("code")["date"].shift(-1)
    work["rank_today"] = work.groupby("date")[score_col].rank(method="first", ascending=False)
    date_groups = {dt: sub for dt, sub in work.groupby("date", sort=False)}
    dates = sorted(date_groups.keys())

    positions: Dict[str, FactorPosition] = {}
    trades: List[dict] = []
    orders: List[dict] = []
    monitor_rows: List[dict] = []
    equity_rows: List[dict] = []
    equity = 1.0
    cash = 1.0
    cost_rate = cost_bp / 10000.0

    for dt in dates[:-1]:
        day = date_groups.get(dt)
        if day.empty:
            continue
        next_dt = pd.to_datetime(day["next_date"].dropna().min()) if day["next_date"].notna().any() else None
        if pd.isna(next_dt):
            continue
        candidates = _score_candidates(day, score_col, min_score)
        top = candidates.head(top_n).copy()
        top_codes = set(top["code"].astype(str))
        day_by_code = day.set_index(day["code"].astype(str), drop=False)

        for code, pos in list(positions.items()):
            if code not in day_by_code.index:
                continue
            row = day_by_code.loc[code]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            pos.days_held += 1
            close_px = float(pd.to_numeric(row.get("close"), errors="coerce"))
            if np.isfinite(close_px):
                pos.peak_close = max(pos.peak_close, close_px) if np.isfinite(pos.peak_close) else max(close_px, pos.entry_price)
            current_score = float(pd.to_numeric(row.get(score_col), errors="coerce"))
            rank_today = float(pd.to_numeric(row.get("rank_today"), errors="coerce"))
            exit_reason = None
            if pos.days_held >= max_hold_days:
                exit_reason = "max_hold"
            elif pos.days_held >= min_hold_days:
                rank_broken = pd.notna(rank_today) and rank_today > top_n * exit_rank_mult + turnover_buffer
                score_faded = (code not in top_codes) and (not np.isfinite(current_score) or current_score < max(min_score, pos.entry_score - 10.0))
                if rank_broken:
                    exit_reason = "rank_drop"
                elif score_faded:
                    exit_reason = "score_fade"
            if exit_reason:
                exit_px = float(pd.to_numeric(row.get("next_open"), errors="coerce"))
                if not np.isfinite(exit_px):
                    continue
                pnl_pct = exit_px / pos.entry_price - 1.0 - (2 * cost_rate)
                cash += pos.units * exit_px * (1.0 - cost_rate)
                trades.append(
                    {
                        "code": code,
                        "entry_signal_date": pos.entry_signal_date,
                        "entry_date": pos.entry_date,
                        "entry_price": pos.entry_price,
                        "exit_date": next_dt,
                        "exit_price": exit_px,
                        "pnl_pct": float(pnl_pct),
                        "days_held": pos.days_held,
                        "entry_score": pos.entry_score,
                        "exit_score": current_score,
                        "weight": pos.target_weight,
                        "exit_reason": exit_reason,
                    }
                )
                orders.append({"signal_date": dt, "execute_date": next_dt, "code": code, "side": "SELL", "price": exit_px, "weight": pos.target_weight, "score": current_score, "reason": exit_reason})
                positions.pop(code, None)

        available_slots = max(0, top_n - len(positions))
        buy_df = top[~top["code"].astype(str).isin(positions.keys())].head(available_slots).copy()
        if not buy_df.empty:
            weight = min(max_position_weight, base_total_exposure / max(top_n, 1))
            marked_value = cash
            for code, pos in positions.items():
                row = day_by_code.loc[code] if code in day_by_code.index else None
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                price = float(pd.to_numeric(row.get("next_open"), errors="coerce")) if row is not None else np.nan
                if np.isfinite(price):
                    marked_value += pos.units * price
            for _, row in buy_df.iterrows():
                entry_px = float(pd.to_numeric(row.get("next_open"), errors="coerce"))
                if not np.isfinite(entry_px):
                    continue
                code = str(row["code"])
                score = float(pd.to_numeric(row.get(score_col), errors="coerce"))
                allocation = marked_value * weight
                units = allocation / entry_px
                cash -= allocation * (1.0 + cost_rate)
                positions[code] = FactorPosition(
                    code=code,
                    entry_signal_date=pd.Timestamp(dt),
                    entry_date=next_dt,
                    entry_price=entry_px,
                    entry_score=score,
                    target_weight=weight,
                    units=units,
                    allocation=allocation,
                    peak_close=float(pd.to_numeric(row.get("close"), errors="coerce")),
                )
                orders.append({"signal_date": dt, "execute_date": next_dt, "code": code, "side": "BUY", "price": entry_px, "weight": weight, "score": score, "reason": score_col})

        equity = cash
        for code, pos in positions.items():
            row = day_by_code.loc[code] if code in day_by_code.index else None
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            price = float(pd.to_numeric(row.get("next_open"), errors="coerce")) if row is not None else np.nan
            if np.isfinite(price):
                equity += pos.units * price
        monitor_rows.append(
            {
                "date": dt,
                "positions": len(positions),
                "cash_weight": max(0.0, 1.0 - sum(p.target_weight for p in positions.values())),
                "top_score": float(pd.to_numeric(top[score_col], errors="coerce").mean()) if not top.empty else np.nan,
                "equity": equity,
            }
        )
        equity_rows.append({"date": next_dt, "equity": equity})

    latest_date = work["date"].max()
    latest_candidates = _score_candidates(work[work["date"] == latest_date].copy(), score_col, min_score).head(top_n)
    latest_day = work[work["date"] == latest_date].set_index(work[work["date"] == latest_date]["code"].astype(str), drop=False)
    latest_positions = pd.DataFrame(
        [
            {
                "code": p.code,
                "entry_signal_date": p.entry_signal_date,
                "entry_date": p.entry_date,
                "entry_price": p.entry_price,
                "entry_score": p.entry_score,
                "target_weight": p.target_weight,
                "days_held": p.days_held,
            }
            for p in positions.values()
        ]
    )
    for code, pos in list(positions.items()):
        if code not in latest_day.index:
            continue
        row = latest_day.loc[code]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        exit_px = float(pd.to_numeric(row.get("close"), errors="coerce"))
        if not np.isfinite(exit_px):
            continue
        pnl_pct = exit_px / pos.entry_price - 1.0 - (2 * cost_rate)
        cash += pos.units * exit_px * (1.0 - cost_rate)
        trades.append(
            {
                "code": code,
                "entry_signal_date": pos.entry_signal_date,
                "entry_date": pos.entry_date,
                "entry_price": pos.entry_price,
                "exit_date": latest_date,
                "exit_price": exit_px,
                "pnl_pct": float(pnl_pct),
                "days_held": pos.days_held,
                "entry_score": pos.entry_score,
                "exit_score": float(pd.to_numeric(row.get(score_col), errors="coerce")),
                "weight": pos.target_weight,
                "exit_reason": "period_end_mark",
            }
        )
    if positions:
        equity_rows.append({"date": latest_date, "equity": cash})
    return pd.DataFrame(trades), pd.DataFrame(monitor_rows), pd.DataFrame(equity_rows), pd.DataFrame(orders), latest_candidates, latest_positions


def _portfolio_metrics(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    orders: Optional[pd.DataFrame] = None,
    period_start: Optional[pd.Timestamp] = None,
    period_end: Optional[pd.Timestamp] = None,
) -> dict:
    if trades is None or trades.empty:
        return {"trade_count": 0, "hit_rate": np.nan, "expectancy": np.nan, "avg_hold_days": np.nan, "total_return": np.nan, "annualized_return": np.nan, "max_drawdown": np.nan, "return_drawdown": np.nan, "turnover": 0.0}
    eq = equity.copy()
    if not eq.empty:
        eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
        equity_values = pd.concat([pd.Series([1.0]), eq["equity"].dropna()], ignore_index=True)
        peak = equity_values.cummax()
        dd = equity_values / peak - 1.0
        total_return = float(eq["equity"].iloc[-1] - 1.0)
        max_dd = float(dd.min())
    else:
        total_return = float((1.0 + trades["pnl_pct"] * trades.get("weight", 1.0)).prod() - 1.0)
        max_dd = np.nan
    start = pd.to_datetime(period_start, errors="coerce")
    end = pd.to_datetime(period_end, errors="coerce")
    days = int((end - start).days) if pd.notna(start) and pd.notna(end) and end > start else 0
    annualized = float((1.0 + total_return) ** (365.0 / days) - 1.0) if days > 0 and total_return > -1.0 else np.nan
    turnover = float(pd.to_numeric(orders["weight"], errors="coerce").abs().sum() / 2.0) if orders is not None and not orders.empty and "weight" in orders.columns else 0.0
    return {
        "trade_count": int(len(trades)),
        "hit_rate": float((trades["pnl_pct"] > 0).mean()),
        "expectancy": float(trades["pnl_pct"].mean()),
        "avg_hold_days": float(trades["days_held"].mean()),
        "total_return": total_return,
        "annualized_return": annualized,
        "max_drawdown": max_dd,
        "return_drawdown": float(total_return / abs(max_dd)) if pd.notna(max_dd) and max_dd < 0 else np.nan,
        "turnover": turnover,
    }


def _compact_factor_candidates(
    latest_candidates: pd.DataFrame,
    latest_positions: pd.DataFrame,
    score_col: str,
    strategy_name: str,
) -> pd.DataFrame:
    if latest_candidates is None or latest_candidates.empty:
        return pd.DataFrame()
    out = latest_candidates.copy()
    held = set(latest_positions["code"].astype(str)) if latest_positions is not None and not latest_positions.empty else set()
    out["recommended_action"] = np.where(out["code"].astype(str).isin(held), "HOLD", "BUY")
    out["strategy"] = strategy_name
    out["target_weight"] = 0.15
    keep = [
        "date",
        "code",
        "close",
        "open",
        "next_open",
        "next_date",
        "rank_today",
        score_col,
        "lowvol_score",
        "low_vol_20d_score",
        "low_vol_60d_score",
        "low_atr_score",
        "low_downside_vol_20d_score",
        "low_gap_score",
        "mdd60_score",
        "amount_20d",
        "turnover_20d",
        "tradable",
        "recommended_action",
        "target_weight",
        "strategy",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].reset_index(drop=True)


def _build_factor_daily_actions(
    latest_positions: pd.DataFrame,
    latest_candidates: pd.DataFrame,
    panel: pd.DataFrame,
    as_of_date: pd.Timestamp,
    score_col: str,
    strategy_name: str,
    top_n: int = 3,
) -> pd.DataFrame:
    positions = latest_positions.copy() if latest_positions is not None else pd.DataFrame()
    candidates = latest_candidates.copy() if latest_candidates is not None else pd.DataFrame()
    day = panel[pd.to_datetime(panel["date"]) == pd.Timestamp(as_of_date)].copy()
    day_by_code = day.set_index(day["code"].astype(str), drop=False) if not day.empty else pd.DataFrame()
    pos_codes = positions["code"].astype(str).tolist() if not positions.empty else []
    cand_codes = candidates["code"].astype(str).tolist() if not candidates.empty else []
    buy_codes = [code for code in cand_codes if code not in set(pos_codes)][: max(0, int(top_n) - len(pos_codes))]

    rows = []
    for _, pos in positions.iterrows():
        code = str(pos["code"])
        row = day_by_code.loc[code] if code in day_by_code.index else pd.Series(dtype="object")
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        current_price = float(pd.to_numeric(row.get("close"), errors="coerce")) if not row.empty else np.nan
        entry_price = float(pd.to_numeric(pos.get("entry_price"), errors="coerce"))
        pnl = current_price / entry_price - 1.0 if np.isfinite(current_price) and np.isfinite(entry_price) and entry_price > 0 else np.nan
        rows.append(
            {
                "as_of_date": pd.Timestamp(as_of_date).date(),
                "action": "HOLD",
                "code": code,
                "target_weight": float(pd.to_numeric(pos.get("target_weight"), errors="coerce")),
                "strategy": strategy_name,
                "entry_date": pos.get("entry_date"),
                "entry_price": entry_price,
                "current_price": current_price,
                "strategy_score": float(pd.to_numeric(row.get(score_col), errors="coerce")) if not row.empty else np.nan,
                "pnl_pct": pnl,
                "reason": "existing_position_hold",
            }
        )

    cand_by_code = candidates.set_index(candidates["code"].astype(str), drop=False) if not candidates.empty else pd.DataFrame()
    for code in buy_codes:
        cand = cand_by_code.loc[code]
        if isinstance(cand, pd.DataFrame):
            cand = cand.iloc[-1]
        rows.append(
            {
                "as_of_date": pd.Timestamp(as_of_date).date(),
                "action": "BUY",
                "code": code,
                "target_weight": 0.15,
                "strategy": strategy_name,
                "entry_date": pd.NaT,
                "entry_price": np.nan,
                "current_price": float(pd.to_numeric(cand.get("close"), errors="coerce")),
                "strategy_score": float(pd.to_numeric(cand.get(score_col), errors="coerce")),
                "pnl_pct": np.nan,
                "reason": score_col,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {"SELL": 0, "HOLD": 1, "BUY": 2}
    out["action_order"] = out["action"].map(order).fillna(9)
    return out.sort_values(["action_order", "target_weight"], ascending=[True, False]).drop(columns=["action_order"]).reset_index(drop=True)


def _build_factor_action_history(
    orders: pd.DataFrame,
    daily_actions: pd.DataFrame,
    as_of_date: pd.Timestamp,
    recent_days: int = 10,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date) if start_date else pd.Timestamp(as_of_date) - pd.Timedelta(days=recent_days)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp(as_of_date)
    frames = []
    if orders is not None and not orders.empty:
        hist = orders.copy()
        hist["action_date"] = pd.to_datetime(hist["execute_date"], errors="coerce")
        hist = hist[(hist["action_date"] >= start) & (hist["action_date"] <= end)].copy()
        if not hist.empty:
            hist["action"] = hist["side"].astype(str).str.upper()
            hist = hist.rename(columns={"weight": "target_weight"})
            frames.append(hist[["action_date", "action", "code", "target_weight", "score", "reason"]])
    if daily_actions is not None and not daily_actions.empty:
        holds = daily_actions[daily_actions["action"].eq("HOLD")].copy()
        if not holds.empty:
            holds["action_date"] = pd.to_datetime(holds["as_of_date"], errors="coerce")
            holds["score"] = holds["strategy_score"]
            frames.append(holds[["action_date", "action", "code", "target_weight", "score", "reason"]])
    if not frames:
        return pd.DataFrame(columns=["action_date", "action", "code", "target_weight", "score", "reason"])
    out = pd.concat(frames, ignore_index=True)
    order = {"SELL": 0, "HOLD": 1, "BUY": 2}
    out["action_order"] = out["action"].map(order).fillna(9)
    return out.sort_values(["action_date", "action_order", "code"], ascending=[False, True, True]).drop(columns=["action_order"]).reset_index(drop=True)


def run_factor_lowvol_daily(
    all_csvs: List[str],
    cfg_path: Optional[str] = None,
    cfg_overrides: Optional[dict] = None,
    outdir: str = "./output/factor_lowvol_daily",
    cache_dir: Optional[str] = None,
    refresh_external: bool = False,
    min_amount_20d: float = 20_000_000.0,
    action_recent_days: int = 10,
    action_start: Optional[str] = None,
    action_end: Optional[str] = None,
):
    os.makedirs(outdir, exist_ok=True)
    strategy_name = "pure_lowvol"
    score_col = "score_pure_lowvol"
    print("[factor_lowvol_daily] load config/data", flush=True)
    base_cfg = merge_config(load_config(cfg_path), cfg_overrides)
    df_all = load_market_csv_multi(all_csvs)
    print(f"[factor_lowvol_daily] market rows={len(df_all):,}, codes={df_all['code'].nunique():,}", flush=True)
    df_ind = compute_indicator_panel(df_all, base_cfg)
    df_ind["code"] = df_ind["code"].astype(str).str.split(".").str[-1].str.zfill(6)
    df_ind["date"] = pd.to_datetime(df_ind["date"], errors="coerce")
    idx_state = build_index_state_from_panel(df_ind, base_cfg, by_bucket=False)
    idx_state["date"] = pd.to_datetime(idx_state["date"], errors="coerce")
    print("[factor_lowvol_daily] build lowvol factor panel", flush=True)
    panel = df_ind.merge(idx_state[["date", "market_state_index"]], on="date", how="left")
    panel = build_lowvol_factor_panel(panel, min_amount_20d=min_amount_20d)
    trades, monitor, equity, orders, latest_candidates, latest_positions = simulate_factor_portfolio(panel, score_col=score_col)
    latest_date = pd.to_datetime(panel["date"].max())
    metrics = _portfolio_metrics(trades, equity, orders, pd.to_datetime(panel["date"].min()), latest_date)
    compact_candidates = _compact_factor_candidates(latest_candidates, latest_positions, score_col, strategy_name)
    actions = _build_factor_daily_actions(latest_positions, latest_candidates, panel, latest_date, score_col, strategy_name)
    history = _build_factor_action_history(orders, actions, latest_date, recent_days=action_recent_days, start_date=action_start, end_date=action_end)

    recent_orders = pd.DataFrame()
    if orders is not None and not orders.empty:
        recent_orders = orders.copy()
        recent_orders["execute_date"] = pd.to_datetime(recent_orders["execute_date"], errors="coerce")
        start = pd.Timestamp(action_start) if action_start else latest_date - pd.Timedelta(days=action_recent_days)
        end = pd.Timestamp(action_end) if action_end else latest_date
        recent_orders = recent_orders[(recent_orders["execute_date"] >= start) & (recent_orders["execute_date"] <= end)].reset_index(drop=True)

    latest_day = panel[pd.to_datetime(panel["date"]) == latest_date]
    strategy = pd.DataFrame(
        [
            {
                "as_of_date": latest_date.date(),
                "strategy": strategy_name,
                "score_col": score_col,
                "description": "ranked low-volatility composite: 20d/60d volatility, ATR%, downside volatility, gap risk, 60d drawdown",
                "top_n": 3,
                "min_score": 58.0,
                "min_hold_days": 15,
                "base_total_exposure": 0.45,
                "max_position_weight": 0.26,
                "market_state_index": str(latest_day["market_state_index"].mode().iloc[0]) if "market_state_index" in latest_day.columns and not latest_day["market_state_index"].dropna().empty else "",
                "candidate_count": int(pd.to_numeric(latest_day.get(score_col), errors="coerce").ge(58.0).sum()) if score_col in latest_day.columns else 0,
                "position_count": int(len(latest_positions)),
            }
        ]
    )
    summary = pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])

    strategy.to_csv(os.path.join(outdir, "pure_lowvol_daily_strategy.csv"), index=False, encoding="utf-8-sig")
    actions.to_csv(os.path.join(outdir, "pure_lowvol_daily_actions.csv"), index=False, encoding="utf-8-sig")
    history.to_csv(os.path.join(outdir, "pure_lowvol_action_history.csv"), index=False, encoding="utf-8-sig")
    compact_candidates.to_csv(os.path.join(outdir, "pure_lowvol_latest_candidates.csv"), index=False, encoding="utf-8-sig")
    latest_positions.to_csv(os.path.join(outdir, "pure_lowvol_positions_state.csv"), index=False, encoding="utf-8-sig")
    recent_orders.to_csv(os.path.join(outdir, "pure_lowvol_orders_next_open.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(os.path.join(outdir, "pure_lowvol_summary.csv"), index=False, encoding="utf-8-sig")
    print("[factor_lowvol_daily] wrote compact daily files", flush=True)
    return {
        "strategy": strategy,
        "actions": actions,
        "history": history,
        "latest_candidates": compact_candidates,
        "latest_positions": latest_positions,
        "recent_orders": recent_orders,
        "summary": summary,
    }


def run_factor_research(
    all_csvs: List[str],
    cfg_path: Optional[str] = None,
    cfg_overrides: Optional[dict] = None,
    outdir: str = "./output/factor_research",
    cache_dir: Optional[str] = None,
    refresh_external: bool = False,
    extra_factor_csvs: Optional[List[str]] = None,
    train_months: int = 36,
    val_months: int = 6,
    test_months: int = 6,
    step_months: int = 3,
    min_amount_20d: float = 20_000_000.0,
    skip_diagnostics: bool = False,
):
    os.makedirs(outdir, exist_ok=True)
    print("[factor_research] load config/data", flush=True)
    base_cfg = merge_config(load_config(cfg_path), cfg_overrides)
    df_all = load_market_csv_multi(all_csvs)
    print(f"[factor_research] market rows={len(df_all):,}, codes={df_all['code'].nunique():,}", flush=True)
    df_ind = compute_indicator_panel(df_all, base_cfg)
    df_ind["code"] = df_ind["code"].astype(str).str.split(".").str[-1].str.zfill(6)
    df_ind["date"] = pd.to_datetime(df_ind["date"], errors="coerce")
    print(f"[factor_research] indicator rows={len(df_ind):,}", flush=True)
    idx_state = build_index_state_from_panel(df_ind, base_cfg, by_bucket=False)
    idx_state["date"] = pd.to_datetime(idx_state["date"], errors="coerce")
    external = load_external_factor_data(cache_dir=cache_dir or os.path.join(outdir, "factor_cache"), refresh=refresh_external, extra_csvs=extra_factor_csvs)

    print("[factor_research] build classic factor panel", flush=True)
    panel = df_ind.merge(idx_state[["date", "market_state_index"]], on="date", how="left")
    panel = build_classic_factor_panel(panel, external=external, min_amount_20d=min_amount_20d)
    panel = add_forward_returns(panel)

    factor_cols = sorted({c for cols in FACTOR_GROUPS.values() for c in cols if c in panel.columns})
    factor_cols += [c for c in ["momentum_score", "reversal_score", "lowvol_score", "value_score_v2", "quality_score_v2", "growth_score", "liquidity_score_v2", "score_quality_value_momentum_lowvol"] if c in panel.columns]
    factor_cols = list(dict.fromkeys(factor_cols))

    print(f"[factor_research] factor diagnostics factors={len(factor_cols)}", flush=True)
    if skip_diagnostics:
        coverage = pd.read_csv(os.path.join(outdir, "factor_coverage_report.csv")) if os.path.exists(os.path.join(outdir, "factor_coverage_report.csv")) else pd.DataFrame()
        ic = pd.read_csv(os.path.join(outdir, "factor_ic_report.csv")) if os.path.exists(os.path.join(outdir, "factor_ic_report.csv")) else pd.DataFrame()
        layers = pd.read_csv(os.path.join(outdir, "factor_layer_return_report.csv")) if os.path.exists(os.path.join(outdir, "factor_layer_return_report.csv")) else pd.DataFrame()
        stability = pd.read_csv(os.path.join(outdir, "factor_stability_report.csv")) if os.path.exists(os.path.join(outdir, "factor_stability_report.csv")) else pd.DataFrame()
        print("[factor_research] reused existing factor diagnostic reports", flush=True)
    else:
        coverage = factor_coverage_report(panel, factor_cols)
        coverage.to_csv(os.path.join(outdir, "factor_coverage_report.csv"), index=False, encoding="utf-8-sig")
        print("[factor_research] wrote factor_coverage_report.csv", flush=True)
        ic = factor_ic_report(panel, factor_cols)
        ic.to_csv(os.path.join(outdir, "factor_ic_report.csv"), index=False, encoding="utf-8-sig")
        print("[factor_research] wrote factor_ic_report.csv", flush=True)
        layers = factor_layer_return_report(panel, factor_cols)
        layers.to_csv(os.path.join(outdir, "factor_layer_return_report.csv"), index=False, encoding="utf-8-sig")
        print("[factor_research] wrote factor_layer_return_report.csv", flush=True)
        stability = factor_stability_report(panel, factor_cols)
        stability.to_csv(os.path.join(outdir, "factor_stability_report.csv"), index=False, encoding="utf-8-sig")
        print("[factor_research] wrote factor_stability_report.csv", flush=True)

    windows = _generate_windows(panel["date"], train_months, val_months, test_months, step_months)
    if not windows:
        raise RuntimeError("factor research: insufficient windows for walk-forward")

    print(f"[factor_research] walk-forward windows={len(windows)}", flush=True)
    metrics_rows = []
    latest_outputs = None
    for window_id, (_, _, vl_s, vl_e, te_s, te_e) in enumerate(windows, start=1):
        print(f"[factor_research] walk-forward window={window_id}/{len(windows)} test={pd.Timestamp(te_s).date()}..{pd.Timestamp(te_e).date()}", flush=True)
        val_panel = panel[(panel["date"] >= pd.Timestamp(vl_s)) & (panel["date"] < pd.Timestamp(vl_e))].copy()
        test_panel = panel[(panel["date"] >= pd.Timestamp(te_s)) & (panel["date"] < pd.Timestamp(te_e))].copy()
        val_scores = []
        test_outputs = {}
        for name in STRATEGY_WEIGHTS:
            score_col = f"score_{name}"
            if score_col not in panel.columns:
                continue
            if not pd.to_numeric(panel[score_col], errors="coerce").notna().any():
                continue
            val_trades, _, val_equity, val_orders, _, _ = simulate_factor_portfolio(val_panel, score_col=score_col)
            val_metric = _portfolio_metrics(val_trades, val_equity, val_orders, pd.Timestamp(vl_s), pd.Timestamp(vl_e))
            val_score = (val_metric.get("hit_rate") or 0.0) * 100.0 + (val_metric.get("expectancy") or 0.0) * 1000.0 + min(val_metric.get("trade_count") or 0, 200) * 0.01
            val_scores.append((name, val_score))
            trades, monitor, equity, orders, latest_candidates, latest_positions = simulate_factor_portfolio(test_panel, score_col=score_col)
            test_outputs[name] = (trades, monitor, equity, orders, latest_candidates, latest_positions)
            m = _portfolio_metrics(trades, equity, orders, pd.Timestamp(te_s), pd.Timestamp(te_e))
            metrics_rows.append(
                {
                    "window_id": window_id,
                    "variant": name,
                    "selected_variant": name,
                    "val_start": pd.Timestamp(vl_s).date(),
                    "val_end": pd.Timestamp(vl_e).date(),
                    "test_start": pd.Timestamp(te_s).date(),
                    "test_end": pd.Timestamp(te_e).date(),
                    **m,
                }
            )
        if val_scores:
            selected = max(val_scores, key=lambda x: x[1])[0]
            trades, monitor, equity, orders, latest_candidates, latest_positions = test_outputs[selected]
            m = _portfolio_metrics(trades, equity, orders, pd.Timestamp(te_s), pd.Timestamp(te_e))
            metrics_rows.append(
                {
                    "window_id": window_id,
                    "variant": "validation_selected",
                    "selected_variant": selected,
                    "val_start": pd.Timestamp(vl_s).date(),
                    "val_end": pd.Timestamp(vl_e).date(),
                    "test_start": pd.Timestamp(te_s).date(),
                    "test_end": pd.Timestamp(te_e).date(),
                    **m,
                }
            )
            latest_outputs = test_outputs[selected]

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(os.path.join(outdir, "portfolio_walkforward_metrics.csv"), index=False, encoding="utf-8-sig")
    summary = (
        metrics.groupby("variant")[["trade_count", "hit_rate", "expectancy", "avg_hold_days", "total_return", "annualized_return", "max_drawdown", "return_drawdown", "turnover"]]
        .mean()
        .reset_index()
        .sort_values(["return_drawdown", "expectancy"], ascending=[False, False])
    )
    summary.to_csv(os.path.join(outdir, "portfolio_walkforward_summary.csv"), index=False, encoding="utf-8-sig")

    if latest_outputs is not None:
        trades, monitor, equity, orders, latest_candidates, latest_positions = latest_outputs
        trades.to_csv(os.path.join(outdir, "portfolio_backtest_trades.csv"), index=False, encoding="utf-8-sig")
        monitor.to_csv(os.path.join(outdir, "portfolio_backtest_daily_monitor.csv"), index=False, encoding="utf-8-sig")
        equity.to_csv(os.path.join(outdir, "portfolio_equity_curve.csv"), index=False, encoding="utf-8-sig")
        orders.to_csv(os.path.join(outdir, "portfolio_orders_next_open.csv"), index=False, encoding="utf-8-sig")
        latest_candidates.to_csv(os.path.join(outdir, "portfolio_latest_candidates.csv"), index=False, encoding="utf-8-sig")
        latest_positions.to_csv(os.path.join(outdir, "portfolio_positions_state.csv"), index=False, encoding="utf-8-sig")

    return {
        "panel": panel,
        "coverage": coverage,
        "ic": ic,
        "layers": layers,
        "stability": stability,
        "metrics": metrics,
        "summary": summary,
    }
