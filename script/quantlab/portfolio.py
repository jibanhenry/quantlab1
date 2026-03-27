# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .backtest import backtest_simple, score_candidates
from .config import load_config, merge_config
from .io_utils import load_market_csv_multi
from .market_state import build_index_state_from_panel
from .signals import assemble_signals, compute_indicators
from .valuation import add_valuation_features


@dataclass
class PortfolioPosition:
    code: str
    entry_signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_score: float
    target_weight: float
    stop_ref: float
    strategy: str
    days_held: int = 0
    peak_close: float = np.nan


def _portfolio_cfg(cfg: dict) -> dict:
    return cfg.get("portfolio", {}) if isinstance(cfg, dict) else {}


def _strategy_profile_weights(profile: str) -> Dict[str, float]:
    profiles = {
        "blended": {
            "trend_score": 0.24,
            "momentum_score": 0.16,
            "medium_term_score": 0.20,
            "signal_score": 0.14,
            "quality_score": 0.12,
            "value_factor_score": 0.08,
            "risk_score": 0.04,
            "liquidity_score": 0.02,
        },
        "breakout": {
            "trend_score": 0.22,
            "momentum_score": 0.10,
            "medium_term_score": 0.28,
            "signal_score": 0.20,
            "quality_score": 0.12,
            "value_factor_score": 0.02,
            "risk_score": 0.03,
            "liquidity_score": 0.03,
        },
        "pullback": {
            "trend_score": 0.28,
            "momentum_score": 0.06,
            "medium_term_score": 0.12,
            "signal_score": 0.22,
            "quality_score": 0.12,
            "value_factor_score": 0.10,
            "risk_score": 0.07,
            "liquidity_score": 0.03,
        },
        "mid_momo": {
            "trend_score": 0.18,
            "momentum_score": 0.18,
            "medium_term_score": 0.30,
            "signal_score": 0.10,
            "quality_score": 0.12,
            "value_factor_score": 0.04,
            "risk_score": 0.05,
            "liquidity_score": 0.03,
        },
        "quality_trend": {
            "trend_score": 0.26,
            "momentum_score": 0.08,
            "medium_term_score": 0.20,
            "signal_score": 0.10,
            "quality_score": 0.22,
            "value_factor_score": 0.06,
            "risk_score": 0.05,
            "liquidity_score": 0.03,
        },
        "value_trend": {
            "trend_score": 0.22,
            "momentum_score": 0.06,
            "medium_term_score": 0.18,
            "signal_score": 0.10,
            "quality_score": 0.10,
            "value_factor_score": 0.24,
            "risk_score": 0.07,
            "liquidity_score": 0.03,
        },
        "pullback_quality_combo": {
            "trend_score": 0.27,
            "momentum_score": 0.07,
            "medium_term_score": 0.16,
            "signal_score": 0.16,
            "quality_score": 0.18,
            "value_factor_score": 0.08,
            "risk_score": 0.05,
            "liquidity_score": 0.03,
        },
    }
    return profiles.get(profile, profiles["blended"])


def _profile_filter_mask(df: pd.DataFrame, profile: str) -> pd.Series:
    close = pd.to_numeric(df.get("close"), errors="coerce")
    ema20 = pd.to_numeric(df.get("ema20"), errors="coerce")
    ema50 = pd.to_numeric(df.get("ema50"), errors="coerce")
    ema200 = pd.to_numeric(df.get("ema200"), errors="coerce")
    breakout = pd.to_numeric(df.get("breakout20_atr"), errors="coerce")
    roc20 = pd.to_numeric(df.get("roc20"), errors="coerce")
    roc60 = pd.to_numeric(df.get("roc60"), errors="coerce")
    dist_high20 = pd.to_numeric(df.get("dist_high20"), errors="coerce")
    clv = pd.to_numeric(df.get("clv"), errors="coerce")
    body_ratio = pd.to_numeric(df.get("body_ratio"), errors="coerce")
    value_score = pd.to_numeric(df.get("value_score"), errors="coerce")
    s1 = pd.to_numeric(df.get("s1_entry", 0), errors="coerce").fillna(0.0)
    s2 = pd.to_numeric(df.get("s2_entry", 0), errors="coerce").fillna(0.0)
    trend_ok = (ema50 > ema200) & (close > ema20)

    if profile == "breakout":
        return (s2 == 1) | (breakout > 0) | (close > pd.to_numeric(df.get("boll_up"), errors="coerce"))
    if profile == "pullback":
        return ((s1 == 1) | ((trend_ok) & (dist_high20 > -0.08) & (dist_high20 < -0.01)))
    if profile == "mid_momo":
        return trend_ok & (roc20 > 0) & (roc60 > 0)
    if profile == "quality_trend":
        return trend_ok & (clv > 0) & (body_ratio > 0.35)
    if profile == "value_trend":
        return trend_ok & value_score.notna() & (value_score > 0.55)
    if profile == "pullback_quality_combo":
        quality_ok = trend_ok & (clv > 0) & (body_ratio > 0.35)
        pullback_ok = ((s1 == 1) | (trend_ok & (dist_high20 > -0.08) & (dist_high20 < -0.01)))
        return quality_ok | pullback_ok
    return pd.Series(True, index=df.index)


def compute_indicator_panel(df_all: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    parts = []
    for _, sub in tqdm(df_all.groupby("code"), desc="portfolio.compute_indicators", total=df_all["code"].nunique()):
        sub = sub.sort_values("date").reset_index(drop=True)
        parts.append(compute_indicators(sub, cfg))
    df_ind = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return add_valuation_features(df_ind, cfg)


def build_signal_panel(df_ind: pd.DataFrame, cfg: dict, by_bucket: bool = False) -> pd.DataFrame:
    idx_state = build_index_state_from_panel(df_ind, cfg, by_bucket=by_bucket)
    panel = []
    for code, sub in tqdm(df_ind.groupby("code"), desc="portfolio.assemble_signals", total=df_ind["code"].nunique()):
        sig = assemble_signals(sub.sort_values("date").reset_index(drop=True), idx_state, cfg)
        sig["code"] = str(code)
        panel.append(sig)
    out = pd.concat(panel, ignore_index=True) if panel else pd.DataFrame()
    return score_candidates(out, cfg) if not out.empty else out


def add_multifactor_scores(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if panel is None or panel.empty:
        return panel

    out = panel.copy()
    p_cfg = _portfolio_cfg(cfg)

    out["trend_raw"] = (
        (out["ema20"] > out["ema50"]).astype(float) * 0.35
        + (out["ema50"] > out["ema200"]).astype(float) * 0.35
        + np.clip(pd.to_numeric(out["adx14"], errors="coerce") / 50.0, 0.0, 1.0) * 0.20
        + np.clip((pd.to_numeric(out["plus_di14"], errors="coerce") - pd.to_numeric(out["minus_di14"], errors="coerce")) / 50.0, -1.0, 1.0) * 0.10
    )
    out["momentum_raw"] = (
        pd.to_numeric(out["roc10"], errors="coerce").fillna(0.0) * 0.35
        + pd.to_numeric(out["macd_hist"], errors="coerce").fillna(0.0) * 0.35
        + pd.to_numeric(out["cci20"], errors="coerce").fillna(0.0) * 0.15
        + ((pd.to_numeric(out["close"], errors="coerce") / pd.to_numeric(out["ema20"], errors="coerce")) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 0.15
    )
    out["medium_term_raw"] = (
        pd.to_numeric(out.get("roc20"), errors="coerce").fillna(0.0) * 0.30
        + pd.to_numeric(out.get("roc60"), errors="coerce").fillna(0.0) * 0.30
        + pd.to_numeric(out.get("breakout20_atr"), errors="coerce").clip(-3.0, 3.0).fillna(0.0) * 0.20
        + pd.to_numeric(out.get("dist_high60"), errors="coerce").fillna(-0.10) * 0.10
        + (1.0 + pd.to_numeric(out.get("mdd60"), errors="coerce").fillna(-0.30)) * 0.10
    )
    out["signal_raw"] = (
        out.get("s2_entry", 0).astype(float) * 1.0
        + out.get("s1_entry", 0).astype(float) * 0.8
        + out.get("s3_long_entry", 0).astype(float) * 0.6
        + out.get("s4_pyramid", 0).astype(float) * 0.4
        + np.where(out["market_state_index"].eq("trend_ok"), 0.2, 0.0)
    )
    out["quality_raw"] = (
        pd.to_numeric(out.get("clv"), errors="coerce").fillna(0.0) * 0.30
        + pd.to_numeric(out.get("body_ratio"), errors="coerce").fillna(0.0) * 0.20
        + pd.to_numeric(out.get("lower_wick_ratio"), errors="coerce").fillna(0.0) * 0.10
        + pd.to_numeric(out.get("rvol20"), errors="coerce").clip(0.0, 4.0).fillna(1.0) * 0.15
        + (-pd.to_numeric(out.get("gap_atr14"), errors="coerce").clip(0.0, 3.0).fillna(0.0)) * 0.15
        + (-pd.to_numeric(out.get("tr_pct"), errors="coerce").clip(0.0, 0.2).fillna(0.0)) * 0.10
    )
    out["risk_raw"] = -pd.to_numeric(out["atr_pct"], errors="coerce").fillna(np.nan)
    out["liquidity_raw"] = pd.to_numeric(out["turnover"], errors="coerce").fillna(np.nan)
    out["value_raw"] = pd.to_numeric(out["value_score"], errors="coerce").fillna(np.nan)

    grouped = out.groupby("date", group_keys=False)
    out["trend_score"] = grouped["trend_raw"].rank(pct=True).fillna(0.5) * 100.0
    out["momentum_score"] = grouped["momentum_raw"].rank(pct=True).fillna(0.5) * 100.0
    out["medium_term_score"] = grouped["medium_term_raw"].rank(pct=True).fillna(0.5) * 100.0
    out["signal_score"] = grouped["signal_raw"].rank(pct=True).fillna(0.5) * 100.0
    out["quality_score"] = grouped["quality_raw"].rank(pct=True).fillna(0.5) * 100.0
    out["value_factor_score"] = grouped["value_raw"].rank(pct=True).fillna(0.5) * 100.0
    out["risk_score"] = grouped["risk_raw"].rank(pct=True).fillna(0.5) * 100.0
    out["liquidity_score"] = grouped["liquidity_raw"].rank(pct=True).fillna(0.5) * 100.0

    weights = {
        "trend_score": float(p_cfg.get("trend_weight", 0.28)),
        "momentum_score": float(p_cfg.get("momentum_weight", 0.24)),
        "signal_score": float(p_cfg.get("signal_weight", 0.18)),
        "medium_term_score": float(p_cfg.get("medium_term_weight", 0.20)),
        "quality_score": float(p_cfg.get("quality_weight", 0.12)),
        "value_factor_score": float(p_cfg.get("value_weight", 0.12)),
        "risk_score": float(p_cfg.get("risk_weight", 0.10)),
        "liquidity_score": float(p_cfg.get("liquidity_weight", 0.08)),
    }
    total = sum(weights.values())
    if total <= 0:
        total = 1.0
        weights["trend_score"] = 1.0
    for key in weights:
        weights[key] /= total

    out["multifactor_score"] = 0.0
    for col, weight in weights.items():
        out["multifactor_score"] += out[col].fillna(50.0) * weight

    for profile in ("blended", "breakout", "pullback", "mid_momo", "quality_trend", "value_trend", "pullback_quality_combo"):
        profile_weights = _strategy_profile_weights(profile)
        score = pd.Series(0.0, index=out.index, dtype=float)
        total_weight = 0.0
        for col, weight in profile_weights.items():
            score = score + out[col].fillna(50.0) * weight
            total_weight += weight
        out[f"profile_score_{profile}"] = score / max(total_weight, 1e-9)

    out["composite_reason"] = (
        "trend=" + out["trend_score"].round(1).astype(str)
        + "|mom=" + out["momentum_score"].round(1).astype(str)
        + "|mid=" + out["medium_term_score"].round(1).astype(str)
        + "|signal=" + out["signal_score"].round(1).astype(str)
        + "|quality=" + out["quality_score"].round(1).astype(str)
        + "|value=" + out["value_factor_score"].round(1).astype(str)
        + "|risk=" + out["risk_score"].round(1).astype(str)
    )
    return out


def _choose_strategy(row: pd.Series) -> str:
    if int(row.get("s2_entry", 0)) == 1:
        return "S2"
    if int(row.get("s1_entry", 0)) == 1:
        return "S1"
    if int(row.get("s3_long_entry", 0)) == 1:
        return "S3"
    return "MF"


def _stop_for_row(row: pd.Series) -> float:
    strategy = _choose_strategy(row)
    if strategy == "S2":
        return float(row.get("s2_stop", np.nan))
    if strategy == "S1":
        return float(row.get("s1_stop", np.nan))
    if strategy == "S3":
        return float(row.get("s3_stop", np.nan))
    return float(row.get("close", np.nan) - 1.5 * row.get("atr14", np.nan))


def _eligible_universe(day_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if day_df.empty:
        return day_df
    p_cfg = _portfolio_cfg(cfg)
    profile = str(p_cfg.get("strategy_profile", "blended"))
    min_score = float(p_cfg.get("min_score", 55.0))
    market_ok = day_df["market_state_index"].isin(["trend_ok", "neutral", "range_bias"])
    has_signal = (day_df.get("s1_entry", 0) == 1) | (day_df.get("s2_entry", 0) == 1) | (day_df.get("s3_long_entry", 0) == 1)
    if profile in {"regime_switch", "meta_switch_rsi_vol"}:
        out = day_df.copy()
        regime = str(out["market_state_index"].mode().iloc[0]) if "market_state_index" in out.columns and not out["market_state_index"].dropna().empty else "neutral"
        if profile == "meta_switch_rsi_vol":
            median_rsi = float(pd.to_numeric(out.get("rsi14"), errors="coerce").median())
            median_rvol = float(pd.to_numeric(out.get("rvol20"), errors="coerce").median())
            if regime == "trend_ok" and np.isfinite(median_rsi) and np.isfinite(median_rvol) and median_rsi >= 58.0 and median_rvol >= 1.0:
                active_profile = "quality_trend"
            elif regime == "trend_ok" and np.isfinite(median_rsi) and median_rsi >= 52.0:
                active_profile = "pullback_quality_combo"
            elif regime == "range_bias" or (np.isfinite(median_rsi) and median_rsi < 48.0):
                active_profile = "pullback"
            else:
                active_profile = "pullback_quality_combo"
        else:
            if regime == "trend_ok":
                active_profile = "quality_trend"
            elif regime == "range_bias":
                active_profile = "pullback"
            else:
                active_profile = "pullback_quality_combo"
        score_col = f"profile_score_{active_profile}"
        if score_col not in out.columns:
            score_col = "multifactor_score"
        strategy_score = pd.to_numeric(out[score_col], errors="coerce")
        score_ok = strategy_score >= min_score
        profile_ok = _profile_filter_mask(out, active_profile)
        out = out[market_ok & profile_ok & (has_signal | score_ok)].copy()
        out["strategy_profile"] = active_profile
        out["strategy_score"] = pd.to_numeric(out[score_col], errors="coerce")
        if "confidence" not in out.columns:
            out["confidence"] = pd.to_numeric(out.get("technical_score", np.nan), errors="coerce")
        return out.sort_values(["strategy_score", "candidate_score", "confidence"], ascending=[False, False, False])
    score_col = f"profile_score_{profile}"
    if score_col not in day_df.columns:
        score_col = "multifactor_score"
    strategy_score = pd.to_numeric(day_df[score_col], errors="coerce")
    score_ok = strategy_score >= min_score
    profile_ok = _profile_filter_mask(day_df, profile)
    out = day_df[market_ok & profile_ok & (has_signal | score_ok)].copy()
    out["strategy_profile"] = profile
    out["strategy_score"] = pd.to_numeric(out[score_col], errors="coerce")
    if "confidence" not in out.columns:
        out["confidence"] = pd.to_numeric(out.get("technical_score", np.nan), errors="coerce")
    out = out.sort_values(["strategy_score", "candidate_score", "confidence"], ascending=[False, False, False])
    return out


def _target_total_exposure(top_df: pd.DataFrame, cfg: dict) -> float:
    p_cfg = _portfolio_cfg(cfg)
    min_total = float(p_cfg.get("min_total_exposure", 0.20))
    base_total = float(p_cfg.get("base_total_exposure", 0.55))
    max_total = float(p_cfg.get("max_total_exposure", 0.85))
    if top_df is None or top_df.empty:
        return min_total
    score_col = "strategy_score" if "strategy_score" in top_df.columns else "multifactor_score"
    avg_score = float(pd.to_numeric(top_df[score_col], errors="coerce").head(int(p_cfg.get("top_n", 3))).mean())
    score_floor = float(p_cfg.get("min_score", 55.0))
    score_cap = 92.0
    strength = np.clip((avg_score - score_floor) / max(score_cap - score_floor, 1e-6), 0.0, 1.0)
    market_state = str(top_df["market_state_index"].mode().iloc[0]) if "market_state_index" in top_df.columns and not top_df["market_state_index"].dropna().empty else "neutral"
    if market_state == "trend_ok":
        target = base_total + (max_total - base_total) * strength
    elif market_state == "range_bias":
        target = min(base_total, 0.35)
    else:
        target = min_total + (base_total - min_total) * strength
    return float(np.clip(target, min_total, max_total))


def _allocate_target_weights(cands: pd.DataFrame, total_exposure: float, cfg: dict) -> pd.DataFrame:
    out = cands.copy()
    if out.empty or total_exposure <= 0:
        out["target_weight"] = 0.0
        return out
    p_cfg = _portfolio_cfg(cfg)
    min_w = float(p_cfg.get("min_position_weight", 0.10))
    max_w = float(p_cfg.get("max_position_weight", 0.35))
    score_col = "strategy_score" if "strategy_score" in out.columns else "multifactor_score"
    scores = pd.to_numeric(out[score_col], errors="coerce").fillna(float(p_cfg.get("min_score", 55.0)))
    raw = (scores - float(p_cfg.get("min_score", 55.0))).clip(lower=1.0)
    weights = raw / raw.sum()
    out["target_weight"] = weights * total_exposure
    out["target_weight"] = out["target_weight"].clip(lower=min_w, upper=max_w)
    scale = total_exposure / max(out["target_weight"].sum(), 1e-9)
    out["target_weight"] = (out["target_weight"] * scale).clip(upper=max_w)
    return out


def _score_exit_threshold(pos: PortfolioPosition, cfg: dict) -> float:
    p_cfg = _portfolio_cfg(cfg)
    base_floor = float(p_cfg.get("exit_score_floor", p_cfg.get("min_score", 55.0) - 2.0))
    score_drop = float(p_cfg.get("exit_score_drop", 10.0))
    return max(base_floor, pos.entry_score - score_drop)


def _trend_breakdown(row: pd.Series) -> bool:
    close = float(pd.to_numeric(row.get("close"), errors="coerce"))
    ema20 = float(pd.to_numeric(row.get("ema20"), errors="coerce"))
    ema50 = float(pd.to_numeric(row.get("ema50"), errors="coerce"))
    adx14 = float(pd.to_numeric(row.get("adx14"), errors="coerce"))
    dist_high20 = float(pd.to_numeric(row.get("dist_high20"), errors="coerce"))
    if not np.isfinite(close):
        return False
    weak_price = np.isfinite(ema20) and close < ema20
    weak_trend = np.isfinite(ema20) and np.isfinite(ema50) and ema20 < ema50
    trend_exhaust = np.isfinite(adx14) and adx14 < 16.0
    deep_pullback = np.isfinite(dist_high20) and dist_high20 < -0.12
    return bool((weak_price and weak_trend) or (weak_price and deep_pullback) or (weak_trend and trend_exhaust))


def _trailing_exit(pos: PortfolioPosition, row: pd.Series, cfg: dict) -> bool:
    p_cfg = _portfolio_cfg(cfg)
    close = float(pd.to_numeric(row.get("close"), errors="coerce"))
    ema20 = float(pd.to_numeric(row.get("ema20"), errors="coerce"))
    atr_pct = float(pd.to_numeric(row.get("atr_pct"), errors="coerce"))
    if not np.isfinite(close) or not np.isfinite(pos.entry_price):
        return False
    peak_close = pos.peak_close if np.isfinite(pos.peak_close) else max(close, pos.entry_price)
    open_profit = peak_close / pos.entry_price - 1.0
    giveback = 1.0 - close / peak_close if peak_close > 0 else 0.0
    min_open_profit = float(p_cfg.get("trail_min_open_profit", 0.10))
    giveback_limit = float(p_cfg.get("trail_giveback_pct", 0.06))
    atr_buffer = float(p_cfg.get("trail_atr_buffer", 1.2))
    ema_break = np.isfinite(ema20) and close < ema20
    atr_break = np.isfinite(atr_pct) and open_profit > 0 and giveback > max(giveback_limit, atr_pct * atr_buffer)
    return bool(open_profit >= min_open_profit and (ema_break or atr_break))


def simulate_multifactor_portfolio(panel: pd.DataFrame, cfg: dict, cost_bp: float = 2.0):
    if panel is None or panel.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty, empty

    p_cfg = _portfolio_cfg(cfg)
    top_n = int(p_cfg.get("top_n", 10))
    min_hold_days = int(p_cfg.get("min_hold_days", 7))
    max_hold_days = int(p_cfg.get("max_hold_days", 20))
    exit_rank_mult = float(p_cfg.get("exit_rank_mult", 4.0))
    turnover_buffer = float(p_cfg.get("turnover_buffer", 0.0))
    exit_score_floor = float(p_cfg.get("exit_score_floor", p_cfg.get("min_score", 55.0) - 2.0))
    exit_score_drop = float(p_cfg.get("exit_score_drop", 10.0))
    signal_exit_score_floor = float(p_cfg.get("signal_exit_score_floor", exit_score_floor + 2.0))
    exit_style = str(p_cfg.get("exit_style", "rank"))

    panel = panel.sort_values(["date", "code"]).reset_index(drop=True).copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["next_open"] = panel.groupby("code")["open"].shift(-1)
    panel["next_date"] = panel.groupby("code")["date"].shift(-1)
    rank_col = "strategy_score" if "strategy_score" in panel.columns else "multifactor_score"
    panel["rank_today"] = panel.groupby("date")[rank_col].rank(method="first", ascending=False)

    positions: Dict[str, PortfolioPosition] = {}
    trades: List[dict] = []
    orders: List[dict] = []
    monitor_rows: List[dict] = []
    equity_rows: List[dict] = []

    dates = sorted(panel["date"].dropna().unique())
    equity = 1.0

    for dt in dates[:-1]:
        day_df = panel[panel["date"] == dt].copy()
        if day_df.empty:
            continue
        next_dt = pd.to_datetime(day_df["next_date"].dropna().min()) if day_df["next_date"].notna().any() else None
        if pd.isna(next_dt):
            continue

        eligible = _eligible_universe(day_df, cfg)
        top_df = eligible.head(top_n).copy()
        target_exposure = _target_total_exposure(top_df, cfg)
        top_df = _allocate_target_weights(top_df, target_exposure, cfg)
        top_codes = set(top_df["code"].astype(str))
        row_map = {str(r["code"]): r for _, r in day_df.iterrows()}

        sells: List[Tuple[str, str, PortfolioPosition, pd.Series]] = []
        holds: List[dict] = []
        for code, pos in list(positions.items()):
            row = row_map.get(code)
            if row is None:
                continue
            pos.days_held += 1
            close_px = float(pd.to_numeric(row.get("close"), errors="coerce"))
            if np.isfinite(close_px):
                pos.peak_close = max(pos.peak_close, close_px) if np.isfinite(pos.peak_close) else max(close_px, pos.entry_price)
            rank_today = float(row.get("rank_today", np.nan))
            current_score = float(row.get("strategy_score", row.get("multifactor_score", np.nan)))
            stop_ref = _stop_for_row(row)
            score_threshold = _score_exit_threshold(pos, cfg)
            no_signal = int(row.get("s1_entry", 0)) + int(row.get("s2_entry", 0)) + int(row.get("s3_long_entry", 0)) == 0
            exit_reason = None
            if pd.notna(stop_ref) and float(row.get("close", np.nan)) < stop_ref:
                exit_reason = "hit_stop"
            elif pos.days_held >= max_hold_days:
                exit_reason = "max_hold"
            elif pos.days_held >= min_hold_days:
                rank_broken = (
                    pd.notna(rank_today)
                    and rank_today > (top_n * exit_rank_mult + turnover_buffer)
                    and current_score < max(score_threshold, pos.entry_score - exit_score_drop * 0.5)
                )
                score_faded = code not in top_codes and current_score < score_threshold
                signal_off = code not in top_codes and no_signal and current_score < signal_exit_score_floor
                trend_broken = code not in top_codes and _trend_breakdown(row)

                if exit_style == "score":
                    if score_faded:
                        exit_reason = "score_fade"
                    elif _trailing_exit(pos, row, cfg):
                        exit_reason = "trail_exit"
                    elif signal_off:
                        exit_reason = "signal_off"
                    elif rank_broken:
                        exit_reason = "rank_drop"
                elif exit_style == "trend":
                    if trend_broken:
                        exit_reason = "trend_break"
                    elif _trailing_exit(pos, row, cfg):
                        exit_reason = "trail_exit"
                    elif score_faded:
                        exit_reason = "score_fade"
                    elif signal_off:
                        exit_reason = "signal_off"
                    elif rank_broken:
                        exit_reason = "rank_drop"
                else:
                    if _trailing_exit(pos, row, cfg):
                        exit_reason = "trail_exit"
                    elif rank_broken:
                        exit_reason = "rank_drop"
                    elif score_faded:
                        exit_reason = "score_fade"
                    elif signal_off:
                        exit_reason = "signal_off"
                    elif trend_broken:
                        exit_reason = "trend_break"

            if exit_reason:
                sells.append((code, exit_reason, pos, row))
            else:
                holds.append(
                    {
                        "signal_date": dt,
                        "execute_date": next_dt,
                        "code": code,
                        "action": "HOLD",
                        "score": float(row.get("strategy_score", row.get("multifactor_score", np.nan))),
                        "strategy": pos.strategy,
                        "reason": str(row.get("composite_reason", "")),
                    }
                )

        for code, reason, pos, row in sells:
            exit_px = float(row.get("next_open", np.nan))
            if not np.isfinite(exit_px):
                continue
            pnl_pct = (exit_px - pos.entry_price) / pos.entry_price - (2 * cost_bp / 10000.0)
            equity *= (1.0 + pnl_pct * pos.target_weight)
            trades.append(
                {
                    "code": code,
                    "strategy": pos.strategy,
                    "entry_signal_date": pos.entry_signal_date,
                    "entry_date": pos.entry_date,
                    "entry_price": pos.entry_price,
                    "exit_date": next_dt,
                    "exit_price": exit_px,
                    "pnl_pct": float(pnl_pct),
                    "days_held": pos.days_held,
                    "entry_score": pos.entry_score,
                    "weight": pos.target_weight,
                    "exit_score": float(row.get("strategy_score", row.get("multifactor_score", np.nan))),
                    "exit_reason": reason,
                }
            )
            orders.append(
                {
                    "signal_date": dt,
                    "execute_date": next_dt,
                    "code": code,
                    "side": "SELL",
                    "price": exit_px,
                    "weight": pos.target_weight,
                    "score": float(row.get("strategy_score", row.get("multifactor_score", np.nan))),
                    "strategy": pos.strategy,
                    "reason": reason,
                }
            )
            positions.pop(code, None)

        available_slots = max(0, top_n - len(positions))
        current_exposure = float(sum(pos.target_weight for pos in positions.values()))
        available_exposure = max(0.0, target_exposure - current_exposure)
        buy_df = top_df[~top_df["code"].astype(str).isin(positions.keys())].head(available_slots).copy()
        if not buy_df.empty:
            buy_df = _allocate_target_weights(buy_df, available_exposure, cfg)
        for _, row in buy_df.iterrows():
            entry_px = float(row.get("next_open", np.nan))
            if not np.isfinite(entry_px):
                continue
            target_weight = float(row.get("target_weight", 0.0))
            if target_weight <= 0:
                continue
            code = str(row["code"])
            strategy = _choose_strategy(row)
            pos = PortfolioPosition(
                code=code,
                entry_signal_date=dt,
                entry_date=next_dt,
                entry_price=entry_px,
                entry_score=float(row.get("strategy_score", row.get("multifactor_score", np.nan))),
                target_weight=target_weight,
                stop_ref=_stop_for_row(row),
                strategy=strategy,
                peak_close=float(row.get("close", entry_px)) if np.isfinite(float(pd.to_numeric(row.get("close"), errors="coerce"))) else entry_px,
            )
            positions[code] = pos
            orders.append(
                {
                    "signal_date": dt,
                    "execute_date": next_dt,
                    "code": code,
                    "side": "BUY",
                    "price": entry_px,
                    "weight": target_weight,
                    "score": float(row.get("strategy_score", row.get("multifactor_score", np.nan))),
                    "strategy": strategy,
                    "reason": str(row.get("composite_reason", "")),
                }
            )

        monitor_rows.append(
            {
                "signal_date": dt,
                "candidate_count": int(len(eligible)),
                "top_n": top_n,
                "held_count": int(len(positions)),
                "target_exposure": target_exposure,
                "realized_exposure": float(sum(pos.target_weight for pos in positions.values())),
                "avg_top_score": float(top_df["strategy_score"].mean()) if not top_df.empty else np.nan,
                "avg_held_score": float(np.mean([p.entry_score for p in positions.values()])) if positions else np.nan,
            }
        )
        equity_rows.append({
            "date": next_dt,
            "equity": equity,
            "positions": int(len(positions)),
            "cash_weight": max(0.0, 1.0 - float(sum(pos.target_weight for pos in positions.values()))),
        })

    orders_df = pd.DataFrame(orders)
    trades_df = pd.DataFrame(trades)
    monitor_df = pd.DataFrame(monitor_rows)
    equity_df = pd.DataFrame(equity_rows)

    latest_day = dates[-2] if len(dates) >= 2 else dates[-1]
    latest_candidates = _eligible_universe(panel[panel["date"] == latest_day].copy(), cfg).head(top_n).copy()
    latest_target_exposure = _target_total_exposure(latest_candidates, cfg)
    latest_candidates = _allocate_target_weights(latest_candidates, latest_target_exposure, cfg)
    if latest_candidates.empty:
        latest_candidates["recommended_action"] = pd.Series(dtype="object")
        latest_candidates["strategy"] = pd.Series(dtype="object")
    else:
        latest_candidates["recommended_action"] = np.where(
            latest_candidates["code"].astype(str).isin(positions.keys()),
            "HOLD",
            "BUY",
        )
        latest_candidates["strategy"] = latest_candidates.apply(_choose_strategy, axis=1)
    latest_positions = pd.DataFrame(
        [
            {
                "code": p.code,
                "entry_date": p.entry_date,
                "entry_price": p.entry_price,
                "entry_score": p.entry_score,
                "target_weight": p.target_weight,
                "strategy": p.strategy,
                "days_held": p.days_held,
            }
            for p in positions.values()
        ]
    )

    return panel, trades_df, monitor_df, equity_df, orders_df, latest_candidates, latest_positions


def summarize_portfolio_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["metric", "value"])
    values = {
        "trades": int(len(trades)),
        "hit_rate": float((trades["pnl_pct"] > 0).mean()),
        "expectancy": float(trades["pnl_pct"].mean()),
        "avg_hold_days": float(trades["days_held"].mean()),
        "avg_entry_score": float(trades["entry_score"].mean()),
    }
    return pd.DataFrame({"metric": list(values.keys()), "value": list(values.values())})


def run_portfolio_daily(
    all_csvs: List[str],
    cfg_path: Optional[str] = None,
    cfg_overrides: Optional[dict] = None,
    outdir: str = "./output",
):
    os.makedirs(outdir, exist_ok=True)
    cfg = merge_config(load_config(cfg_path), cfg_overrides)
    df_all = load_market_csv_multi(all_csvs)
    df_ind = compute_indicator_panel(df_all, cfg)
    panel = build_signal_panel(df_ind, cfg, by_bucket=False)
    panel = add_multifactor_scores(panel, cfg)
    panel, trades, monitor, equity, orders, latest_candidates, latest_positions = simulate_multifactor_portfolio(panel, cfg)

    panel.to_csv(os.path.join(outdir, "portfolio_signal_panel.csv"), index=False, encoding="utf-8-sig")
    trades.to_csv(os.path.join(outdir, "portfolio_backtest_trades.csv"), index=False, encoding="utf-8-sig")
    monitor.to_csv(os.path.join(outdir, "portfolio_backtest_daily_monitor.csv"), index=False, encoding="utf-8-sig")
    equity.to_csv(os.path.join(outdir, "portfolio_equity_curve.csv"), index=False, encoding="utf-8-sig")
    orders.to_csv(os.path.join(outdir, "portfolio_orders_next_open.csv"), index=False, encoding="utf-8-sig")
    latest_candidates.to_csv(os.path.join(outdir, "portfolio_latest_candidates.csv"), index=False, encoding="utf-8-sig")
    latest_positions.to_csv(os.path.join(outdir, "portfolio_positions_state.csv"), index=False, encoding="utf-8-sig")
    summarize_portfolio_trades(trades).to_csv(os.path.join(outdir, "portfolio_summary.csv"), index=False, encoding="utf-8-sig")
    return {
        "panel": panel,
        "trades": trades,
        "monitor": monitor,
        "equity": equity,
        "orders": orders,
        "latest_candidates": latest_candidates,
        "latest_positions": latest_positions,
    }


def _daily_strategy_from_regime_label(regime_label: str) -> str:
    if regime_label == "weak_range_regime":
        return "pullback_quality_combo"
    if regime_label == "pullback_regime":
        return "pullback"
    if regime_label == "quality_trend_regime":
        return "quality_trend"
    return "pullback"


def _build_net_daily_actions(
    latest_positions: pd.DataFrame,
    latest_candidates: pd.DataFrame,
    panel: pd.DataFrame,
    as_of_date: pd.Timestamp,
    strategy_profile: str,
    top_n: int = 3,
) -> pd.DataFrame:
    current_close = {}
    as_of_rows = panel[pd.to_datetime(panel["date"]) == pd.Timestamp(as_of_date)].copy()
    if not as_of_rows.empty:
        current_close = {
            str(row["code"]): float(pd.to_numeric(row.get("close"), errors="coerce"))
            for _, row in as_of_rows.iterrows()
        }

    pos_map = {}
    if latest_positions is not None and not latest_positions.empty:
        for _, row in latest_positions.iterrows():
            pos_map[str(row["code"])] = row

    cand_map = {}
    if latest_candidates is not None and not latest_candidates.empty:
        for _, row in latest_candidates.iterrows():
            code = str(row["code"])
            cand_map[code] = row

    current_codes = list(pos_map.keys())
    hold_codes = current_codes.copy()
    available_slots = max(0, int(top_n) - len(current_codes))
    candidate_codes = [str(x) for x in latest_candidates["code"].tolist()] if latest_candidates is not None and not latest_candidates.empty else []
    buy_codes = [code for code in candidate_codes if code not in pos_map][:available_slots]

    rows = []
    for code in hold_codes:
        pos = pos_map[code]
        cand = cand_map.get(code, pd.Series(dtype="object"))
        close_px = current_close.get(code, np.nan)
        entry_px = float(pd.to_numeric(pos.get("entry_price"), errors="coerce"))
        unrealized = close_px / entry_px - 1.0 if np.isfinite(close_px) and np.isfinite(entry_px) and entry_px > 0 else np.nan
        rows.append(
            {
                "as_of_date": pd.Timestamp(as_of_date).date(),
                "action": "HOLD",
                "code": code,
                "target_weight": float(pd.to_numeric(cand.get("target_weight", pos.get("target_weight")), errors="coerce")),
                "strategy": str(cand.get("strategy", pos.get("strategy", ""))),
                "strategy_profile": strategy_profile,
                "entry_date": pos.get("entry_date"),
                "entry_price": entry_px,
                "current_price": close_px,
                "pnl_pct": unrealized,
                "reason": "existing_position_hold",
            }
        )
    for code in buy_codes:
        cand = cand_map[code]
        rows.append(
            {
                "as_of_date": pd.Timestamp(as_of_date).date(),
                "action": "BUY",
                "code": code,
                "target_weight": float(pd.to_numeric(cand.get("target_weight"), errors="coerce")),
                "strategy": str(cand.get("strategy", "")),
                "strategy_profile": strategy_profile,
                "entry_date": pd.NaT,
                "entry_price": np.nan,
                "current_price": float(pd.to_numeric(cand.get("close"), errors="coerce")),
                "pnl_pct": np.nan,
                "reason": str(cand.get("composite_reason", "")),
            }
        )
    action_order = {"SELL": 0, "HOLD": 1, "BUY": 2}
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["action_order"] = out["action"].map(action_order).fillna(9)
    out = out.sort_values(["action_order", "target_weight"], ascending=[True, False]).drop(columns=["action_order"])
    return out


def _build_action_history(
    orders: pd.DataFrame,
    trades: pd.DataFrame,
    daily_actions: pd.DataFrame,
    as_of_date: pd.Timestamp,
    recent_days: int = 10,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    orders = orders.copy() if orders is not None else pd.DataFrame()
    trades = trades.copy() if trades is not None else pd.DataFrame()
    daily_actions = daily_actions.copy() if daily_actions is not None else pd.DataFrame()

    if start_date:
        start = pd.Timestamp(start_date)
    else:
        start = pd.Timestamp(as_of_date) - pd.Timedelta(days=recent_days)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp(as_of_date)

    if not orders.empty:
        orders["action_date"] = pd.to_datetime(orders["execute_date"], errors="coerce")
        orders["action"] = orders["side"].astype(str).str.upper()
        sell_trade_map = {}
        if not trades.empty:
            trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
            for _, row in trades.iterrows():
                sell_trade_map[(str(row["code"]), pd.Timestamp(row["exit_date"]))] = float(pd.to_numeric(row.get("pnl_pct"), errors="coerce"))
        orders["pnl_pct"] = orders.apply(
            lambda row: sell_trade_map.get((str(row["code"]), pd.Timestamp(row["action_date"])), np.nan)
            if str(row["action"]).upper() == "SELL"
            else np.nan,
            axis=1,
        )
        orders_hist = orders[(orders["action_date"] >= start) & (orders["action_date"] <= end)].copy()
        orders_hist = orders_hist.rename(columns={"price": "current_price", "weight": "target_weight"})
        keep_cols = ["action_date", "action", "code", "target_weight", "strategy", "reason", "pnl_pct"]
        orders_hist = orders_hist[keep_cols]
    else:
        orders_hist = pd.DataFrame(columns=["action_date", "action", "code", "target_weight", "strategy", "reason", "pnl_pct"])

    if not daily_actions.empty:
        daily_actions["action_date"] = pd.to_datetime(daily_actions["as_of_date"], errors="coerce")
        hold_hist = daily_actions[(daily_actions["action"] == "HOLD") & (daily_actions["action_date"] >= start) & (daily_actions["action_date"] <= end)].copy()
        hold_hist = hold_hist.rename(columns={"current_price": "current_price"})
        hold_hist = hold_hist[["action_date", "action", "code", "target_weight", "strategy", "reason", "pnl_pct"]]
    else:
        hold_hist = pd.DataFrame(columns=["action_date", "action", "code", "target_weight", "strategy", "reason", "pnl_pct"])

    out = pd.concat([orders_hist, hold_hist], ignore_index=True)
    if out.empty:
        return out
    action_order = {"SELL": 0, "HOLD": 1, "BUY": 2}
    out["action_order"] = out["action"].map(action_order).fillna(9)
    out = out.sort_values(["action_date", "action_order", "code"], ascending=[False, True, True]).drop(columns=["action_order"])
    return out


def run_portfolio_regime_daily(
    all_csvs: List[str],
    cfg_path: Optional[str] = None,
    cfg_overrides: Optional[dict] = None,
    outdir: str = "./output",
    regime_lookback_months: int = 3,
    action_recent_days: int = 10,
    action_start: Optional[str] = None,
    action_end: Optional[str] = None,
):
    os.makedirs(outdir, exist_ok=True)
    cfg = merge_config(load_config(cfg_path), cfg_overrides)
    df_all = load_market_csv_multi(all_csvs)
    df_ind = compute_indicator_panel(df_all, cfg)
    panel = build_signal_panel(df_ind, cfg, by_bucket=False)
    panel = add_multifactor_scores(panel, cfg)

    latest_date = pd.to_datetime(panel["date"].max())
    lookback_start = latest_date - pd.DateOffset(months=regime_lookback_months)
    recent_panel = panel[panel["date"] >= lookback_start].copy()
    features = _window_feature_vector(recent_panel)
    regime_label = _label_window_regime(features)
    strategy_profile = _daily_strategy_from_regime_label(regime_label)

    profile_cfg = merge_config(
        cfg,
        {
            "portfolio": {
                "strategy_profile": strategy_profile,
                "top_n": 3,
                "min_score": 58.0,
                "min_hold_days": 15,
                "max_hold_days": 9999,
                "exit_rank_mult": 9.0 if strategy_profile == "pullback" else 8.5,
                "turnover_buffer": 18.0 if strategy_profile == "pullback" else 17.0,
                "base_total_exposure": 0.40,
                "max_total_exposure": 0.60,
                "max_position_weight": 0.26 if strategy_profile == "pullback" else 0.25,
            }
        },
    )

    panel_out, trades, monitor, equity, orders, latest_candidates, latest_positions = simulate_multifactor_portfolio(panel, profile_cfg)
    summary_df = summarize_portfolio_trades(trades)
    regime_df = pd.DataFrame(
        [
            {
                "as_of_date": latest_date.date(),
                "lookback_start": lookback_start.date(),
                "lookback_months": regime_lookback_months,
                "regime_label": regime_label,
                "strategy_profile": strategy_profile,
                **features,
            }
        ]
    )

    actions_df = _build_net_daily_actions(
        latest_positions=latest_positions,
        latest_candidates=latest_candidates,
        panel=panel_out,
        as_of_date=latest_date,
        strategy_profile=strategy_profile,
    )
    history_df = _build_action_history(
        orders=orders,
        trades=trades,
        daily_actions=actions_df,
        as_of_date=latest_date,
        recent_days=action_recent_days,
        start_date=action_start,
        end_date=action_end,
    )

    regime_df.to_csv(os.path.join(outdir, "portfolio_daily_regime.csv"), index=False, encoding="utf-8-sig")
    actions_df.to_csv(os.path.join(outdir, "portfolio_daily_actions.csv"), index=False, encoding="utf-8-sig")
    history_df.to_csv(os.path.join(outdir, "portfolio_action_history.csv"), index=False, encoding="utf-8-sig")
    panel_out.to_csv(os.path.join(outdir, "portfolio_signal_panel.csv"), index=False, encoding="utf-8-sig")
    trades.to_csv(os.path.join(outdir, "portfolio_backtest_trades.csv"), index=False, encoding="utf-8-sig")
    monitor.to_csv(os.path.join(outdir, "portfolio_backtest_daily_monitor.csv"), index=False, encoding="utf-8-sig")
    equity.to_csv(os.path.join(outdir, "portfolio_equity_curve.csv"), index=False, encoding="utf-8-sig")
    orders.to_csv(os.path.join(outdir, "portfolio_orders_next_open.csv"), index=False, encoding="utf-8-sig")
    latest_candidates.to_csv(os.path.join(outdir, "portfolio_latest_candidates.csv"), index=False, encoding="utf-8-sig")
    latest_positions.to_csv(os.path.join(outdir, "portfolio_positions_state.csv"), index=False, encoding="utf-8-sig")
    summary_df.to_csv(os.path.join(outdir, "portfolio_summary.csv"), index=False, encoding="utf-8-sig")
    return {
        "regime": regime_df,
        "actions": actions_df,
        "history": history_df,
        "latest_candidates": latest_candidates,
        "latest_positions": latest_positions,
        "orders": orders,
        "summary": summary_df,
    }


def _month_add(dt64: np.datetime64, months: int) -> np.datetime64:
    ts = pd.Timestamp(dt64)
    y = ts.year + (ts.month - 1 + months) // 12
    m = (ts.month - 1 + months) % 12 + 1
    d = min(ts.day, [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return np.datetime64(pd.Timestamp(year=y, month=m, day=d).date())


def _generate_windows(dates: pd.Series, train_m: int, val_m: int, test_m: int, step_m: int):
    start = pd.to_datetime(dates.min()).to_datetime64()
    end = pd.to_datetime(dates.max()).to_datetime64()
    windows = []
    cur = start
    while True:
        tr_e = _month_add(cur, train_m)
        vl_e = _month_add(tr_e, val_m)
        te_e = _month_add(vl_e, test_m)
        if te_e > end:
            break
        windows.append((cur, tr_e, tr_e, vl_e, vl_e, te_e))
        cur = _month_add(cur, step_m)
    return windows


def _slice_panel(panel: pd.DataFrame, start: np.datetime64, end: np.datetime64) -> pd.DataFrame:
    return panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] < pd.Timestamp(end))].copy()


def _slice_df(df: pd.DataFrame, start: np.datetime64, end: np.datetime64) -> pd.DataFrame:
    return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] < pd.Timestamp(end))].copy()


def _portfolio_variants(base_cfg: dict) -> List[Tuple[str, dict]]:
    variant_set = str(_portfolio_cfg(base_cfg).get("variant_set", "full"))
    meta_compare = [
        ("strat_pullback_longhold", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 9.0, "turnover_buffer": 18.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
        ("strat_quality_trend_longhold", {"portfolio": {"strategy_profile": "quality_trend", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 8.0, "turnover_buffer": 16.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.24}}),
        ("strat_pullback_quality_combo", {"portfolio": {"strategy_profile": "pullback_quality_combo", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 8.5, "turnover_buffer": 17.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.25}}),
        ("strat_meta_switch_rsi_vol", {"portfolio": {"strategy_profile": "meta_switch_rsi_vol", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 8.5, "turnover_buffer": 17.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.25}}),
    ]
    regime_compare = [
        ("strat_pullback_longhold", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 9.0, "turnover_buffer": 18.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
        ("strat_quality_trend_longhold", {"portfolio": {"strategy_profile": "quality_trend", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 8.0, "turnover_buffer": 16.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.24}}),
        ("strat_pullback_quality_combo", {"portfolio": {"strategy_profile": "pullback_quality_combo", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 8.5, "turnover_buffer": 17.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.25}}),
        ("strat_regime_switch", {"portfolio": {"strategy_profile": "regime_switch", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 8.5, "turnover_buffer": 17.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.25}}),
    ]
    if variant_set == "meta_compare":
        return meta_compare
    combo_compare = [
        ("strat_pullback_longhold", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 9.0, "turnover_buffer": 18.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
        ("strat_quality_trend_longhold", {"portfolio": {"strategy_profile": "quality_trend", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 8.0, "turnover_buffer": 16.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.24}}),
        ("strat_pullback_quality_combo", {"portfolio": {"strategy_profile": "pullback_quality_combo", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 8.5, "turnover_buffer": 17.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.25}}),
    ]
    if variant_set == "regime_compare":
        return regime_compare
    pullback_compare = [
        ("strat_pullback_hold15", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 7, "max_hold_days": 15, "exit_rank_mult": 5.5, "turnover_buffer": 9.0, "exit_score_floor": 61.0, "exit_score_drop": 11.0, "signal_exit_score_floor": 64.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
        ("strat_pullback_hold30", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 10, "max_hold_days": 30, "exit_rank_mult": 7.0, "turnover_buffer": 12.0, "exit_score_floor": 59.0, "exit_score_drop": 13.0, "signal_exit_score_floor": 62.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
        ("strat_pullback_hold45", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 45, "exit_rank_mult": 8.0, "turnover_buffer": 15.0, "exit_score_floor": 58.0, "exit_score_drop": 15.0, "signal_exit_score_floor": 61.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
    ]
    if variant_set == "combo_compare":
        return combo_compare
    if variant_set == "pullback_compare":
        return pullback_compare
    if variant_set == "pullback_exit_compare":
        return [
            ("strat_pullback_rank_exit", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 45, "exit_style": "rank", "exit_rank_mult": 8.0, "turnover_buffer": 15.0, "exit_score_floor": 58.0, "exit_score_drop": 15.0, "signal_exit_score_floor": 61.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
            ("strat_pullback_score_exit", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 45, "exit_style": "score", "exit_rank_mult": 10.0, "turnover_buffer": 20.0, "exit_score_floor": 61.0, "exit_score_drop": 10.0, "signal_exit_score_floor": 64.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
            ("strat_pullback_trend_exit", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 45, "exit_style": "trend", "exit_rank_mult": 12.0, "turnover_buffer": 24.0, "exit_score_floor": 59.0, "exit_score_drop": 12.0, "signal_exit_score_floor": 63.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
        ]
    if variant_set == "pullback_opt_compare":
        return [
            ("strat_pullback_rank_exit", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 45, "exit_style": "rank", "exit_rank_mult": 8.0, "turnover_buffer": 15.0, "exit_score_floor": 58.0, "exit_score_drop": 15.0, "signal_exit_score_floor": 61.0, "trail_min_open_profit": 1.0, "trail_giveback_pct": 1.0, "trail_atr_buffer": 99.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
            ("strat_pullback_rank_trail", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 45, "exit_style": "rank", "exit_rank_mult": 8.5, "turnover_buffer": 16.0, "exit_score_floor": 57.0, "exit_score_drop": 16.0, "signal_exit_score_floor": 60.0, "trail_min_open_profit": 0.10, "trail_giveback_pct": 0.06, "trail_atr_buffer": 1.1, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
        ]
    return [
        ("strat_blended", {"portfolio": {"strategy_profile": "blended", "top_n": 3, "min_score": 58.0, "min_hold_days": 7, "max_hold_days": 20, "exit_rank_mult": 4.5, "turnover_buffer": 6.0, "base_total_exposure": 0.45, "max_total_exposure": 0.65, "max_position_weight": 0.28}}),
        ("strat_blended_longhold", {"portfolio": {"strategy_profile": "blended", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 8.0, "turnover_buffer": 16.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
        ("strat_breakout", {"portfolio": {"strategy_profile": "breakout", "top_n": 3, "min_score": 60.0, "min_hold_days": 7, "max_hold_days": 18, "exit_rank_mult": 5.0, "turnover_buffer": 6.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.30}}),
        ("strat_pullback", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 10, "max_hold_days": 24, "exit_rank_mult": 5.0, "turnover_buffer": 8.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
        ("strat_pullback_hold15", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 7, "max_hold_days": 15, "exit_rank_mult": 5.5, "turnover_buffer": 9.0, "exit_score_floor": 61.0, "exit_score_drop": 11.0, "signal_exit_score_floor": 64.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
        ("strat_pullback_hold20", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 10, "max_hold_days": 20, "exit_rank_mult": 6.0, "turnover_buffer": 10.0, "exit_score_floor": 60.0, "exit_score_drop": 12.0, "signal_exit_score_floor": 63.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
        ("strat_pullback_hold30", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 10, "max_hold_days": 30, "exit_rank_mult": 7.0, "turnover_buffer": 12.0, "exit_score_floor": 59.0, "exit_score_drop": 13.0, "signal_exit_score_floor": 62.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
        ("strat_pullback_hold45", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 45, "exit_rank_mult": 8.0, "turnover_buffer": 15.0, "exit_score_floor": 58.0, "exit_score_drop": 15.0, "signal_exit_score_floor": 61.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
        ("strat_pullback_longhold", {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 9.0, "turnover_buffer": 18.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}}),
        ("strat_mid_momo", {"portfolio": {"strategy_profile": "mid_momo", "top_n": 3, "min_score": 58.0, "min_hold_days": 7, "max_hold_days": 20, "exit_rank_mult": 5.0, "turnover_buffer": 7.0, "base_total_exposure": 0.45, "max_total_exposure": 0.65, "max_position_weight": 0.28}}),
        ("strat_quality_trend", {"portfolio": {"strategy_profile": "quality_trend", "top_n": 3, "min_score": 58.0, "min_hold_days": 10, "max_hold_days": 24, "exit_rank_mult": 5.5, "turnover_buffer": 8.0, "base_total_exposure": 0.45, "max_total_exposure": 0.65, "max_position_weight": 0.26}}),
        ("strat_quality_trend_longhold", {"portfolio": {"strategy_profile": "quality_trend", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 8.0, "turnover_buffer": 16.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.24}}),
        ("strat_value_trend", {"portfolio": {"strategy_profile": "value_trend", "top_n": 3, "min_score": 56.0, "min_hold_days": 10, "max_hold_days": 24, "exit_rank_mult": 5.0, "turnover_buffer": 7.0, "base_total_exposure": 0.35, "max_total_exposure": 0.55, "max_position_weight": 0.24}}),
    ]


def _score_variant(trades: pd.DataFrame) -> float:
    if trades is None or trades.empty:
        return -1e9
    hit_rate = float((trades["pnl_pct"] > 0).mean())
    expectancy = float(trades["pnl_pct"].mean())
    trade_count = len(trades)
    return hit_rate * 100.0 + expectancy * 1000.0 + min(trade_count, 300) * 0.01


def _meta_model_base_variants(base_cfg: dict) -> List[Tuple[str, dict]]:
    return [
        ("strat_pullback_longhold", merge_config(base_cfg, {"portfolio": {"strategy_profile": "pullback", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 9.0, "turnover_buffer": 18.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.26}})),
        ("strat_quality_trend_longhold", merge_config(base_cfg, {"portfolio": {"strategy_profile": "quality_trend", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 8.0, "turnover_buffer": 16.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.24}})),
        ("strat_pullback_quality_combo", merge_config(base_cfg, {"portfolio": {"strategy_profile": "pullback_quality_combo", "top_n": 3, "min_score": 58.0, "min_hold_days": 15, "max_hold_days": 9999, "exit_rank_mult": 8.5, "turnover_buffer": 17.0, "base_total_exposure": 0.40, "max_total_exposure": 0.60, "max_position_weight": 0.25}})),
    ]


def _window_feature_vector(panel: pd.DataFrame) -> dict:
    if panel is None or panel.empty:
        return {
            "trend_share": 0.0,
            "range_share": 0.0,
            "neutral_share": 1.0,
            "median_rsi14": 50.0,
            "median_rvol20": 1.0,
            "median_adx14": 20.0,
            "median_atr_pct": 0.03,
            "median_roc20": 0.0,
            "mean_signal_score": 50.0,
            "mean_quality_score": 50.0,
        }
    daily = panel.groupby("date").agg(
        state=("market_state_index", lambda s: s.mode().iloc[0] if not s.dropna().empty else "neutral"),
        median_rsi14=("rsi14", "median"),
        median_rvol20=("rvol20", "median"),
        median_adx14=("adx14", "median"),
        median_atr_pct=("atr_pct", "median"),
        median_roc20=("roc20", "median"),
        mean_signal_score=("signal_score", "mean"),
        mean_quality_score=("quality_score", "mean"),
    ).reset_index(drop=True)
    state_share = daily["state"].value_counts(normalize=True)
    return {
        "trend_share": float(state_share.get("trend_ok", 0.0)),
        "range_share": float(state_share.get("range_bias", 0.0)),
        "neutral_share": float(state_share.get("neutral", 0.0)),
        "median_rsi14": float(pd.to_numeric(daily["median_rsi14"], errors="coerce").median()),
        "median_rvol20": float(pd.to_numeric(daily["median_rvol20"], errors="coerce").median()),
        "median_adx14": float(pd.to_numeric(daily["median_adx14"], errors="coerce").median()),
        "median_atr_pct": float(pd.to_numeric(daily["median_atr_pct"], errors="coerce").median()),
        "median_roc20": float(pd.to_numeric(daily["median_roc20"], errors="coerce").median()),
        "mean_signal_score": float(pd.to_numeric(daily["mean_signal_score"], errors="coerce").mean()),
        "mean_quality_score": float(pd.to_numeric(daily["mean_quality_score"], errors="coerce").mean()),
    }


def _pick_variant_by_meta_model(history: List[dict], features: dict, fallback_name: str) -> str:
    if len(history) < 6:
        return fallback_name
    cols = list(features.keys())
    hist_df = pd.DataFrame(history)
    feat_df = hist_df[cols].apply(pd.to_numeric, errors="coerce")
    med = feat_df.median()
    scale = (feat_df.quantile(0.75) - feat_df.quantile(0.25)).replace(0, 1.0).fillna(1.0)
    x = ((pd.Series(features)[cols] - med) / scale).astype(float)
    z = (feat_df.sub(med, axis=1)).div(scale, axis=1).fillna(0.0)
    dists = ((z - x) ** 2).sum(axis=1) ** 0.5
    work = hist_df.copy()
    work["dist"] = dists.values
    work = work.sort_values(["dist", "label_score"], ascending=[True, False]).head(min(5, len(work)))
    vote = {}
    for _, row in work.iterrows():
        weight = 1.0 / max(float(row["dist"]), 0.05)
        vote[row["label"]] = vote.get(row["label"], 0.0) + weight
    if not vote:
        return fallback_name
    return max(vote.items(), key=lambda kv: kv[1])[0]


def _label_window_regime(features: dict) -> str:
    trend_share = float(features.get("trend_share", 0.0))
    range_share = float(features.get("range_share", 0.0))
    median_rsi14 = float(features.get("median_rsi14", 50.0))
    median_rvol20 = float(features.get("median_rvol20", 1.0))
    mean_quality_score = float(features.get("mean_quality_score", 50.0))
    mean_signal_score = float(features.get("mean_signal_score", 50.0))

    if trend_share >= 0.45 and median_rsi14 >= 56.0 and mean_quality_score >= 55.0:
        return "quality_trend_regime"
    if trend_share >= 0.30 and 49.0 <= median_rsi14 <= 58.0 and median_rvol20 <= 1.20:
        return "pullback_regime"
    if trend_share >= 0.20 and median_rsi14 >= 52.0 and median_rvol20 >= 0.95 and mean_signal_score >= 50.0:
        return "combo_regime"
    if range_share >= 0.25 or median_rsi14 < 48.0:
        return "weak_range_regime"
    return "mixed_transition"


def run_portfolio_regime_analysis(
    all_csvs: List[str],
    cfg_path: Optional[str] = None,
    cfg_overrides: Optional[dict] = None,
    outdir: str = "./output",
    train_months: int = 24,
    val_months: int = 3,
    test_months: int = 6,
    step_months: int = 1,
):
    os.makedirs(outdir, exist_ok=True)
    base_cfg = merge_config(load_config(cfg_path), cfg_overrides)
    df_all = load_market_csv_multi(all_csvs)
    df_ind = compute_indicator_panel(df_all, base_cfg)
    panel = build_signal_panel(df_ind, base_cfg, by_bucket=False)
    panel = add_multifactor_scores(panel, base_cfg)
    windows = _generate_windows(panel["date"], train_months, val_months, test_months, step_months)
    if not windows:
        raise RuntimeError("portfolio regime analysis: insufficient windows")

    base_variants = _meta_model_base_variants(base_cfg)
    variant_rows = []
    window_rows = []
    for window_id, (_, _, vl_s, vl_e, te_s, te_e) in enumerate(windows, start=1):
        val_panel = _slice_panel(panel, vl_s, vl_e)
        test_panel = _slice_panel(panel, te_s, te_e)
        features = _window_feature_vector(val_panel)
        regime_label = _label_window_regime(features)
        scored_variants = []
        for name, cfg_variant in base_variants:
            _, trades, _, _, _, _, _ = simulate_multifactor_portfolio(test_panel, cfg_variant)
            score = _score_variant(trades)
            hit_rate = float((trades["pnl_pct"] > 0).mean()) if not trades.empty else np.nan
            expectancy = float(trades["pnl_pct"].mean()) if not trades.empty else np.nan
            trade_count = int(len(trades))
            avg_hold_days = float(trades["days_held"].mean()) if not trades.empty else np.nan
            variant_rows.append(
                {
                    "window_id": window_id,
                    "variant": name,
                    "val_start": pd.Timestamp(vl_s).date(),
                    "val_end": pd.Timestamp(vl_e).date(),
                    "test_start": pd.Timestamp(te_s).date(),
                    "test_end": pd.Timestamp(te_e).date(),
                    "regime_label": regime_label,
                    **features,
                    "score": score,
                    "hit_rate": hit_rate,
                    "expectancy": expectancy,
                    "trade_count": trade_count,
                    "avg_hold_days": avg_hold_days,
                }
            )
            scored_variants.append((name, score, hit_rate, expectancy, trade_count, avg_hold_days))
        best_name, best_score, best_hit_rate, best_expectancy, best_trade_count, best_avg_hold = max(scored_variants, key=lambda x: x[1])
        window_rows.append(
            {
                "window_id": window_id,
                "val_start": pd.Timestamp(vl_s).date(),
                "val_end": pd.Timestamp(vl_e).date(),
                "test_start": pd.Timestamp(te_s).date(),
                "test_end": pd.Timestamp(te_e).date(),
                "regime_label": regime_label,
                **features,
                "best_variant": best_name,
                "best_score": best_score,
                "best_hit_rate": best_hit_rate,
                "best_expectancy": best_expectancy,
                "best_trade_count": best_trade_count,
                "best_avg_hold_days": best_avg_hold,
            }
        )

    variant_df = pd.DataFrame(variant_rows)
    window_df = pd.DataFrame(window_rows)
    stability = (
        variant_df.groupby(["regime_label", "variant"])[["hit_rate", "expectancy", "trade_count", "avg_hold_days", "score"]]
        .mean()
        .reset_index()
    )
    win_counts = (
        window_df.groupby(["regime_label", "best_variant"])
        .size()
        .reset_index(name="win_count")
    )
    regime_summary = stability.merge(
        win_counts,
        left_on=["regime_label", "variant"],
        right_on=["regime_label", "best_variant"],
        how="left",
    ).drop(columns=["best_variant"])
    regime_summary["win_count"] = regime_summary["win_count"].fillna(0).astype(int)
    regime_summary = regime_summary.sort_values(["regime_label", "win_count", "expectancy"], ascending=[True, False, False])

    variant_df.to_csv(os.path.join(outdir, "portfolio_regime_variant_results.csv"), index=False, encoding="utf-8-sig")
    window_df.to_csv(os.path.join(outdir, "portfolio_regime_windows.csv"), index=False, encoding="utf-8-sig")
    regime_summary.to_csv(os.path.join(outdir, "portfolio_regime_summary.csv"), index=False, encoding="utf-8-sig")
    return regime_summary


def run_portfolio_walkforward(
    all_csvs: List[str],
    cfg_path: Optional[str] = None,
    cfg_overrides: Optional[dict] = None,
    outdir: str = "./output",
    train_months: int = 36,
    val_months: int = 3,
    test_months: int = 12,
    step_months: int = 12,
):
    os.makedirs(outdir, exist_ok=True)
    base_cfg = merge_config(load_config(cfg_path), cfg_overrides)
    df_all = load_market_csv_multi(all_csvs)
    df_ind = compute_indicator_panel(df_all, base_cfg)
    panel = build_signal_panel(df_ind, base_cfg, by_bucket=False)
    panel = add_multifactor_scores(panel, base_cfg)
    windows = _generate_windows(panel["date"], train_months, val_months, test_months, step_months)
    if not windows:
        raise RuntimeError("portfolio walkforward: insufficient windows")

    metrics = []
    best_outputs = None
    variant_set = str(_portfolio_cfg(base_cfg).get("variant_set", "full"))
    if variant_set == "meta_model_compare":
        selector_history: List[dict] = []
        base_variants = _meta_model_base_variants(base_cfg)
        for window_id, (_, _, vl_s, vl_e, te_s, te_e) in enumerate(windows, start=1):
            val_panel = _slice_panel(panel, vl_s, vl_e)
            test_panel = _slice_panel(panel, te_s, te_e)
            test_df_ind = _slice_df(df_ind, te_s, te_e)
            baseline_idx_state = build_index_state_from_panel(test_df_ind, base_cfg, by_bucket=False)
            baseline_data = {str(code): sub.copy() for code, sub in test_df_ind.groupby("code")}
            _, baseline_trades, _, _ = backtest_simple(baseline_data, baseline_idx_state, base_cfg)
            baseline_hit = float((baseline_trades["pnl_pct"] > 0).mean()) if not baseline_trades.empty else np.nan
            baseline_exp = float(baseline_trades["pnl_pct"].mean()) if not baseline_trades.empty else np.nan
            baseline_cnt = int(len(baseline_trades))

            val_scores = []
            test_results = {}
            for name, cfg_variant in base_variants:
                _, val_trades, _, _, _, _, _ = simulate_multifactor_portfolio(val_panel, cfg_variant)
                val_score = _score_variant(val_trades)
                val_scores.append((name, val_score))
                _, trades, monitor, equity, orders, latest_candidates, latest_positions = simulate_multifactor_portfolio(test_panel, cfg_variant)
                test_results[name] = (trades, monitor, equity, orders, latest_candidates, latest_positions)
                metrics.append(
                    {
                        "window_id": window_id,
                        "variant": name,
                        "selected_variant": name,
                        "val_start": pd.Timestamp(vl_s).date(),
                        "val_end": pd.Timestamp(vl_e).date(),
                        "test_start": pd.Timestamp(te_s).date(),
                        "test_end": pd.Timestamp(te_e).date(),
                        "hit_rate": float((trades["pnl_pct"] > 0).mean()) if not trades.empty else np.nan,
                        "expectancy": float(trades["pnl_pct"].mean()) if not trades.empty else np.nan,
                        "trade_count": int(len(trades)),
                        "avg_hold_days": float(trades["days_held"].mean()) if not trades.empty else np.nan,
                        "baseline_hit_rate": baseline_hit,
                        "baseline_expectancy": baseline_exp,
                        "baseline_trade_count": baseline_cnt,
                    }
                )

            fallback_name = max(val_scores, key=lambda x: x[1])[0]
            features = _window_feature_vector(val_panel)
            chosen_name = _pick_variant_by_meta_model(selector_history, features, fallback_name)
            trades, monitor, equity, orders, latest_candidates, latest_positions = test_results[chosen_name]
            metrics.append(
                {
                    "window_id": window_id,
                    "variant": "strat_meta_model_selector",
                    "selected_variant": chosen_name,
                    "val_start": pd.Timestamp(vl_s).date(),
                    "val_end": pd.Timestamp(vl_e).date(),
                    "test_start": pd.Timestamp(te_s).date(),
                    "test_end": pd.Timestamp(te_e).date(),
                    "hit_rate": float((trades["pnl_pct"] > 0).mean()) if not trades.empty else np.nan,
                    "expectancy": float(trades["pnl_pct"].mean()) if not trades.empty else np.nan,
                    "trade_count": int(len(trades)),
                    "avg_hold_days": float(trades["days_held"].mean()) if not trades.empty else np.nan,
                    "baseline_hit_rate": baseline_hit,
                    "baseline_expectancy": baseline_exp,
                    "baseline_trade_count": baseline_cnt,
                }
            )
            label_name = max(
                (
                    (name, _score_variant(test_results[name][0]), float(test_results[name][0]["pnl_pct"].mean()) if not test_results[name][0].empty else -1e9)
                    for name, _ in base_variants
                ),
                key=lambda x: x[1],
            )[0]
            selector_history.append({**features, "label": label_name, "label_score": max((_score_variant(test_results[name][0]) for name, _ in base_variants), default=-1e9)})
            best_outputs = (trades, monitor, equity, orders, latest_candidates, latest_positions)

        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(outdir, "portfolio_walkforward_metrics.csv"), index=False, encoding="utf-8-sig")
        summary = (
            metrics_df.groupby("variant")[["hit_rate", "expectancy", "trade_count", "avg_hold_days", "baseline_hit_rate", "baseline_expectancy", "baseline_trade_count"]]
            .mean()
            .reset_index()
            .sort_values(["hit_rate", "expectancy"], ascending=[False, False])
        )
        summary.to_csv(os.path.join(outdir, "portfolio_walkforward_summary.csv"), index=False, encoding="utf-8-sig")

        if best_outputs is not None:
            trades, monitor, equity, orders, latest_candidates, latest_positions = best_outputs
            trades.to_csv(os.path.join(outdir, "portfolio_backtest_trades.csv"), index=False, encoding="utf-8-sig")
            monitor.to_csv(os.path.join(outdir, "portfolio_backtest_daily_monitor.csv"), index=False, encoding="utf-8-sig")
            equity.to_csv(os.path.join(outdir, "portfolio_equity_curve.csv"), index=False, encoding="utf-8-sig")
            orders.to_csv(os.path.join(outdir, "portfolio_orders_next_open.csv"), index=False, encoding="utf-8-sig")
            latest_candidates.to_csv(os.path.join(outdir, "portfolio_latest_candidates.csv"), index=False, encoding="utf-8-sig")
            latest_positions.to_csv(os.path.join(outdir, "portfolio_positions_state.csv"), index=False, encoding="utf-8-sig")
        return metrics_df

    for window_id, (_, _, vl_s, vl_e, te_s, te_e) in enumerate(windows, start=1):
        val_panel = _slice_panel(panel, vl_s, vl_e)
        best_name = None
        best_cfg = None
        best_score = -1e9
        for name, override in _portfolio_variants(base_cfg):
            cfg_variant = merge_config(base_cfg, override)
            _, trades, _, _, _, _, _ = simulate_multifactor_portfolio(val_panel, cfg_variant)
            score = _score_variant(trades)
            if score > best_score:
                best_score = score
                best_name = name
                best_cfg = cfg_variant

        test_panel = _slice_panel(panel, te_s, te_e)
        _, trades, monitor, equity, orders, latest_candidates, latest_positions = simulate_multifactor_portfolio(test_panel, best_cfg)
        test_df_ind = _slice_df(df_ind, te_s, te_e)
        baseline_idx_state = build_index_state_from_panel(test_df_ind, base_cfg, by_bucket=False)
        baseline_data = {str(code): sub.copy() for code, sub in test_df_ind.groupby("code")}
        _, baseline_trades, _, _ = backtest_simple(baseline_data, baseline_idx_state, base_cfg)
        metrics.append(
            {
                "window_id": window_id,
                "variant": best_name,
                "val_start": pd.Timestamp(vl_s).date(),
                "val_end": pd.Timestamp(vl_e).date(),
                "test_start": pd.Timestamp(te_s).date(),
                "test_end": pd.Timestamp(te_e).date(),
                "hit_rate": float((trades["pnl_pct"] > 0).mean()) if not trades.empty else np.nan,
                "expectancy": float(trades["pnl_pct"].mean()) if not trades.empty else np.nan,
                "trade_count": int(len(trades)),
                "avg_hold_days": float(trades["days_held"].mean()) if not trades.empty else np.nan,
                "baseline_hit_rate": float((baseline_trades["pnl_pct"] > 0).mean()) if not baseline_trades.empty else np.nan,
                "baseline_expectancy": float(baseline_trades["pnl_pct"].mean()) if not baseline_trades.empty else np.nan,
                "baseline_trade_count": int(len(baseline_trades)),
            }
        )
        best_outputs = (trades, monitor, equity, orders, latest_candidates, latest_positions)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(outdir, "portfolio_walkforward_metrics.csv"), index=False, encoding="utf-8-sig")
    summary = (
        metrics_df.groupby("variant")[["hit_rate", "expectancy", "trade_count", "avg_hold_days", "baseline_hit_rate", "baseline_expectancy", "baseline_trade_count"]]
        .mean()
        .reset_index()
        .sort_values(["hit_rate", "expectancy"], ascending=[False, False])
    )
    summary.to_csv(os.path.join(outdir, "portfolio_walkforward_summary.csv"), index=False, encoding="utf-8-sig")

    if best_outputs is not None:
        trades, monitor, equity, orders, latest_candidates, latest_positions = best_outputs
        trades.to_csv(os.path.join(outdir, "portfolio_backtest_trades.csv"), index=False, encoding="utf-8-sig")
        monitor.to_csv(os.path.join(outdir, "portfolio_backtest_daily_monitor.csv"), index=False, encoding="utf-8-sig")
        equity.to_csv(os.path.join(outdir, "portfolio_equity_curve.csv"), index=False, encoding="utf-8-sig")
        orders.to_csv(os.path.join(outdir, "portfolio_orders_next_open.csv"), index=False, encoding="utf-8-sig")
        latest_candidates.to_csv(os.path.join(outdir, "portfolio_latest_candidates.csv"), index=False, encoding="utf-8-sig")
        latest_positions.to_csv(os.path.join(outdir, "portfolio_positions_state.csv"), index=False, encoding="utf-8-sig")

    return metrics_df
