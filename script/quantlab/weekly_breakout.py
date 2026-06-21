# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .io_utils import load_market_csv_multi
from .weekly_research import aggregate_daily_to_weekly


def _write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _safe_num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def build_ma20_breakout_panel(daily_df: pd.DataFrame, min_amount_20w: float = 20_000_000.0) -> pd.DataFrame:
    weekly = aggregate_daily_to_weekly(daily_df)
    weekly = weekly.sort_values(["code", "date"]).reset_index(drop=True)
    g = weekly.groupby("code", sort=False)
    close = _safe_num(weekly, "close")
    weekly["ma20"] = g["close"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(20, min_periods=20).mean())
    weekly["amount_20w"] = _safe_num(weekly, "amount").groupby(weekly["code"]).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    weekly["list_age_weeks"] = g.cumcount() + 1
    weekly["prev_close"] = g["close"].shift(1)
    weekly["prev_ma20"] = g["ma20"].shift(1)
    weekly["next_date"] = g["date"].shift(-1)
    weekly["next_date_2"] = g["date"].shift(-2)
    weekly["next_open"] = g["open"].shift(-1)
    weekly["next_open_2"] = g["open"].shift(-2)
    pct_chg = _safe_num(weekly, "pct_chg")
    weekly["tradable"] = (
        weekly["list_age_weeks"].ge(26)
        & weekly["amount_20w"].ge(float(min_amount_20w))
        & close.notna()
        & _safe_num(weekly, "open").notna()
        & ~weekly["code"].astype(str).str.startswith("399")
        & ~(pct_chg >= 9.5).fillna(False)
        & ~(pct_chg <= -9.5).fillna(False)
    )
    weekly["above_ma20"] = close > weekly["ma20"]
    weekly["buy_signal"] = (
        weekly["tradable"].fillna(False)
        & weekly["ma20"].notna()
        & weekly["prev_ma20"].notna()
        & (weekly["prev_close"] <= weekly["prev_ma20"])
        & (close > weekly["ma20"])
    )
    weekly["sell_signal"] = (
        weekly["ma20"].notna()
        & weekly["prev_ma20"].notna()
        & (weekly["prev_close"] >= weekly["prev_ma20"])
        & (close < weekly["ma20"])
    )
    active_parts = []
    for _, sub in weekly.groupby("code", sort=False):
        in_pos = False
        active = []
        for _, row in sub.iterrows():
            if bool(row.get("buy_signal")):
                in_pos = True
            elif bool(row.get("sell_signal")):
                in_pos = False
            active.append(in_pos)
        part = sub.copy()
        part["active_position"] = active
        active_parts.append(part)
    weekly = pd.concat(active_parts, ignore_index=True) if active_parts else weekly
    return weekly


def extract_breakout_trades(panel: pd.DataFrame, cost_bp: float = 2.0) -> pd.DataFrame:
    cost = float(cost_bp) / 10000.0
    rows = []
    for code, sub in panel.sort_values(["code", "date"]).groupby("code", sort=False):
        in_pos = False
        entry = None
        for _, row in sub.iterrows():
            if not in_pos and bool(row.get("buy_signal")):
                entry_px = float(pd.to_numeric(row.get("next_open"), errors="coerce"))
                entry_date = row.get("next_date")
                if np.isfinite(entry_px) and pd.notna(entry_date):
                    in_pos = True
                    entry = {
                        "code": str(code),
                        "entry_signal_date": row.get("date"),
                        "entry_date": entry_date,
                        "entry_price": entry_px,
                        "entry_close": row.get("close"),
                        "entry_ma20": row.get("ma20"),
                    }
                continue
            if in_pos and bool(row.get("sell_signal")):
                exit_px = float(pd.to_numeric(row.get("next_open"), errors="coerce"))
                exit_date = row.get("next_date")
                if entry is not None and np.isfinite(exit_px) and pd.notna(exit_date):
                    pnl = exit_px / float(entry["entry_price"]) - 1.0 - 2.0 * cost
                    rows.append(
                        {
                            **entry,
                            "exit_signal_date": row.get("date"),
                            "exit_date": exit_date,
                            "exit_price": exit_px,
                            "exit_close": row.get("close"),
                            "exit_ma20": row.get("ma20"),
                            "pnl_pct": float(pnl),
                            "weeks_held": int(max(1, round((pd.Timestamp(exit_date) - pd.Timestamp(entry["entry_date"])).days / 7))),
                        }
                    )
                in_pos = False
                entry = None
    return pd.DataFrame(rows)


def simulate_breakout_portfolio(
    panel: pd.DataFrame,
    total_exposure: float = 0.45,
    cost_bp: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = panel.sort_values(["date", "code"]).copy()
    work = work[
        work["tradable"].fillna(False)
        & work["active_position"].fillna(False)
        & _safe_num(work, "next_open").notna()
        & _safe_num(work, "next_open_2").notna()
    ].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    cost = float(cost_bp) / 10000.0
    equity = 1.0
    rows = []
    holdings = []
    for dt, sub in work.groupby("date", sort=False):
        sub = sub.copy()
        ret = _safe_num(sub, "next_open_2") / _safe_num(sub, "next_open") - 1.0
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
        if ret.empty:
            continue
        basket_ret = float(ret.mean()) - 2.0 * cost
        equity *= 1.0 + float(total_exposure) * basket_ret
        exit_date = pd.to_datetime(sub["next_date_2"].dropna().max()) if sub["next_date_2"].notna().any() else pd.Timestamp(dt)
        rows.append(
            {
                "date": exit_date,
                "signal_date": dt,
                "equity": float(equity),
                "active_count": int(len(ret)),
                "basket_ret": basket_ret,
                "exposure": float(total_exposure),
            }
        )
        top_sample = sub.sort_values("amount_20w", ascending=False).head(20).copy()
        top_sample["signal_date"] = dt
        holdings.append(top_sample[["signal_date", "date", "code", "close", "ma20", "next_open", "next_open_2", "amount_20w"]])
    equity_df = pd.DataFrame(rows)
    holdings_df = pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()
    return equity_df, holdings_df


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak.replace(0, np.nan) - 1.0
    return float(dd.min()) if len(dd) else np.nan


def summarize_breakout(trades: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    pnl = pd.to_numeric(trades.get("pnl_pct"), errors="coerce") if trades is not None and not trades.empty else pd.Series(dtype=float)
    eq = equity.sort_values("date").copy() if equity is not None and not equity.empty else pd.DataFrame()
    if eq.empty:
        portfolio = {
            "annual_return": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "final_equity": np.nan,
            "weeks": 0,
            "avg_active_count": np.nan,
        }
    else:
        eq["date"] = pd.to_datetime(eq["date"], errors="coerce")
        years = max((eq["date"].max() - eq["date"].min()).days / 365.25, 1e-9)
        final_equity = float(eq["equity"].iloc[-1])
        weekly_ret = pd.to_numeric(eq["equity"], errors="coerce").pct_change().dropna()
        portfolio = {
            "annual_return": float(final_equity ** (1.0 / years) - 1.0) if final_equity > 0 else np.nan,
            "max_drawdown": _max_drawdown(pd.to_numeric(eq["equity"], errors="coerce")),
            "sharpe": float(weekly_ret.mean() / weekly_ret.std(ddof=1) * math.sqrt(52)) if len(weekly_ret) > 2 and weekly_ret.std(ddof=1) > 0 else np.nan,
            "final_equity": final_equity,
            "weeks": int(len(eq)),
            "avg_active_count": float(pd.to_numeric(eq.get("active_count"), errors="coerce").mean()),
        }
    row: Dict[str, float] = {
        "strategy": "ma20_weekly_breakout",
        "trade_count": int(len(pnl)),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "trade_expectancy": float(pnl.mean()) if len(pnl) else np.nan,
        "trade_median": float(pnl.median()) if len(pnl) else np.nan,
        "avg_weeks_held": float(pd.to_numeric(trades.get("weeks_held"), errors="coerce").mean()) if trades is not None and not trades.empty else np.nan,
        **portfolio,
    }
    return pd.DataFrame([row])


def run_weekly_breakout_experiment(
    csvs: List[str],
    outdir: str,
    min_amount_20w: float = 20_000_000.0,
    total_exposure: float = 0.45,
    cost_bp: float = 2.0,
) -> Dict[str, pd.DataFrame]:
    os.makedirs(outdir, exist_ok=True)
    daily = load_market_csv_multi(csvs)
    panel = build_ma20_breakout_panel(daily, min_amount_20w=min_amount_20w)
    trades = extract_breakout_trades(panel, cost_bp=cost_bp)
    equity, holdings = simulate_breakout_portfolio(panel, total_exposure=total_exposure, cost_bp=cost_bp)
    summary = summarize_breakout(trades, equity)
    _write_csv(summary, os.path.join(outdir, "weekly_ma20_breakout_summary.csv"))
    _write_csv(equity, os.path.join(outdir, "weekly_ma20_breakout_equity.csv"))
    _write_csv(trades, os.path.join(outdir, "weekly_ma20_breakout_trades.csv"))
    _write_csv(holdings, os.path.join(outdir, "weekly_ma20_breakout_holdings_sample.csv"))
    return {
        "panel": panel,
        "summary": summary,
        "equity": equity,
        "trades": trades,
        "holdings": holdings,
    }
