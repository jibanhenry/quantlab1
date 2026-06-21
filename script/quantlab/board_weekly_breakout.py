# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _safe_num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _normalize_board_weekly(raw: pd.DataFrame, board_type: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = raw.copy()
    if "industry" in out.columns and "board" not in out.columns:
        out = out.rename(columns={"industry": "board"})
    if "board" not in out.columns:
        raise ValueError("weekly board kline must contain industry or board column")
    out["board_type"] = board_type
    out["board"] = out["board"].astype(str).str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "board_code" not in out.columns:
        out["board_code"] = ""
    for col in ["open", "high", "low", "close", "pct_chg", "change", "volume", "amount", "amplitude", "turnover"]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    keep = ["date", "board_type", "board", "board_code", "open", "high", "low", "close", "pct_chg", "change", "volume", "amount", "amplitude", "turnover"]
    out = out[keep].dropna(subset=["date", "board", "open", "close"])
    return out.drop_duplicates(["board_type", "board", "date"], keep="last").sort_values(["board_type", "board", "date"]).reset_index(drop=True)


def build_board_ma20_panel(weekly_df: pd.DataFrame, board_type: str, min_amount_20w: float = 0.0) -> pd.DataFrame:
    weekly = _normalize_board_weekly(weekly_df, board_type)
    if weekly.empty:
        return weekly
    g = weekly.groupby(["board_type", "board"], sort=False)
    close = _safe_num(weekly, "close")
    weekly["ma20"] = g["close"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(20, min_periods=20).mean())
    weekly["amount_20w"] = _safe_num(weekly, "amount").groupby([weekly["board_type"], weekly["board"]]).rolling(20, min_periods=10).mean().reset_index(level=[0, 1], drop=True)
    weekly["list_age_weeks"] = g.cumcount() + 1
    weekly["prev_close"] = g["close"].shift(1)
    weekly["prev_ma20"] = g["ma20"].shift(1)
    weekly["next_date"] = g["date"].shift(-1)
    weekly["next_date_2"] = g["date"].shift(-2)
    weekly["next_open"] = g["open"].shift(-1)
    weekly["next_open_2"] = g["open"].shift(-2)
    weekly["tradable"] = weekly["list_age_weeks"].ge(20) & weekly["amount_20w"].ge(float(min_amount_20w))
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
    parts = []
    for _, sub in weekly.groupby(["board_type", "board"], sort=False):
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
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else weekly


def extract_board_trades(panel: pd.DataFrame, cost_bp: float = 2.0) -> pd.DataFrame:
    cost = float(cost_bp) / 10000.0
    rows = []
    for keys, sub in panel.sort_values(["board_type", "board", "date"]).groupby(["board_type", "board"], sort=False):
        board_type, board = keys
        in_pos = False
        entry = None
        for _, row in sub.iterrows():
            if not in_pos and bool(row.get("buy_signal")):
                entry_px = float(pd.to_numeric(row.get("next_open"), errors="coerce"))
                entry_date = row.get("next_date")
                if np.isfinite(entry_px) and pd.notna(entry_date):
                    in_pos = True
                    entry = {
                        "board_type": board_type,
                        "board": board,
                        "board_code": row.get("board_code"),
                        "entry_signal_date": row.get("date"),
                        "entry_date": entry_date,
                        "entry_price": entry_px,
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
                            "pnl_pct": float(pnl),
                            "weeks_held": int(max(1, round((pd.Timestamp(exit_date) - pd.Timestamp(entry["entry_date"])).days / 7))),
                        }
                    )
                in_pos = False
                entry = None
    return pd.DataFrame(rows)


def simulate_board_portfolio(panel: pd.DataFrame, total_exposure: float = 0.45, cost_bp: float = 2.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = panel.sort_values(["date", "board_type", "board"]).copy()
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
    for dt, sub in work.groupby("date", sort=True):
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
        sample = sub.sort_values("amount_20w", ascending=False).head(50).copy()
        sample["signal_date"] = dt
        holdings.append(sample[["signal_date", "date", "board_type", "board", "board_code", "close", "ma20", "next_open", "next_open_2", "amount_20w"]])
    return pd.DataFrame(rows), pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak.replace(0, np.nan) - 1.0
    return float(dd.min()) if len(dd) else np.nan


def summarize_board_breakout(name: str, trades: pd.DataFrame, equity: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    pnl = pd.to_numeric(trades.get("pnl_pct"), errors="coerce") if trades is not None and not trades.empty else pd.Series(dtype=float)
    eq = equity.sort_values("date").copy() if equity is not None and not equity.empty else pd.DataFrame()
    if eq.empty:
        annual_return = max_drawdown = sharpe = final_equity = np.nan
        weeks = 0
        avg_active_count = np.nan
        start_date = end_date = pd.NaT
    else:
        eq["date"] = pd.to_datetime(eq["date"], errors="coerce")
        start_date = eq["date"].min()
        end_date = eq["date"].max()
        years = max((end_date - start_date).days / 365.25, 1e-9)
        final_equity = float(eq["equity"].iloc[-1])
        weekly_ret = pd.to_numeric(eq["equity"], errors="coerce").pct_change().dropna()
        annual_return = float(final_equity ** (1.0 / years) - 1.0) if final_equity > 0 else np.nan
        max_drawdown = _max_drawdown(pd.to_numeric(eq["equity"], errors="coerce"))
        sharpe = float(weekly_ret.mean() / weekly_ret.std(ddof=1) * math.sqrt(52)) if len(weekly_ret) > 2 and weekly_ret.std(ddof=1) > 0 else np.nan
        weeks = int(len(eq))
        avg_active_count = float(pd.to_numeric(eq.get("active_count"), errors="coerce").mean())
    return pd.DataFrame(
        [
            {
                "strategy": name,
                "board_count": int(panel["board"].nunique()) if panel is not None and not panel.empty else 0,
                "start_date": start_date,
                "end_date": end_date,
                "trade_count": int(len(pnl)),
                "win_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
                "trade_expectancy": float(pnl.mean()) if len(pnl) else np.nan,
                "trade_median": float(pnl.median()) if len(pnl) else np.nan,
                "avg_weeks_held": float(pd.to_numeric(trades.get("weeks_held"), errors="coerce").mean()) if trades is not None and not trades.empty else np.nan,
                "annual_return": annual_return,
                "max_drawdown": max_drawdown,
                "sharpe": sharpe,
                "final_equity": final_equity,
                "weeks": weeks,
                "avg_active_count": avg_active_count,
            }
        ]
    )


def run_board_weekly_breakout_experiment(
    industry_kline_path: str,
    concept_kline_path: str,
    outdir: str,
    min_amount_20w: float = 0.0,
    total_exposure: float = 0.45,
    cost_bp: float = 2.0,
) -> Dict[str, pd.DataFrame]:
    os.makedirs(outdir, exist_ok=True)
    jobs = [
        ("industry", industry_kline_path, "industry_ma20_weekly_breakout"),
        ("concept", concept_kline_path, "concept_ma20_weekly_breakout"),
    ]
    summaries: List[pd.DataFrame] = []
    result: Dict[str, pd.DataFrame] = {}
    for board_type, path, strategy_name in jobs:
        raw = pd.read_csv(path)
        panel = build_board_ma20_panel(raw, board_type=board_type, min_amount_20w=min_amount_20w)
        trades = extract_board_trades(panel, cost_bp=cost_bp)
        equity, holdings = simulate_board_portfolio(panel, total_exposure=total_exposure, cost_bp=cost_bp)
        summary = summarize_board_breakout(strategy_name, trades, equity, panel)
        prefix = f"{board_type}_ma20_breakout"
        _write_csv(summary, os.path.join(outdir, f"{prefix}_summary.csv"))
        _write_csv(equity, os.path.join(outdir, f"{prefix}_equity.csv"))
        _write_csv(trades, os.path.join(outdir, f"{prefix}_trades.csv"))
        _write_csv(holdings, os.path.join(outdir, f"{prefix}_holdings_sample.csv"))
        summaries.append(summary)
        result[f"{board_type}_summary"] = summary
        result[f"{board_type}_equity"] = equity
        result[f"{board_type}_trades"] = trades
        result[f"{board_type}_holdings"] = holdings
    combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    _write_csv(combined, os.path.join(outdir, "board_ma20_breakout_summary.csv"))
    result["summary"] = combined
    return result
