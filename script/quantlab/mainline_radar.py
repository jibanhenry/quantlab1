# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd


def _write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _safe_num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _rank_pct_by_date(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    vals = pd.to_numeric(df[col], errors="coerce")
    return vals.groupby(df["date"]).rank(pct=True, method="average", ascending=ascending) * 100.0


def _normalize_board_weekly(raw: pd.DataFrame, board_type: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = raw.copy()
    if "industry" in out.columns and "board" not in out.columns:
        out = out.rename(columns={"industry": "board"})
    if "board" not in out.columns:
        raise ValueError("weekly board data must contain industry or board column")
    if "board_code" not in out.columns:
        out["board_code"] = ""
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["board"] = out["board"].astype(str).str.strip()
    out["board_code"] = out["board_code"].astype(str).str.strip()
    out["board_type"] = board_type
    for col in ["open", "high", "low", "close", "pct_chg", "change", "volume", "amount", "amplitude", "turnover"]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    keep = ["date", "board_type", "board", "board_code", "open", "high", "low", "close", "pct_chg", "change", "volume", "amount", "amplitude", "turnover"]
    return out[keep].dropna(subset=["date", "board", "close"]).drop_duplicates(["board_type", "board", "date"], keep="last").sort_values(["board_type", "board", "date"]).reset_index(drop=True)


def load_board_weekly(industry_kline_path: str, concept_kline_path: str) -> pd.DataFrame:
    frames = []
    if industry_kline_path and os.path.exists(industry_kline_path):
        frames.append(_normalize_board_weekly(pd.read_csv(industry_kline_path), "industry"))
    if concept_kline_path and os.path.exists(concept_kline_path):
        frames.append(_normalize_board_weekly(pd.read_csv(concept_kline_path), "concept"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["board_type", "board", "date"]).reset_index(drop=True)


def build_mainline_radar_panel(weekly: pd.DataFrame, min_weeks: int = 30) -> pd.DataFrame:
    if weekly is None or weekly.empty:
        return pd.DataFrame()
    out = weekly.sort_values(["board_type", "board", "date"]).reset_index(drop=True).copy()
    keys = [out["board_type"], out["board"]]
    g = out.groupby(["board_type", "board"], sort=False)
    close = _safe_num(out, "close")
    amount = _safe_num(out, "amount")
    out["list_age_weeks"] = g.cumcount() + 1
    out["ret_1w"] = g["close"].pct_change(1, fill_method=None)
    for n in [4, 8, 12, 20]:
        out[f"ret_{n}w"] = g["close"].pct_change(n, fill_method=None)
    for n in [5, 10, 20]:
        out[f"ma{n}"] = g["close"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(n, min_periods=n).mean())
    out["ma5_slope_3w"] = out["ma5"] / g["ma5"].shift(3) - 1.0
    out["ma10_slope_3w"] = out["ma10"] / g["ma10"].shift(3) - 1.0
    out["ma20_slope_3w"] = out["ma20"] / g["ma20"].shift(3) - 1.0
    out["prev_close"] = g["close"].shift(1)
    out["prev_ma20"] = g["ma20"].shift(1)
    out["above_ma10"] = close > out["ma10"]
    out["above_ma20"] = close > out["ma20"]
    out["ma20_breakout"] = out["above_ma20"] & (out["prev_close"] <= out["prev_ma20"])
    out["recent_ma20_breakout"] = out["ma20_breakout"].astype(float).groupby(keys).rolling(3, min_periods=1).max().reset_index(level=[0, 1], drop=True).fillna(0).gt(0)
    out["amount_20w"] = amount.groupby(keys).rolling(20, min_periods=10).mean().reset_index(level=[0, 1], drop=True)
    out["amount_4w"] = amount.groupby(keys).rolling(4, min_periods=3).mean().reset_index(level=[0, 1], drop=True)
    out["amount_rvol_1w"] = amount / out["amount_20w"].replace(0, np.nan)
    out["amount_rvol_4w"] = out["amount_4w"] / out["amount_20w"].replace(0, np.nan)
    out["positive_weeks_4w"] = out["ret_1w"].gt(0).astype(float).groupby(keys).rolling(4, min_periods=4).sum().reset_index(level=[0, 1], drop=True)
    out["vol_8w"] = out["ret_1w"].groupby(keys).rolling(8, min_periods=6).std().reset_index(level=[0, 1], drop=True)
    out["vol_20w"] = out["ret_1w"].groupby(keys).rolling(20, min_periods=12).std().reset_index(level=[0, 1], drop=True)
    out["high20_close"] = g["close"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(20, min_periods=12).max())
    out["dist_high_20w"] = close / out["high20_close"].replace(0, np.nan) - 1.0
    out["mdd_8w"] = close / g["close"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(8, min_periods=6).max()).replace(0, np.nan) - 1.0
    out["fwd_ret_4w"] = g["close"].shift(-4) / close - 1.0
    out["fwd_ret_8w"] = g["close"].shift(-8) / close - 1.0

    for col in ["ret_4w", "ret_8w", "ret_12w", "ret_20w"]:
        out[f"{col}_rank"] = _rank_pct_by_date(out, col)
    out["rs_base_score"] = (
        out["ret_4w_rank"] * 0.30
        + out["ret_8w_rank"] * 0.25
        + out["ret_12w_rank"] * 0.25
        + out["ret_20w_rank"] * 0.20
    )
    out["rs_base_score_4w_ago"] = g["rs_base_score"].shift(4)
    out["rs_improve"] = out["rs_base_score"] - out["rs_base_score_4w_ago"]
    out["rs_improve_rank"] = _rank_pct_by_date(out, "rs_improve")
    out["rs_score"] = (out["rs_base_score"] * 0.80 + out["rs_improve_rank"] * 0.20).clip(0, 100)

    out["trend_score"] = 0.0
    out.loc[out["above_ma10"].fillna(False), "trend_score"] += 20.0
    out.loc[out["above_ma20"].fillna(False), "trend_score"] += 20.0
    out.loc[_safe_num(out, "ma5_slope_3w").gt(0), "trend_score"] += 15.0
    out.loc[_safe_num(out, "ma10_slope_3w").gt(0), "trend_score"] += 15.0
    out.loc[_safe_num(out, "ma20_slope_3w").gt(0), "trend_score"] += 10.0
    out.loc[out["recent_ma20_breakout"].fillna(False), "trend_score"] += 20.0

    rvol1 = _safe_num(out, "amount_rvol_1w")
    rvol4 = _safe_num(out, "amount_rvol_4w")
    vol1_score = ((rvol1 - 0.80) / 1.00 * 100.0).clip(0, 100)
    vol4_score = ((rvol4 - 0.90) / 0.80 * 100.0).clip(0, 100)
    pulse_penalty = np.where(rvol1.gt(3.0), 20.0, 0.0)
    out["volume_score"] = (vol1_score * 0.45 + vol4_score * 0.55 - pulse_penalty).clip(0, 100)

    pos_score = (_safe_num(out, "positive_weeks_4w") / 4.0 * 100.0).clip(0, 100)
    mdd_score = ((_safe_num(out, "mdd_8w") + 0.15) / 0.15 * 100.0).clip(0, 100)
    vol_ratio = (_safe_num(out, "vol_8w") / _safe_num(out, "vol_20w").replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    vol_score = ((1.40 - vol_ratio) / 0.80 * 100.0).clip(0, 100)
    out["stability_score"] = (pos_score * 0.45 + mdd_score * 0.35 + vol_score * 0.20).clip(0, 100)

    dist_high = _safe_num(out, "dist_high_20w")
    overheated = out["ret_4w"].gt(0.22) | out["ret_8w"].gt(0.38) | dist_high.gt(-0.005)
    not_too_far = dist_high.ge(-0.18)
    out["heat_score"] = 100.0
    out.loc[dist_high.lt(-0.18), "heat_score"] = 45.0
    out.loc[dist_high.between(-0.18, -0.08, inclusive="left"), "heat_score"] = 75.0
    out.loc[overheated.fillna(False), "heat_score"] = 55.0
    out.loc[not_too_far.fillna(False) & ~overheated.fillna(False), "heat_score"] = 100.0

    out["mainline_score"] = (
        out["rs_score"] * 0.35
        + out["trend_score"] * 0.25
        + out["volume_score"] * 0.20
        + out["stability_score"] * 0.10
        + out["heat_score"] * 0.10
    ).clip(0, 100)
    out["rs_pass"] = out["rs_score"].ge(70) & out["rs_improve"].ge(0)
    out["trend_pass"] = out["trend_score"].ge(70)
    out["volume_pass"] = out["volume_score"].ge(55) & (rvol1.ge(1.15) | rvol4.ge(1.08))
    out["quality_pass_count"] = out[["rs_pass", "trend_pass", "volume_pass"]].sum(axis=1)
    out["signal_grade"] = "C"
    out.loc[out["mainline_score"].ge(70) & out["quality_pass_count"].ge(2), "signal_grade"] = "B"
    out.loc[out["mainline_score"].ge(80) & out["rs_pass"] & out["trend_pass"] & out["volume_pass"], "signal_grade"] = "A"
    out.loc[out["list_age_weeks"].lt(int(min_weeks)), "signal_grade"] = ""
    out.loc[out["list_age_weeks"].lt(int(min_weeks)), "mainline_score"] = np.nan

    reasons = []
    for row in out.itertuples(index=False):
        parts = []
        if bool(getattr(row, "rs_pass", False)):
            parts.append("相对强度领先且改善")
        if bool(getattr(row, "trend_pass", False)):
            parts.append("趋势结构转强")
        if bool(getattr(row, "volume_pass", False)):
            parts.append("成交额温和放大")
        if bool(getattr(row, "recent_ma20_breakout", False)):
            parts.append("近3周突破MA20")
        if getattr(row, "heat_score", np.nan) < 70:
            parts.append("热度/位置需谨慎")
        reasons.append("；".join(parts))
    out["signal_reason"] = reasons
    return out.sort_values(["date", "mainline_score"], ascending=[True, False]).reset_index(drop=True)


def _latest_candidates(panel: pd.DataFrame, top_n: int) -> pd.DataFrame:
    latest_date = panel["date"].max()
    latest = panel[panel["date"].eq(latest_date) & panel["mainline_score"].notna()].copy()
    latest = latest.sort_values(["signal_grade", "mainline_score", "rs_score", "amount_20w"], ascending=[True, False, False, False])
    keep = [
        "date", "board_type", "board", "board_code", "signal_grade", "mainline_score",
        "rs_score", "trend_score", "volume_score", "stability_score", "heat_score",
        "quality_pass_count", "rs_pass", "trend_pass", "volume_pass", "signal_reason",
        "close", "ret_4w", "ret_8w", "ret_12w", "ret_20w", "rs_improve",
        "ma20_breakout", "recent_ma20_breakout", "amount_rvol_1w", "amount_rvol_4w",
        "positive_weeks_4w", "dist_high_20w", "fwd_ret_4w", "fwd_ret_8w",
    ]
    keep = [c for c in keep if c in latest.columns]
    return latest[keep].head(int(top_n)).reset_index(drop=True)


def _history_frame(panel: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "date", "board_type", "board", "board_code", "signal_grade", "mainline_score",
        "rs_score", "trend_score", "volume_score", "stability_score", "heat_score",
        "quality_pass_count", "rs_pass", "trend_pass", "volume_pass", "signal_reason",
        "ret_4w", "ret_8w", "ret_12w", "ret_20w", "rs_improve", "amount_rvol_1w",
        "amount_rvol_4w", "dist_high_20w", "fwd_ret_4w", "fwd_ret_8w",
    ]
    keep = [c for c in keep if c in panel.columns]
    return panel[panel["mainline_score"].notna()][keep].sort_values(["date", "mainline_score"], ascending=[True, False]).reset_index(drop=True)


def _backtest_summary(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grade_name, sub in [
        ("A", history[history["signal_grade"].eq("A")]),
        ("B", history[history["signal_grade"].eq("B")]),
        ("A_or_B", history[history["signal_grade"].isin(["A", "B"])]),
        ("top20_each_week", history.sort_values(["date", "mainline_score"], ascending=[True, False]).groupby("date", as_index=False).head(20)),
    ]:
        for ret_col in ["fwd_ret_4w", "fwd_ret_8w"]:
            ret = pd.to_numeric(sub.get(ret_col), errors="coerce").dropna()
            rows.append(
                {
                    "bucket": grade_name,
                    "horizon": ret_col,
                    "signals": int(len(ret)),
                    "mean_return": float(ret.mean()) if len(ret) else np.nan,
                    "median_return": float(ret.median()) if len(ret) else np.nan,
                    "win_rate": float(ret.gt(0).mean()) if len(ret) else np.nan,
                    "worst_return": float(ret.min()) if len(ret) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _fmt_pct(value) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "-"


def _build_summary_md(latest: pd.DataFrame, backtest: pd.DataFrame, panel: pd.DataFrame) -> str:
    latest_date = pd.to_datetime(panel["date"].max()).strftime("%Y-%m-%d") if not panel.empty else "-"
    lines: List[str] = []
    lines.append("# 主线启动雷达")
    lines.append("")
    lines.append(f"- 数据最新周频日期: {latest_date}")
    lines.append(f"- 覆盖板块数: {panel[['board_type', 'board']].drop_duplicates().shape[0] if not panel.empty else 0}")
    lines.append("- 口径: 稳健启动，要求相对强度、趋势结构、成交额确认共同改善。")
    lines.append("")
    lines.append("## 最新 Top 20")
    lines.append("")
    if latest.empty:
        lines.append("暂无候选。")
    else:
        lines.append("| 排名 | 等级 | 类型 | 板块 | 分数 | 近4周 | 近8周 | 成交1周/20周 | 原因 |")
        lines.append("|---:|---|---|---|---:|---:|---:|---:|---|")
        for i, row in latest.head(20).iterrows():
            lines.append(
                f"| {i + 1} | {row.get('signal_grade', '')} | {row.get('board_type', '')} | {row.get('board', '')} | "
                f"{float(row.get('mainline_score', np.nan)):.1f} | {_fmt_pct(row.get('ret_4w'))} | {_fmt_pct(row.get('ret_8w'))} | "
                f"{float(row.get('amount_rvol_1w', np.nan)):.2f} | {row.get('signal_reason', '')} |"
            )
    lines.append("")
    lines.append("## 历史回放")
    lines.append("")
    if backtest.empty:
        lines.append("历史回放暂无有效样本。")
    else:
        lines.append("| 分组 | 周期 | 样本 | 均值 | 中位数 | 胜率 | 最差 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in backtest.itertuples(index=False):
            lines.append(
                f"| {row.bucket} | {row.horizon} | {row.signals} | {_fmt_pct(row.mean_return)} | "
                f"{_fmt_pct(row.median_return)} | {_fmt_pct(row.win_rate)} | {_fmt_pct(row.worst_return)} |"
            )
    lines.append("")
    lines.append("## 评分原理")
    lines.append("")
    lines.append("- 相对强度 35%: 4/8/12/20 周收益横截面排名，并奖励近 4 周排名改善。")
    lines.append("- 趋势启动 25%: MA10/MA20 站上、均线上行、近 3 周 MA20 突破。")
    lines.append("- 成交确认 20%: 1 周与 4 周成交额相对 20 周均值温和放大，极端脉冲扣分。")
    lines.append("- 稳定性 10%: 正收益周数、近 8 周回撤与波动收敛。")
    lines.append("- 不过热过滤 10%: 排除距离 20 周高点过远或短期涨幅过热的末端信号。")
    return "\n".join(lines) + "\n"


def run_mainline_radar(
    industry_kline_path: str,
    concept_kline_path: str,
    outdir: str,
    top_n: int = 20,
    min_weeks: int = 30,
) -> Dict[str, pd.DataFrame]:
    os.makedirs(outdir, exist_ok=True)
    weekly = load_board_weekly(industry_kline_path, concept_kline_path)
    panel = build_mainline_radar_panel(weekly, min_weeks=min_weeks)
    latest = _latest_candidates(panel, top_n=top_n) if not panel.empty else pd.DataFrame()
    history = _history_frame(panel) if not panel.empty else pd.DataFrame()
    backtest = _backtest_summary(history) if not history.empty else pd.DataFrame()
    _write_csv(latest, os.path.join(outdir, "mainline_latest.csv"))
    _write_csv(history, os.path.join(outdir, "mainline_rank_history.csv"))
    _write_csv(backtest, os.path.join(outdir, "mainline_backtest_summary.csv"))
    _write_text(_build_summary_md(latest, backtest, panel), os.path.join(outdir, "mainline_summary.md"))
    return {
        "panel": panel,
        "mainline_latest": latest,
        "mainline_rank_history": history,
        "mainline_backtest_summary": backtest,
    }
