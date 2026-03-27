# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .buckets import monthly_freeze_bucket_map
from .config import load_config, merge_config
from .io_utils import load_market_csv_multi, save_outputs
from .signals import compute_indicators
from .market_state import build_index_state_from_panel
from .backtest import backtest_simple
from .valuation import add_valuation_features

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _month_add(dt64: np.datetime64, months: int) -> np.datetime64:
    ts = pd.Timestamp(dt64)
    y = ts.year + (ts.month - 1 + months) // 12
    m = (ts.month - 1 + months) % 12 + 1
    d = min(ts.day, [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return np.datetime64(pd.Timestamp(year=y, month=m, day=d).date())


def _generate_walk_forward_windows(
    dates: pd.Series, train_m: int, val_m: int, test_m: int, step_m: int
) -> List[Tuple[np.datetime64, np.datetime64, np.datetime64, np.datetime64, np.datetime64, np.datetime64]]:
    start = pd.to_datetime(dates.min()).to_datetime64()
    end = pd.to_datetime(dates.max()).to_datetime64()
    windows = []
    cur_train_start = start
    while True:
        train_end = _month_add(cur_train_start, train_m)
        val_end = _month_add(train_end, val_m)
        test_end = _month_add(val_end, test_m)
        if test_end > end:
            break
        windows.append((cur_train_start, train_end, train_end, val_end, val_end, test_end))
        cur_train_start = _month_add(cur_train_start, step_m)
    return windows


def _sample_params(base: dict) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg["S1"]["atr_mul"] = round(random.uniform(1.4, 2.2), 2)
    cfg["S1"]["atr_mul_pullback"] = round(random.uniform(0.6, 1.2), 2)
    cfg["S1"]["rsi_bull_low"] = random.choice([40, 42, 45, 48, 50])
    cfg["S2"]["bw_quantile"] = round(random.uniform(0.15, 0.35), 2)
    cfg["S2"]["atr_stop_mul"] = round(random.uniform(0.8, 1.4), 2)
    cfg["S3"]["rsi_buy"] = random.choice([28, 30, 32, 35, 38, 40])
    cfg["S3"]["atr_stop_mul"] = round(random.uniform(1.0, 1.6), 2)
    cfg["stock_state"]["adx_trend_th"] = random.choice([22, 25, 28])
    cfg["stock_state"]["adx_range_th"] = random.choice([18, 20, 22])
    return cfg


def _slice_by_dates(df: pd.DataFrame, start: np.datetime64, end: np.datetime64) -> pd.DataFrame:
    m = (df["date"] >= pd.Timestamp(start)) & (df["date"] < pd.Timestamp(end))
    return df.loc[m].copy()


def _run_once(df_all: pd.DataFrame, cfg: dict, by_bucket: bool):
    df_ind = []
    for _, sub in df_all.groupby("code"):
        df_ind.append(compute_indicators(sub.sort_values("date"), cfg))
    df_ind = pd.concat(df_ind, ignore_index=False) if df_ind else pd.DataFrame()
    if df_ind.empty:
        return df_ind, df_ind, df_ind, df_ind
    df_ind = add_valuation_features(df_ind.reset_index(drop=True), cfg)
    idx_state = build_index_state_from_panel(df_ind, cfg, by_bucket=by_bucket)
    data = {str(c): g for c, g in df_ind.groupby("code")}
    return backtest_simple(data, idx_state, cfg, cost_bp=2.0)


def _trade_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    if trades is None or trades.empty:
        return {
            "hit_rate": 0.0,
            "expectancy": 0.0,
            "trade_count": 0.0,
            "avg_pnl": 0.0,
            "avg_hold_days": np.nan,
        }
    hold_days = (
        pd.to_datetime(trades["exit_date"], errors="coerce") - pd.to_datetime(trades["entry_date"], errors="coerce")
    ).dt.days
    return {
        "hit_rate": float((trades["pnl_pct"] > 0).mean()),
        "expectancy": float(trades["pnl_pct"].mean()),
        "trade_count": float(len(trades)),
        "avg_pnl": float(trades["pnl_pct"].mean()),
        "avg_hold_days": float(hold_days.mean()) if not hold_days.empty else np.nan,
    }


def _candidate_topn_hit_rate(cands: pd.DataFrame, top_n: int = 10) -> float:
    if cands is None or cands.empty or "candidate_score" not in cands.columns:
        return np.nan
    sub = cands.sort_values("candidate_score", ascending=False).head(top_n)
    if "predicted_return" in sub.columns:
        return float((sub["predicted_return"] > 0).mean())
    if "confidence" in sub.columns:
        return float((sub["confidence"] >= sub["confidence"].median()).mean())
    return np.nan


def _strategy_breakdown(trades: pd.DataFrame, variant_name: str, window_id: int) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["variant", "window_id", "strategy", "trade_count", "hit_rate", "expectancy"])
    grouped = (
        trades.groupby("strategy")["pnl_pct"]
        .agg(
            trade_count="count",
            hit_rate=lambda s: float((s > 0).mean()),
            expectancy="mean",
        )
        .reset_index()
    )
    grouped.insert(0, "window_id", window_id)
    grouped.insert(0, "variant", variant_name)
    return grouped


def _year_breakdown(trades: pd.DataFrame, variant_name: str, window_id: int) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["variant", "window_id", "year", "trade_count", "hit_rate", "expectancy"])
    tmp = trades.copy()
    tmp["year"] = pd.to_datetime(tmp["entry_date"], errors="coerce").dt.year
    grouped = (
        tmp.groupby("year")["pnl_pct"]
        .agg(
            trade_count="count",
            hit_rate=lambda s: float((s > 0).mean()),
            expectancy="mean",
        )
        .reset_index()
    )
    grouped.insert(0, "window_id", window_id)
    grouped.insert(0, "variant", variant_name)
    return grouped


def _variant_specs() -> List[Tuple[str, dict]]:
    return [
        ("baseline", {"valuation": {"enabled": False, "mode": "rank_only", "ml_weight": 0.0}}),
        ("value_pb", {"valuation": {"enabled": True, "mode": "rank_only", "pb_weight": 1.0, "ps_weight": 0.0}}),
        ("value_ps", {"valuation": {"enabled": True, "mode": "rank_only", "pb_weight": 0.0, "ps_weight": 1.0}}),
        ("value_pbps", {"valuation": {"enabled": True, "mode": "rank_only", "pb_weight": 0.5, "ps_weight": 0.5}}),
        ("value_soft_filter", {"valuation": {"enabled": True, "mode": "soft_filter", "pb_weight": 0.5, "ps_weight": 0.5}}),
        ("value_ml_combo", {"valuation": {"enabled": True, "mode": "rank_only", "pb_weight": 0.5, "ps_weight": 0.5, "ml_weight": 0.2, "tech_weight": 0.6, "value_weight": 0.2}}),
    ]


def _score_for_selection(trades: pd.DataFrame) -> float:
    m = _trade_metrics(trades)
    if m["trade_count"] <= 0:
        return -1e9
    return m["hit_rate"] * 100.0 + m["expectancy"] * 1000.0 + min(m["trade_count"], 20.0) * 0.1


def _tune_bucket_on_validation(
    df_bucket: pd.DataFrame,
    base_cfg: dict,
    train_start: np.datetime64,
    val_start: np.datetime64,
    val_end: np.datetime64,
    trials: int,
    by_bucket: bool,
) -> dict:
    del train_start  # reserved for future train-time feature/model fitting
    best_score = -1e9
    best_cfg = base_cfg
    for _ in tqdm(range(trials), desc="  [bucket] 随机搜索", leave=False):
        cfg = _sample_params(base_cfg)
        df_v = _slice_by_dates(df_bucket, val_start, val_end)
        if df_v.empty:
            continue
        if "bucket_id" not in df_v.columns:
            df_v = df_v.copy()
            df_v["bucket_id"] = "B"
        _, trades, _, _ = _run_once(df_v, cfg, by_bucket=by_bucket)
        score = _score_for_selection(trades)
        if score > best_score:
            best_score = score
            best_cfg = cfg
    return best_cfg


def run_quarterly_tuning(
    all_in_one_csv: str,
    outdir: str = "./output",
    cfg_path: Optional[str] = None,
    cfg_overrides: Optional[dict] = None,
    bucket_mode: str = "size",
    train_months: int = 6,
    val_months: int = 3,
    step_months: int = 3,
    trials: int = 30,
):
    os.makedirs(outdir, exist_ok=True)
    base_cfg = merge_config(load_config(cfg_path), cfg_overrides)

    print("[Q1/7] 加载CSV（可多文件，以逗号分隔）...")
    csvs = [p.strip() for p in all_in_one_csv.split(",") if p.strip()]
    df = load_market_csv_multi(csvs)

    print(f"[Q2/7] 生成 {bucket_mode} 分桶（月度冻结）...")
    bucket_map = monthly_freeze_bucket_map(df, mode=bucket_mode, k=3, code_industry=None)
    df = df.merge(bucket_map, on="code", how="left")
    if "bucket_id" not in df.columns:
        df["bucket_id"] = "ALL"

    print("[Q3/7] 构造 train/val/test 滚动窗口...")
    windows = _generate_walk_forward_windows(df["date"], train_months, val_months, step_months, step_months)
    print(f"  窗口数：{len(windows)}")
    if not windows:
        raise RuntimeError("没有足够数据构造 train/val/test 窗口。")

    tuned_windows: List[dict] = []
    metrics_rows: List[dict] = []
    strategy_rows: List[pd.DataFrame] = []
    year_rows: List[pd.DataFrame] = []
    latest_outputs = None

    for window_id, (tr_s, tr_e, vl_s, vl_e, te_s, te_e) in enumerate(windows, start=1):
        print(f"[Q4/7] Window {window_id}: train={pd.Timestamp(tr_s).date()}->{pd.Timestamp(tr_e).date()}  val={pd.Timestamp(vl_s).date()}->{pd.Timestamp(vl_e).date()}  test={pd.Timestamp(te_s).date()}->{pd.Timestamp(te_e).date()}")
        tuned_by_bucket: Dict[str, dict] = {}
        for bucket_id, sub in tqdm(df.groupby("bucket_id"), desc=f"[window {window_id}] 按桶调参"):
            tuned_by_bucket[str(bucket_id)] = _tune_bucket_on_validation(
                sub, base_cfg, tr_s, vl_s, vl_e, trials=trials, by_bucket=True
            )

        tuned_windows.append(
            {
                "window_id": window_id,
                "train_start": str(pd.Timestamp(tr_s).date()),
                "train_end": str(pd.Timestamp(tr_e).date()),
                "val_start": str(pd.Timestamp(vl_s).date()),
                "val_end": str(pd.Timestamp(vl_e).date()),
                "test_start": str(pd.Timestamp(te_s).date()),
                "test_end": str(pd.Timestamp(te_e).date()),
                "configs": tuned_by_bucket,
            }
        )

        df_test = _slice_by_dates(df, te_s, te_e)
        for variant_name, variant_override in _variant_specs():
            sig_parts, trd_parts, summ_parts, cand_parts = [], [], [], []
            for bucket_id, sub in df_test.groupby("bucket_id"):
                bucket_cfg = merge_config(tuned_by_bucket.get(str(bucket_id), base_cfg), variant_override)
                sig, trd, summ, cand = _run_once(sub, bucket_cfg, by_bucket=True)
                if sig is not None and not sig.empty:
                    sig_parts.append(sig)
                if trd is not None and not trd.empty:
                    trd_parts.append(trd)
                if summ is not None and not summ.empty:
                    summ_parts.append(summ)
                if cand is not None and not cand.empty:
                    cand_parts.append(cand)

            signals_all = pd.concat(sig_parts, ignore_index=True) if sig_parts else pd.DataFrame()
            trades_all = pd.concat(trd_parts, ignore_index=True) if trd_parts else pd.DataFrame()
            summary_all = pd.concat(summ_parts, ignore_index=True) if summ_parts else pd.DataFrame()
            cands_all = pd.concat(cand_parts, ignore_index=True) if cand_parts else pd.DataFrame()

            m = _trade_metrics(trades_all)
            metrics_rows.append(
                {
                    "window_id": window_id,
                    "variant": variant_name,
                    "train_start": pd.Timestamp(tr_s).date(),
                    "train_end": pd.Timestamp(tr_e).date(),
                    "val_start": pd.Timestamp(vl_s).date(),
                    "val_end": pd.Timestamp(vl_e).date(),
                    "test_start": pd.Timestamp(te_s).date(),
                    "test_end": pd.Timestamp(te_e).date(),
                    "hit_rate": m["hit_rate"],
                    "expectancy": m["expectancy"],
                    "trade_count": m["trade_count"],
                    "avg_hold_days": m["avg_hold_days"],
                    "top10_candidate_hit_proxy": _candidate_topn_hit_rate(cands_all, top_n=10),
                }
            )
            strategy_rows.append(_strategy_breakdown(trades_all, variant_name, window_id))
            year_rows.append(_year_breakdown(trades_all, variant_name, window_id))

            if window_id == len(windows) and variant_name == "value_pbps":
                latest_outputs = (signals_all, trades_all, summary_all, cands_all)

    print("[Q5/7] 写出最优参数与滚动实验报告...")
    tuned_path = os.path.join(outdir, f"tuned_config_quarterly_{pd.to_datetime(df['date'].max()):%Y%m%d}.json")
    with open(tuned_path, "w", encoding="utf-8") as f:
        json.dump({"windows": tuned_windows}, f, ensure_ascii=False, indent=2)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(outdir, "walk_forward_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    summary_variant = (
        metrics_df.groupby("variant")[["hit_rate", "expectancy", "trade_count", "avg_hold_days", "top10_candidate_hit_proxy"]]
        .mean()
        .reset_index()
        .sort_values(["hit_rate", "expectancy"], ascending=[False, False])
    )
    summary_path = os.path.join(outdir, "walk_forward_variant_summary.csv")
    summary_variant.to_csv(summary_path, index=False, encoding="utf-8-sig")

    strategy_df = pd.concat(strategy_rows, ignore_index=True) if strategy_rows else pd.DataFrame()
    if not strategy_df.empty:
        strategy_df.to_csv(os.path.join(outdir, "walk_forward_strategy_breakdown.csv"), index=False, encoding="utf-8-sig")

    year_df = pd.concat(year_rows, ignore_index=True) if year_rows else pd.DataFrame()
    if not year_df.empty:
        year_df.to_csv(os.path.join(outdir, "walk_forward_year_breakdown.csv"), index=False, encoding="utf-8-sig")

    print("[Q6/7] 输出最后测试窗的推荐结果...")
    if latest_outputs is not None:
        signals_all, trades_all, summary_all, cands_all = latest_outputs
        save_outputs(
            signals_all,
            trades_all,
            summary_all,
            cands_all,
            outdir,
            save_signals=True,
            save_trades=True,
            save_summary=True,
            save_candidates=True,
        )

    print("[Q7/7] 完成。")
