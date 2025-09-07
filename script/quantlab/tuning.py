# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, random, warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .config import load_config
from .io_utils import load_market_csv_multi, save_outputs
from .signals import compute_indicators
from .market_state import build_index_state_from_panel
from .backtest import backtest_simple
from .buckets import monthly_freeze_bucket_map

warnings.filterwarnings("ignore", category=RuntimeWarning)

def _month_add(dt64: np.datetime64, months: int) -> np.datetime64:
    ts = pd.Timestamp(dt64)
    y = ts.year + (ts.month - 1 + months) // 12
    m = (ts.month - 1 + months) % 12 + 1
    d = min(ts.day, [31,29 if y%4==0 and (y%100!=0 or y%400==0) else 28,31,30,31,30,31,31,30,31,30,31][m-1])
    return np.datetime64(pd.Timestamp(year=y, month=m, day=d).date())

def _generate_windows(dates: pd.Series, train_m: int, val_m: int, step_m: int) -> List[Tuple[np.datetime64, np.datetime64, np.datetime64, np.datetime64]]:
    start = pd.to_datetime(dates.min()).to_datetime64()
    end   = pd.to_datetime(dates.max()).to_datetime64()
    windows = []
    cur_train_start = start
    while True:
        train_end = _month_add(cur_train_start, train_m)
        val_end   = _month_add(train_end, val_m)
        if val_end > end: break
        windows.append((cur_train_start, train_end, train_end, val_end))
        cur_train_start = _month_add(cur_train_start, step_m)
    return windows

def _sample_params(base: dict) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg['S1']['atr_mul'] = round(random.uniform(1.4, 2.2), 2)
    cfg['S1']['atr_mul_pullback'] = round(random.uniform(0.6, 1.2), 2)
    cfg['S1']['rsi_bull_low'] = random.choice([40, 42, 45, 48, 50])
    cfg['S2']['bw_quantile'] = round(random.uniform(0.15, 0.35), 2)
    cfg['S2']['atr_stop_mul'] = round(random.uniform(0.8, 1.4), 2)
    cfg['S3']['rsi_buy'] = random.choice([28, 30, 32, 35, 38, 40])
    cfg['S3']['atr_stop_mul'] = round(random.uniform(1.0, 1.6), 2)
    cfg['stock_state']['adx_trend_th'] = random.choice([22, 25, 28])
    cfg['stock_state']['adx_range_th'] = random.choice([18, 20, 22])
    return cfg

def _slice_by_dates(df: pd.DataFrame, start: np.datetime64, end: np.datetime64) -> pd.DataFrame:
    m = (df['date'] >= pd.Timestamp(start)) & (df['date'] < pd.Timestamp(end))
    return df.loc[m].copy()

def _eval_expectancy(trades: pd.DataFrame):
    if trades is None or trades.empty: return (0.0, 0.0, 0)
    exp_ = float(trades['pnl_pct'].mean())
    winr = float((trades['pnl_pct'] > 0).mean())
    ntrd = int(trades.shape[0])
    return (exp_, winr, ntrd)

def _run_once(df_all: pd.DataFrame, cfg: dict, by_bucket: bool):
    df_ind = []
    for code, sub in df_all.groupby('code'):
        df_ind.append(compute_indicators(sub, cfg))
    df_ind = pd.concat(df_ind, ignore_index=False) if len(df_ind)>0 else pd.DataFrame()
    if df_ind.empty:
        return df_ind, df_ind, df_ind, df_ind
    idx_state = build_index_state_from_panel(df_ind, cfg, by_bucket=by_bucket)
    data = {str(c): g for c, g in df_ind.groupby('code')}
    signals, trades, summary, cands = backtest_simple(data, idx_state, cfg, cost_bp=2.0)
    return signals, trades, summary, cands

def _tune_bucket(df_bucket: pd.DataFrame, base_cfg: dict, windows: List[Tuple], trials: int, by_bucket: bool) -> dict:
    best_score = -1e9; best_cfg = base_cfg
    for _ in tqdm(range(trials), desc="  [bucket] 随机搜索", leave=False):
        cfg = _sample_params(base_cfg)
        scores = []
        for (tr_s, tr_e, vl_s, vl_e) in windows:
            df_v = _slice_by_dates(df_bucket, vl_s, vl_e)
            if df_v.empty: continue
            if 'bucket_id' not in df_v.columns:
                df_v = df_v.copy(); df_v['bucket_id'] = 'B'
            _, trades, _, _ = _run_once(df_v, cfg, by_bucket=by_bucket)
            exp_, winr, ntrd = _eval_expectancy(trades)
            score = exp_ * 1000.0 + winr * 10.0 + (0.0 if ntrd<=0 else 1.0)
            scores.append(score)
        if len(scores)==0: continue
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score; best_cfg = cfg
    return best_cfg

def run_quarterly_tuning(all_in_one_csv: str,
                         outdir: str = "./output",
                         cfg_path: Optional[str] = None,
                         bucket_mode: str = "size",
                         train_months: int = 6,
                         val_months: int = 3,
                         step_months: int = 3,
                         trials: int = 30):
    os.makedirs(outdir, exist_ok=True)
    base_cfg = load_config(cfg_path)
    print("[Q1/6] 加载CSV（可多文件，以逗号分隔）...")
    csvs = [p.strip() for p in all_in_one_csv.split(",")]
    df = load_market_csv_multi(csvs)

    print(f"[Q2/6] 生成 {bucket_mode} 分桶（月度冻结） ...")
    bucket_map = monthly_freeze_bucket_map(df, mode=bucket_mode, k=3, code_industry=None)
    df = df.merge(bucket_map, on='code', how='left')
    if 'bucket_id' not in df.columns:
        df['bucket_id'] = 'ALL'

    print("[Q3/6] 构造滚动窗口 ...")
    windows = _generate_windows(df['date'], train_months, val_months, step_months)
    print(f"  窗口数：{len(windows)}")

    tuned_by_bucket: Dict[str, dict] = {}
    by_bucket = True
    for b, sub in tqdm(df.groupby('bucket_id'), desc="[Q4/6] 按桶调参"):
        best_cfg = _tune_bucket(sub, base_cfg, windows, trials=trials, by_bucket=by_bucket)
        tuned_by_bucket[str(b)] = best_cfg

    save_path = os.path.join(outdir, f"tuned_config_quarterly_{pd.to_datetime(df['date'].max()):%Y%m%d}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(tuned_by_bucket, f, ensure_ascii=False, indent=2)
    print(f"[Q5/6] 调参结果写出：{save_path}")

    if len(windows) > 0:
        _, _, vl_s, vl_e = windows[-1]
        df_last = df[(df['date']>=pd.Timestamp(vl_s)) & (df['date']<pd.Timestamp(vl_e))].copy()
        signals_all, trades_all, summary_all, cands_all = [], [], [], []
        for b, sub in df_last.groupby('bucket_id'):
            cfg_b = tuned_by_bucket.get(str(b), base_cfg)
            sig, trd, summ, cand = _run_once(sub, cfg_b, by_bucket=True)
            if sig is not None and not sig.empty: signals_all.append(sig)
            if trd is not None and not trd.empty: trades_all.append(trd)
            if summ is not None and not summ.empty: summary_all.append(summ)
            if cand is not None and not cand.empty: cands_all.append(cand)
        signals_all = pd.concat(signals_all, ignore_index=True) if signals_all else pd.DataFrame()
        trades_all = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
        summary_all = pd.concat(summary_all, ignore_index=True) if summary_all else pd.DataFrame()
        cands_all = pd.concat(cands_all, ignore_index=True) if cands_all else pd.DataFrame()
        save_outputs(signals_all, trades_all, summary_all, cands_all, outdir,
                     save_signals=True, save_trades=True, save_summary=True, save_candidates=True)
        print("[Q6/6] 用最优参数回放最后验证窗并写出结果。")
