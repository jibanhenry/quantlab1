# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd
from typing import List

CHN2ENG = {
    "code": "code","日期": "date","开盘": "open","最高": "high","最低": "low","收盘": "close",
    "前收": "preclose","成交量": "volume","成交额": "amount","换手率": "turnover",
    "涨跌幅": "pct_chg","pbMRQ": "pb_mrq","psTTM": "ps_ttm",
}
NEEDED = ['code','date','open','high','low','close','preclose','volume','amount','turnover','pct_chg','pb_mrq','ps_ttm']

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={cn: CHN2ENG[cn] for cn in CHN2ENG if cn in df.columns})
    for c in NEEDED:
        if c not in df.columns:
            df[c] = np.nan
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_market_csv_multi(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p: continue
        local = pd.read_csv(p)
        local = _normalize_cols(local)
        frames.append(local)
    if not frames:
        raise FileNotFoundError("未读取到任何CSV，请检查 --csv 参数路径")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(['code','date']).drop_duplicates(subset=['code','date'], keep='last').reset_index(drop=True)
    cols = list(dict.fromkeys(NEEDED + [c for c in df.columns if c not in NEEDED]))
    return df[cols]

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def save_outputs(signals_daily: pd.DataFrame,
                 trades_ledger: pd.DataFrame,
                 strategy_summary: pd.DataFrame,
                 candidates_today: pd.DataFrame,
                 outdir: str,
                 save_signals: bool=True,
                 save_trades: bool=True,
                 save_summary: bool=True,
                 save_candidates: bool=True):
    ensure_dir(outdir)
    if save_signals and signals_daily is not None and not signals_daily.empty:
        signals_daily.to_csv(os.path.join(outdir, "signals_daily.csv"), index=False, encoding='utf-8-sig')
    if save_trades and trades_ledger is not None and not trades_ledger.empty:
        trades_ledger.to_csv(os.path.join(outdir, "trades_ledger.csv"), index=False, encoding='utf-8-sig')
    if save_summary and strategy_summary is not None and not strategy_summary.empty:
        strategy_summary.to_csv(os.path.join(outdir, "strategy_summary.csv"), index=False, encoding='utf-8-sig')
    if save_candidates and candidates_today is not None and not candidates_today.empty:
        last_day = pd.to_datetime(candidates_today['date'].max()).strftime("%Y%m%d")
        candidates_today.to_csv(os.path.join(outdir, f"candidates_{last_day}.csv"), index=False, encoding='utf-8-sig')
