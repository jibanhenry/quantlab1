# -*- coding: utf-8 -*-
import os
import pandas as pd
from typing import Dict, Optional, List
from tqdm.auto import tqdm
from .config import load_config
from .io_utils import load_market_csv_multi, save_outputs
from .signals import compute_indicators
from .market_state import build_index_state_from_panel
from .backtest import backtest_simple

def _group_to_dict(df_ind: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    res = {}
    for code, sub in df_ind.groupby('code'):
        res[str(code)] = sub.copy()
    return res

def _compute_indicators_with_progress(df_all: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    results: List[pd.DataFrame] = []
    for code, sub in tqdm(df_all.groupby('code'), desc="[指标] compute per-code"):
        results.append(compute_indicators(sub, cfg))
    return pd.concat(results, ignore_index=False)

def daily_run(all_csvs: List[str],
              cfg_path: Optional[str]=None,
              outdir: str="./output",
              bucket_map_csv: Optional[str]=None,
              save_signals: bool=True,
              save_trades: bool=True,
              save_summary: bool=True,
              save_candidates: bool=True):
    cfg = load_config(cfg_path)
    print("[1/5] 读取多个CSV并合并去重...")
    df_all = load_market_csv_multi(all_csvs)

    print("[2/5] 计算技术指标（按股票进度展示）...")
    df_ind = _compute_indicators_with_progress(df_all, cfg)

    if bucket_map_csv and os.path.exists(bucket_map_csv):
        print(f"[3/5] 加载分桶映射：{bucket_map_csv}")
        try:
            bm = pd.read_csv(bucket_map_csv)
            df_ind = df_ind.merge(bm, on='code', how='left')
        except Exception as e:
            print(f"  分桶映射加载失败（忽略）：{e}")
    else:
        print("[3/5] 未提供分桶映射，跳过。")

    print("[4/5] 构造市场气候（全市场/分桶）...")
    idx_state = build_index_state_from_panel(df_ind, cfg, by_bucket=('bucket_id' in df_ind.columns))

    print("[5/5] 回测与聚合（按股票进度展示）...")
    indicator_dict = _group_to_dict(df_ind)
    signals, trades, summary, cands = backtest_simple(indicator_dict, idx_state, cfg, cost_bp=2.0)

    save_outputs(signals, trades, summary, cands, outdir,
                 save_signals=save_signals,
                 save_trades=save_trades,
                 save_summary=save_summary,
                 save_candidates=save_candidates)
    print(f"完成。输出目录：{outdir}")
