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
    res: Dict[str, pd.DataFrame] = {}
    for code, sub in df_ind.groupby('code'):
        res[str(code)] = sub.copy()
    return res

def _compute_indicators_with_progress(df_all: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """按股票分组计算技术指标，带进度条，合并回一个大表。"""
    out = []
    for code, sub in tqdm(df_all.groupby('code'), desc='compute_indicators(code)', total=df_all['code'].nunique()):
        sub = sub.sort_values('date').reset_index(drop=True)
        ind = compute_indicators(sub, cfg)
        out.append(ind)
    df_ind = pd.concat(out, axis=0, ignore_index=True)
    return df_ind

def _ensure_keys(df: pd.DataFrame) -> pd.DataFrame:
    """确保存在 ['code','date'] 两列，类型正确。"""
    d = df.copy()
    # 统一 code
    if 'code' not in d.columns:
        if 'symbol' in d.columns:
            d['code'] = d['symbol'].astype(str)
        else:
            raise KeyError("neither 'code' nor 'symbol' found in DataFrame")
    else:
        d['code'] = d['code'].astype(str)
    # 统一 date
    if 'date' not in d.columns:
        if 'entry_date' in d.columns:
            d['date'] = pd.to_datetime(d['entry_date'])
        else:
            # 如果只有 'day' 或其他，请按你本地字段补充
            raise KeyError("neither 'date' nor 'entry_date' found in DataFrame")
    else:
        d['date'] = pd.to_datetime(d['date'])
    return d

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
        try:
            bm = pd.read_csv(bucket_map_csv)
            if 'code' in bm.columns and 'bucket_id' in bm.columns:
                print(f"[3/5] 载入分桶映射：{bucket_map_csv}")
                df_ind = df_ind.merge(bm, on='code', how='left')
        except Exception as e:
            print(f"[3/5] 分桶映射载入失败（跳过 by_bucket）：{e}")

    print("[4/5] 构造市场气候（全市场/分桶）...")
    idx_state = build_index_state_from_panel(df_ind, cfg, by_bucket=('bucket_id' in df_ind.columns))

    print("[5/5] 回测与聚合（按股票进度展示）...")
    indicator_dict = _group_to_dict(df_ind)
    signals, trades, summary, cands = backtest_simple(indicator_dict, idx_state, cfg, cost_bp=2.0)

    # ========= 修复点 1：候选与指标合并后再做 ML 预测 =========
    try:
        from .model import add_predictions_to_candidates, predict_for_code_date
        # 确保候选有 ['code','date']，并与 df_ind 合并获得完整特征
        cands_keys = _ensure_keys(cands)[['code','date']]
        df_ind_keys = _ensure_keys(df_ind)[['code','date']]
        # 只把候选的键 join 到 df_ind 上拿全量特征
        cands_features = cands.merge(df_ind, on=['code','date'], how='left', suffixes=('', '_ind'))
        # 喂给模型
        cands = add_predictions_to_candidates(cands_features)
    except Exception as _ml_e:
        print(f"[ML] prediction step skipped (candidates): {_ml_e}")

    # ========= 修复点 2：给账本 trades 也追加 ML 预测 =========
    try:
        trades = _ensure_keys(trades)
        key_df = trades[['code','date']].drop_duplicates()
        pred_ledger = predict_for_code_date(df_ind, key_df)
        trades = trades.merge(pred_ledger, on=['code','date'], how='left')
    except Exception as _e:
        print(f"[ML] prediction step skipped (trades ledger): {_e}")

    save_outputs(signals, trades, summary, cands, outdir,
                 save_signals=save_signals,
                 save_trades=save_trades,
                 save_summary=save_summary,
                 save_candidates=save_candidates)
    print(f"完成。输出目录：{outdir}")

def main():
    # 允许通过命令行执行本文件进行日常跑批
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csvs', required=True, help='多个CSV，逗号分隔；也可由上游 main 计算得到')
    ap.add_argument('--cfg', default=None, help='配置文件路径（yaml）。为空则用默认配置')
    ap.add_argument('--outdir', default='./output', help='输出目录')
    ap.add_argument('--bucket_map_csv', default=None, help='可选：代码到分桶ID的映射表')
    ap.add_argument('--save_signals', type=int, default=1)
    ap.add_argument('--save_trades', type=int, default=1)
    ap.add_argument('--save_summary', type=int, default=1)
    ap.add_argument('--save_candidates', type=int, default=1)
    args = ap.parse_args()

    csvs = [s.strip() for s in args.csvs.split(',') if s.strip()]
    daily_run(csvs,
              cfg_path=args.cfg,
              outdir=args.outdir,
              bucket_map_csv=args.bucket_map_csv,
              save_signals=bool(args.save_signals),
              save_trades=bool(args.save_trades),
              save_summary=bool(args.save_summary),
              save_candidates=bool(args.save_candidates))

if __name__ == "__main__":
    main()