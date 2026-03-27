# -*- coding: utf-8 -*-
"""
Train ML model on ALL historical trades (no holdout) and produce IN-SAMPLE metrics.
Also exports quintile thresholds for daily predicted_bin tagging.
"""

# 允许直接运行 train_ml.py：自动把父目录塞进 sys.path，让 "from .xxx" 可用
if __name__ == "__main__" and __package__ is None:
    import sys, pathlib
    pkg_parent = pathlib.Path(__file__).resolve().parent.parent
    if str(pkg_parent) not in sys.path:
        sys.path.insert(0, str(pkg_parent))
    __package__ = "quantlab"

import argparse
import glob
import json
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from .config import load_config
from .io_utils import load_market_csv_multi
from . import model as model_mod
from .model import train_regressor, save_model, DEFAULT_FEATURES
from .signals import compute_indicators
from .valuation import add_valuation_features


# ============== 增强点：把 market_state_* & s*_pos 纳入训练特征 ==============
EXTRA_FEATURES = [
    "market_state_index", "market_state_stock",
    "s1_pos", "s2_pos", "s3_pos",
]
# 训练前动态扩展 model.DEFAULT_FEATURES（不用改 model.py 文件）
model_mod.DEFAULT_FEATURES[:] = list(dict.fromkeys(model_mod.DEFAULT_FEATURES + EXTRA_FEATURES))
# ======================================================================


def _expand_csvs(csvs_arg: str) -> List[str]:
    parts = [p.strip() for p in csvs_arg.split(',') if p.strip()]
    files: List[str] = []
    for p in parts:
        if any(ch in p for ch in ['*', '?', '[']):
            files.extend(sorted(glob.glob(p)))
        else:
            files.append(p)
    files = [f for f in dict.fromkeys(files) if os.path.exists(f)]
    return files


def _pick_strategy_column(trades: pd.DataFrame) -> str:
    """从 trades 中自动识别策略列名，统一映射为 'strategy'。不存在则创建 'unknown'。"""
    candidates = ["strategy", "strategy_name", "sid", "signal_id", "signal", "strategy_id"]
    for c in candidates:
        if c in trades.columns:
            trades["strategy"] = trades[c].astype(str)
            return "strategy"
    trades["strategy"] = "unknown"
    return "strategy"


def _ensure_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """若训练表中缺少扩展特征，则补 0.0 列，确保与 pipeline 候选的列集合一致。"""
    for col in EXTRA_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _build_training_frame(ledger_csv: str, csvs: List[str], cfg_path: Optional[str]) -> pd.DataFrame:
    """将 trades_ledger 的进场快照与当日指标对齐，构建训练样本。"""
    trades = pd.read_csv(ledger_csv, parse_dates=['entry_date', 'exit_date'])
    if trades.empty:
        raise RuntimeError("trades_ledger.csv is empty.")

    trades['code'] = trades['symbol'].astype(str)
    trades['date'] = pd.to_datetime(trades['entry_date'])
    _pick_strategy_column(trades)

    df_all = load_market_csv_multi(csvs)
    cfg = load_config(cfg_path)  # cfg_path 可为 None
    ind_parts = []
    for _, sub in df_all.groupby("code"):
        ind_parts.append(compute_indicators(sub.sort_values("date").reset_index(drop=True), cfg))
    ind = pd.concat(ind_parts, ignore_index=True) if ind_parts else pd.DataFrame()
    ind = add_valuation_features(ind, cfg)

    keys = ['code', 'date']
    feat_cols = [c for c in ind.columns if c not in ('date', 'code')]
    features = ind[keys + feat_cols].copy()

    features['code'] = features['code'].astype(str)
    features['date'] = pd.to_datetime(features['date'])

    train_df = pd.merge(
        trades[['code', 'date', 'pnl_pct', 'strategy']],
        features,
        on=['code', 'date'],
        how='inner'
    ).copy()

    train_df = train_df.dropna(subset=['pnl_pct'])

    num_cols = [c for c in train_df.columns if c not in ('code', 'date', 'strategy')]
    for c in num_cols:
        if c != 'pnl_pct':
            train_df[c] = pd.to_numeric(train_df[c], errors='coerce')
    train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    train_df = _ensure_extra_features(train_df)
    return train_df


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    var = float(np.var(y_true))
    r2 = float(1.0 - np.sum(err ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)) if var > 1e-12 else float('nan')
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float('nan')
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson}


def _quantile_report(y_true: np.ndarray, y_pred: np.ndarray, q: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "p": y_pred}).sort_values("p").reset_index(drop=True)
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q, labels=list(range(1, q + 1)))
    rep = (
        df.groupby("bin")["y"]
        .agg(
            count="count",
            mean_y="mean",
            win_rate=lambda s: float((s > 0).mean()),
            avg_win=lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0,
            avg_loss=lambda s: float(s[s <= 0].mean()) if (s <= 0).any() else 0.0,
        )
        .reset_index()
    )
    rep["bin"] = rep["bin"].astype(int)
    return rep


def _quantile_report_detail(df_eval: pd.DataFrame, q: int = 5, by: str = "strategy") -> pd.DataFrame:
    df = df_eval.sort_values("p").reset_index(drop=True).copy()
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q, labels=list(range(1, q + 1)))
    rep = (
        df.groupby(["bin", by])["y"]
        .agg(
            count="count",
            mean_y="mean",
            win_rate=lambda s: float((s > 0).mean()),
            avg_win=lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0,
            avg_loss=lambda s: float(s[s <= 0].mean()) if (s <= 0).any() else 0.0,
        )
        .reset_index()
    )
    rep["bin"] = rep["bin"].astype(int)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--csvs',
        default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2020-2025_all.csv,"
                "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv",
        help='Comma-separated paths or globs to market CSVs (default: data/*.csv)'
    )
    ap.add_argument('--cfg', default=None, help='quantlab config path (default: config.yaml)')
    ap.add_argument('--ledger', default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/output/quantlab/trades_ledger.csv",
                    help='Path to trades_ledger.csv (default: ./output/trades_ledger.csv)')
    ap.add_argument('--out', default="./models/ml_return_model.pkl", help='Output model path')
    ap.add_argument('--xgb', type=int, default=True, help='1 to use XGBoost, 0 to use RandomForest, None=auto')
    ap.add_argument('--metrics_out', default="./models/ml_metrics.json", help='Where to save metrics JSON')
    ap.add_argument('--quintile_csv', default="./models/ml_quintiles.csv", help='Where to save quintile analysis CSV')
    ap.add_argument('--quintile_by_strategy_csv', default="./models/ml_quintiles_by_strategy.csv",
                    help='Where to save quintile-by-strategy CSV')
    ap.add_argument('--quintile_thresholds', default="./models/ml_quintile_thresholds.json",
                    help='Where to save quintile thresholds JSON')
    args = ap.parse_args()

    csv_files = _expand_csvs(args.csvs)
    if not csv_files:
        raise SystemExit(f"No market CSVs found for pattern(s): {args.csvs}")

    train_df = _build_training_frame(args.ledger, csv_files, args.cfg)
    if train_df.empty:
        raise SystemExit("No training samples after merging indicators and ledger.")

    model = train_regressor(train_df, target_col='pnl_pct', use_xgb=(None if args.xgb is None else bool(args.xgb)))
    out_path = save_model(model, args.out)

    # —— 预测阶段使用“训练时的特征顺序” —— #
    y_true = train_df['pnl_pct'].to_numpy(dtype=float)
    feature_order = getattr(model, "_feature_names", None)
    if feature_order is not None:
        cols = [c for c in feature_order if c in train_df.columns]
    else:
        cols = [c for c in DEFAULT_FEATURES if c in train_df.columns]

    X = train_df[cols].apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y_pred = model.predict(X.values)

    # 指标 & 报表
    metrics = _regression_metrics(y_true, y_pred)
    rep = _quantile_report(y_true, y_pred, q=5)

    df_eval = pd.DataFrame({"y": y_true, "p": y_pred, "strategy": train_df["strategy"].astype(str)})
    rep_by_strategy = _quantile_report_detail(df_eval, q=5, by="strategy")

    # ===== 导出训练集预测的五分位阈值 + 预测分布统计 =====
    qs = df_eval["p"].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
    qs_str_keys = {str(k): float(v) for k, v in qs.items()}
    pred_stats = {
        "mean": float(np.mean(y_pred)),
        "std":  float(np.std(y_pred)),
        "min":  float(np.min(y_pred)),
        "max":  float(np.max(y_pred)),
        "median": float(np.median(y_pred)),
    }

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.quintile_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.quintile_by_strategy_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.quintile_thresholds), exist_ok=True)

    with open(args.metrics_out, 'w', encoding='utf-8') as f:
        json.dump({"samples": int(len(y_true)), "features": len(cols), "metrics": metrics},
                  f, ensure_ascii=False, indent=2)
    rep.to_csv(args.quintile_csv, index=False, encoding='utf-8-sig')
    rep_by_strategy.to_csv(args.quintile_by_strategy_csv, index=False, encoding='utf-8-sig')
    with open(args.quintile_thresholds, 'w', encoding='utf-8') as f:
        json.dump({"quintile_thresholds": qs_str_keys, "pred_stats": pred_stats}, f, ensure_ascii=False, indent=2)

    # 额外打印一行阈值摘要（排查“全 5”更直观）
    print(f"[train_ml] Quintile thresholds (q20/q40/q60/q80): "
          f"{qs_str_keys.get('0.2'):.6g}, {qs_str_keys.get('0.4'):.6g}, "
          f"{qs_str_keys.get('0.6'):.6g}, {qs_str_keys.get('0.8'):.6g}")
    print(f"[train_ml] Pred stats mean={pred_stats['mean']:.6g} std={pred_stats['std']:.6g} "
          f"min={pred_stats['min']:.6g} med={pred_stats['median']:.6g} max={pred_stats['max']:.6g}")

    print(f"[train_ml] Model saved to: {out_path}. Samples: {len(train_df)}  Features used: {len(cols)}")
    print(f"[train_ml] In-sample metrics: MAE={metrics['mae']:.6f}  RMSE={metrics['rmse']:.6f}  R2={metrics['r2']:.4f}  Pearson={metrics['pearson']:.4f}")
    print(f"[train_ml] Quintile analysis saved to: {args.quintile_csv}")
    print(f"[train_ml] Quintile-by-Strategy analysis saved to: {args.quintile_by_strategy_csv}")
    print(f"[train_ml] Quintile thresholds saved to: {args.quintile_thresholds}")
    print(f"[train_ml] Metrics JSON saved to: {args.metrics_out}")


if __name__ == '__main__':
    main()
