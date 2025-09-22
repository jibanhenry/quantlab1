# -*- coding: utf-8 -*-
"""
ML integration for return prediction, priority XGBoost with RF fallback.
- Train: see scripts/train_ml.py
- Inference:
    - add_predictions_to_candidates(df) to append:
        'predicted_return', 'predicted_bin' (abs), 'predicted_bin_abs', 'predicted_bin_rel'
    - predict_for_code_date(df_ind, keys_df) for trades ledger enrichment
"""
from __future__ import annotations
import os
import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load

# Try XGBoost first; if unavailable, fallback to RandomForest
_USE_XGB_DEFAULT = True
try:
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
    _USE_XGB_DEFAULT = False

from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------------
# Paths (relative to this file's directory: quantlab/)
# ---------------------------------------------------------------------
DEFAULT_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "models", "ml_return_model.pkl")
DEFAULT_THRESH_PATH = os.path.join(os.path.dirname(__file__), "models", "ml_quintile_thresholds.json")

# ---------------------------------------------------------------------
# Feature set (intersection will be used at runtime)
# ---------------------------------------------------------------------
DEFAULT_FEATURES: List[str] = [
    "open","close","preclose","volume","amount","turnover","pb_mrq","ps_ttm",
    "ema20","ema50","ema200",
    "macd_dif","macd_dea","macd_hist",
    "rsi14","atr14","atr_pct",
    "boll_mid","boll_up","boll_low","boll_bw",
    "obv",
    "plus_di14","minus_di14","adx14",
    "cci20","roc10","kdj_k","kdj_d","wr14","psar",
    "market_state_index","market_state_stock",
    "s1_pos","s2_pos","s3_pos",
]

def _select_features(df: pd.DataFrame, feature_list: Optional[List[str]]=None) -> Tuple[pd.DataFrame, List[str]]:
    feats = feature_list or DEFAULT_FEATURES
    use_cols = [c for c in feats if c in df.columns]
    # 安全数值化 + 清洗
    X = df[use_cols].apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, use_cols

# ---------------------------------------------------------------------
# Thresholds loading / bin assignment
# ---------------------------------------------------------------------
def _load_quintile_thresholds(thresh_path: Optional[str] = None):
    """
    Returns:
      thresholds: [q20, q40, q60, q80] or None
      pred_stats: dict or None (keys: mean, std, min, max, median)
      path_used: str
    """
    p = thresh_path or os.getenv("QUANTLAB_ML_THRESHOLDS", DEFAULT_THRESH_PATH)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        qs = data.get("quintile_thresholds") or data
        keys = ["0.2", "0.4", "0.6", "0.8"]
        thresholds = [float(qs[k]) for k in keys]
        pred_stats = data.get("pred_stats") or None
        return thresholds, pred_stats, p
    except Exception:
        return None, None, p

def _assign_bin_from_thresholds(p_value: float, thresholds: list) -> int:
    # thresholds: [q20, q40, q60, q80]
    if thresholds is None or len(thresholds) != 4:
        return 0  # 未分配/未知
    if p_value != p_value:  # NaN
        return 0
    q20, q40, q60, q80 = thresholds
    if p_value <= q20: return 1
    if p_value <= q40: return 2
    if p_value <= q60: return 3
    if p_value <= q80: return 4
    return 5

def _intra_day_bin(series: pd.Series, q: int = 5) -> pd.Series:
    """当日相对排名分桶（不依赖阈值）。"""
    r = series.rank(method="first")
    jitter = 1e-9 * np.random.RandomState(42).randn(len(r))  # 避免重复边界
    return pd.qcut(r + jitter, q, labels=list(range(1, q+1))).astype(int)

# ---------------------------------------------------------------------
# Inference on candidates (daily)
# ---------------------------------------------------------------------
def add_predictions_to_candidates(cands: Optional[pd.DataFrame],
                                  model_path: Optional[str]=None,
                                  min_features: int=5,
                                  thresholds_path: Optional[str]=None) -> pd.DataFrame:
    if cands is None or cands.empty:
        return cands

    model_p = model_path or os.getenv("QUANTLAB_ML_MODEL", DEFAULT_MODEL_PATH)
    if not os.path.exists(model_p):
        print(f"[ML] model not found at {model_p}; skip predictions.")
        return cands

    # Load model
    try:
        reg = load(model_p)
    except Exception as e:
        print(f"[ML] failed to load model: {e}; skip predictions.")
        return cands

    # Build feature matrix
    X, cols = _select_features(cands)
    if len(cols) < min_features:
        print(f"[ML] only {len(cols)} usable features in candidates; need >= {min_features}. Skip.")
        return cands

    # Align to training-time column order if available
    feat_order = getattr(reg, "_feature_names", None)
    if feat_order:
        missing = [c for c in feat_order if c not in X.columns]
        extra = [c for c in X.columns if c not in feat_order]
        for col in missing:
            X[col] = 0.0
        X = X[feat_order]
        print(f"[ML][debug] features matched={len(feat_order)-len(missing)}/{len(feat_order)}, "
              f"missing={len(missing)}, extra_ignored={len(extra)}")

    # Predict
    try:
        y_pred = reg.predict(X.to_numpy(dtype=float))
    except Exception as e:
        print(f"[ML] prediction error: {e}; skip predictions.")
        return cands

    out = cands.copy()
    out['predicted_return'] = y_pred.astype(float)

    # Load thresholds (+ training pred stats) and assign bins
    thresholds, pred_stats, th_path = _load_quintile_thresholds(thresholds_path)
    print(f"[ML][debug] thresholds_file={th_path}  thresholds={thresholds}  pred_stats={pred_stats}")

    # Absolute bin (cross-day comparable)
    if thresholds:
        use_calib = os.getenv("QUANTLAB_ML_CALIBRATE_Z", "0").lower() in ("1","true","yes")
        if use_calib and pred_stats and pred_stats.get("std", 0):
            mu, sigma = float(pred_stats["mean"]), float(pred_stats["std"])
            z_preds = (out['predicted_return'] - mu) / sigma
            z_thresholds = [ (t - mu) / sigma for t in thresholds ]
            out['predicted_bin_abs'] = [ _assign_bin_from_thresholds(z, z_thresholds) for z in z_preds ]
            print(f"[ML][debug] used z-calibration with mu={mu:.6g}, sigma={sigma:.6g}")
        else:
            out['predicted_bin_abs'] = [ _assign_bin_from_thresholds(v, thresholds) for v in out['predicted_return'] ]
    else:
        out['predicted_bin_abs'] = 0  # no thresholds

    # Relative bin (within-day ranking)
    try:
        out['predicted_bin_rel'] = _intra_day_bin(out['predicted_return'])
    except Exception as e:
        print(f"[ML][debug] relative bin failed: {e}")
        out['predicted_bin_rel'] = 0

    # Backward-compat: keep original column name
    out['predicted_bin'] = out['predicted_bin_abs']

    # Quick distribution debug
    try:
        mn = float(np.nanmin(out['predicted_return']))
        md = float(np.nanmedian(out['predicted_return']))
        mx = float(np.nanmax(out['predicted_return']))
        print(f"[ML][debug] preds(min/med/max)={mn:.6f}/{md:.6f}/{mx:.6f}")
        vc = out['predicted_bin_abs'].value_counts(normalize=True, dropna=False)
        if len(vc) and vc.max() >= 0.8:
            dom = int(vc.idxmax())
            print(f"[ML][warn] {vc.max()*100:.1f}% samples fell into ABS bin={dom}. "
                  f"Likely thresholds/scale mismatch. You can rely on 'predicted_bin_rel' for same-day ranking, "
                  f"or enable z-calibration via QUANTLAB_ML_CALIBRATE_Z=1.")
    except Exception:
        pass

    return out

# ---------------------------------------------------------------------
# Inference for trades ledger (by code+date)
# ---------------------------------------------------------------------
def predict_for_code_date(df_ind: pd.DataFrame,
                          keys_df: pd.DataFrame,
                          model_path: Optional[str]=None,
                          thresholds_path: Optional[str]=None) -> pd.DataFrame:
    """
    Predict for given (code, date) keys using the indicator panel df_ind.
    Returns a DataFrame with ['code','date','predicted_return','predicted_bin_abs','predicted_bin_rel'].
    """
    if keys_df is None or keys_df.empty:
        return pd.DataFrame(columns=['code','date','predicted_return','predicted_bin_abs','predicted_bin_rel'])

    model_p = model_path or os.getenv("QUANTLAB_ML_MODEL", DEFAULT_MODEL_PATH)
    if not os.path.exists(model_p):
        print(f"[ML] model not found at {model_p}; cannot predict for ledger.")
        return pd.DataFrame(columns=['code','date','predicted_return','predicted_bin_abs','predicted_bin_rel'])

    try:
        reg = load(model_p)
    except Exception as e:
        print(f"[ML] failed to load model: {e}; cannot predict for ledger.")
        return pd.DataFrame(columns=['code','date','predicted_return','predicted_bin_abs','predicted_bin_rel'])

    # Prepare keys and base table
    keys = keys_df[['code','date']].copy()
    keys['code'] = keys['code'].astype(str)
    keys['date'] = pd.to_datetime(keys['date'])

    base = df_ind.copy()
    base['code'] = base['code'].astype(str)
    base['date'] = pd.to_datetime(base['date'])

    feat = pd.merge(keys, base, on=['code','date'], how='left')

    # Feature matrix and alignment
    X, cols = _select_features(feat)
    feat_order = getattr(reg, "_feature_names", None)
    if feat_order:
        for col in feat_order:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feat_order]

    # Predict
    try:
        y_pred = reg.predict(X.to_numpy(dtype=float))
    except Exception as e:
        print(f"[ML] prediction error on ledger: {e}")
        return pd.DataFrame(columns=['code','date','predicted_return','predicted_bin_abs','predicted_bin_rel'])

    out = keys.copy()
    out['predicted_return'] = y_pred.astype(float)

    # Absolute bin (thresholds)
    thresholds, pred_stats, th_path = _load_quintile_thresholds(thresholds_path)
    if thresholds:
        use_calib = os.getenv("QUANTLAB_ML_CALIBRATE_Z", "0").lower() in ("1","true","yes")
        if use_calib and pred_stats and pred_stats.get("std", 0):
            mu, sigma = float(pred_stats["mean"]), float(pred_stats["std"])
            z_vals = (out['predicted_return'] - mu) / sigma
            z_thresholds = [ (t - mu) / sigma for t in thresholds ]
            out['predicted_bin_abs'] = [ _assign_bin_from_thresholds(z, z_thresholds) for z in z_vals ]
        else:
            out['predicted_bin_abs'] = [ _assign_bin_from_thresholds(v, thresholds) for v in out['predicted_return'] ]
    else:
        out['predicted_bin_abs'] = 0

    # Relative bin
    try:
        out['predicted_bin_rel'] = _intra_day_bin(out['predicted_return'])
    except Exception:
        out['predicted_bin_rel'] = 0

    return out

# ---------------------------------------------------------------------
# Training API
# ---------------------------------------------------------------------
def train_regressor(df_features: pd.DataFrame, target_col: str="pnl_pct",
                    use_xgb: Optional[bool]=None,
                    random_state: int=2024):
    X, cols = _select_features(df_features)
    y = df_features[target_col].astype(float).values

    # env override
    if use_xgb is None:
        env = os.getenv("QUANTLAB_USE_XGBOOST", "auto").lower()
        if env in ("1", "true", "yes"):
            use_xgb = True
        elif env in ("0", "false", "no"):
            use_xgb = False
        else:
            use_xgb = _USE_XGB_DEFAULT

    if use_xgb and _HAS_XGB:
        reg = XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
    else:
        reg = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
            oob_score=False,
        )

    reg.fit(X, y)
    try:
        reg._feature_names = cols  # type: ignore[attr-defined]
    except Exception:
        pass
    return reg

def save_model(model, path: Optional[str]=None) -> str:
    p = path or DEFAULT_MODEL_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    dump(model, p)
    return p