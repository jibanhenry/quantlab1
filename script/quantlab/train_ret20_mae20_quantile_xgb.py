# -*- coding: utf-8 -*-
import os
import gc
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.metrics import mean_absolute_error

import indicators as ind
from io_utils import load_market_csv_multi


# =========================
# 默认配置：PyCharm 直接运行即可
# =========================

TRAIN_CSV = "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2020-2025_all.csv"
TEST_CSV  = "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv"

DATE_COL = "date"
SYM_COL  = "code"

N = 20
Q_RET = 0.7
Q_MAE = 0.1

VALID_DAYS = 126
OUTDIR = "model_outputs_v2"

SEED = 42
N_ESTIMATORS = 3000
LEARNING_RATE = 0.03
MAX_DEPTH = 6
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
REG_LAMBDA = 1.0
MIN_CHILD_WEIGHT = 10.0
EARLY_STOPPING_ROUNDS = 150

EPS = 1e-12
USE_V2_FEATURES = True

# ✅ 关键开关：TEST 数据往往没有足够历史支撑 252 窗口
USE_LONG_WINDOW = False

# ✅ 关键策略：特征缺失的处理方式
MISSING_FEATURE_STRATEGY = "impute"

# ✅ 内存优化：分批处理 code（batch 越小越省内存，但越慢）
CODE_BATCH_SIZE = 500

# ✅ 内存优化：只保留必要列（建议 True）
KEEP_ONLY_OHLCV = True

# ✅ 内存优化：强制降精度（建议 True）
DOWNCAST_NUMERIC = True


# =========================
# 工具函数
# =========================

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def _maybe_keep_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if not KEEP_ONLY_OHLCV:
        return df
    keep = [DATE_COL, SYM_COL, "open", "high", "low", "close", "volume"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    if not DOWNCAST_NUMERIC:
        return df

    out = df.copy()

    # code/date
    if SYM_COL in out.columns:
        out[SYM_COL] = out[SYM_COL].astype(str)
    if DATE_COL in out.columns:
        out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")

    # OHLCV
    for c in ["open", "high", "low", "close"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    if "volume" in out.columns:
        v = pd.to_numeric(out["volume"], errors="coerce")
        # volume 用 float32 更省内存且足够
        out["volume"] = v.astype("float32")

    return out


def read_csv(path: str) -> pd.DataFrame:
    """
    用 io_utils 读取并标准化列名
    期望输出至少包含：date, code, open, high, low, close, volume
    """
    df = load_market_csv_multi([path])

    # 基础清洗
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if SYM_COL in df.columns:
        df[SYM_COL] = df[SYM_COL].astype(str)

    # 强制把 OHLCV 转成数值
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 内存优化
    df = _maybe_keep_ohlcv(df)
    df = _downcast(df)
    return df


def infer_feature_columns(df: pd.DataFrame, user_features: Optional[List[str]] = None) -> List[str]:
    if user_features:
        missing = [c for c in user_features if c not in df.columns]
        if missing:
            raise ValueError(f"Provided features not found in data: {missing}")
        return user_features

    exclude = {
        DATE_COL, SYM_COL,
        "open", "high", "low", "close", "volume", "amount",
        "entry",
        f"ret_{N}", f"mae_{N}",
        "ret20", "mae20",
    }

    feats: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == "O":
            continue
        feats.append(c)

    if not feats:
        raise ValueError("No feature columns inferred. Please check whether indicators were computed.")
    return feats


def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    return out


def diagnose_dropna(df: pd.DataFrame, needed_cols: List[str], name: str, topk: int = 30):
    miss_rate = df[needed_cols].isna().mean().sort_values(ascending=False)
    print(f"\n[DIAG] {name} missing-rate top{topk}:")
    for c, r in miss_rate.head(topk).items():
        if r > 0:
            print(f"  {c:25s}  missing={r:.3f}")
    print(f"[DIAG] {name} rows before cleaning: {len(df)}")


def fill_features_by_train_stats(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_df.copy()
    te = test_df.copy()
    med = tr[feature_cols].median(numeric_only=True)
    tr[feature_cols] = tr[feature_cols].fillna(med)
    te[feature_cols] = te[feature_cols].fillna(med)
    return tr, te


# =========================
# 指标计算（内存优化版：按 code 分批处理）
# =========================

def _calc_one_stock(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values(DATE_COL).copy()

    close = g["close"]
    vol = g["volume"]

    # 基础
    g["ema_10"] = ind.ema(close, 10)
    g["ema_20"] = ind.ema(close, 20)
    g["ema_60"] = ind.ema(close, 60)

    g["sma_10"] = ind.sma(close, 10)
    g["sma_20"] = ind.sma(close, 20)
    g["sma_60"] = ind.sma(close, 60)

    # 注意：true_range/atr 需要 high/low/close 列，直接传 g
    g["tr"] = ind.true_range(g[["high", "low", "close"]])
    g["atr_14"] = ind.atr(g[["high", "low", "close"]], 14)
    g["atr_pct_14"] = g["atr_14"] / (close.abs() + EPS)

    g["rsi_14"] = ind.rsi(close, 14)

    dif, dea, hist = ind.macd(close, 12, 26, 9)
    g["macd_dif"] = dif
    g["macd_dea"] = dea
    g["macd_hist"] = hist

    mid, up, low, bw = ind.bollinger(close, 20, 2.0)
    g["boll_mid"] = mid
    g["boll_up"] = up
    g["boll_low"] = low
    g["boll_bw"] = bw

    g["obv"] = ind.obv(close, vol)
    g["cci_20"] = ind.cci(g[["high", "low", "close"]], 20)
    g["roc_10"] = ind.roc(close, 10)

    k, d, j = ind.kdj(g[["high", "low", "close"]], 9, 3, 3)
    g["kdj_k"] = k
    g["kdj_d"] = d
    g["kdj_j"] = j

    g["willr_14"] = ind.williams_r(g[["high", "low", "close"]], 14)

    pdi, mdi, adx = ind.dmi_adx(g[["high", "low", "close"]], 14)
    g["pdi_14"] = pdi
    g["mdi_14"] = mdi
    g["adx_14"] = adx

    g["psar"] = ind.psar(g[["high", "low"]], 0.02, 0.2)

    # V2
    if USE_V2_FEATURES:
        g["mdd_60"] = ind.rolling_max_drawdown(close, window=60)
        g["gap_atr_14"] = ind.gap_atr(g[["open", "high", "low", "close"]], atr_n=14)

        g["tr_pct"] = ind.tr_pct(g[["high", "low", "close"]])

        if USE_LONG_WINDOW:
            g["tr_pctile_252"] = ind.tr_pctile(g[["high", "low", "close"]], window=252)
        else:
            g["tr_pctile_252"] = np.nan

        g["clv"] = ind.close_location_value(g[["open", "high", "low", "close"]])
        uw, lw, br = ind.wick_ratios(g[["open", "high", "low", "close"]])
        g["upper_wick_r"] = uw
        g["lower_wick_r"] = lw
        g["body_r"] = br

        g["dist_to_high_20"] = ind.dist_to_high(close, window=20)

        g["breakout_atr_20_14"] = ind.breakout_strength_atr(
            g[["open", "high", "low", "close"]], lookback=20, atr_n=14
        )

        g["rvol_20"] = ind.rvol(vol, window=20)
        g["vol_z20"] = ind.vol_zscore(vol, window=20)

        g["rav"] = ind.range_adjusted_volume(g)
        g["rav_rel_20"] = ind.rav_relative(g, window=20)

        g["vp_div_5_5_20"] = ind.volume_price_divergence(g, price_mom=5, vol_short=5, vol_long=20)

        g["atr_ratio_14_60"] = ind.atr_ratio(g, atr_short=14, atr_long=60)

        g["boll_bw2"] = ind.bollinger_bandwidth(close, n=20, k=2.0)

        if USE_LONG_WINDOW:
            _, bw_q = ind.bollinger_bw_quantile(close, n=20, k=2.0, q_window=252)
            g["boll_bw_q252"] = bw_q
        else:
            g["boll_bw_q252"] = np.nan

    # 内存优化：把新增列尽量压到 float32（不改变结果逻辑）
    if DOWNCAST_NUMERIC:
        new_num = g.select_dtypes(include=[np.number]).columns
        for c in new_num:
            if c not in ["open", "high", "low", "close", "volume"]:
                g[c] = g[c].astype("float32")

    return g


def add_indicator_features_batched(df: pd.DataFrame, batch_size: int = CODE_BATCH_SIZE) -> pd.DataFrame:
    """
    ✅ 内存优化：不要对全市场直接 groupby.apply
    而是按 code 分批处理，拼接结果
    """
    out_list = []
    df = df.sort_values([SYM_COL, DATE_COL])

    codes = df[SYM_COL].dropna().astype(str).unique().tolist()
    total = len(codes)
    if total == 0:
        return df

    for i in range(0, total, batch_size):
        batch_codes = set(codes[i:i + batch_size])
        part = df[df[SYM_COL].isin(batch_codes)].copy()

        # 按 code 分组逐个算，避免 apply 产生巨大中间对象
        res_parts = []
        for code, g in part.groupby(SYM_COL, sort=False):
            res_parts.append(_calc_one_stock(g))

        part_feat = pd.concat(res_parts, axis=0, ignore_index=False)
        part_feat = sanitize_numeric(part_feat)
        out_list.append(part_feat)

        # 释放中间对象
        del part, res_parts, part_feat
        gc.collect()

        print(f"[INFO] indicator batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} done ({min(i+batch_size, total)}/{total} codes)")

    out = pd.concat(out_list, axis=0, ignore_index=False)
    out = out.sort_values([SYM_COL, DATE_COL])
    return out


# =========================
# 标签构造
# =========================

def add_ret_mae_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out = out.sort_values([SYM_COL, DATE_COL])
    g = out.groupby(SYM_COL, group_keys=False)

    out["entry"] = g["open"].shift(-1)
    out[f"ret_{N}"] = g["close"].shift(-N) / (out["entry"] + EPS) - 1.0

    future_low_min = g["low"].shift(-1).rolling(N, min_periods=N).min()
    out[f"mae_{N}"] = future_low_min / (out["entry"] + EPS) - 1.0

    out = sanitize_numeric(out)
    if DOWNCAST_NUMERIC:
        for c in [f"ret_{N}", f"mae_{N}", "entry"]:
            if c in out.columns:
                out[c] = out[c].astype("float32")
    return out


def split_train_valid_by_last_dates(train_df: pd.DataFrame, valid_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.sort_values([DATE_COL, SYM_COL])
    uniq_dates = train_df[DATE_COL].drop_duplicates().sort_values()
    if len(uniq_dates) <= valid_days:
        valid_dates = uniq_dates
    else:
        valid_dates = uniq_dates.iloc[-valid_days:]

    valid = train_df[train_df[DATE_COL].isin(valid_dates)].copy()
    train = train_df[~train_df[DATE_COL].isin(valid_dates)].copy()
    return train, valid


# =========================
# 训练：xgb.train（兼容旧版）
# =========================

def _quantile_huber_obj(q: float, delta: float = 1.0):
    q = float(q)
    delta = float(delta)

    def obj(preds: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        e = y - preds
        abs_e = np.abs(e)

        huber_grad_e = np.where(abs_e <= delta, e, delta * np.sign(e))
        w = np.where(e >= 0, q, 1.0 - q)

        grad = -w * huber_grad_e
        hess = w * np.where(abs_e <= delta, 1.0, 0.0)
        hess = np.clip(hess, 1e-6, None)
        return grad, hess

    return obj


def train_quantile_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    quantile: float,
    seed: int = SEED
) -> xgb.Booster:

    dtrain = xgb.DMatrix(X_train, label=y_train.values)
    dvalid = xgb.DMatrix(X_valid, label=y_valid.values)

    params = {
        "seed": seed,
        "eta": LEARNING_RATE,
        "max_depth": MAX_DEPTH,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "lambda": REG_LAMBDA,
        "min_child_weight": MIN_CHILD_WEIGHT,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",   # ✅ 通常比 auto 更省内存也更快
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=N_ESTIMATORS,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        obj=_quantile_huber_obj(quantile, delta=1.0),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False
    )
    return booster


# =========================
# 评估
# =========================

def quantile_coverage(y_true: np.ndarray, y_pred_q: np.ndarray) -> float:
    return float(np.mean(y_true <= y_pred_q))


def top_bucket_mean(y_true: np.ndarray, score: np.ndarray, top_frac: float = 0.1) -> float:
    n = len(y_true)
    if n == 0:
        return np.nan
    k = max(1, int(n * top_frac))
    idx = np.argsort(score)[-k:]
    return float(np.mean(y_true[idx]))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("xgboost version:", getattr(xgb, "__version__", "unknown"))

    # 1) 读取数据（已做 keep cols + downcast）
    train_raw = read_csv(TRAIN_CSV)
    test_raw = read_csv(TEST_CSV)

    required = {DATE_COL, SYM_COL, "open", "high", "low", "close", "volume"}
    for name, df in [("train", train_raw), ("test", test_raw)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[DEBUG] {name} columns:", df.columns.tolist()[:120])
            raise ValueError(f"{name} data missing columns: {missing}. Please check io_utils output column names.")

    # 2) 指标（✅ 分批处理，显著降内存）
    train_feat = add_indicator_features_batched(train_raw, batch_size=CODE_BATCH_SIZE)
    del train_raw
    gc.collect()

    test_feat = add_indicator_features_batched(test_raw, batch_size=CODE_BATCH_SIZE)
    del test_raw
    gc.collect()

    # 3) 标签
    train = add_ret_mae_labels(train_feat)
    del train_feat
    gc.collect()

    test = add_ret_mae_labels(test_feat)
    del test_feat
    gc.collect()

    y_ret_col = f"ret_{N}"
    y_mae_col = f"mae_{N}"

    # 4) 特征列
    feature_cols = infer_feature_columns(train)
    print("[INFO] feature count:", len(feature_cols))
    print("[INFO] feature sample:", feature_cols[:40])

    # 5) 清理策略
    must_have = ["entry", y_ret_col, y_mae_col]

    if MISSING_FEATURE_STRATEGY == "drop_strict":
        needed = feature_cols + must_have
        train = train.dropna(subset=needed).copy()
        test = test.dropna(subset=needed).copy()
    else:
        train = train.dropna(subset=must_have).copy()
        test = test.dropna(subset=must_have).copy()

        train = sanitize_numeric(train)
        test = sanitize_numeric(test)

        train, test = fill_features_by_train_stats(train, test, feature_cols)

    print(f"[INFO] rows after cleaning: train={len(train)}, test={len(test)}")
    if len(test) == 0:
        print("[ERROR] TEST has 0 usable samples. Usually: test file too short (< N+1 days per code).")
        return

    # 6) 验证集切分
    train_part, valid_part = split_train_valid_by_last_dates(train, VALID_DAYS)
    del train
    gc.collect()

    X_train = train_part[feature_cols]
    X_valid = valid_part[feature_cols]
    X_test = test[feature_cols]

    y_train_ret = train_part[y_ret_col]
    y_valid_ret = valid_part[y_ret_col]
    y_test_ret = test[y_ret_col]

    y_train_mae = train_part[y_mae_col]
    y_valid_mae = valid_part[y_mae_col]
    y_test_mae = test[y_mae_col]

    # 释放不需要的表
    del train_part, valid_part
    gc.collect()

    # 7) 训练
    model_ret = train_quantile_xgb(X_train, y_train_ret, X_valid, y_valid_ret, quantile=Q_RET, seed=SEED)
    model_mae = train_quantile_xgb(X_train, y_train_mae, X_valid, y_valid_mae, quantile=Q_MAE, seed=SEED)

    # 释放训练集矩阵（训练完成后就不需要了）
    del X_train, y_train_ret, y_train_mae
    gc.collect()

    # 8) 预测
    dvalid = xgb.DMatrix(X_valid)
    dtest = xgb.DMatrix(X_test)

    pred_ret_valid = model_ret.predict(dvalid)
    pred_mae_valid = model_mae.predict(dvalid)

    pred_ret_test = model_ret.predict(dtest)
    pred_mae_test = model_mae.predict(dtest)

    # 9) 评估
    valid_ret_pin = pinball_loss(y_valid_ret.values, pred_ret_valid, Q_RET)
    valid_mae_pin = pinball_loss(y_valid_mae.values, pred_mae_valid, Q_MAE)
    test_ret_pin = pinball_loss(y_test_ret.values, pred_ret_test, Q_RET)
    test_mae_pin = pinball_loss(y_test_mae.values, pred_mae_test, Q_MAE)

    valid_ret_cov = quantile_coverage(y_valid_ret.values, pred_ret_valid)
    valid_mae_cov = quantile_coverage(y_valid_mae.values, pred_mae_valid)
    test_ret_cov = quantile_coverage(y_test_ret.values, pred_ret_test)
    test_mae_cov = quantile_coverage(y_test_mae.values, pred_mae_test)

    valid_top_ret = top_bucket_mean(y_valid_ret.values, pred_ret_valid, top_frac=0.1)
    test_top_ret = top_bucket_mean(y_test_ret.values, pred_ret_test, top_frac=0.1)

    valid_top_mae = top_bucket_mean(y_valid_mae.values, pred_mae_valid, top_frac=0.1)
    test_top_mae = top_bucket_mean(y_test_mae.values, pred_mae_test, top_frac=0.1)

    print("========== Quantile XGBoost Training Summary ==========")
    print(f"[VALID] Return Q{Q_RET} ret{N}: pinball={valid_ret_pin:.6f}, MAE={mean_absolute_error(y_valid_ret, pred_ret_valid):.6f}, coverage={valid_ret_cov:.4f}")
    print(f"[VALID] Risk   Q{Q_MAE} mae{N}: pinball={valid_mae_pin:.6f}, MAE={mean_absolute_error(y_valid_mae, pred_mae_valid):.6f}, coverage={valid_mae_cov:.4f}")
    print(f"[TEST ] Return Q{Q_RET} ret{N}: pinball={test_ret_pin:.6f}, MAE={mean_absolute_error(y_test_ret, pred_ret_test):.6f}, coverage={test_ret_cov:.4f}")
    print(f"[TEST ] Risk   Q{Q_MAE} mae{N}: pinball={test_mae_pin:.6f}, MAE={mean_absolute_error(y_test_mae, pred_mae_test):.6f}, coverage={test_mae_cov:.4f}")

    print("---------- Ranking sanity checks (top 10%) ----------")
    print(f"[VALID] top10% by pred_ret: mean(true ret{N})={valid_top_ret:.6f}")
    print(f"[TEST ] top10% by pred_ret: mean(true ret{N})={test_top_ret:.6f}")
    print(f"[VALID] top10% by pred_mae: mean(true mae{N})={valid_top_mae:.6f}  (越接近0越好)")
    print(f"[TEST ] top10% by pred_mae: mean(true mae{N})={test_top_mae:.6f}  (越接近0越好)")

    # 10) 保存模型
    ret_path = os.path.join(OUTDIR, f"xgb_ret_q{Q_RET}_N{N}.json")
    mae_path = os.path.join(OUTDIR, f"xgb_mae_q{Q_MAE}_N{N}.json")
    model_ret.save_model(ret_path)
    model_mae.save_model(mae_path)
    print(f"Saved model: {ret_path}")
    print(f"Saved model: {mae_path}")

    # 11) 保存测试集预测结果
    out_pred = test[[DATE_COL, SYM_COL, "entry", y_ret_col, y_mae_col]].copy()
    out_pred[f"pred_ret_q{Q_RET}_N{N}"] = pred_ret_test
    out_pred[f"pred_mae_q{Q_MAE}_N{N}"] = pred_mae_test
    pred_path = os.path.join(OUTDIR, f"test_preds_N{N}.csv")
    out_pred.to_csv(pred_path, index=False)
    print(f"Saved test predictions: {pred_path}")

    # 12) 保存特征列清单
    feat_path = os.path.join(OUTDIR, "feature_cols.txt")
    with open(feat_path, "w", encoding="utf-8") as f:
        for c in feature_cols:
            f.write(c + "\n")
    print(f"Saved feature list: {feat_path}")


if __name__ == "__main__":
    main()