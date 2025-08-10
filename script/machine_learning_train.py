#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
1) 从 2025_06_daily.csv 增量更新 test_samples.csv（按 code 分组、基于日期的增量）
   - day1=t-6 … day7=t0；label：未来5天最高价 ≥ t0*(1+threshold)
   - 仅生成 event_date 大于 test_samples 中“该 code 的最大 event_date”的新样本
2) 合并 train_val_samples.csv + 最新 test_samples.csv（按 ['code','event_date'] 或仅 event_date 去重）
3) 始终用全量数据训练；若给 --val_start/--val_end，则在该时间窗评估（但仍参与训练）
4) 类别不平衡：类别权重；支持 hgb/xgb/sgd（sgd 分批）
5) 进度：关键阶段日志；--verbose 开启 tqdm 进度条
6) 保存模型：统一为 **joblib** 格式（.joblib）
   - 既支持传入“目录”并自动命名（trainer_YYYYMMDD.joblib）
   - 也支持传入完整文件路径（自动补 .joblib 扩展名）
7) 保存报告：默认写入输出目录，文件名为 模型名+日期+_report.json
   验证窗可选导出：--val_pred_out 导出含 y_prob 的明细
8) 新增：阈值扫描模块
   - 在验证窗对一系列阈值计算 precision/recall/f1/TP/FP/TN/FN
   - 支持多种选优准则（如最佳F1；或在 precision>=P / recall>=R 约束下最大化另一项）
   - 导出扫描表 CSV，并将推荐阈值写入报告
"""

import os
import json
import argparse
import warnings
import time
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 可选：XGBoost（大数据更稳更快）
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# 可选：tqdm 进度条
try:
    from tqdm.auto import tqdm
    def _pbar(iterable=None, total=None, desc=None, disable=False):
        return tqdm(iterable=iterable, total=total, desc=desc, disable=disable)
except Exception:
    def _pbar(iterable=None, total=None, desc=None, disable=False):
        return iterable if iterable is not None else range(total or 0)

# 允许中英文列名
DATE_CANDS  = ["date", "Date", "trade_date", "datetime", "timestamp", "日期"]
PRICE_CANDS = ["close", "Close", "adj_close", "Adj Close", "AdjClose", "adjclose",
               "price", "Price", "close_price", "收盘"]

def _ts(): return time.strftime("%Y-%m-%d %H:%M:%S")
def _today_str(): return time.strftime("%Y%m%d")
def _log(msg: str, force: bool = False, verbose: bool = True):
    if force or verbose:
        print(f"[{_ts()}] {msg}")

class Stopwatch:
    def __init__(self): self.t0 = time.time()
    def lap(self): t=time.time();dt=t-self.t0;self.t0=t;return dt

# —— 中文列名标准化为英文 —— #
def _normalize_daily_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "日期":"date","开盘":"open","最高":"high","最低":"low","收盘":"close","前收":"preclose",
        "成交量":"volume","成交额":"amount","换手率":"turnover","涨跌幅":"pct_chg"
    }
    to_rename = {c: rename_map[c] for c in df.columns if c in rename_map}
    if to_rename: df = df.rename(columns=to_rename)
    return df

def _detect_date_col(df: pd.DataFrame) -> str:
    for c in DATE_CANDS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            return c
    c0 = df.columns[0]
    df[c0] = pd.to_datetime(df[c0], errors="coerce")
    return c0

def _detect_price_col(df: pd.DataFrame) -> str:
    for c in PRICE_CANDS:
        if c in df.columns:
            return c
    for c in df.columns:
        if "close" in c.lower() or c == "收盘":
            return c
    raise ValueError("无法找到价格列（如 'close' 或 '收盘'）。")

# —— 单序列滑窗 —— #
def _make_dayN_windows_one_series(df: pd.DataFrame, date_col: str, price_col: str,
                                  window_days: int, horizon_days: int, threshold: float) -> pd.DataFrame:
    df = df.sort_values(date_col).reset_index(drop=True)
    num_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols: return pd.DataFrame(columns=["event_date","label"])
    n = len(df); last_anchor = n - 1 - horizon_days
    if last_anchor < window_days - 1: return pd.DataFrame(columns=["event_date","label"])
    recs = []
    for anchor in range(window_days-1, last_anchor+1):
        start = anchor-(window_days-1); end = anchor+1
        feat = {"event_date": df.loc[anchor, date_col]}
        for offset, idx in enumerate(range(start, end), start=1):
            tag = f"day{offset}"; row = df.loc[idx, num_cols]
            for col in num_cols: feat[f"{tag}_{col}"] = row[col]
        price_t0 = df.loc[anchor, price_col]
        future_max = df.loc[anchor+1: anchor+horizon_days, price_col].max()
        feat["label"] = 1 if future_max >= (1.0+threshold)*price_t0 else 0
        recs.append(feat)
    return pd.DataFrame.from_records(recs)

# —— 基于“每只股票最大 event_date”的增量构造 —— #
def _build_incremental_from_daily(
    daily_df: pd.DataFrame, test_path: str, date_col: str, price_col: str,
    window_days: int, horizon_days: int, threshold: float, verbose: bool
) -> pd.DataFrame:
    """
    仅为“新增日期”生成样本：
      - 若存在 code：对每个 code 取 test_samples 的最大 event_date = max_d
        · 切 daily：date >= max_d - (window_days-1)，构造窗口
        · 构造后再过滤 event_date > max_d
      - 若不存在 code：用全局最大 event_date 同理处理
    返回：仅包含“新增样本”的 DataFrame（含 code 列时保留 code）
    """
    has_code = "code" in daily_df.columns
    max_map = {}
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path, parse_dates=["event_date"])
        if has_code and "code" in test_df.columns:
            test_df["code"] = test_df["code"].astype(str)
            max_ser = test_df.groupby("code")["event_date"].max()
            max_map = max_ser.to_dict()
        else:
            if not test_df.empty:
                max_map = {"__GLOBAL__": pd.to_datetime(test_df["event_date"]).max()}
    else:
        max_map = {}

    if has_code:
        groups = daily_df.groupby("code", sort=False)
        parts = []
        n_groups = groups.ngroups
        _log(f"增量模式：按 code 处理 {n_groups} 组", force=True)
        for i, (code, g) in enumerate(_pbar(groups, total=n_groups, desc="按code增量", disable=not verbose), 1):
            max_d = max_map.get(str(code), None)
            if max_d is None:
                base = g.copy()
            else:
                cutoff = max_d - pd.Timedelta(days=window_days-1)
                base = g[g[date_col] >= cutoff].copy()
            if base.empty: continue
            part = _make_dayN_windows_one_series(base, date_col, price_col, window_days, horizon_days, threshold)
            if max_map and max_d is not None and not part.empty:
                part = part[part["event_date"] > max_d]
            if not part.empty:
                part.insert(0, "code", str(code))
                parts.append(part)
            if not verbose and (i % 50 == 0):
                _log(f"  已处理 {i}/{n_groups} 个代码...", force=True)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["code","event_date","label"])
    else:
        max_d = max_map.get("__GLOBAL__", None)
        if max_d is None:
            base = daily_df.copy()
        else:
            cutoff = max_d - pd.Timedelta(days=window_days-1)
            base = daily_df[daily_df[date_col] >= cutoff].copy()
        part = _make_dayN_windows_one_series(base, date_col, price_col, window_days, horizon_days, threshold)
        if max_d is not None and not part.empty:
            part = part[part["event_date"] > max_d]
        return part

def _merge_incremental(new_df: pd.DataFrame, out_csv: str, verbose: bool = True) -> pd.DataFrame:
    """
    增量合并 test_samples.csv：
      - 主键：['event_date']；如有 code 列则 ['code','event_date']
      - 仍做一次去重覆盖（兜底）
    """
    sw = Stopwatch()
    has_code = "code" in new_df.columns
    keys = ["event_date"] if not has_code else ["code","event_date"]

    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv, parse_dates=["event_date"])
        if has_code and "code" in old.columns:
            old["code"] = old["code"].astype(str)
        all_cols = sorted(set(old.columns) | set(new_df.columns))
        old = old.reindex(columns=all_cols)
        new = new_df.reindex(columns=all_cols)
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.sort_values(keys).drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    else:
        merged = new_df.sort_values(keys).reset_index(drop=True)

    front = (["code"] if has_code else []) + ["event_date", "label"]
    others = [c for c in merged.columns if c not in front]
    merged = merged[front + sorted(others)]
    _log(f"test_samples 合并完成：行数={len(merged)}，用时 {sw.lap():.2f}s", force=True)
    return merged

# —— 类别不平衡 & 评估 —— #
def _class_weights(y: np.ndarray) -> dict:
    classes = np.unique(y)
    if len(classes) == 1: return {int(classes[0]): 1.0}
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(wi) for c, wi in zip(classes, w)}

def _eval_probs(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    """
    概率评估（ROC/PR） + 阈值上的分类指标。
    注意：precision/recall/f1 用的是 y_pred（概率经阈值转换），不是 y_prob。
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    y_pred = (y_prob >= thr).astype(int)
    multi = len(np.unique(y_true)) > 1

    out = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if multi else None,
        "average_precision": float(average_precision_score(y_true, y_prob)) if multi else None,
        "threshold": float(thr)
    }
    if multi:
        out.update({
            "f1": float(f1_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
        })
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
            out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
        except Exception:
            pass
    else:
        cnt1 = int((y_true == 1).sum())
        out.update({"positive_count": cnt1, "negative_count": int(len(y_true) - cnt1)})
    return out

# —— 阈值扫描（新增模块） —— #
def threshold_scan(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thr_min: float = 0.0,
    thr_max: float = 1.0,
    thr_step: float = 0.01,
    # 选优策略：
    # best_f1: 直接取F1最高
    # max_recall_at_precision_ge: 在 precision >= target 时，最大化 recall
    # max_precision_at_recall_ge: 在 recall >= target 时，最大化 precision
    strategy: str = "best_f1",
    target: float = 0.2
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    在验证集上对一系列阈值进行扫描，产出每个阈值的分类指标，并返回“推荐阈值”。

    参数
    ----
    y_true, y_prob : 验证集真实标签与预测概率
    thr_min, thr_max, thr_step : 阈值扫描范围与步长
    strategy : 选优策略（见上注释）
    target   : 当 strategy 需要阈值约束时使用（如 precision>=target）

    返回
    ----
    (scan_df, best_info)
      scan_df  : 每行是一个阈值及其 precision/recall/f1/TP/FP/TN/FN/pred_pos 等
      best_info: 推荐阈值与关键指标，如 {"recommended_threshold": 0.73, "f1":0.12, "precision":..., "recall":...}
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    if len(np.unique(y_true)) < 2:
        # 单类无法计算分类指标，直接返回空结果
        return pd.DataFrame(columns=["threshold","precision","recall","f1","tp","fp","tn","fn","pred_pos","support_pos_rate"]), {}

    thresholds = np.arange(thr_min, thr_max + 1e-12, thr_step, dtype=float)
    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        pred_pos = tp + fp
        support_pos_rate = (tp + fn) / max(1, len(y_true))
        rows.append({
            "threshold": round(float(thr), 6),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "pred_pos": int(pred_pos),
            "support_pos_rate": float(support_pos_rate)
        })
    scan_df = pd.DataFrame(rows)

    # —— 选优策略 —— #
    best_row = None
    if strategy == "best_f1":
        best_row = scan_df.sort_values(["f1","precision","recall"], ascending=[False, False, False]).head(1)
    elif strategy == "max_recall_at_precision_ge":
        cand = scan_df[scan_df["precision"] >= float(target)]
        best_row = cand.sort_values(["recall","f1"], ascending=[False, False]).head(1) if len(cand)>0 else pd.DataFrame()
    elif strategy == "max_precision_at_recall_ge":
        cand = scan_df[scan_df["recall"] >= float(target)]
        best_row = cand.sort_values(["precision","f1"], ascending=[False, False]).head(1) if len(cand)>0 else pd.DataFrame()
    else:
        # 未知策略，则退回 best_f1
        best_row = scan_df.sort_values(["f1","precision","recall"], ascending=[False, False, False]).head(1)

    best_info = {}
    if best_row is not None and len(best_row) == 1:
        r = best_row.iloc[0].to_dict()
        best_info = {
            "recommended_threshold": float(r["threshold"]),
            "precision": float(r["precision"]),
            "recall": float(r["recall"]),
            "f1": float(r["f1"]),
            "tp": int(r["tp"]), "fp": int(r["fp"]), "tn": int(r["tn"]), "fn": int(r["fn"]),
            "pred_pos": int(r["pred_pos"])
        }
    return scan_df, best_info

# —— 三种训练器 —— #
def train_hgb(X_all: pd.DataFrame, y_all: np.ndarray, verbose: bool = True):
    sw = Stopwatch()
    cw = _class_weights(y_all)
    model = HistGradientBoostingClassifier(
        class_weight=cw, max_iter=600, learning_rate=0.05, max_depth=8,
        early_stopping=True, validation_fraction=0.1, random_state=42
    )
    _log("开始训练 HGB...", force=True)
    model.fit(X_all, y_all)
    _log(f"HGB 训练完成，用时 {sw.lap():.2f}s", force=True)
    return {"type": "hgb", "model": model}

def train_xgb(X_all: pd.DataFrame, y_all: np.ndarray, verbose: bool = True):
    if not HAS_XGB: raise RuntimeError("未安装 xgboost，请使用 --trainer hgb 或 sgd")
    sw = Stopwatch()
    pos = max(1, int((y_all == 1).sum())); neg = max(1, int((y_all == 0).sum()))
    spw = neg / pos
    dtrain = xgb.DMatrix(X_all, label=y_all)
    params = dict(objective="binary:logistic", eval_metric="auc", eta=0.05, max_depth=8,
                  subsample=0.8, colsample_bytree=0.8, lambda_=1.0, alpha=0.0,
                  scale_pos_weight=spw, tree_method="hist")
    _log("开始训练 XGBoost...", force=True)
    bst = xgb.train(params, dtrain, num_boost_round=400, verbose_eval=False)
    _log(f"XGBoost 训练完成，用时 {sw.lap():.2f}s", force=True)
    return {"type": "xgb", "booster": bst}

def train_sgd_partial(X_all: pd.DataFrame, y_all: np.ndarray, batch_size: int = 20000, epochs: int = 3, verbose: bool = True):
    sw = Stopwatch()
    numeric_cols = list(X_all.columns)
    pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                               ("sc", StandardScaler())]), numeric_cols)], remainder="drop")
    cw = _class_weights(y_all)
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1,
                        learning_rate="optimal", class_weight=cw, random_state=42)
    pre.fit(X_all.iloc[:min(10000, len(X_all))])
    classes = np.array([0,1], dtype=int)
    order = np.arange(len(X_all)); rng = np.random.default_rng(42)
    _log(f"开始训练 SGD（epochs={epochs}, batch_size={batch_size}）...", force=True)
    for ep in _pbar(range(epochs), desc="SGD Epoch", disable=not verbose):
        rng.shuffle(order)
        batches = range(0, len(order), batch_size)
        for s in _pbar(batches, total=len(range(0, len(order), batch_size)), desc=f"Epoch {ep+1} Batches", disable=not verbose):
            e = min(len(order), s + batch_size)
            Xi = pre.transform(X_all.iloc[order[s:e]])
            yi = y_all[order[s:e]]
            clf.partial_fit(Xi, yi, classes=classes)
        if not verbose: _log(f"  完成 epoch {ep+1}/{epochs}", force=True)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    _log(f"SGD 训练完成，用时 {sw.lap():.2f}s", force=True)
    return {"type": "sgd", "model": pipe}

# —— 路径解析（模型 .joblib 与报告 .json） —— #
def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _resolve_model_path_joblib(trainer: str, model_out: str) -> str:
    """
    返回 joblib 模型保存路径：
      - 若模型路径是目录或为空/'auto'：使用 <dir>/<trainer>_<YYYYMMDD>.joblib
      - 若是文件：若无 .joblib 扩展名则自动补上
    """
    today = _today_str()
    trigger_auto = (not model_out) or (model_out.strip().lower() in ("auto",))
    if trigger_auto or os.path.isdir(model_out):
        base_dir = model_out if os.path.isdir(model_out) else ""
        fname = f"{trainer}_{today}.joblib"
        return os.path.join(base_dir, fname) if base_dir else fname
    # 显式文件路径
    if not model_out.lower().endswith(".joblib"):
        model_out = model_out + ".joblib"
    return model_out

def _resolve_report_path(trainer: str, report_out: str) -> str:
    """
    返回报告输出路径：
      - 若 report_out 是目录或为空/'auto'/'training_report.json'：使用 <dir>/<trainer>_<YYYYMMDD>_report.json
      - 若是文件：若无 .json 扩展名则自动补上
    """
    today = _today_str()
    default_names = {"training_report.json", "", "auto", None}
    if report_out and os.path.isdir(report_out):
        return os.path.join(report_out, f"{trainer}_{today}_report.json")
    if (report_out in default_names) or (isinstance(report_out, str) and report_out.strip().lower() in ("", "auto", "training_report.json")):
        return f"{trainer}_{today}_report.json"
    if not report_out.lower().endswith(".json"):
        report_out = report_out + ".json"
    return report_out

# =========================
# 四、主流程
# =========================

def main():
    ap = argparse.ArgumentParser()
    # —— 你指定的默认保存位置（模型与报告） ——
    default_model_path = "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/model/"
    default_report_dir = "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/output/"

    # 默认路径（方便直接在 PyCharm 运行）
    ap.add_argument("--daily", type=str, default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv")
    ap.add_argument("--test_out", type=str, default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/test_samples.csv")
    ap.add_argument("--train_val", type=str, default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/train_val_samples.csv")

    # 现在的默认：给出一个具体的模型路径，但仍支持“传目录则自动命名”
    ap.add_argument("--model_out", type=str, default=default_model_path,
                    help="模型保存路径；给目录则自动命名为 <trainer>_<YYYYMMDD>.joblib")

    # 报告默认改为输出目录；给目录则自动命名
    ap.add_argument("--report_out", type=str, default=default_report_dir,
                    help="报告保存路径或目录；给目录则自动命名为 <trainer>_<YYYYMMDD>_report.json")

    ap.add_argument("--features_out", type=str,
                    default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/output/feature_columns.json")

    # 仅用于生成 test_samples 的参数
    ap.add_argument("--window_days",  type=int, default=7)
    ap.add_argument("--horizon_days", type=int, default=5)
    ap.add_argument("--threshold",    type=float, default=0.30)

    # 模型与 SGD 训练参数
    ap.add_argument("--trainer", type=str, default="xgb", choices=["hgb", "xgb", "sgd"])
    ap.add_argument("--sgd_batch_size", type=int, default=20000)
    ap.add_argument("--sgd_epochs",     type=int, default=3)

    # 评估时间窗（仍然参与训练）
    ap.add_argument("--val_start", type=str, default="2025-02-01")
    ap.add_argument("--val_end",   type=str, default="2025-08-05")
    # 评估增强：主阈值与导出明细
    ap.add_argument("--dec_thr", type=float, default=0.5, help="主判定阈值，用于计算precision/recall/f1和y_pred导出")
    ap.add_argument("--val_pred_out", type=str, default=None, help="验证窗预测明细CSV（含y_prob/y_pred），留空则不导出")

    # —— 阈值扫描参数（新增） —— #
    ap.add_argument("--scan_enable", action="store_true", default=True, help="启用阈值扫描（默认开启）")
    ap.add_argument("--scan_min", type=float, default=0.0, help="扫描阈值最小值")
    ap.add_argument("--scan_max", type=float, default=1.0, help="扫描阈值最大值")
    ap.add_argument("--scan_step", type=float, default=0.01, help="扫描阈值步长")
    ap.add_argument("--scan_strategy", type=str, default="best_f1",
                    choices=["best_f1","max_recall_at_precision_ge","max_precision_at_recall_ge"],
                    help="选优策略")
    ap.add_argument("--scan_target", type=float, default=0.2,
                    help="当策略需要约束（如 precision>=target 或 recall>=target）时的目标值")
    ap.add_argument("--scan_out", type=str, default=None,
                    help="阈值扫描结果CSV路径；留空时写到报告目录下，命名为 <trainer>_<YYYYMMDD>_threshold_scan.csv")

    # 日志
    ap.add_argument("--verbose", action="store_true", help="显示详细进度（默认关闭）")
    args = ap.parse_args()
    verbose = bool(args.verbose)

    # A) 读取 daily，并标准化列名
    if not os.path.exists(args.daily):
        raise FileNotFoundError(f"找不到 daily 文件：{args.daily}")
    _log("读取 daily...", force=True)
    daily_df = pd.read_csv(args.daily, low_memory=False)
    daily_df = _normalize_daily_columns(daily_df)
    if "code" in daily_df.columns:
        daily_df["code"] = daily_df["code"].astype(str)
    date_col  = _detect_date_col(daily_df)
    price_col = _detect_price_col(daily_df)

    # B) 基于日期的增量生成新样本，并合并
    _log("基于日期的增量生成新样本...", force=True)
    new_only = _build_incremental_from_daily(
        daily_df, args.test_out, date_col, price_col,
        window_days=args.window_days, horizon_days=args.horizon_days, threshold=args.threshold,
        verbose=verbose
    )
    if new_only.empty:
        _log("没有检测到比现有 test_samples 更新的样本（可能 daily 没有新数据）。", force=True)
        if os.path.exists(args.test_out):
            updated_test = pd.read_csv(args.test_out, parse_dates=["event_date"])
        else:
            updated_test = pd.DataFrame(columns=["event_date","label"])
    else:
        _log(f"新增样本数：{len(new_only)}，开始合并到 test_samples.csv...", force=True)
        updated_test = _merge_incremental(new_only, args.test_out, verbose=verbose)
        _ensure_dir(args.test_out)
        updated_test.to_csv(args.test_out, index=False)
        _log(f"✅ test_samples.csv 已更新：{args.test_out}（行数={len(updated_test)}）", force=True)

    # C) 合并 train_val + 最新 test（按 ['code','event_date'] 或仅 event_date 去重）
    if not os.path.exists(args.train_val):
        raise FileNotFoundError(f"找不到 train_val_samples：{args.train_val}")
    _log("读取并合并 train_val + test...", force=True)
    tv = pd.read_csv(args.train_val, parse_dates=["event_date"])
    if "code" in updated_test.columns and "code" in tv.columns:
        tv["code"] = tv["code"].astype(str)
    ts = updated_test.copy()
    df = pd.concat([tv, ts], ignore_index=True)
    keys = ["event_date"] if "code" not in df.columns else ["code","event_date"]
    df = df.sort_values(keys).drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    _log(f"合并完成，总样本={len(df)}", force=True)

    # 特征列：day1_*..day{window_days}_*
    skip = {"event_date","label","code"}
    feat_cols = [c for c in df.columns if c not in skip and any(c.startswith(f"day{k}_") for k in range(1, args.window_days+1))]
    if not feat_cols:
        raise ValueError("未找到 day1_*…dayN_* 特征列。")

    # D) 解析模型输出路径（统一 joblib）
    model_path = _resolve_model_path_joblib(args.trainer, (args.model_out or "").strip())
    _ensure_dir(model_path)

    # 同步解析报告输出路径（默认：模型名+日期，写到你指定的输出目录）
    report_path = _resolve_report_path(args.trainer, (args.report_out or "").strip())
    _ensure_dir(report_path)

    # E) 全量训练并保存（joblib）
    X_all = df[feat_cols]
    y_all = df["label"].astype(int).values
    _log(f"开始训练（trainer={args.trainer}）...", force=True)
    if args.trainer == "hgb":
        pack = train_hgb(X_all, y_all, verbose=verbose)
        payload = {"type": "hgb", "model": pack["model"], "feature_cols": feat_cols}
    elif args.trainer == "xgb":
        pack = train_xgb(X_all, y_all, verbose=verbose)
        payload = {"type": "xgb", "booster": pack["booster"], "feature_cols": feat_cols}
    else:
        pack = train_sgd_partial(X_all, y_all, batch_size=args.sgd_batch_size, epochs=args.sgd_epochs, verbose=verbose)
        payload = {"type": "sgd", "model": pack["model"], "feature_cols": feat_cols}

    joblib.dump(payload, model_path)
    _log(f"训练已完成并保存模型到：{model_path}", force=True)

    # F) 指定时间窗评估（仍然参与训练）
    metrics = None
    rows_info = {"total": int(len(df)), "train_used": int(len(df))}
    y_val = None; y_prob = None; mask_val = None
    if args.val_start and args.val_end:
        val_start = pd.to_datetime(args.val_start); val_end = pd.to_datetime(args.val_end)
        mask_val = (df["event_date"] >= val_start) & (df["event_date"] <= val_end)
        X_val, y_val = X_all[mask_val], y_all[mask_val]
        rows_info["val_period_rows"] = int(mask_val.sum())
        _log(f"开始在验证时间段评估：{args.val_start} ~ {args.val_end}（样本={rows_info['val_period_rows']}）", force=True)
        if len(X_val) > 0:
            if args.trainer == "hgb":
                y_prob = payload["model"].predict_proba(X_val)[:,1]
            elif args.trainer == "xgb":
                dval = xgb.DMatrix(X_val, label=y_val)
                y_prob = payload["booster"].predict(dval)
            else:
                y_prob = payload["model"].predict_proba(X_val)[:,1]
            metrics = _eval_probs(y_val, y_prob, thr=args.dec_thr)
            _log("评估完成。", force=True)

            # 可选导出验证窗明细
            if args.val_pred_out:
                cols = []
                if "code" in df.columns:
                    cols.append(df.loc[mask_val, "code"].reset_index(drop=True).rename("code"))
                cols.append(df.loc[mask_val, "event_date"].reset_index(drop=True).rename("event_date"))
                detail = pd.DataFrame(cols[0])
                for c in cols[1:]:
                    detail = pd.concat([detail, c], axis=1)
                detail["y_true"] = y_val
                detail["y_prob"] = y_prob
                detail["y_pred"] = (detail["y_prob"] >= args.dec_thr).astype(int)
                out_dir = os.path.dirname(args.val_pred_out)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                detail.to_csv(args.val_pred_out, index=False)
                _log(f"验证窗预测明细已导出：{args.val_pred_out}（{len(detail)} 行）", force=True)
        else:
            _log("验证时间段内没有样本，跳过评估。", force=True)

    # G) 输出报告与特征列
    report = {
        "trainer": args.trainer,
        "model_path": model_path,
        "rows": rows_info,
        "window_days": args.window_days,
        "horizon_days": args.horizon_days,
        "threshold": args.threshold,
        "decision_threshold": args.dec_thr,
        "validation_window": {"start": args.val_start, "end": args.val_end} if args.val_start and args.val_end else None,
        "metrics_on_validation_window": metrics
    }

    # G2) 阈值扫描（仅当启用且评估窗口内有样本时）
    if args.scan_enable and (y_val is not None) and (y_prob is not None) and (len(y_val) > 0):
        scan_df, best_info = threshold_scan(
            y_true=y_val, y_prob=y_prob,
            thr_min=args.scan_min, thr_max=args.scan_max, thr_step=args.scan_step,
            strategy=args.scan_strategy, target=args.scan_target
        )
        # 决定扫描表输出路径
        if args.scan_out and args.scan_out.strip():
            scan_out_path = args.scan_out
            if not scan_out_path.lower().endswith(".csv"):
                scan_out_path += ".csv"
        else:
            # 写到报告目录同级
            report_dir = os.path.dirname(report_path)
            fname = f"{args.trainer}_{_today_str()}_threshold_scan.csv"
            scan_out_path = os.path.join(report_dir if report_dir else ".", fname)
        _ensure_dir(scan_out_path)
        scan_df.to_csv(scan_out_path, index=False)
        _log(f"阈值扫描表已导出：{scan_out_path}（{len(scan_df)} 行）", force=True)

        # 将推荐阈值写入报告
        report["threshold_scan"] = {
            "strategy": args.scan_strategy,
            "target": args.scan_target,
            "scan_min": args.scan_min,
            "scan_max": args.scan_max,
            "scan_step": args.scan_step,
            "recommended": best_info,
            "scan_out_csv": scan_out_path
        }
        if best_info:
            _log(f"推荐阈值：{best_info}", force=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(args.features_out, "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feat_cols}, f, ensure_ascii=False, indent=2)

    print(f"[done] model -> {model_path}")
    print(f"[done] report -> {report_path}")
    print(f"[done] features -> {args.features_out}")

if __name__ == "__main__":
    main()