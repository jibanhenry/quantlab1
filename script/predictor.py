#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预测自测脚本（仅用 2025_06_daily.csv）
- 自动从原始日线构造 day1..dayN 扁平特征（N 从模型的 feature_cols 自动推断）
- 兼容 HGB / XGB / SGD 三类模型（.joblib 的 dict 打包）
- 只保留高置信度样本（y_prob >= --thresh），输出 code(如有)、event_date、y_prob、y_pred
"""

import os
import sys
import re
import argparse
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---- tqdm 进度条（可选） ----
try:
    from tqdm.auto import tqdm
    def _pbar(iterable=None, total=None, desc=None, disable=False, unit=None):
        return tqdm(iterable=iterable, total=total, desc=desc, disable=disable, unit=unit)
except Exception:
    def _pbar(iterable=None, total=None, desc=None, disable=False, unit=None):
        return iterable if iterable is not None else range(total or 0)

# ---- 列名候选 ----
DATE_CANDS  = ["event_date", "date", "Date", "trade_date", "datetime", "timestamp", "日期"]
PRICE_CANDS = ["close", "Close", "adj_close", "Adj Close", "AdjClose", "adjclose",
               "price", "Price", "close_price", "收盘"]

def _log(msg: str):
    print(msg, flush=True)

def _normalize_daily_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把常见中文列改为英文；若已是英文则忽略。"""
    rename_map = {
        "日期":"date","开盘":"open","最高":"high","最低":"low","收盘":"close","前收":"preclose",
        "成交量":"volume","成交额":"amount","换手率":"turnover","涨跌幅":"pct_chg"
    }
    to_rename = {c: rename_map[c] for c in df.columns if c in rename_map}
    if to_rename:
        df = df.rename(columns=to_rename)
    return df

def _detect_date_col(df: pd.DataFrame) -> str:
    """自动识别日期列（若存在 event_date 也会识别）。"""
    for c in DATE_CANDS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            return c
    c0 = df.columns[0]
    df[c0] = pd.to_datetime(df[c0], errors="coerce")
    return c0

def _infer_window_days_from_features(feature_cols: List[str]) -> int:
    """根据 feature_cols 推断最大的 dayN（如 day7_xxx -> N=7），找不到则默认 7。"""
    pat = re.compile(r"^day(\d+)_")
    max_n = 0
    for c in feature_cols:
        m = pat.match(c)
        if m:
            try:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
            except Exception:
                pass
    return max_n if max_n > 0 else 7

def _make_dayN_windows_one_series(df: pd.DataFrame, date_col: str, window_days: int) -> pd.DataFrame:
    """
    仅构造 day1..dayN 的扁平特征（不生成 label）。
    day1 = t-(N-1), ..., dayN = t0
    返回列：event_date + dayk_<原数值列>
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    # 所有数值列（排除日期、code）
    num_cols = [c for c in df.columns
                if c not in (date_col, "code") and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return pd.DataFrame(columns=["event_date"])
    n = len(df)
    last_anchor = n - 1
    if last_anchor < window_days - 1:
        return pd.DataFrame(columns=["event_date"])

    recs = []
    for anchor in range(window_days - 1, last_anchor + 1):
        start = anchor - (window_days - 1)
        end = anchor + 1
        feat = {"event_date": df.loc[anchor, date_col]}
        for offset, idx in enumerate(range(start, end), start=1):
            tag = f"day{offset}"
            row = df.loc[idx, num_cols]
            for col in num_cols:
                feat[f"{tag}_{col}"] = row[col]
        recs.append(feat)
    return pd.DataFrame.from_records(recs)

def _windows_from_daily_for_features(
    daily_df: pd.DataFrame,
    feature_cols: List[str],
    show_progress: bool = True
) -> pd.DataFrame:
    """
    从 daily 构造 day1..dayN 特征，并与 feature_cols 对齐。
    带进度条：按 code 分组遍历时显示“生成 day1..dayN 特征: xx%|...”
    """
    window_days = _infer_window_days_from_features(feature_cols)
    date_col = _detect_date_col(daily_df)

    if "code" in daily_df.columns:
        daily_df["code"] = daily_df["code"].astype(str)
        parts = []
        codes = list(daily_df["code"].unique())
        pbar = _pbar(total=len(codes),
                     desc=f"生成 day1..day{window_days} 特征",
                     disable=not show_progress,
                     unit="stock")
        for code in codes:
            g = daily_df[daily_df["code"] == code].copy()
            if g.empty:
                pbar.update(1)
                continue
            w = _make_dayN_windows_one_series(g, date_col, window_days)
            if not w.empty:
                w.insert(0, "code", str(code))
                parts.append(w)
            pbar.update(1)
        pbar.close()
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["code", "event_date"])
    else:
        # 单序列：一次性生成，同时模拟一个按行推进的进度条
        g = daily_df.sort_values(date_col)
        n = len(g)
        step = max(1, n // 100)
        pbar = _pbar(total=n, desc=f"生成 day1..day{window_days} 特征", disable=not show_progress, unit="row")
        w = _make_dayN_windows_one_series(g, date_col, window_days)
        for i in range(0, n, step):
            pbar.update(min(step, n - i))
        pbar.close()
        out = w

    # 对齐模型所需特征列（缺则补 NaN）
    if out.empty:
        return out
    missing = [c for c in feature_cols if c not in out.columns]
    for c in missing:
        out[c] = np.nan
    cols = (["code"] if "code" in out.columns else []) + ["event_date"] + feature_cols
    return out[cols]

def _load_model_payload(model_path: str) -> dict:
    """加载 .joblib 模型包，兼容 XGB/HGB/SGD 的统一结构。"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在：{model_path}")
    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "type" not in payload:
        raise ValueError("不识别的模型包结构，请确认为训练脚本导出的 joblib 字典格式。")

    if payload.get("type") == "xgb":
        try:
            import xgboost as xgb  # noqa
        except Exception as e:
            raise RuntimeError(
                "加载到 XGB 模型，但当前解释器无法导入 xgboost。\n"
                f"请安装：{sys.executable} -m pip install xgboost\n原始错误：{e}"
            )
        # 既支持直接打包 booster，也兼容只存了路径的情形（若你后来有这种保存方式）
        if "booster" not in payload and "model_path" in payload:
            mp = payload["model_path"]
            if not os.path.exists(mp):
                raise FileNotFoundError(f"XGB 模型 JSON 不存在：{mp}")
            booster = xgb.Booster()
            booster.load_model(mp)
            payload["booster"] = booster

    if "feature_cols" not in payload or not isinstance(payload["feature_cols"], list):
        raise ValueError("模型包缺少 feature_cols。")

    return payload

def _predict_proba(payload: dict, X_feat: pd.DataFrame) -> np.ndarray:
    """统一预测概率接口。"""
    mtype = payload.get("type")
    if mtype in ("hgb", "sgd"):
        return payload["model"].predict_proba(X_feat.astype(float))[:, 1]
    elif mtype == "xgb":
        import xgboost as xgb
        dmat = xgb.DMatrix(X_feat, nthread=-1)
        return payload["booster"].predict(dmat)
    else:
        raise ValueError(f"未知模型类型：{mtype}")

def predict_windows(daily_df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    """从日线生成窗口特征 -> 预测概率 -> 产出 (code?, event_date, y_prob, y_pred)"""
    feats = payload["feature_cols"]
    win = _windows_from_daily_for_features(daily_df, feats, show_progress=True)
    if win.empty:
        return pd.DataFrame(columns=(["code"] if "code" in daily_df.columns else []) + ["event_date", "y_prob", "y_pred"])
    X = win[feats]
    y_prob = _predict_proba(payload, X)
    out = win[(["code"] if "code" in win.columns else []) + ["event_date"]].copy()
    out["y_prob"] = y_prob
    # y_pred 先不阈值筛选，在 main 里根据 --thresh 再做过滤
    out["y_pred"] = 0  # 先占位，main 中按阈值覆盖
    return out

def main():
    p = argparse.ArgumentParser(description="执行模型预测模块自测")
    p.add_argument("--daily",
                   default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv",
                   help="原始日线 CSV 路径")
    p.add_argument("--model",
                   default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/model/xgb_20250809.joblib",
                   help="已训练模型文件路径 (.joblib)")
    p.add_argument("--thresh", default=0.85, type=float,
                   help="筛选高置信度的阈值（y_prob >= thresh）")

    args = p.parse_args()

    # 读取数据
    if not os.path.exists(args.daily):
        raise FileNotFoundError(f"找不到 daily：{args.daily}")
    df = pd.read_csv(args.daily, low_memory=False)
    df = _normalize_daily_columns(df)
    if "code" in df.columns:
        df["code"] = df["code"].astype(str)

    # 载入模型
    payload = _load_model_payload(args.model)

    _log("[info] 正在从 daily 生成 day1..dayN 特征（按模型自动推断 N）...")
    df_pred = predict_windows(df, payload)
    if df_pred.empty:
        _log("[warn] 没有生成可预测的窗口样本。")
        return

    # 根据阈值给 y_pred，并仅保留高置信度样本
    thr = float(args.thresh)
    df_pred["y_pred"] = (df_pred["y_prob"] >= thr).astype(int)
    df_high = df_pred[df_pred["y_prob"] >= thr].copy()

    # 排序&展示
    sort_cols = (["code"] if "code" in df_high.columns else []) + ["event_date", "y_prob"]
    df_high = df_high.sort_values(sort_cols, ascending=[True] * (len(sort_cols) - 1) + [False]).reset_index(drop=True)

    _log(f"[info] 高置信度（y_prob >= {thr:.2f}）命中 {len(df_high)} 条记录。示例：")
    print(df_high.head(50).to_string(index=False))

if __name__ == "__main__":
    main()