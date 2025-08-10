#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型查看/打分脚本（只用 2025_06_daily.csv）
- 自动从原始日线构造 day1..dayN 扁平特征（N 从模型的 feature_cols 自动推断）
- 兼容三类模型：HGB / XGB / SGD（.joblib 中的 dict）
- 两种模式：
  * ranges：根据开始/结束时间 + 多个概率阈值区间，输出(event_date, code?, y_prob, y_pred, close_t, close_t+1..close_t+5)
  * series：根据开始/结束时间 + 多个代码，输出该/这些代码的概率时间序列
"""

import os
import sys
import re
import argparse
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ========== 进度条（tqdm 可选） ==========
try:
    from tqdm.auto import tqdm
    def _pbar(iterable=None, total=None, desc=None, disable=False):
        return tqdm(iterable=iterable, total=total, desc=desc, disable=disable)
except Exception:
    def _pbar(iterable=None, total=None, desc=None, disable=False):
        # 无 tqdm 时的降级：直接返回原可迭代对象
        return iterable if iterable is not None else range(total or 0)

# ================= 基础工具 =================

def _log(msg: str):
    print(msg, flush=True)

def _parse_datesafe(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return pd.to_datetime(s, errors="coerce")

def _parse_prob_ranges(s: str) -> List[Tuple[float, float]]:
    """
    解析形如 "0.8-1.0,0.6-0.8" 的区间串 -> [(0.8,1.0),(0.6,0.8)]
    非法项自动忽略，结果为空则回退为 [(0.0, 1.0)]。
    """
    out = []
    if not s:
        return [(0.0, 1.0)]
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                lo = float(a)
                hi = float(b)
                if lo <= hi and 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0:
                    out.append((lo, hi))
            except Exception:
                pass
    return out if out else [(0.0, 1.0)]

# 允许中英文列名（与训练脚本一致）
DATE_CANDS  = ["event_date", "date", "Date", "trade_date", "datetime", "timestamp", "日期"]
PRICE_CANDS = ["close", "Close", "adj_close", "Adj Close", "AdjClose", "adjclose",
               "price", "Price", "close_price", "收盘"]

def _normalize_daily_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "日期":"date","开盘":"open","最高":"high","最低":"low","收盘":"close","前收":"preclose",
        "成交量":"volume","成交额":"amount","换手率":"turnover","涨跌幅":"pct_chg"
    }
    to_rename = {c: rename_map[c] for c in df.columns if c in rename_map}
    if to_rename:
        df = df.rename(columns=to_rename)
    return df

def _detect_date_col(df: pd.DataFrame) -> str:
    for c in DATE_CANDS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            return c
    c0 = df.columns[0]
    df[c0] = pd.to_datetime(df[c0], errors="coerce")
    return c0

def _detect_price_col(df: pd.DataFrame) -> Optional[str]:
    for c in PRICE_CANDS:
        if c in df.columns:
            return c
    for c in df.columns:
        if "close" in c.lower() or c == "收盘":
            return c
    return None

def _infer_window_days_from_features(feature_cols: List[str]) -> int:
    pat = re.compile(r"^day(\d+)_")
    max_n = 0
    for c in feature_cols:
        m = pat.match(c)
        if m:
            try:
                n = int(m.group(1))
                max_n = max(max_n, n)
            except Exception:
                pass
    return max_n if max_n > 0 else 7

def _make_dayN_windows_one_series(df: pd.DataFrame, date_col: str, window_days: int) -> pd.DataFrame:
    df = df.sort_values(date_col).reset_index(drop=True)
    num_cols = [c for c in df.columns if c not in (date_col, "code") and pd.api.types.is_numeric_dtype(df[c])]
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
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
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
        groups = list(daily_df.groupby("code", sort=False))
        parts = []
        pbar = _pbar(total=len(groups),
                     desc=f"生成 day1..day{window_days} 特征",
                     disable=not show_progress)
        for code, g in groups:
            g = g.sort_values(date_col)
            if start_ts is not None:
                cutoff = start_ts - pd.Timedelta(days=window_days - 1)
                g = g[g[date_col] >= cutoff]
            if end_ts is not None:
                g = g[g[date_col] <= end_ts]
            if not g.empty:
                w = _make_dayN_windows_one_series(g, date_col, window_days)
                if not w.empty:
                    if start_ts is not None:
                        w = w[w["event_date"] >= start_ts]
                    if end_ts is not None:
                        w = w[w["event_date"] <= end_ts]
                    if not w.empty:
                        w.insert(0, "code", str(code))
                        parts.append(w)
            pbar.update(1)
        pbar.close()
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=(["code", "event_date"]))
    else:
        g = daily_df.sort_values(date_col)
        if start_ts is not None:
            cutoff = start_ts - pd.Timedelta(days=window_days - 1)
            g = g[g[date_col] >= cutoff]
        if end_ts is not None:
            g = g[g[date_col] <= end_ts]
        # 单序列也给个大致的进度：按行数粗略估计
        n = len(g)
        pbar = _pbar(total=n, desc=f"生成 day1..day{window_days} 特征", disable=not show_progress)
        w = _make_dayN_windows_one_series(g, date_col, window_days)
        for i in range(0, n, max(1, n // 100)):
            pbar.update(min(max(1, n // 100), n - i))
        pbar.close()
        if start_ts is not None:
            w = w[w["event_date"] >= start_ts]
        if end_ts is not None:
            w = w[w["event_date"] <= end_ts]
        out = w

    # 对齐模型所需特征列（缺则补 NaN）
    if out.empty:
        return out
    missing = [c for c in feature_cols if c not in out.columns]
    for c in missing:
        out[c] = np.nan
    cols = (["code"] if "code" in out.columns else []) + ["event_date"] + feature_cols
    return out[cols]

def _build_future_close_table(
    daily_df: pd.DataFrame,
    horizon_days: int = 5
) -> pd.DataFrame:
    """
    生成每个锚点日期的收盘价列：close_t（当天）与 close_t+1..close_t+H（未来）
    若存在 code，则为每个 code 单独计算。
    """
    date_col = _detect_date_col(daily_df)
    price_col = _detect_price_col(daily_df)
    if price_col is None:
        return pd.DataFrame()

    if "code" in daily_df.columns:
        daily_df["code"] = daily_df["code"].astype(str)
        cols = []
        for code, g in daily_df.groupby("code", sort=False):
            g = g.sort_values(date_col).reset_index(drop=True)
            base = pd.DataFrame({
                "code": str(code),
                "event_date": g[date_col],
                "close_t": g[price_col].values
            })
            for k in range(1, horizon_days + 1):
                base[f"close_t+{k}"] = g[price_col].shift(-k).values
            cols.append(base)
        fut = pd.concat(cols, ignore_index=True) if cols else pd.DataFrame()
    else:
        g = daily_df.sort_values(date_col).reset_index(drop=True)
        base = pd.DataFrame({"event_date": g[date_col], "close_t": g[price_col].values})
        for k in range(1, horizon_days + 1):
            base[f"close_t+{k}"] = g[price_col].shift(-k).values
        fut = base
    return fut

def _load_model_payload(model_path: str) -> dict:
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
    if "feature_cols" not in payload or not isinstance(payload["feature_cols"], list):
        raise ValueError("模型包缺少 feature_cols。")
    return payload

def _predict_proba(payload: dict, X_feat: pd.DataFrame) -> np.ndarray:
    mtype = payload.get("type")
    if mtype in ("hgb", "sgd"):
        return payload["model"].predict_proba(X_feat.astype(float))[:, 1]
    elif mtype == "xgb":
        import xgboost as xgb
        dmat = xgb.DMatrix(X_feat, nthread=-1)
        return payload["booster"].predict(dmat)
    else:
        raise ValueError(f"未知模型类型：{mtype}")

# ================= 主流程 =================

def main():
    ap = argparse.ArgumentParser()

    # —— 保留你的默认路径 ——
    ap.add_argument("--model_path", type=str, default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/model/xgb_20250809.joblib", help="模型 .joblib 路径")
    ap.add_argument("--data_csv",   type=str, default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv", help="原始日线 CSV 路径")

    ap.add_argument("--mode", choices=["ranges", "series"], default="ranges",
                    help="ranges：按多概率区间筛选并附带未来5天收盘；series：按代码输出概率序列")

    ap.add_argument("--start", type=str, default="2025-06-01", help="开始日期（YYYY-MM-DD）")
    ap.add_argument("--end",   type=str, default="2025-08-07", help="结束日期（YYYY-MM-DD）")

    # ranges 模式
    ap.add_argument("--prob_ranges", type=str, default="0.85-1.0",
                    help="多个概率区间，用逗号分隔，如 '0.8-1.0,0.6-0.8'")
    ap.add_argument("--dec_thr",  type=float, default=0.85, help="判定阈值（用于生成 y_pred）")

    # series 模式
    ap.add_argument("--codes", type=str, default="",
                    help="多个代码，用逗号分隔；留空则自动选样本最多的一个")

    # 通用保存
    ap.add_argument("--save", action="store_true", default=True, help="是否保存结果到CSV")
    ap.add_argument("--out_csv", type=str, default=None, help="保存路径；留空则用默认文件名")
    ap.add_argument("--output_dir", type=str, default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/output/", help="保存目录（当 --out_csv 为空时使用）")

    ap.add_argument("--verbose", action="store_true", help="显示更多日志（默认关闭）")

    args = ap.parse_args()
    verbose = bool(args.verbose)

    # 1) 加载模型
    payload = _load_model_payload(args.model_path)
    feature_cols = payload["feature_cols"]
    mtype = payload["type"]
    _log(f"[info] 已加载模型：type={mtype}，特征数={len(feature_cols)}")

    # 2) 读取 daily，规范列
    if not os.path.exists(args.data_csv):
        raise FileNotFoundError(f"找不到数据文件：{args.data_csv}")
    daily = pd.read_csv(args.data_csv, low_memory=False)
    daily = _normalize_daily_columns(daily)
    if "code" in daily.columns:
        daily["code"] = daily["code"].astype(str)

    # 3) 时间窗解析
    start_ts = _parse_datesafe(args.start)
    end_ts   = _parse_datesafe(args.end)

    # 4) 从 daily 构造与模型特征对齐的窗口特征（带进度条）
    win = _windows_from_daily_for_features(daily, feature_cols, start_ts, end_ts, show_progress=True)
    if win.empty:
        _log("[warn] 在当前时间窗内构造不出有效锚点（可能数据不足以形成 N 天窗口）。")
        if args.save:
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = args.out_csv or os.path.join(args.output_dir, f"{args.mode}_empty.csv")
            pd.DataFrame().to_csv(out_path, index=False)
            _log(f"[done] 已输出空结果：{out_path}")
        return

    # 5) 预测概率 & 阈值
    X = win[feature_cols]
    y_prob = _predict_proba(payload, X)
    base = win[(["code"] if "code" in win.columns else []) + ["event_date"]].copy()
    base["y_prob"] = y_prob
    base["y_pred"] = (base["y_prob"] >= float(args.dec_thr)).astype(int)

    # 6) ranges / series
    if args.mode == "ranges":
        # 加入当天与未来5天收盘
        fut = _build_future_close_table(daily, horizon_days=5)
        if not fut.empty:
            join_keys = ["event_date"] + (["code"] if "code" in base.columns and "code" in fut.columns else [])
            base = base.merge(fut, on=join_keys, how="left")

        ranges = _parse_prob_ranges(args.prob_ranges)
        blocks = []
        for (lo, hi) in ranges:
            sel = base[(base["y_prob"] >= lo) & (base["y_prob"] <= hi)].copy()
            sel.insert(0, "range", f"[{lo:.2f},{hi:.2f}]")
            blocks.append(sel)
            _log(f"[info] 区间 [{lo:.2f}, {hi:.2f}] 命中 {len(sel)} 条")
        result = pd.concat(blocks, ignore_index=True) if blocks else base.iloc[0:0].copy()

        # 统一列顺序（含 close_t）
        front = ["range"] + (["code"] if "code" in result.columns else []) + ["event_date", "y_prob", "y_pred"]
        price_cols = ["close_t"] + [f"close_t+{k}" for k in range(1, 6)]
        exist_price = [c for c in price_cols if c in result.columns]
        result = result[front + exist_price]

        # 排序：先区间、再概率降序、再日期升序
        result = result.sort_values(["range", "y_prob", "event_date"], ascending=[True, False, True]).reset_index(drop=True)

        # 输出
        if args.save:
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = args.out_csv or os.path.join(args.output_dir, "ranges_result.csv")
            result.to_csv(out_path, index=False)
            _log(f"[done] 结果已导出：{out_path}")
        else:
            print(result.head(30).to_string(index=False))

    else:  # series
        if "code" not in base.columns:
            raise ValueError("数据中没有 code 列，无法进行 series 分析。")

        codes = [c.strip() for c in str(args.codes).split(",") if c.strip()]
        if not codes:
            # 自动选择样本最多的一个代码
            freq = base["code"].value_counts()
            if freq.empty:
                _log("[warn] 没有可用代码。")
                if args.save:
                    os.makedirs(args.output_dir, exist_ok=True)
                    out_path = args.out_csv or os.path.join(args.output_dir, "series_empty.csv")
                    pd.DataFrame().to_csv(out_path, index=False)
                    _log(f"[done] 已输出空结果：{out_path}")
                return
            codes = [freq.index[0]]
            _log(f"[info] 未指定 --codes，自动选择样本最多的代码：{codes[0]}")

        res_list = []
        for code in codes:
            ser = base[base["code"] == code].copy()
            ser = ser.sort_values("event_date").reset_index(drop=True)
            ser.insert(0, "code", code)
            res_list.append(ser)
            _log(f"[info] 代码 {code}：记录数 {len(ser)}")
        result = pd.concat(res_list, ignore_index=True) if res_list else base.iloc[0:0].copy()

        if args.save:
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = args.out_csv or os.path.join(args.output_dir, "series_result.csv")
            result.to_csv(out_path, index=False)
            _log(f"[done] 序列已导出：{out_path}")
        else:
            print(result.head(50).to_string(index=False))

if __name__ == "__main__":
    main()