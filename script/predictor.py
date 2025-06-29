# predictor.py
# -*- coding: utf-8 -*-
"""
执行模型预测模块

提供 5 大功能函数：
1. daily_to_windows    — 将原始日线转成模型输入的“前7天窗口”格式
2. load_model          — 加载 .joblib 模型
3. predict_windows     — 批量跑 predict_proba，得到每条窗口的置信度
4. filter_by_threshold — 按概率阈值 & 时间区间筛选高置信度记录，并排序
5. get_prev7_probs     — 查询指定(code, event_date)窗口的7天预测概率序列
6. get_single_proba    — 查询指定(code, event_date)窗口的预测概率
"""

import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from typing import List, Union
from sklearn.base import ClassifierMixin


def daily_to_windows(
    df_daily: pd.DataFrame,
    raw_cols: List[str],
    seq_len: int = 7
) -> pd.DataFrame:
    """
    将日线 DataFrame 转成“前 seq_len 天”多个窗口。
    返回包含 ['code','event_date', day1_<col> … day7_<col>] 的 DataFrame。
    """
    recs = []
    df_daily = df_daily.copy()
    df_daily["code"] = df_daily["code"].astype(str).str.zfill(6)
    for code, grp in tqdm(df_daily.groupby("code", sort=False), desc="daily→windows"):
        grp = grp.sort_values("日期").reset_index(drop=True)
        for idx in range(seq_len, len(grp)):
            win = grp.iloc[idx-seq_len:idx]
            if win[raw_cols].isnull().any().any():
                continue
            fv = win[raw_cols].to_numpy().flatten()
            rec = {
                "code": code,
                "event_date": grp.at[idx, "日期"]
            }
            for d in range(seq_len):
                base = d * len(raw_cols)
                for i, col in enumerate(raw_cols):
                    rec[f"day{d+1}_{col}"] = fv[base + i]
            recs.append(rec)
    return pd.DataFrame.from_records(recs)


def load_model(path: str):
    """加载 .joblib 格式的模型文件"""
    return joblib.load(path)


def predict_windows(
    df_windows: pd.DataFrame,
    model,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    对 df_windows 中所有窗口批量 predict_proba，
    返回原 df_windows + 新增一列 pred_proba。
    """
    X = df_windows[feature_cols].to_numpy()
    proba = model.predict_proba(X)[:, 1]
    df = df_windows.copy()
    df["pred_proba"] = proba
    return df


def filter_by_threshold(
    df_pred: pd.DataFrame,
    threshold: float,
    start_date: Union[pd.Timestamp, str] = None,
    end_date:   Union[pd.Timestamp, str] = None
) -> pd.DataFrame:
    """
    筛出 pred_proba > threshold，
    可选地按 event_date 区间过滤，最后按 pred_proba 降序返回。
    """
    df = df_pred[df_pred["pred_proba"] > threshold].copy()
    if start_date is not None:
        df = df[df["event_date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["event_date"] <= pd.to_datetime(end_date)]
    return df.sort_values("pred_proba", ascending=False).reset_index(drop=True)


def get_prev7_probs(
    df_daily: pd.DataFrame,
    code: str,
    event_date: pd.Timestamp,
    raw_cols: List[str],
    seq_len: int,
    model: ClassifierMixin
) -> pd.Series:
    """
    对给定 code + event_date，返回一共 8 个窗口的预测概率：
      prob_day1 … prob_day7  （分别对应 offset=7 … offset=1）
      prob_event            （offset=0，即事件日窗口）
    df_daily: 包含 'code','日期' 以及 raw_cols 那几列的原始日线
    """
    # 1. 准备
    dt = pd.to_datetime(event_date)
    daily = df_daily[df_daily["code"] == code]\
              .sort_values("日期").reset_index(drop=True)
    idx = daily.index[daily["日期"] == dt]
    if len(idx) == 0:
        raise KeyError(f"{code} 没有 {event_date} 这一天的数据")
    idx = idx[0]
    if idx < seq_len:
        raise ValueError(f"{code}@{event_date} 无法回溯 {seq_len} 天，共只有 {idx} 天")

    # 2. 对 8 个 offset 批量收集特征
    windows = []
    for offset in range(seq_len, -1, -1):  # seq_len … 0
        start = idx - offset - seq_len
        end   = idx - offset
        # 比如 offset=0, 切 [idx-7:idx]；offset=7, 切 [idx-14:idx-7]
        win = daily.iloc[start:end][raw_cols].to_numpy().flatten()
        windows.append(win)
    X = np.stack(windows, axis=0)  # (8, seq_len*len(raw_cols))

    # 3. 批量预测
    prob = model.predict_proba(X)[:, 1]  # (8,)

    # 4. 返回 Series
    idxs = [f"prob_day{d}" for d in range(1, seq_len+1)] + ["prob_event"]
    return pd.Series(prob.tolist(), index=idxs)


def get_single_proba(
    df_pred:    pd.DataFrame,
    code:       str,
    event_date: Union[pd.Timestamp, str]
) -> float:
    """
    直接从 df_pred 中读取指定窗口的 pred_proba 值。
    """
    dt = pd.to_datetime(event_date)
    row = df_pred[(df_pred["code"] == code) & (df_pred["event_date"] == dt)]
    if row.empty:
        raise KeyError(f"No window for {code} @ {event_date}")
    return float(row["pred_proba"].iloc[0])


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="执行模型预测模块自测")
    p.add_argument("--daily",  default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv",
                   help="原始日线 CSV 路径")
    p.add_argument("--model",  default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/model/gbdt_model.joblib",
                   help="已训练模型文件路径 (.joblib)")
    p.add_argument("--thresh", default=0.85, type=float,
                   help="筛选高置信度的阈值")
    args = p.parse_args()

    # 加载日线
    df_daily = pd.read_csv(args.daily, parse_dates=["日期"])
    raw_cols = [
        "开盘","最高","最低","收盘","前收",
        "成交量","成交额","换手率","涨跌幅",
        "pbMRQ","psTTM"
    ]

    # 1. 转窗口
    df_win = daily_to_windows(df_daily, raw_cols)
    feature_cols = [c for c in df_win.columns if c not in ("code", "event_date")]

    # 2. 加载模型 & 批量预测
    mdl     = load_model(args.model)
    df_pred = predict_windows(df_win, mdl, feature_cols)

    # 3. 筛高置信度并输出
    df_high = filter_by_threshold(df_pred, args.thresh)
    print("高置信度样本 (code, event_date, pred_proba):")
    print(df_high[["code", "event_date", "pred_proba"]].head(20))