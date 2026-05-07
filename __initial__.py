# -*- coding: utf-8 -*-
"""
stock_project/__initial__.py

一键执行：先更新日线、再批量预测、最后输出“从 --start 到现在”的高置信度窗口，
并为每条高置信度样本计算“前7天 + 事件日”的预测概率序列。

兼容你当前的 script/predictor.py（使用其公开的 predict_windows 以及内部下划线函数）。
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# ---- 你自己的模块路径保持不变 ----
from script.data_updater import update_data
import script.predictor as pred  # 用它里的函数，包括 _ 开头的


# ---------- 辅助：从日线构造指定 code 的锚点序列并计算概率 ----------

def _get_prev7_anchor_windows_one_code(daily_norm: pd.DataFrame,
                                       code: str,
                                       event_date: pd.Timestamp,
                                       feature_cols: list,
                                       seq_len: int = 7) -> pd.DataFrame:
    """
    对单个 code：
      1) 使用 predictor 的窗口函数把整支股票转成按锚点的 day1..dayN 特征（N=seq_len）
      2) 找到 event_date 对应的锚点行在全表 w 中的索引 idx
      3) 取 [idx-(seq_len-1) .. idx] 之间的这些锚点（最多 seq_len 个），用于“前7天+事件日”的概率序列
    返回：包含这些锚点行（列里有 event_date + feature_cols）
    """
    g = daily_norm[daily_norm["code"] == code].copy()
    if g.empty:
        raise ValueError(f"代码 {code} 在日线中没有记录")

    win_tmp = pred._make_dayN_windows_one_series(
        g, date_col=pred._detect_date_col(g), window_days=seq_len
    )
    if win_tmp.empty:
        raise ValueError(f"{code} 在指定序列长度 {seq_len} 下无法形成窗口")

    missing = [c for c in feature_cols if c not in win_tmp.columns]
    for c in missing:
        win_tmp[c] = np.nan

    win_tmp = win_tmp[["event_date"] + feature_cols].copy()
    event_date = pd.to_datetime(event_date)
    pos = win_tmp.index[win_tmp["event_date"] == event_date]
    if len(pos) == 0:
        raise ValueError(f"{code} 在锚点表中找不到事件日 {event_date.date()}（数据不足或不连续）")
    idx = pos[0]

    start_idx = max(0, idx - (seq_len - 1))
    sliced = win_tmp.loc[start_idx:idx].copy()
    return sliced.reset_index(drop=True)


def get_prev7_probs(daily_df: pd.DataFrame,
                    code: str,
                    event_date: pd.Timestamp,
                    payload: dict,
                    seq_len: int = 7) -> dict:
    """
    计算“前7天 + 事件日”的概率序列。
    返回：{'prob_day1':..., 'prob_day2':..., ..., 'prob_day7':..., 'prob_event':...}
    """
    feature_cols = payload["feature_cols"]
    daily_norm = pred._normalize_daily_columns(daily_df.copy())
    if "code" in daily_norm.columns:
        daily_norm["code"] = daily_norm["code"].astype(str)

    anchors = _get_prev7_anchor_windows_one_code(
        daily_norm, code, event_date, feature_cols, seq_len=seq_len
    )
    X = anchors[feature_cols]
    probs = pred._predict_proba(payload, X)

    out = {}
    k = len(probs)
    # 填充前 (seq_len-1) 天
    take_prev = min(seq_len - 1, max(0, k - 1))
    for i in range(take_prev):
        out[f"prob_day{i+1}"] = float(probs[i])
    # 事件日前一天（若存在且尚未覆盖）
    if k >= 2 and (seq_len - 1) >= 1:
        out[f"prob_day{seq_len-1}"] = float(probs[-2])
    # 事件日
    out["prob_event"] = float(probs[-1])
    return out


# ---------- 主流程 ----------

def full_refresh_and_predict(
    daily_csv: str,
    model_path: str,
    threshold: float = 0.85,
    seq_len: int = 7,
    start: str = "2025-07-01",
):
    """
    1) 调用 update_data 更新 daily_csv（增量或首次全量）
    2) 读取最新 daily_csv，规范列名
    3) 用 predictor.predict_windows 批量生成 (code?, event_date, y_prob)
    4) 过滤出 event_date >= --start 的记录
    5) 根据阈值筛高置信度，返回 (df_daily, df_pred_all, df_high, payload, seq_len)
    """
    # 1) 更新日线
    update_data(output_file=daily_csv)

    # 2) 读取 & 规范
    df_daily = pd.read_csv(daily_csv, low_memory=False)
    df_daily = pred._normalize_daily_columns(df_daily)
    if "code" in df_daily.columns:
        df_daily["code"] = df_daily["code"].astype(str)

    # 3) 载入模型并批量预测
    payload = pred._load_model_payload(model_path)
    df_pred_all = pred.predict_windows(df_daily, payload)

    # 4) 按开始日期过滤
    if start:
        start_ts = pd.to_datetime(start)
        df_pred_all = df_pred_all[df_pred_all["event_date"] >= start_ts].copy()

    # 5) 根据阈值生成 y_pred 并筛高置信度（按日期倒序）
    df_pred_all["y_pred"] = (df_pred_all["y_prob"] >= float(threshold)).astype(int)
    df_high = df_pred_all[df_pred_all["y_prob"] >= float(threshold)].copy()

    sort_cols = (["code"] if "code" in df_high.columns else []) + ["event_date"]
    df_high = df_high.sort_values(sort_cols, ascending=[True] * (len(sort_cols) - 1) + [False]).reset_index(drop=True)

    return df_daily, df_pred_all, df_high, payload, seq_len


def main():
    p = argparse.ArgumentParser(
        prog="stock_project",
        description="一键更新日线 + 批量预测 + 输出高置信度窗口及其前7天+事件日概率"
    )
    p.add_argument("--daily-csv",
                   default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv",
                   help="本地存储的当月日线 CSV 路径")
    p.add_argument("--model",
                   default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/model/xgb_20250831.joblib",
                   help=".joblib 格式的训练好模型文件")
    p.add_argument("--threshold", type=float, default=0.85,
                   help="筛选高置信度阈值（y_prob >= threshold）")
    p.add_argument("--seq-len",   type=int, default=7,
                   help="滑窗天数（默认为7）")
    p.add_argument("--start",     type=str, default="2025-07-01",
                   help="开始日期（含），缺省表示从 2025-07-01 到现在")
    p.add_argument("--save-output", type=str, default="./output/",
                   help="可选：保存高置信度结果到该目录，文件名自动生成为 predict_YYYYMMDD.csv")

    args = p.parse_args()

    # 核心流程
    df_daily, df_pred, df_high, payload, seq_len = full_refresh_and_predict(
        daily_csv=args.daily_csv,
        model_path=args.model,
        threshold=args.threshold,
        seq_len=args.seq_len,
        start=args.start,
    )

    # 输出高置信度窗口的基本信息（按日期倒序）
    print(f"\n从 {args.start} 起，全部窗口共 {len(df_pred)} 条，高置信度 (>= {args.threshold}) 共 {len(df_high)} 条：")
    show_cols = (["code"] if "code" in df_high.columns else []) + ["event_date", "y_prob"]
    print(df_high[show_cols].sort_values("event_date", ascending=False).head(100).to_string(index=False))

    # 为每条高置信度记录拼接“前7天 + 事件日”的概率序列
    if len(df_high) == 0:
        print("\n（无高置信度记录，无需输出序列。）")
        return

    records = []
    for _, row in df_high.iterrows():
        code = row["code"] if "code" in row else None
        evd  = row["event_date"]
        try:
            if code is None:
                # 无 code 列（单序列数据）的情况：无法按 code 回溯，这里直接跳过或可扩展成单序列逻辑
                continue
            prev7 = get_prev7_probs(
                daily_df=df_daily,
                code=str(code),
                event_date=evd,
                payload=payload,
                seq_len=args.seq_len,
            )
        except Exception as e:
            print(f"跳过 {code} @ {pd.to_datetime(evd).date()}：{e}")
            continue

        rec = {
            "code":       str(code),
            "event_date": evd,
            "y_prob":     float(row["y_prob"]),
        }
        rec.update(prev7)  # 合并 prob_day1..prob_day7, prob_event
        records.append(rec)

    if records:
        df_out = pd.DataFrame(records).sort_values("event_date", ascending=False).reset_index(drop=True)
        print("\n高置信度窗口及其前7天+事件日概率（按事件日倒序）：")
        print(df_out.head(100).to_string(index=False))

        # 可选保存
        if args.save_output:
            try:
                os.makedirs(args.save_output, exist_ok=True)
                today_str = datetime.now().strftime("%Y%m%d")
                save_path = os.path.join(args.save_output, f"predict_{today_str}.csv")
                df_out.to_csv(save_path, index=False, encoding="utf-8-sig")
                print(f"\n已将高置信度结果保存到 {save_path}")
            except Exception as e:
                print(f"\n保存结果失败：{e}")
    else:
        print("\n未生成可用的“前7天+事件日”概率序列（可能回溯不足或单序列数据）。")


if __name__ == "__main__":
    main()
