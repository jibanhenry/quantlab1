# -*- coding: utf-8 -*-
"""
stock_project/__initial__.py

一键执行：先更新日线、再批量预测、最后输出高置信度窗口及其前7天+事件日的预测概率。
"""
import argparse
import pandas as pd
import numpy as np

from script.data_updater import update_data
from script.predictor import (
    daily_to_windows,
    load_model,
    predict_windows,
    filter_by_threshold,
    get_prev7_probs,
)

def full_refresh_and_predict(
    daily_csv: str,
    model_path: str,
    threshold: float = 0.85,
    raw_cols: list[str] = None,
    seq_len: int = 7,
):
    """
    1) 调用 update_data 更新 daily_csv（增量或首次全量）
    2) 读取最新 daily_csv，转为窗口格式
    3) 批量 predict_proba
    4) 筛出高置信度，返回 (df_daily, df_pred, df_high)
    """
    # 1) 更新日线
    #update_data(output_file=daily_csv)

    # 2) 读取整月日线
    df_daily = pd.read_csv(daily_csv, parse_dates=["日期"])
    df_daily["code"] = df_daily["code"].astype(str).str.zfill(6)

    # 默认特征列
    if raw_cols is None:
        raw_cols = [
            "开盘", "最高", "最低", "收盘", "前收",
            "成交量", "成交额", "换手率", "涨跌幅",
            "pbMRQ", "psTTM",
        ]

    # 3) 转窗口 + 预测
    df_win = daily_to_windows(df_daily, raw_cols, seq_len=seq_len)
    feat_cols = [c for c in df_win.columns if c not in ("code","event_date")]

    model = load_model(model_path)
    df_pred = predict_windows(df_win, model, feat_cols)

    # 4) 筛高置信度
    df_high = filter_by_threshold(df_pred, threshold)

    return df_daily, df_pred, df_high, model, raw_cols, seq_len


def main():
    p = argparse.ArgumentParser(
        prog="stock_project",
        description="一键更新日线 + 批量预测 + 输出高置信度窗口及其前7天+事件日概率"
    )
    p.add_argument("--daily-csv", default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv",
                   help="本地存储的当月日线 CSV 路径")
    p.add_argument("--model",     default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/model/gbdt_model.joblib",
                   help=".joblib 格式的训练好模型文件")
    p.add_argument("--threshold", type=float, default=0.85,
                   help="筛选高置信度阈值")
    p.add_argument("--seq-len",   type=int, default=7,
                   help="滑窗天数（默认为7）")
    args = p.parse_args()

    # 核心流程
    df_daily, df_pred, df_high, model, raw_cols, seq_len = full_refresh_and_predict(
        daily_csv=args.daily_csv,
        model_path=args.model,
        threshold=args.threshold,
        seq_len=args.seq_len,
    )

    # 输出高置信度窗口的基本信息
    print(f"\n全部窗口共 {len(df_pred)} 条，高置信度 (> {args.threshold}) 共 {len(df_high)} 条：")
    print(df_high[["code","event_date","pred_proba"]].to_string(index=False))

    # 再为每条高置信度记录拼接前7天和事件日的概率序列
    records = []
    for _, row in df_high.iterrows():
        code = row["code"]
        evd = row["event_date"]
        try:
            prev7 = get_prev7_probs(
                df_daily=df_daily,
                code=code,
                event_date=evd,
                raw_cols=raw_cols,
                seq_len=seq_len,
                model=model
            )
        except (KeyError, ValueError) as e:
            # 如果回溯不足或其他问题，则跳过
            print(f"跳过 {code} @ {evd}：{e}")
            continue

        rec = {
            "code":       code,
            "event_date": evd,
            "pred_proba": row["pred_proba"]
        }
        # 添加 prob_day1…prob_day7 和 prob_event
        rec.update(prev7.to_dict())
        records.append(rec)

    df_out = pd.DataFrame(records)
    print("\n高置信度窗口及其前7天+事件日概率：")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()