#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import baostock as bs
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from tqdm import tqdm

def fetch_all_codes_baostock() -> list:
    """获取所有 A 股代码（6 位，不带后缀）"""
    rs = bs.query_stock_basic(code_name="")
    codes = []
    while rs.error_code == "0" and rs.next():
        full = rs.get_row_data()[0]       # 格式如 'sh.600000'
        parts = full.split('.')
        if len(parts) == 2:
            codes.append(parts[1])        # 取 '600000'
    return codes

def fetch_hist_baostock(code: str,
                        start_date: str,
                        end_date: str,
                        sleep: float = 0.2) -> pd.DataFrame:
    """
    拉取单只股票的日线（前复权），返回中文列名 DataFrame
    """
    prefix = f"sh.{code}" if code.startswith("6") else f"sz.{code}"
    fields = ",".join([
        "date","open","high","low","close",
        "preclose","volume","amount","turn",
        "pctChg","pbMRQ","psTTM"
    ])
    rs = bs.query_history_k_data_plus(
        prefix, fields,
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="2"
    )
    records = []
    while rs.error_code == "0" and rs.next():
        records.append(rs.get_row_data())
    if not records:
        # 返回一个空的 DataFrame，列名和后面一致
        cols = ["日期","开盘","最高","最低","收盘","前收",
                "成交量","成交额","换手率","涨跌幅",
                "pbMRQ","psTTM"]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(records, columns=fields.split(","))
    df.rename(columns={
        "date":     "日期",
        "open":     "开盘",
        "high":     "最高",
        "low":      "最低",
        "close":    "收盘",
        "preclose": "前收",
        "volume":   "成交量",
        "amount":   "成交额",
        "turn":     "换手率",
        "pctChg":   "涨跌幅",
        "pbMRQ":    "pbMRQ",
        "psTTM":    "psTTM"
    }, inplace=True)

    # 类型转换
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    num_cols = ["开盘","最高","最低","收盘","前收",
                "成交量","成交额","换手率","涨跌幅",
                "pbMRQ","psTTM"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    time.sleep(sleep)
    return df

def update_data(start_date: str = None,
                end_date:   str = None,
                output_file: str = "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv"):
    """
    增量拉取日线并保存到 output_file。
    - 如果 output_file 存在：自动从文件最后一天 +1 开始补
    - 如果不存在：从 start_date（或默认 2020-01-01）开始全量拉
    - end_date 默认为今天
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 1. 判断增量区间
    if os.path.exists(output_file):
        df_exist = pd.read_csv(output_file, parse_dates=["日期"])
        df_exist["code"] = df_exist["code"].astype(str).str.zfill(6)
        last_day = df_exist["日期"].max().date()
        start_date = (last_day + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        df_exist = None
        if start_date is None:
            start_date = "2020-01-01"

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"增量区间：{start_date} → {end_date}")

    # 2. 准备代码列表
    if df_exist is not None:
        # 先强制转 str，再 zfill
        codes = [
            str(c).zfill(6)
            for c in df_exist["code"].astype(str).unique()
        ]
    else:
        codes = fetch_all_codes_baostock()

    # 3. 登录并下载
    bs.login()
    buffers = []
    for code in tqdm(codes, desc="增量下载日线"):
        df_inc = fetch_hist_baostock(code, start_date, end_date)
        if not df_inc.empty:
            df_inc.insert(0, "code", code)
            buffers.append(df_inc)
    bs.logout()

    # 4. 合并去重并保存
    if buffers:
        df_new = pd.concat(buffers, ignore_index=True)
        if df_exist is not None:
            df_all = pd.concat([df_exist, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.drop_duplicates(subset=["code","日期"], keep="last", inplace=True)
        df_all.sort_values(["code","日期"], inplace=True)
        df_all.to_csv(output_file, index=False)
        print(f"完成：新增 {len(df_new)} 条，文件共 {len(df_all)} 条，已保存到 {output_file}")
    else:
        print("未获取到任何新数据。")

def main():
    # 你可以在这里加入 argparse 来让 start_date/end_date/output_file 可由命令行传入
    update_data()

if __name__ == "__main__":
    main()