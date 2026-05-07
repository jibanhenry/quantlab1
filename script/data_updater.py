#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import baostock as bs
import pandas as pd
import time
import os
import sys
from datetime import datetime, timedelta
from tqdm import tqdm

DAILY_COLUMNS = ["日期","开盘","最高","最低","收盘","前收",
                 "成交量","成交额","换手率","涨跌幅",
                 "pbMRQ","psTTM"]

BAOSTOCK_RETRY_ATTEMPTS = 5
BAOSTOCK_RETRY_BASE_SLEEP = 3


def _empty_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=DAILY_COLUMNS)


def _normalize_code(code) -> str:
    return str(code).split(".")[-1].zfill(6)


def _baostock_symbol(code: str) -> str:
    code = _normalize_code(code)
    if code.startswith(("5", "6", "9")):
        return f"sh.{code}"
    if code.startswith(("4", "8")):
        return f"bj.{code}"
    return f"sz.{code}"


def _sleep_before_retry(attempt: int) -> None:
    time.sleep(BAOSTOCK_RETRY_BASE_SLEEP * attempt)


def login_baostock_with_retry(attempts: int = BAOSTOCK_RETRY_ATTEMPTS):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            lg = bs.login()
            if lg.error_code == "0":
                return lg
            last_error = f"{lg.error_code} {lg.error_msg}"
        except Exception as exc:
            last_error = str(exc)

        print(f"baostock 登录失败，第 {attempt}/{attempts} 次：{last_error}")
        try:
            bs.logout()
        except Exception:
            pass
        if attempt < attempts:
            _sleep_before_retry(attempt)

    raise RuntimeError(f"baostock 登录失败，已重试 {attempts} 次：{last_error}")


def fetch_stock_basic_baostock() -> dict:
    """获取当前在市普通 A 股代码及 IPO 日期"""
    last_error = None
    for attempt in range(1, BAOSTOCK_RETRY_ATTEMPTS + 1):
        rs = bs.query_stock_basic(code_name="")
        if rs.error_code == "0":
            break
        last_error = f"{rs.error_code} {rs.error_msg}"
        print(f"query_stock_basic 失败，第 {attempt}/{BAOSTOCK_RETRY_ATTEMPTS} 次：{last_error}")
        if attempt < BAOSTOCK_RETRY_ATTEMPTS:
            _sleep_before_retry(attempt)
    else:
        raise RuntimeError(f"query_stock_basic 失败：{last_error}")

    if rs.error_code != "0":
        raise RuntimeError(f"query_stock_basic 失败：{rs.error_code} {rs.error_msg}")
    stock_basic = {}
    while rs.error_code == "0" and rs.next():
        row = dict(zip(rs.fields, rs.get_row_data()))
        if row.get("type") != "1" or row.get("status") != "1":
            continue
        full = row["code"]       # 格式如 'sh.600000'
        parts = full.split('.')
        if len(parts) == 2:
            code = _normalize_code(parts[1])        # 取 '600000'
            stock_basic[code] = row.get("ipoDate") or None
    return dict(sorted(stock_basic.items()))


def fetch_all_codes_baostock() -> list:
    """获取当前在市普通 A 股代码（6 位，不带后缀）"""
    return list(fetch_stock_basic_baostock().keys())

def fetch_hist_baostock(code: str,
                        start_date: str,
                        end_date: str,
                        sleep: float = 0.2) -> pd.DataFrame:
    """
    拉取单只股票的日线（前复权），返回中文列名 DataFrame
    """
    code = _normalize_code(code)
    prefix = _baostock_symbol(code)
    fields = ",".join([
        "date","open","high","low","close",
        "preclose","volume","amount","turn",
        "pctChg","pbMRQ","psTTM"
    ])
    last_error = None
    for attempt in range(1, BAOSTOCK_RETRY_ATTEMPTS + 1):
        rs = bs.query_history_k_data_plus(
            prefix, fields,
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2"
        )
        if rs.error_code == "0":
            break
        last_error = f"{rs.error_code} {rs.error_msg}"
        print(f"{prefix} 日线查询失败，第 {attempt}/{BAOSTOCK_RETRY_ATTEMPTS} 次：{last_error}")
        if attempt < BAOSTOCK_RETRY_ATTEMPTS:
            _sleep_before_retry(attempt)
    else:
        raise RuntimeError(f"{prefix} 日线查询失败：{last_error}")

    records = []
    while rs.error_code == "0" and rs.next():
        records.append(rs.get_row_data())
    if not records:
        return _empty_daily_frame()

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
                output_file: str = "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv",
                sleep: float = 0.05,
                include_new_codes: bool = True):
    """
    增量拉取日线并保存到 output_file。
    - 如果 output_file 存在：按每个 code 自己的最后一天 +1 开始补
    - 如果不存在：从 start_date（或默认 2020-01-01）开始全量拉
    - include_new_codes=True 时额外纳入 baostock 当前在市普通股票
    - end_date 默认为今天
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 1. 判断增量区间
    default_start = start_date or "2020-01-01"
    if os.path.exists(output_file):
        df_exist = pd.read_csv(output_file, parse_dates=["日期"])
        df_exist["code"] = df_exist["code"].map(_normalize_code)
        df_exist["日期"] = pd.to_datetime(df_exist["日期"], errors="coerce")
        last_by_code = df_exist.groupby("code")["日期"].max()
    else:
        df_exist = None
        last_by_code = pd.Series(dtype="datetime64[ns]")

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    end_ts = pd.to_datetime(end_date)

    # 2. 登录并准备代码列表。既保留已有代码，也纳入数据源当前代码表。
    login_baostock_with_retry()

    buffers = []
    try:
        existing_codes = [] if df_exist is None else df_exist["code"].dropna().unique().tolist()
        if df_exist is None or include_new_codes:
            print("正在从 baostock 获取当前在市普通 A 股代码及 IPO 日期...")
            stock_basic = fetch_stock_basic_baostock()
            source_codes = list(stock_basic.keys())
            print(f"代码表获取完成：{len(source_codes)} 个当前在市普通 A 股")
        else:
            stock_basic = {}
            source_codes = []
        codes = sorted(set(existing_codes) | set(source_codes))

        start_by_code = {}
        for code in codes:
            last_day = last_by_code.get(code)
            if pd.notna(last_day):
                code_start = (last_day.date() + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                code_start = stock_basic.get(code) or default_start
            if pd.to_datetime(code_start) <= end_ts:
                start_by_code[code] = code_start

        if start_by_code:
            min_start = min(start_by_code.values())
            print(f"增量区间：{min_start} → {end_date}（按 code 分别补齐，待更新 {len(start_by_code)} 个代码）")
        else:
            print(f"所有代码已更新到 {end_date} 或更晚，无需下载。")

        # 3. 下载
        for code, code_start in tqdm(
            start_by_code.items(),
            desc="增量下载日线",
            file=sys.stdout,
            dynamic_ncols=True,
        ):
            try:
                df_inc = fetch_hist_baostock(code, code_start, end_date, sleep=sleep)
            except Exception as exc:
                print(f"\n[WARN] 跳过 {code}（{code_start} → {end_date}）：{exc}")
                continue
            if not df_inc.empty:
                df_inc.insert(0, "code", code)
                buffers.append(df_inc)
    finally:
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
