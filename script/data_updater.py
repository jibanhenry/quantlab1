#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import baostock as bs
import pandas as pd
import time
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from tqdm import tqdm

DAILY_COLUMNS = ["日期","开盘","最高","最低","收盘","前收",
                 "成交量","成交额","换手率","涨跌幅",
                 "pbMRQ","psTTM"]
NUMERIC_COLUMNS = ["开盘","最高","最低","收盘","前收",
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
    if code.startswith(("4", "8", "920")):
        return f"bj.{code}"
    if code.startswith(("5", "6", "9")):
        return f"sh.{code}"
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
        if rs.error_code == "10001001":
            try:
                bs.logout()
            except Exception:
                pass
            login_baostock_with_retry()
            continue
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


def _chunked(seq, size: int):
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _coerce_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["code"] + DAILY_COLUMNS)
    df = df.copy()
    if "code" not in df.columns:
        df.insert(0, "code", "")
    df["code"] = df["code"].map(_normalize_code)
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _merge_daily_frames(df_existing: pd.DataFrame, frames: list) -> tuple:
    if frames:
        df_new = pd.concat(frames, ignore_index=True)
        df_new = _coerce_daily_frame(df_new)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
        new_rows = int(len(df_new))
    else:
        df_new = pd.DataFrame(columns=["code"] + DAILY_COLUMNS)
        df_all = df_existing.copy()
        new_rows = 0
    df_all = _coerce_daily_frame(df_all)
    df_all.dropna(subset=["code", "日期"], inplace=True)
    df_all.drop_duplicates(subset=["code","日期"], keep="last", inplace=True)
    df_all.sort_values(["code","日期"], inplace=True)
    return df_all, new_rows


def _save_daily_frame(df: pd.DataFrame, output_file: str) -> None:
    df.to_csv(output_file, index=False, encoding="utf-8-sig")


def fetch_hist_batch(tasks: list, sleep: float = 0.01):
    try:
        login_baostock_with_retry()
    except Exception as exc:
        return len(tasks), [], [{"code": task[0], "error": f"login: {exc}"} for task in tasks]

    buffers = []
    errors = []
    try:
        for code, code_start, end_date in tasks:
            try:
                df_inc = fetch_hist_baostock(code, code_start, end_date, sleep=sleep)
            except Exception as exc:
                errors.append({"code": code, "start_date": code_start, "end_date": end_date, "error": str(exc)})
                continue
            if not df_inc.empty:
                df_inc.insert(0, "code", code)
                buffers.append(df_inc)
        return len(tasks), buffers, errors
    finally:
        bs.logout()


def _process_pool_executor(workers: int) -> ProcessPoolExecutor:
    try:
        ctx = mp.get_context("fork")
        return ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    except (TypeError, ValueError):
        return ProcessPoolExecutor(max_workers=workers)


def update_data(start_date: str = None,
                end_date:   str = None,
                output_file: str = "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv",
                sleep: float = 0.05,
                include_new_codes: bool = True,
                workers: int = 1,
                batch_size: int = 25,
                flush_every: int = 250,
                max_update_codes: int = None):
    """
    增量拉取日线并保存到 output_file。
    - 如果 output_file 存在：按每个 code 自己的最后一天 +1 开始补
    - 如果不存在：从 start_date（或默认 2020-01-01）开始全量拉
    - include_new_codes=True 时额外纳入 baostock 当前在市普通股票
    - end_date 默认为今天
    - workers>1 时使用多进程按批并行拉取，并定期落盘保存进度
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 1. 判断增量区间
    default_start = start_date or "2020-01-01"
    if os.path.exists(output_file):
        df_exist = pd.read_csv(output_file, dtype={"code": str}, parse_dates=["日期"])
        df_exist = _coerce_daily_frame(df_exist)
        last_by_code = df_exist.groupby("code")["日期"].max()
    else:
        df_exist = pd.DataFrame(columns=["code"] + DAILY_COLUMNS)
        df_exist["日期"] = pd.to_datetime(df_exist["日期"])
        last_by_code = pd.Series(dtype="datetime64[ns]")

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    end_ts = pd.to_datetime(end_date)

    # 2. 登录并准备代码列表。既保留已有代码，也纳入数据源当前代码表。
    login_baostock_with_retry()
    parent_logged_in = True

    buffers = []
    try:
        existing_codes = df_exist["code"].dropna().unique().tolist()
        if df_exist.empty or include_new_codes:
            print("正在从 baostock 获取当前在市普通 A 股代码及 IPO 日期...")
            stock_basic = fetch_stock_basic_baostock()
            source_codes = list(stock_basic.keys())
            print(f"代码表获取完成：{len(source_codes)} 个当前在市普通 A 股")
        else:
            stock_basic = {}
            source_codes = []
        codes = sorted(set(existing_codes) | set(source_codes))

        tasks = []
        for code in codes:
            last_day = last_by_code.get(code)
            if pd.notna(last_day):
                code_start = (last_day.date() + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                code_start = stock_basic.get(code) or default_start
            if pd.to_datetime(code_start) <= end_ts:
                tasks.append((code, code_start, end_date))

        if max_update_codes is not None and max_update_codes > 0 and len(tasks) > max_update_codes:
            print(f"限制本轮更新代码数：{max_update_codes}/{len(tasks)}")
            tasks = tasks[:max_update_codes]

        if tasks:
            min_start = min(task[1] for task in tasks)
            print(f"增量区间：{min_start} → {end_date}（按 code 分别补齐，待更新 {len(tasks)} 个代码）")
        else:
            print(f"所有代码已更新到 {end_date} 或更晚，无需下载。")

        # 3. 下载
        workers = max(1, int(workers or 1))
        batch_size = max(1, int(batch_size or 1))
        flush_every = max(1, int(flush_every or 1))
        df_all = df_exist.copy()
        total_new_rows = 0
        errors = []
        if workers > 1:
            try:
                bs.logout()
            except Exception:
                pass
            parent_logged_in = False

        if workers <= 1:
            done = 0
            for code, code_start, task_end_date in tqdm(
                tasks,
                desc="增量下载日线",
                file=sys.stdout,
                dynamic_ncols=True,
            ):
                try:
                    df_inc = fetch_hist_baostock(code, code_start, task_end_date, sleep=sleep)
                except Exception as exc:
                    print(f"\n[WARN] 跳过 {code}（{code_start} → {task_end_date}）：{exc}")
                    errors.append({"code": code, "start_date": code_start, "end_date": task_end_date, "error": str(exc)})
                    done += 1
                    continue
                if not df_inc.empty:
                    df_inc.insert(0, "code", code)
                    buffers.append(df_inc)
                done += 1
                if buffers and done % flush_every == 0:
                    df_all, new_rows = _merge_daily_frames(df_all, buffers)
                    total_new_rows += new_rows
                    _save_daily_frame(df_all, output_file)
                    print(f"\n已保存阶段进度：检查 {done}/{len(tasks)}，新增原始行 {total_new_rows}，文件共 {len(df_all)} 条")
                    buffers = []
        elif tasks:
            task_batches = list(_chunked(tasks, batch_size))
            done = 0
            last_flush_done = 0
            print(f"并行增量下载：workers={workers}，batch_size={batch_size}，batches={len(task_batches)}")
            with _process_pool_executor(workers) as pool:
                futures = [pool.submit(fetch_hist_batch, task_batch, sleep) for task_batch in task_batches]
                for fut in as_completed(futures):
                    batch_done, batch_buffers, batch_errors = fut.result()
                    done += batch_done
                    buffers.extend(batch_buffers)
                    errors.extend(batch_errors)
                    if buffers and (done - last_flush_done >= flush_every or done >= len(tasks)):
                        df_all, new_rows = _merge_daily_frames(df_all, buffers)
                        total_new_rows += new_rows
                        _save_daily_frame(df_all, output_file)
                        last_flush_done = done
                        print(
                            f"已保存阶段进度：检查 {min(done, len(tasks))}/{len(tasks)}，"
                            f"新增原始行 {total_new_rows}，错误 {len(errors)}，文件共 {len(df_all)} 条",
                            flush=True,
                        )
                        buffers = []
    finally:
        if parent_logged_in:
            bs.logout()

    # 4. 合并去重并保存
    if buffers:
        df_all, new_rows = _merge_daily_frames(df_all, buffers)
        total_new_rows += new_rows
        _save_daily_frame(df_all, output_file)
    elif not tasks:
        df_all = df_exist.copy()

    if total_new_rows > 0:
        print(f"完成：新增原始行 {total_new_rows} 条，文件共 {len(df_all)} 条，已保存到 {output_file}")
    else:
        print("未获取到任何新数据。")

    error_path = os.path.join(os.path.dirname(output_file), "daily_update_errors.csv")
    if errors:
        pd.DataFrame(errors).to_csv(error_path, index=False, encoding="utf-8-sig")
        print(f"部分代码更新失败，已记录：{error_path}")
        raise RuntimeError(f"日线更新存在 {len(errors)} 个代码失败，详见 {error_path}")
    elif os.path.exists(error_path):
        os.remove(error_path)

    return df_all

def main():
    # 你可以在这里加入 argparse 来让 start_date/end_date/output_file 可由命令行传入
    update_data()

if __name__ == "__main__":
    main()
