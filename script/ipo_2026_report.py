#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import baostock as bs
import pandas as pd

DAILY_COLUMNS = [
    "日期", "开盘", "最高", "最低", "收盘", "前收",
    "成交量", "成交额", "换手率", "涨跌幅", "pbMRQ", "psTTM",
]
NUMERIC_COLUMNS = [
    "开盘", "最高", "最低", "收盘", "前收",
    "成交量", "成交额", "换手率", "涨跌幅", "pbMRQ", "psTTM",
]


def normalize_code(code) -> str:
    return str(code).split(".")[-1].strip().zfill(6)


def baostock_symbol(code: str) -> str:
    code = normalize_code(code)
    if code.startswith(("4", "8", "920")):
        return f"bj.{code}"
    if code.startswith(("5", "6", "9")):
        return f"sh.{code}"
    return f"sz.{code}"


def board_from_code(code: str) -> str:
    code = normalize_code(code)
    if code.startswith("688"):
        return "科创板"
    if code.startswith(("300", "301")):
        return "创业板"
    if code.startswith(("000", "001", "002", "003")):
        return "深市主板"
    if code.startswith(("600", "601", "603", "605")):
        return "沪市主板"
    if code.startswith(("4", "8", "920")):
        return "北交所"
    return "其他"


def to_float(value):
    if value is None:
        return math.nan
    text = str(value).replace(",", "").replace("元", "").strip()
    if text in {"", "-", "--", "nan", "None", "NaT"}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def fetch_stock_basic() -> pd.DataFrame:
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败：{lg.error_code} {lg.error_msg}")
    try:
        rs = bs.query_stock_basic(code_name="")
        if rs.error_code != "0":
            raise RuntimeError(f"query_stock_basic 失败：{rs.error_code} {rs.error_msg}")
        rows = []
        while rs.error_code == "0" and rs.next():
            row = dict(zip(rs.fields, rs.get_row_data()))
            if row.get("type") != "1":
                continue
            code = normalize_code(row.get("code", ""))
            rows.append({
                "code": code,
                "name": row.get("code_name") or "",
                "ipo_date": row.get("ipoDate") or "",
                "out_date": row.get("outDate") or "",
                "status": row.get("status") or "",
            })
        return pd.DataFrame(rows).drop_duplicates("code")
    finally:
        bs.logout()


def fetch_hist_with_active_login(code: str, start_date: str, end_date: str):
    fields = ",".join([
        "date", "open", "high", "low", "close", "preclose",
        "volume", "amount", "turn", "pctChg", "pbMRQ", "psTTM",
    ])
    rs = bs.query_history_k_data_plus(
        baostock_symbol(code),
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",
    )
    if rs.error_code != "0":
        return f"{rs.error_code} {rs.error_msg}", pd.DataFrame(columns=["code"] + DAILY_COLUMNS)

    records = []
    while rs.error_code == "0" and rs.next():
        records.append(rs.get_row_data())
    if not records:
        return "", pd.DataFrame(columns=["code"] + DAILY_COLUMNS)

    df = pd.DataFrame(records, columns=fields.split(","))
    df.rename(columns={
        "date": "日期",
        "open": "开盘",
        "high": "最高",
        "low": "最低",
        "close": "收盘",
        "preclose": "前收",
        "volume": "成交量",
        "amount": "成交额",
        "turn": "换手率",
        "pctChg": "涨跌幅",
        "pbMRQ": "pbMRQ",
        "psTTM": "psTTM",
    }, inplace=True)
    df.insert(0, "code", normalize_code(code))
    return "", df


def fetch_hist_batch(tasks):
    lg = bs.login()
    if lg.error_code != "0":
        return len(tasks), [], [{"code": task[0], "error": f"login {lg.error_code} {lg.error_msg}"} for task in tasks]
    buffers = []
    errors = []
    try:
        for code, start_date, end_date in tasks:
            try:
                error, df = fetch_hist_with_active_login(code, start_date, end_date)
                if error:
                    errors.append({"code": code, "error": error})
                if not df.empty:
                    buffers.append(df)
            except Exception as exc:
                errors.append({"code": code, "error": str(exc)})
        return len(tasks), buffers, errors
    finally:
        bs.logout()


def chunked(seq, chunks):
    chunks = max(1, chunks)
    size = max(1, math.ceil(len(seq) / chunks))
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def update_daily_data(daily_csv: Path, stock_basic: pd.DataFrame, end_date: str, workers: int) -> pd.DataFrame:
    daily_csv.parent.mkdir(parents=True, exist_ok=True)
    if daily_csv.exists():
        df_exist = pd.read_csv(daily_csv, dtype={"code": str}, parse_dates=["日期"])
        df_exist["code"] = df_exist["code"].map(normalize_code)
    else:
        df_exist = pd.DataFrame(columns=["code"] + DAILY_COLUMNS)
        df_exist["日期"] = pd.to_datetime(df_exist["日期"])

    df_exist["日期"] = pd.to_datetime(df_exist["日期"], errors="coerce")
    last_by_code = df_exist.groupby("code")["日期"].max()
    stock_basic = stock_basic.copy()
    stock_basic["code"] = stock_basic["code"].map(normalize_code)
    stock_basic["ipo_date"] = pd.to_datetime(stock_basic["ipo_date"], errors="coerce")

    existing_codes = set(df_exist["code"].dropna())
    source_codes = set(stock_basic.loc[stock_basic["status"].eq("1"), "code"].dropna())
    codes = sorted(existing_codes | source_codes)
    ipo_by_code = stock_basic.dropna(subset=["ipo_date"]).set_index("code")["ipo_date"].dt.strftime("%Y-%m-%d").to_dict()
    end_ts = pd.to_datetime(end_date)

    tasks = []
    for code in codes:
        last_day = last_by_code.get(code)
        if pd.notna(last_day):
            start_date = (last_day.date() + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start_date = ipo_by_code.get(code, "2020-01-01")
        if pd.to_datetime(start_date) <= end_ts:
            tasks.append((code, start_date, end_date))

    print(f"待增量检查代码数：{len(tasks)}，目标截止日：{end_date}")
    buffers = []
    errors = []
    if tasks:
        task_chunks = list(chunked(tasks, workers * 4))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_hist_batch, task_chunk) for task_chunk in task_chunks]
            done = 0
            for fut in as_completed(futures):
                batch_done, batch_buffers, batch_errors = fut.result()
                done += batch_done
                buffers.extend(batch_buffers)
                errors.extend(batch_errors)
                print(
                    f"增量检查进度：约 {min(done, len(tasks))}/{len(tasks)}，"
                    f"新增非空批次：{len(buffers)}，错误：{len(errors)}",
                    flush=True,
                )

    if buffers:
        df_new = pd.concat(buffers, ignore_index=True)
        df_new["日期"] = pd.to_datetime(df_new["日期"], errors="coerce")
        for col in NUMERIC_COLUMNS:
            df_new[col] = pd.to_numeric(df_new[col], errors="coerce")
        df_all = pd.concat([df_exist, df_new], ignore_index=True)
    else:
        df_new = pd.DataFrame(columns=["code"] + DAILY_COLUMNS)
        df_all = df_exist.copy()

    df_all["code"] = df_all["code"].map(normalize_code)
    df_all["日期"] = pd.to_datetime(df_all["日期"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
    df_all.dropna(subset=["code", "日期"], inplace=True)
    df_all.drop_duplicates(subset=["code", "日期"], keep="last", inplace=True)
    df_all.sort_values(["code", "日期"], inplace=True)
    df_all.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    print(f"日线更新完成：新增 {len(df_new)} 行，合并后 {len(df_all)} 行，保存到 {daily_csv}")

    if errors:
        error_path = Path("output/ipo_2026/daily_update_errors.csv")
        error_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(errors).to_csv(error_path, index=False, encoding="utf-8-sig")
        print(f"部分代码更新失败，已记录：{error_path}")
    else:
        error_path = Path("output/ipo_2026/daily_update_errors.csv")
        if error_path.exists():
            error_path.unlink()
    return df_all


def load_ipo_issue_data() -> pd.DataFrame:
    frames = []
    try:
        import akshare as ak
        cninfo = ak.stock_new_ipo_cninfo()
        if not cninfo.empty:
            code_col = "证劵代码" if "证劵代码" in cninfo.columns else "证券代码"
            frames.append(pd.DataFrame({
                "code": cninfo[code_col].map(normalize_code),
                "issue_name": cninfo.get("证券简称", ""),
                "issue_price": cninfo.get("发行价", pd.NA).map(to_float),
                "issue_source": "cninfo",
            }))
    except Exception as exc:
        print(f"[WARN] akshare stock_new_ipo_cninfo 获取失败：{exc}", file=sys.stderr)

    try:
        import akshare as ak
        ths = ak.stock_ipo_ths(symbol="全部A股")
        if not ths.empty:
            frames.append(pd.DataFrame({
                "code": ths["股票代码"].map(normalize_code),
                "issue_name": ths.get("股票简称", ""),
                "issue_price": ths.get("发行价格", pd.NA).map(to_float),
                "issue_source": "ths",
            }))
    except Exception as exc:
        print(f"[WARN] akshare stock_ipo_ths 获取失败：{exc}", file=sys.stderr)

    if not frames:
        return pd.DataFrame(columns=["code", "issue_name", "issue_price", "issue_source"])
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["code", "issue_price"], na_position="last")
    out = out.drop_duplicates("code", keep="first")
    return out


def pct_vs_open(value, open_price):
    if pd.isna(value) or pd.isna(open_price) or open_price == 0:
        return math.nan
    return (value / open_price - 1.0) * 100.0


def build_ipo_detail(df_daily: pd.DataFrame, stock_basic: pd.DataFrame, issue_df: pd.DataFrame) -> pd.DataFrame:
    basic = stock_basic.copy()
    basic["code"] = basic["code"].map(normalize_code)
    basic["ipo_date"] = pd.to_datetime(basic["ipo_date"], errors="coerce")
    new_basic = basic.loc[basic["ipo_date"].ge(pd.Timestamp("2026-01-01"))].copy()

    daily = df_daily.copy()
    daily["code"] = daily["code"].map(normalize_code)
    daily["日期"] = pd.to_datetime(daily["日期"], errors="coerce")
    daily.sort_values(["code", "日期"], inplace=True)

    issue_df = issue_df.copy()
    issue_df["code"] = issue_df["code"].map(normalize_code)
    issue_by_code = issue_df.set_index("code")

    rows = []
    for _, info in new_basic.sort_values(["ipo_date", "code"]).iterrows():
        code = info["code"]
        hist = daily.loc[daily["code"].eq(code)].sort_values("日期").reset_index(drop=True)
        first = hist.iloc[0] if len(hist) else None
        row = {
            "code": code,
            "name": info.get("name") or "",
            "board": board_from_code(code),
            "ipo_date": info["ipo_date"].date().isoformat() if pd.notna(info["ipo_date"]) else "",
            "available_trading_days": len(hist),
            "issue_price": math.nan,
            "open_price": math.nan,
            "close_day1": math.nan,
            "close_day5": math.nan,
            "close_day30": math.nan,
            "close_day60": math.nan,
        }
        if code in issue_by_code.index:
            row["issue_price"] = issue_by_code.loc[code, "issue_price"]
            if not row["name"]:
                row["name"] = issue_by_code.loc[code, "issue_name"]
        if first is not None:
            row["first_trade_date"] = first["日期"].date().isoformat()
            row["open_price"] = first["开盘"]
            row["close_day1"] = first["收盘"]
            for n in [5, 30, 60]:
                if len(hist) >= n:
                    row[f"close_day{n}"] = hist.iloc[n - 1]["收盘"]
        else:
            row["first_trade_date"] = ""

        row["open_premium_vs_issue_pct"] = pct_vs_open(row["open_price"], row["issue_price"])
        for label in ["day1", "day5", "day30", "day60"]:
            row[f"return_{label}_vs_open_pct"] = pct_vs_open(row[f"close_{label}"], row["open_price"])
        rows.append(row)

    detail = pd.DataFrame(rows)
    rename = {
        "code": "代码",
        "name": "名称",
        "board": "板块",
        "ipo_date": "上市日期",
        "first_trade_date": "首个交易日",
        "available_trading_days": "可用交易日数",
        "issue_price": "发行价",
        "open_price": "开盘价",
        "close_day1": "当日收盘价",
        "close_day5": "5日收盘价",
        "close_day30": "30日收盘价",
        "close_day60": "60日收盘价",
        "open_premium_vs_issue_pct": "开盘较发行价涨跌幅(%)",
        "return_day1_vs_open_pct": "当日收盘较开盘涨跌幅(%)",
        "return_day5_vs_open_pct": "5日收盘较开盘涨跌幅(%)",
        "return_day30_vs_open_pct": "30日收盘较开盘涨跌幅(%)",
        "return_day60_vs_open_pct": "60日收盘较开盘涨跌幅(%)",
    }
    return detail.rename(columns=rename)


def describe_returns(detail: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("开盘较发行价涨跌幅(%)", "开盘溢价"),
        ("当日收盘较开盘涨跌幅(%)", "当日"),
        ("5日收盘较开盘涨跌幅(%)", "5日"),
        ("30日收盘较开盘涨跌幅(%)", "30日"),
        ("60日收盘较开盘涨跌幅(%)", "60日"),
    ]
    rows = []
    for col, name in metrics:
        s = pd.to_numeric(detail[col], errors="coerce").dropna()
        rows.append({
            "指标": name,
            "样本数": int(s.count()),
            "均值(%)": s.mean(),
            "中位数(%)": s.median(),
            "最大值(%)": s.max(),
            "最小值(%)": s.min(),
            "胜率(%)": (s.gt(0).mean() * 100.0) if len(s) else math.nan,
        })
    return pd.DataFrame(rows)


def board_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for board, part in detail.groupby("板块", dropna=False):
        row = {"板块": board, "新股数量": len(part)}
        for col, label in [
            ("当日收盘较开盘涨跌幅(%)", "当日均值(%)"),
            ("5日收盘较开盘涨跌幅(%)", "5日均值(%)"),
            ("30日收盘较开盘涨跌幅(%)", "30日均值(%)"),
            ("60日收盘较开盘涨跌幅(%)", "60日均值(%)"),
        ]:
            s = pd.to_numeric(part[col], errors="coerce")
            row[label] = s.mean()
            row[label.replace("均值", "胜率")] = s.gt(0).mean() * 100.0 if s.notna().any() else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("新股数量", ascending=False)


def write_analysis(detail: pd.DataFrame, summary: pd.DataFrame, board: pd.DataFrame, outdir: Path, latest_date: str) -> None:
    def md_table(df, max_rows=None):
        view = df if max_rows is None else df.head(max_rows)
        return view.to_markdown(index=False, floatfmt=".2f")

    total = len(detail)
    missing_issue = int(detail["发行价"].isna().sum())
    missing_first = int(detail["开盘价"].isna().sum())
    full5 = int(detail["5日收盘价"].notna().sum())
    full30 = int(detail["30日收盘价"].notna().sum())
    full60 = int(detail["60日收盘价"].notna().sum())

    best = detail.sort_values("60日收盘较开盘涨跌幅(%)", ascending=False, na_position="last")
    worst = detail.sort_values("60日收盘较开盘涨跌幅(%)", ascending=True, na_position="last")
    if best["60日收盘较开盘涨跌幅(%)"].notna().sum() == 0:
        best = detail.sort_values("30日收盘较开盘涨跌幅(%)", ascending=False, na_position="last")
        worst = detail.sort_values("30日收盘较开盘涨跌幅(%)", ascending=True, na_position="last")

    content = [
        "# 2026年以来新股表现统计",
        "",
        f"- 日线数据最新日期：{latest_date}",
        f"- 2026年以来新上市股票数：{total}",
        f"- 已满5/30/60个交易日：{full5}/{full30}/{full60}",
        f"- 缺失发行价：{missing_issue}；缺失首日行情：{missing_first}",
        "",
        "## 汇总统计",
        md_table(summary),
        "",
        "## 按板块统计",
        md_table(board),
        "",
        "## 表现最佳 Top 10",
        md_table(best[["代码", "名称", "板块", "上市日期", "可用交易日数", "30日收盘较开盘涨跌幅(%)", "60日收盘较开盘涨跌幅(%)"]], 10),
        "",
        "## 表现最弱 Top 10",
        md_table(worst[["代码", "名称", "板块", "上市日期", "可用交易日数", "30日收盘较开盘涨跌幅(%)", "60日收盘较开盘涨跌幅(%)"]], 10),
        "",
        "## 数据质量说明",
        "- 5/30/60日口径：上市日计为第1个交易日。",
        "- 未满30或60个交易日的新股，对应字段留空。",
        "- 发行价来自 akshare 新股发行接口；日线价格来自 baostock 前复权日线。",
    ]
    (outdir / "ipo_2026_analysis.md").write_text("\n".join(content), encoding="utf-8")


def validate_outputs(df_daily: pd.DataFrame, detail: pd.DataFrame) -> list:
    issues = []
    dup = df_daily.duplicated(["code", "日期"]).sum()
    if dup:
        issues.append(f"日线存在重复 code+日期：{dup}")
    if df_daily["日期"].isna().any():
        issues.append("日线存在无法解析的日期")
    for col in ["开盘", "收盘"]:
        if pd.to_numeric(df_daily[col], errors="coerce").isna().all():
            issues.append(f"日线 {col} 全部无法解析为数值")
    if detail["代码"].duplicated().any():
        issues.append("新股明细存在重复代码")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-csv", default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv")
    ap.add_argument("--outdir", default="output/ipo_2026")
    ap.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    daily_csv = Path(args.daily_csv)

    print("获取 baostock 股票基础信息...")
    stock_basic = fetch_stock_basic()
    stock_basic.to_csv(outdir / "stock_basic_baostock.csv", index=False, encoding="utf-8-sig")

    print("增量更新日线...")
    df_daily = update_daily_data(daily_csv, stock_basic, args.end_date, args.workers)
    latest_date = df_daily["日期"].max().date().isoformat() if not df_daily.empty else ""

    print("获取 IPO 发行价...")
    issue_df = load_ipo_issue_data()
    issue_df.to_csv(outdir / "ipo_issue_sources.csv", index=False, encoding="utf-8-sig")

    print("生成新股明细与统计...")
    detail = build_ipo_detail(df_daily, stock_basic, issue_df)
    summary = describe_returns(detail)
    board = board_summary(detail)
    detail.to_csv(outdir / "ipo_2026_detail.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    summary.to_csv(outdir / "ipo_2026_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    board.to_csv(outdir / "ipo_2026_board_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    write_analysis(detail, summary, board, outdir, latest_date)

    issues = validate_outputs(df_daily, detail)
    if issues:
        print("校验发现问题：")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)
    print(f"完成。输出目录：{outdir}；日线最新日期：{latest_date}；2026新股：{len(detail)}")


if __name__ == "__main__":
    main()
