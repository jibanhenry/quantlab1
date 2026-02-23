import os
import pandas as pd

# 直接复用你现有的扫描逻辑
from pattern_scan import scan_patterns_and_summarize

# 可选：如果你项目里有 io_utils，就用它统一字段（和你训练GRU一致）
try:
    from io_utils import load_market_csv_multi  # type: ignore
except Exception:
    load_market_csv_multi = None


def main():
    # ====== 你要用的CSV ======
    csv_path = "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv"

    # ====== 只保留 entry_date >= 2026-01-01 ======
    min_entry_date = pd.to_datetime("2026-01-01")

    # ====== 输出位置 ======
    outdir = "./pattern_output"
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, "pattern_signals_from_2026.csv")

    # ====== 读数据 ======
    if load_market_csv_multi is not None:
        df = load_market_csv_multi([csv_path])
    else:
        df = pd.read_csv(csv_path)

    # 统一日期格式
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        # 如果你的字段不是 date/code，可以在这里改
        raise ValueError(f"CSV缺少 date 列。当前列: {list(df.columns)[:50]}")

    # ====== 跑4个模式扫描（你的 pattern_scan.py 已经只保留4个） ======
    events, _summary = scan_patterns_and_summarize(
        df,
        horizon=30,              # 这里无所谓，导出信号只用 entry_date
        code_col="code",
        date_col="date",
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        vol_col="volume",
    )

    if len(events) == 0:
        print("No events detected.")
        pd.DataFrame(columns=["code", "entry_date", "pattern"]).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("Saved empty file:", out_csv)
        return

    # ====== 只输出你要的三列，并按 entry_date 过滤 ======
    out = events[["code", "entry_date", "pattern"]].copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce")
    out = out.dropna(subset=["entry_date"]).copy()
    out = out[out["entry_date"] >= min_entry_date].copy()
    out = out.sort_values(["entry_date", "code"]).reset_index(drop=True)

    # ====== 追加：从 entry_date 开始，未来(最多)30个交易日的收盘价统计 ======
    # 口径：
    #   entry_close = entry_date 当天收盘价
    #   fut_* 统计窗口 = entry_date 之后的未来 1..horizon 天（不含 entry_date 当天）
    horizon = 30

    d = df.copy()
    d = d.sort_values(["code", "date"]).copy()

    # 未来窗口（不含当日）的最高/最低/平均收盘价
    # 注意：pandas 的 rolling 默认是“向后看”的窗口。我们要的是“向前看”的未来窗口：
    #   统计区间 = close[t+1 .. t+horizon]（如果未来不足 horizon 天，就用到最新日为止）
    # 实现方式：先 shift(-1) 把 t+1 对齐到 t，然后把序列反转做 rolling，再反转回来。
    fut_max_close = (
        d.groupby("code", sort=False)["close"]
        .transform(lambda s: s.shift(-1)[::-1].rolling(horizon, min_periods=1).max()[::-1])
    )
    fut_min_close = (
        d.groupby("code", sort=False)["close"]
        .transform(lambda s: s.shift(-1)[::-1].rolling(horizon, min_periods=1).min()[::-1])
    )
    fut_mean_close = (
        d.groupby("code", sort=False)["close"]
        .transform(lambda s: s.shift(-1)[::-1].rolling(horizon, min_periods=1).mean()[::-1])
    )

    d["__fut_max_close"] = fut_max_close
    d["__fut_min_close"] = fut_min_close
    d["__fut_mean_close"] = fut_mean_close

    # entry_date -> entry_close
    key_close = (
        d[["code", "date", "close"]]
        .drop_duplicates(["code", "date"])
        .rename(columns={"date": "entry_date", "close": "entry_close"})
    )

    # entry_date -> future stats (aligned at entry_date row)
    key_fut = (
        d[["code", "date", "__fut_max_close", "__fut_min_close", "__fut_mean_close"]]
        .drop_duplicates(["code", "date"])
        .rename(columns={
            "date": "entry_date",
            "__fut_max_close": "fut_max_close_30d",
            "__fut_min_close": "fut_min_close_30d",
            "__fut_mean_close": "fut_mean_close_30d",
        })
    )

    out = out.merge(key_close, on=["code", "entry_date"], how="left")
    out = out.merge(key_fut, on=["code", "entry_date"], how="left")

    # 涨幅/跌幅（以 entry_close 为基准）
    out["fwd_max_close_ret_30d"] = out["fut_max_close_30d"] / (out["entry_close"] + 1e-12) - 1.0
    out["fwd_min_close_ret_30d"] = out["fut_min_close_30d"] / (out["entry_close"] + 1e-12) - 1.0

    # 未来30日均价相对买入价的涨跌幅（>0 表示均价高于买入价）
    out["fwd_mean_close_ret_30d"] = out["fut_mean_close_30d"] / (out["entry_close"] + 1e-12) - 1.0

    # 输出列顺序（更易读）
    out = out[[
        "code", "entry_date", "pattern",
        "entry_close",
        "fut_max_close_30d", "fwd_max_close_ret_30d",
        "fut_min_close_30d", "fwd_min_close_ret_30d",
        "fut_mean_close_30d", "fwd_mean_close_ret_30d",
    ]].copy()

    # ====== 保存并打印 ======
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nExported signals: {len(out):,} rows")
    print("Saved to:", out_csv)
    if len(out) > 0:
        print("\nHead:")
        print(out.head(50).to_string(index=False))


if __name__ == "__main__":
    main()