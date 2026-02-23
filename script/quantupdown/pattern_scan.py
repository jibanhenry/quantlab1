
import numpy as np
import pandas as pd
from typing import Tuple


def _nanmean_safe(a: np.ndarray) -> float:
    """安全版 nanmean。

    当切片为空或全是 NaN 时，返回 np.nan，避免触发 RuntimeWarning: Mean of empty slice。
    """
    if a is None:
        return float("nan")
    a = np.asarray(a)
    if a.size == 0:
        return float("nan")
    if not np.isfinite(a).any():
        return float("nan")
    return float(np.nanmean(a))

# 可选：如果项目里有 io_utils.py（你之前训练 GRU 用过），优先用它做字段标准化
try:
    from io_utils import load_market_csv_multi  # type: ignore
except Exception:
    load_market_csv_multi = None

# ============================================================
# 形态扫描 + 未来N日最高涨幅统计（按股票逐日线数据）
#
# 需要的基础字段（默认列名，可在函数参数中改）：
#   - code: 股票代码
#   - date: 交易日期（可被 pd.to_datetime 转换）
#   - open/high/low/close: OHLC
#   - volume: 成交量
#
# 输出：
#   1) events：每次形态触发的 (code, date, pattern, ...) 以及未来 horizon 天内的最高涨幅
#   2) summary：按 pattern 聚合的次数、均值、分位数、命中率、最小/最大等统计
# ============================================================

def _normalize_columns_for_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将常见字段名映射为本脚本默认使用的字段名：
      code, date, open, high, low, close, volume

    说明：
      - 先把列名统一成小写、去空格
      - 再按常见别名做重命名（例如 ts_code/symbol/ticker -> code；trade_date/datetime/time -> date）
      - 如果最终仍缺少 code 或 date，会抛出更友好的报错，提示你 CSV 的真实列名
    """
    d = df.copy()

    # 1) 统一列名格式
    d.columns = [str(c).strip() for c in d.columns]
    lower_map = {c: str(c).strip().lower() for c in d.columns}
    d = d.rename(columns=lower_map)

    # 2) 常见别名映射
    rename = {}

    # code
    for c in ["code", "ts_code", "symbol", "ticker", "secid", "security_id", "stock", "stock_code"]:
        if c in d.columns:
            rename[c] = "code"
            break

    # date
    for c in ["date", "trade_date", "tradedate", "datetime", "time", "dt", "timestamp"]:
        if c in d.columns:
            rename[c] = "date"
            break

    # OHLCV
    for c in ["open", "o"]:
        if c in d.columns:
            rename[c] = "open"
            break
    for c in ["high", "h"]:
        if c in d.columns:
            rename[c] = "high"
            break
    for c in ["low", "l"]:
        if c in d.columns:
            rename[c] = "low"
            break
    for c in ["close", "c", "adj_close", "adjclose"]:
        if c in d.columns:
            rename[c] = "close"
            break
    for c in ["volume", "vol", "qty", "amount_vol"]:
        if c in d.columns:
            rename[c] = "volume"
            break

    d = d.rename(columns=rename)

    # 3) 兜底检查
    missing = [c for c in ["code", "date"] if c not in d.columns]
    if missing:
        raise ValueError(
            "CSV 缺少关键列: "
            + ",".join(missing)
            + f"。当前列名为: {list(d.columns)[:80]}。"
            + "请检查 CSV 字段名，或在命令行用 --code_col/--date_col 指定实际列名。"
        )

    return d

# ---------------------------
# Utility: forward max return
# ---------------------------
def add_fwd_maxret(df: pd.DataFrame, horizon: int = 30,
                   code_col="code", date_col="date",
                   close_col="close") -> pd.DataFrame:
    """基于“未来最高收盘价”的收益统计。

    计算：
      fwd_max_close_ret_{horizon} = max(close[t+1..t+horizon]) / close[t] - 1

    说明：
      - 使用 close 而不是 high，避免日内尖刺导致的过度乐观。
      - 仍然从“当前行 t 的收盘价”作为基准。
    """
    d = df.sort_values([code_col, date_col]).copy()

    fut_max_close = (
        d.groupby(code_col, sort=False)[close_col]
        .transform(lambda s: s.shift(-1).rolling(horizon, min_periods=1).max())
    )

    d[f"fwd_max_close_ret_{horizon}"] = fut_max_close / (d[close_col] + 1e-12) - 1.0
    return d


def add_fwd_maxret_from_entry(
    df: pd.DataFrame,
    entry_date_col: str,
    horizon: int = 30,
    code_col: str = "code",
    date_col: str = "date",
    close_col: str = "close",
) -> pd.DataFrame:
    """从“入场确认日 entry_date”开始计算未来 N 日最高收盘涨幅。

    计算口径：
      entry_close = close[entry_date]
      fwd_max_close_ret_{horizon} = max(close[entry_date+1 .. entry_date+horizon]) / entry_close - 1

    注意：
      - 这里默认你是在 entry_date 当天收盘后确认形态，并在当日收盘价附近入场（更保守、也更可复现）。
      - 若你希望用 entry_date+1 的开盘价入场，也可以后续再改。
    """
    d = df.sort_values([code_col, date_col]).copy()

    # 预先计算每一行“从该行开始未来 horizon 天”的最高收盘价（不含当日）
    fut_max_close = (
        d.groupby(code_col, sort=False)[close_col]
        .transform(lambda s: s.shift(-1).rolling(horizon, min_periods=1).max())
    )

    # 把 entry_date 映射到 df 的行上（按 code + date）
    key = d[[code_col, date_col, close_col]].drop_duplicates([code_col, date_col]).copy()
    key = key.rename(columns={date_col: entry_date_col, close_col: "entry_close"})

    out = d.merge(key[[code_col, entry_date_col, "entry_close"]], on=[code_col, entry_date_col], how="left")

    out[f"fwd_max_close_ret_{horizon}"] = fut_max_close / (out["entry_close"] + 1e-12) - 1.0
    return out

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _sma(s: pd.Series, win: int) -> pd.Series:
    return s.rolling(win, min_periods=win).mean()

def _true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def add_basic_indicators(df: pd.DataFrame,
                         code_col="code", date_col="date",
                         close_col="close", high_col="high", low_col="low", vol_col="volume") -> pd.DataFrame:
    d = df.sort_values([code_col, date_col]).copy()
    g = d.groupby(code_col, sort=False)

    d["ret1"] = g[close_col].transform(lambda s: s.pct_change())
    d["vol_ma5"] = g[vol_col].transform(lambda s: _sma(s, 5))
    d["vol_ma20"] = g[vol_col].transform(lambda s: _sma(s, 20))
    d["ma20"] = g[close_col].transform(lambda s: _sma(s, 20))
    d["ma30"] = g[close_col].transform(lambda s: _sma(s, 30))
    d["hh20"] = g[close_col].transform(lambda s: s.rolling(20, min_periods=20).max())
    d["ll20"] = g[close_col].transform(lambda s: s.rolling(20, min_periods=20).min())

    # ATR(14) for volatility / tightness checks
    tr = g.apply(lambda x: _true_range(x[high_col], x[low_col], x[close_col])).reset_index(level=0, drop=True)
    d["atr14"] = g.apply(lambda x: tr.loc[x.index].rolling(14, min_periods=14).mean()).reset_index(level=0, drop=True)

    # candle body / range
    d["range"] = (d[high_col] - d[low_col]).replace(0, np.nan)
    d["body"] = (d[close_col] - d["open"]).abs() if "open" in d.columns else np.nan
    # If open col exists, use it; else approximate body with abs(ret1)*close
    if "open" not in d.columns:
        d["body"] = (d["ret1"].abs() * d[close_col]).fillna(np.nan)

    d["body_ratio"] = (d["body"] / (d["range"] + 1e-12)).clip(0, 10)
    return d

# -----------------------------------
# Pattern detectors (rule-based)
# -----------------------------------

def detect_high_tight_flag(
    df: pd.DataFrame,
    code_col="code", date_col="date",
    open_col="open", high_col="high", low_col="low", close_col="close", vol_col="volume",
    # -------- Day0 / Day1 关键形态参数 --------
    day0_ret=0.055,             # Day0 涨幅阈值（默认 5.5%）
    day0_vol_mult=1.5,          # Day0 放量倍数：volume >= vol_ma20 * day0_vol_mult
    day1_range_pct=0.02,        # Day1 振幅阈值： (high-low)/close <= 2%
    day1_body_ratio=0.25,       # Day1 十字星/小实体：body/range <= 0.25
    # -------- 前置“平稳基座”参数 --------
    base_days=20,               # 形态前置基座长度（交易日）
    base_range_pct=0.035,       # 基座期平均振幅占比阈值：mean((high-low)/close) <= 3.5%
    base_vol_cv=0.6,            # 基座期量能稳定性：std(volume)/mean(volume) <= base_vol_cv
    # -------- 旗形整理参数 --------
    flag_days=(3, 10),          # 旗形整理天数范围（含端点）
    band_buffer=0.01,           # 旗形“区间容忍”：允许突破 Day0/Day1 区间上下沿的 1%
    vol_contract=0.8,           # 旗形缩量：旗形期均量 <= 20日均量 * vol_contract
    vol_contract_vs_day0=0.7,   # 旗形缩量（相对 Day0）：旗形期均量 <= Day0 成交量 * 0.7
    # -------- 入场确认参数（更贴近实盘） --------
    breakout_confirm_days=10,   # 旗形结束后，等待向上突破的最大天数
    breakout_buffer=0.01,       # 向上突破阈值：收盘 >= 区间上沿*(1+breakout_buffer)
    breakout_vol_mult=1.2,      # 突破确认日放量：volume >= vol_ma20 * breakout_vol_mult
    hold_buffer=0.003,          # 次日站稳容忍：次日收盘 >= 上沿*(1-hold_buffer)
) -> pd.DataFrame:
    """
    高位紧旗形（High Tight Flag, HTF）检测（更贴近“实盘经验定义”的版本）。

    你给的经验定义：
      - 前置一段时间价格与成交量都相对平稳（基座期）
      - Day0 突然放量上涨（约 5%~6%）
      - Day1 收十字星/小实体，且振幅很小（例如 2%）
      - 随后 3~10 天在 Day0/Day1 的区间内震荡整理，并且量能明显收缩
      - 最后等待“向上突破整理区间上沿”并站稳确认后才入场

    逻辑（按单只股票）：
      0) 基座期（base_days）：平均振幅占比不大，且量能波动不大（用 CV 近似）
      1) Day0：ret >= day0_ret 且放量（相对 20 日均量）
      2) Day1：振幅 <= day1_range_pct 且实体占比 <= day1_body_ratio（十字星/小实体代理）
      3) 旗形整理期 N 天（flag_days）：价格基本在 Day0/Day1 区间内（允许 band_buffer 容忍），同时缩量
      4) 入场：旗形结束后，在 breakout_confirm_days 天内收盘向上突破区间上沿（+breakout_buffer），
              且突破日放量，并在次日仍站稳（hold_buffer）后入场（次日收盘）。

    返回：
      - 触发日 date_col：Day0（用于记录形态起点）
      - entry_date：突破确认并站稳后的入场日（更符合“确认后才买”）
    """
    # 预先计算基础字段（避免重复 groupby 开销）
    d = df.copy()
    d = d.sort_values([code_col, date_col]).copy()
    g = d.groupby(code_col, sort=False)

    # 日收益（按股票）
    ret0 = g[close_col].transform(lambda s: s / (s.shift(1) + 1e-12) - 1.0)

    events = []

    for code, x in d.groupby(code_col, sort=False):
        x = x.sort_values(date_col)
        n = len(x)
        if n < (base_days + 2 + flag_days[0] + 2):
            continue

        dates = x[date_col].to_numpy()
        close_arr = x[close_col].to_numpy(dtype=float)
        open_arr = x[open_col].to_numpy(dtype=float) if open_col in x.columns else np.full(n, np.nan)
        high_arr = x[high_col].to_numpy(dtype=float)
        low_arr = x[low_col].to_numpy(dtype=float)
        vol_arr = x[vol_col].to_numpy(dtype=float)
        vol20_arr = x["vol_ma20"].to_numpy(dtype=float)

        # 关键特征：振幅占比、实体占比
        rng = (high_arr - low_arr)
        rng_pct = rng / (close_arr + 1e-12)
        body = np.abs(close_arr - open_arr)
        # 若 open 缺失，用 ret 近似实体
        if not np.isfinite(open_arr).any():
            # 用前一日 close 近似 open
            prev_close = np.concatenate([[np.nan], close_arr[:-1]])
            body = np.abs(close_arr - prev_close)
        body_ratio = body / (rng + 1e-12)

        # 基座期量能稳定性（CV = std/mean），逐点滚动
        # 注意：只用于过滤，不需要非常精确
        vol_mean = pd.Series(vol_arr).rolling(base_days, min_periods=base_days).mean().to_numpy()
        vol_std = pd.Series(vol_arr).rolling(base_days, min_periods=base_days).std(ddof=0).to_numpy()
        vol_cv = vol_std / (vol_mean + 1e-12)

        # 基座期平均振幅占比
        base_rng_mean = pd.Series(rng_pct).rolling(base_days, min_periods=base_days).mean().to_numpy()

        # Day0 条件：涨幅 + 放量
        day0_big_up = ret0.loc[x.index].to_numpy(dtype=float) >= day0_ret
        day0_vol_ok = vol_arr >= (vol20_arr * day0_vol_mult)

        # Day1 条件：小振幅 + 小实体（十字星代理）
        day1_ok = (rng_pct <= day1_range_pct) & (body_ratio <= day1_body_ratio)

        # 基座期条件：平稳
        base_ok = (base_rng_mean <= base_range_pct) & (vol_cv <= base_vol_cv)

        # 扫描 Day0
        for i in np.where(day0_big_up & day0_vol_ok)[0]:
            # 需要 Day1 存在
            if i + 1 >= n:
                continue

            # 基座期必须在 Day0 前结束（即 i 对应的 base_ok 代表 [i-base_days .. i-1]）
            if i - 1 < 0:
                continue
            if not (i < len(base_ok) and bool(base_ok[i])):
                continue

            # Day1 必须满足十字星/小实体
            if not bool(day1_ok[i + 1]):
                continue

            # 定义整理区间：用 Day0/Day1 的高低点
            band_high = float(np.nanmax([high_arr[i], high_arr[i + 1]]))
            band_low = float(np.nanmin([low_arr[i], low_arr[i + 1]]))
            if not (np.isfinite(band_high) and np.isfinite(band_low) and band_high > band_low):
                continue

            # 旗形整理：在区间内震荡 + 缩量
            for N in range(flag_days[0], flag_days[1] + 1):
                end_flag = i + 1 + N  # Day1 后 N 天整理，最后一天索引
                if end_flag >= n:
                    continue

                win = slice(i + 2, end_flag + 1)  # 整理窗口：Day2 .. Day(1+N)
                if (end_flag - (i + 2) + 1) <= 0:
                    continue

                # 价格必须大体在区间内（允许上下 band_buffer 容忍）
                hi = np.nanmax(high_arr[win])
                lo = np.nanmin(low_arr[win])
                if not (np.isfinite(hi) and np.isfinite(lo)):
                    continue
                if hi > band_high * (1.0 + band_buffer):
                    continue
                if lo < band_low * (1.0 - band_buffer):
                    continue

                # 缩量：整理期均量 <= 20日均量 * vol_contract，且 <= Day0 成交量 * vol_contract_vs_day0
                avg_vol = _nanmean_safe(vol_arr[win])
                avg_vol20 = _nanmean_safe(vol20_arr[win])
                if not (np.isfinite(avg_vol) and np.isfinite(avg_vol20) and np.isfinite(vol_arr[i])):
                    continue
                if not (avg_vol <= avg_vol20 * vol_contract and avg_vol <= vol_arr[i] * vol_contract_vs_day0):
                    continue

                # === 入场：等待向上突破整理区间上沿并站稳 ===
                entry_i = None
                j_start = end_flag + 1
                j_end = min(n - 2, end_flag + breakout_confirm_days)  # 需要 j+1
                for j in range(j_start, j_end + 1):
                    # 1) 收盘突破上沿
                    if close_arr[j] < band_high * (1.0 + breakout_buffer):
                        continue
                    # 2) 突破日放量
                    vj = vol_arr[j]
                    vmj = vol20_arr[j]
                    if not (np.isfinite(vj) and np.isfinite(vmj) and vmj > 0):
                        continue
                    if vj < vmj * breakout_vol_mult:
                        continue
                    # 3) 次日站稳
                    if close_arr[j + 1] < band_high * (1.0 - hold_buffer):
                        continue
                    entry_i = j + 1
                    break

                if entry_i is None:
                    continue

                events.append((code, dates[i], dates[entry_i], "HIGH_TIGHT_FLAG", i, entry_i, N))
                break  # take first N that matches

    out = pd.DataFrame(
        events,
        columns=[code_col, date_col, "entry_date", "pattern", "bar_index", "entry_index", "flag_N"],
    )
    return out


# === MA30 pullback detector ===
def detect_pullback_to_ma30(
    df: pd.DataFrame,
    code_col="code", date_col="date",
    close_col="close", low_col="low", vol_col="volume",
    touch_buffer=0.005,        # 回踩触碰容忍：low <= MA30*(1+touch_buffer)
    hold_buffer=0.003,         # 次日站稳容忍：次日收盘 >= MA30*(1-hold_buffer)
    vol_contract=0.8,          # 缩量回调：回踩日 volume <= vol_ma20 * vol_contract
) -> pd.DataFrame:
    """缩量回调到 MA30（趋势中的再加速）检测。

    逻辑（按单只股票）：
      1) 前提：MA30 已经可用，且回踩当日收盘在 MA30 上方（趋势背景）
      2) 回踩日 Day0：最低价触碰/略破 MA30（touch_buffer 容忍），同时缩量（相对 20 日均量）
      3) 确认入场 Day1：次日收盘仍站在 MA30 附近之上（hold_buffer 容忍）

    返回：
      - date：Day0（回踩当日）
      - entry_date：Day1（确认后买入）
    """
    d = df.sort_values([code_col, date_col]).copy()
    if "ma30" not in d.columns or "vol_ma20" not in d.columns:
        raise ValueError("detect_pullback_to_ma30 需要先调用 add_basic_indicators 计算 ma30 和 vol_ma20")

    events = []
    for code, x in d.groupby(code_col, sort=False):
        x = x.sort_values(date_col)
        n = len(x)
        if n < 35:
            continue

        dates = x[date_col].to_numpy()
        close_arr = x[close_col].to_numpy(dtype=float)
        low_arr = x[low_col].to_numpy(dtype=float)
        ma30 = x["ma30"].to_numpy(dtype=float)
        vol = x[vol_col].to_numpy(dtype=float)
        vol20 = x["vol_ma20"].to_numpy(dtype=float)

        # Day0：趋势背景 + 触碰MA30 + 缩量
        day0 = (
            np.isfinite(ma30) &
            (close_arr > ma30) &
            (low_arr <= ma30 * (1.0 + touch_buffer)) &
            (vol <= vol20 * vol_contract)
        )

        for i in np.where(day0)[0]:
            if i + 1 >= n:
                continue
            # Day1：站稳确认
            if close_arr[i + 1] < ma30[i] * (1.0 - hold_buffer):
                continue
            events.append((code, dates[i], dates[i + 1], "PULLBACK_TO_MA30", i, i + 1))

    return pd.DataFrame(
        events,
        columns=[code_col, date_col, "entry_date", "pattern", "bar_index", "entry_index"],
    )




def detect_false_breakdown(
    df: pd.DataFrame,
    code_col="code", date_col="date",
    close_col="close", low_col="low", vol_col="volume",
    lookback=20,                 # 前低/平台参考窗口（交易日）
    vol_silent=0.8,              # 跌破日“无量”条件：volume <= vol_ma20 * vol_silent
    reclaim_days=3,              # 多少天内快速拉回：reclaim_days 天内收盘回到前低之上
) -> pd.DataFrame:
    """
    假跌破（洗盘/赶人下车）检测（False Breakdown / Shakeout）。

    逻辑（按单只股票）：
      1) 跌破日 Day0：最低价跌破过去 lookback 天的“前低/平台低点”（不含当日），且当天无量（vol_silent）
      2) 快速收回：在接下来 reclaim_days 天内，收盘价重新站回“前低”之上

    触发日返回：Day0（跌破当日）
    """
    d = df.copy()
    g = d.groupby(code_col, sort=False)

    prior_low = g[low_col].transform(lambda s: s.shift(1).rolling(lookback, min_periods=lookback).min())
    breakdown = d[low_col] < prior_low
    vol_ok = d[vol_col] <= d["vol_ma20"] * vol_silent

    day0_mask = breakdown & vol_ok

    events = []
    for code, x in d.sort_values([code_col, date_col]).groupby(code_col, sort=False):
        idx = x.index.to_numpy()
        day0 = day0_mask.loc[idx].to_numpy()
        close_arr = x[close_col].to_numpy()
        prior_low_arr = prior_low.loc[idx].to_numpy()
        dates = x[date_col].to_numpy()
        n = len(x)

        for i in np.where(day0)[0]:
            pl = prior_low_arr[i]
            if not np.isfinite(pl):
                continue
            # reclaim within next reclaim_days
            j_end = min(n - 1, i + reclaim_days)
            reclaim_slice = close_arr[i+1:j_end+1] > pl
            if np.any(reclaim_slice):
                # 形态确认：第一次收盘站回前低之上的那一天
                first_j = int(np.argmax(reclaim_slice))  # 在 slice 内的位置
                entry_i = i + 1 + first_j
                events.append((code, dates[i], dates[entry_i], "FALSE_BREAKDOWN", i, entry_i))

    out = pd.DataFrame(events, columns=[code_col, date_col, "entry_date", "pattern", "bar_index", "entry_index"])
    return out




def detect_rising_three_like(
    df: pd.DataFrame,
    code_col="code", date_col="date",
    close_col="close", open_col="open", vol_col="volume",
    n_days=4,                  # 连续观察窗口长度（交易日）
    small_body_pct=0.012,      # “小阳线”涨幅上限：0 < ret1 < small_body_pct（例如 0.012 表示 +1.2%）
    min_pos_days=3,            # 窗口内至少有多少天满足“小阳线”为正
    vol_slope=1.0,             # 量能温和抬升：窗口末端 vol_ma5 >= 窗口起点 vol_ma5 * vol_slope
) -> pd.DataFrame:
    """
    连续小阳线 + 量能温和放大（趋势延续的代理形态）检测。

    逻辑（按单只股票）：
      - 在过去 n_days 的窗口内，至少 min_pos_days 天为“小阳线”（0 < 日收益 < small_body_pct）
      - 同期量能均线（vol_ma5）较窗口起点不走弱（vol_slope 控制）

    触发日返回：窗口最后一天（即满足条件的当日）
    """
    d = df.copy()
    d = d.sort_values([code_col, date_col])
    g = d.groupby(code_col, sort=False)

    ret1 = g[close_col].transform(lambda s: s.pct_change())
    pos = (ret1 > 0) & (ret1 < small_body_pct)

    # rolling count of "small positive" days
    pos_cnt = g.apply(lambda x: pos.loc[x.index].rolling(n_days, min_periods=n_days).sum()).reset_index(level=0, drop=True)

    vol5 = d["vol_ma5"]
    vol5_ratio = g[vol_col].transform(lambda s: _sma(s, 5))  # same as vol_ma5 but safe
    vol5_start = g.apply(lambda x: vol5_ratio.loc[x.index].shift(n_days-1)).reset_index(level=0, drop=True)

    mask = (pos_cnt >= min_pos_days) & np.isfinite(vol5_ratio) & np.isfinite(vol5_start) & (vol5_ratio >= vol5_start * vol_slope)

    out = d.loc[mask, [code_col, date_col]].copy()
    # 实盘：窗口最后一天收盘确认后，下一交易日入场
    out["entry_date"] = d.groupby(code_col, sort=False)[date_col].shift(-1).loc[mask].values
    out["pattern"] = "SMALL_UP_DAYS_GENTLE_VOL"
    return out

# -----------------------------------
# Run all detectors + summarize
# -----------------------------------
def scan_patterns_and_summarize(
    df: pd.DataFrame,
    horizon: int = 30,
    code_col="code", date_col="date",
    open_col="open", high_col="high", low_col="low", close_col="close", vol_col="volume",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      events: each event row with fwd_max_ret_{horizon}
      summary: per pattern stats (count, mean/median, win rates at thresholds)
    """
    d = df.copy()
    # 1) 统一日期格式并按 (code, date) 排序
    d[date_col] = pd.to_datetime(d[date_col])
    # 2) 计算必要的基础指标（均线、前高前低、ATR、量能均线等）
    d = add_basic_indicators(d, code_col, date_col, close_col, high_col, low_col, vol_col)
    # 3) 计算未来 horizon 天内的“最高收盘涨幅”（用未来最高 close / 当日 close - 1）
    d = add_fwd_maxret(d, horizon, code_col, date_col, close_col)

    # 4) 扫描各类形态，得到触发事件列表（events）
    # 当前保留的形态: HIGH_TIGHT_FLAG, PULLBACK_TO_MA30, FALSE_BREAKDOWN, SMALL_UP_DAYS_GENTLE_VOL
    ev1 = detect_high_tight_flag(d, code_col, date_col, open_col, high_col, low_col, close_col, vol_col)
    # 保留：PULLBACK_TO_MA30
    ev3 = detect_pullback_to_ma30(d, code_col, date_col, close_col, low_col, vol_col)
    ev4 = detect_false_breakdown(d, code_col, date_col, close_col, low_col, vol_col)
    ev6 = detect_rising_three_like(d, code_col, date_col, close_col, open_col, vol_col)

    # Only include nonempty event DataFrames
    event_list = [ev1, ev3, ev4, ev6]
    events = pd.concat([ev for ev in event_list if ev is not None and len(ev) > 0], ignore_index=True)

    # 有些形态的 entry_date 需要下一交易日；若数据尾部不足，会是 NaT，这里先过滤
    events["entry_date"] = pd.to_datetime(events["entry_date"], errors="coerce")
    events = events.dropna(subset=["entry_date"]).copy()

    # 5) 从 entry_date 开始计算未来 horizon 天的最高收盘涨幅，并拼接回 events
    #    注意：这里不能把 d 直接传给 add_fwd_maxret_from_entry，因为 d 本身没有 entry_date 列。
    #    正确做法是：先在 d 的每一行（按 code/date）预计算“从该行开始未来 horizon 天的最高收盘价”，
    #    再把 events.entry_date 对齐到 d 的同一天行，得到 entry_close 与 fut_max_close。
    # 对 d 的每一行：从下一天开始，未来 horizon 天内的最高/最低收盘价
    d_sorted = d.sort_values([code_col, date_col]).copy()
    fut_max_close = (
        d_sorted.groupby(code_col, sort=False)[close_col]
        .transform(lambda s: s.shift(-1).rolling(horizon, min_periods=1).max())
    )
    fut_min_close = (
        d_sorted.groupby(code_col, sort=False)[close_col]
        .transform(lambda s: s.shift(-1).rolling(horizon, min_periods=1).min())
    )
    d_sorted["__fut_max_close"] = fut_max_close
    d_sorted["__fut_min_close"] = fut_min_close

    # entry_date -> entry_close
    key_close = (
        d_sorted[[code_col, date_col, close_col]]
        .drop_duplicates([code_col, date_col])
        .rename(columns={date_col: "entry_date", close_col: "entry_close"})
    )

    # entry_date -> fut_max_close (aligned at entry_date row)
    key_fut = (
        d_sorted[[code_col, date_col, "__fut_max_close"]]
        .drop_duplicates([code_col, date_col])
        .rename(columns={date_col: "entry_date", "__fut_max_close": "fut_max_close"})
    )
    key_fut_min = (
        d_sorted[[code_col, date_col, "__fut_min_close"]]
        .drop_duplicates([code_col, date_col])
        .rename(columns={date_col: "entry_date", "__fut_min_close": "fut_min_close"})
    )

    events = events.merge(key_close, on=[code_col, "entry_date"], how="left")
    events = events.merge(key_fut, on=[code_col, "entry_date"], how="left")
    events = events.merge(key_fut_min, on=[code_col, "entry_date"], how="left")

    # 上行：未来 horizon 天内的最高收盘涨幅
    events[f"fwd_max_close_ret_{horizon}"] = events["fut_max_close"] / (events["entry_close"] + 1e-12) - 1.0
    # 下行：未来 horizon 天内的最低收盘跌幅（通常为负）
    events[f"fwd_min_close_ret_{horizon}"] = events["fut_min_close"] / (events["entry_close"] + 1e-12) - 1.0

    # 兼容旧变量名 fcol（用于后面统计 hit_+X% 的上行口径）
    fcol = f"fwd_max_close_ret_{horizon}"

    # 6) 汇总统计：每个形态出现次数、未来最高/最低收盘涨跌幅的分布与命中率
    # 注：fwd_max/min_ret_{horizon} 在每个 symbol 尾部（未来天数不足时）可能为 NaN，这里用 dropna() 自动忽略
    def _summ(g):
        up = g[f"fwd_max_close_ret_{horizon}"].dropna().to_numpy()
        dn = g[f"fwd_min_close_ret_{horizon}"].dropna().to_numpy()
        # 两个都需要有值才统计（否则说明未来窗口不足或 entry 对齐失败）
        m = min(len(up), len(dn))
        if m == 0:
            return pd.Series({"count": 0})
        up = up[:m]
        dn = dn[:m]

        return pd.Series({
            "count": int(m),
            # 上行（最高收盘涨幅）
            "mean_up": float(np.mean(up)),
            "median_up": float(np.median(up)),
            "p75_up": float(np.quantile(up, 0.75)),
            "p90_up": float(np.quantile(up, 0.90)),
            "min_up": float(np.min(up)),
            "max_up": float(np.max(up)),
            # 下行（最低收盘跌幅，通常为负）
            "mean_dn": float(np.mean(dn)),
            "median_dn": float(np.median(dn)),
            "p25_dn": float(np.quantile(dn, 0.25)),
            "p10_dn": float(np.quantile(dn, 0.10)),
            "min_dn": float(np.min(dn)),
            "max_dn": float(np.max(dn)),
            # 仍保留一些“上行命中率”指标，便于横向比较
            "hit_up_+5%": float(np.mean(up >= 0.05)),
            "hit_up_+10%": float(np.mean(up >= 0.10)),
            "hit_up_+20%": float(np.mean(up >= 0.20)),
        })

    summary = events.groupby("pattern").apply(_summ).sort_values("count", ascending=False).reset_index()
    return events, summary

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Scan K-line patterns and summarize forward max return within horizon.")
    ap.add_argument(
        "--csv1",
        default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2020-2025_all.csv",
        help="First CSV path (e.g., 2020-2025_all.csv)"
    )
    ap.add_argument(
        "--csv2",
        default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv",
        help="Second CSV path (e.g., 2025_06_daily.csv)"
    )
    ap.add_argument("--horizon", type=int, default=30, help="Forward horizon (trading days) for max-high return")
    ap.add_argument("--outdir", default="./pattern_output", help="Output folder for events and summary CSV")
    ap.add_argument("--code_col", default="code", help="股票代码列名（默认 code；如果不同可改）")
    ap.add_argument("--date_col", default="date", help="日期列名（默认 date；如果不同可改）")
    ap.add_argument("--open_col", default="open", help="开盘价列名（默认 open）")
    ap.add_argument("--high_col", default="high", help="最高价列名（默认 high）")
    ap.add_argument("--low_col", default="low", help="最低价列名（默认 low）")
    ap.add_argument("--close_col", default="close", help="收盘价列名（默认 close）")
    ap.add_argument("--vol_col", default="volume", help="成交量列名（默认 volume）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 优先走项目内的 IO 标准化（与你训练 GRU 的数据字段一致）
    if load_market_csv_multi is not None:
        df = load_market_csv_multi([args.csv1, args.csv2])
    else:
        df1 = _normalize_columns_for_patterns(pd.read_csv(args.csv1))
        df2 = _normalize_columns_for_patterns(pd.read_csv(args.csv2))
        df = pd.concat([df1, df2], ignore_index=True)

    print("Loaded columns (head):", list(df.columns)[:40])

    events, summary = scan_patterns_and_summarize(
        df,
        horizon=args.horizon,
        code_col=args.code_col,
        date_col=args.date_col,
        open_col=args.open_col,
        high_col=args.high_col,
        low_col=args.low_col,
        close_col=args.close_col,
        vol_col=args.vol_col,
    )

    up_col = f"fwd_max_close_ret_{args.horizon}"
    dn_col = f"fwd_min_close_ret_{args.horizon}"

    # 额外补充（以 events 原始列为准）
    if len(events) > 0 and (up_col in events.columns) and (dn_col in events.columns):
        extra = (
            events.groupby("pattern")[[up_col, dn_col]]
            .agg(min_up=(up_col, "min"), max_up=(up_col, "max"),
                 min_dn=(dn_col, "min"), max_dn=(dn_col, "max"))
            .reset_index()
        )
        summary = summary.merge(extra, on="pattern", how="left")
    else:
        for c in ["min_up", "max_up", "min_dn", "max_dn"]:
            summary[c] = np.nan

    # Pretty print (count/mean/min/max)
    cols_to_show = [
        "pattern", "count",
        "mean_up", "median_up", "p75_up", "p90_up", "min_up", "max_up",
        "mean_dn", "median_dn", "p25_dn", "p10_dn", "min_dn", "max_dn",
        "hit_up_+5%", "hit_up_+10%", "hit_up_+20%",
    ]
    cols_to_show = [c for c in cols_to_show if c in summary.columns]
    print("\n=== Pattern Summary ===")
    print(summary[cols_to_show].sort_values("count", ascending=False).to_string(index=False))

    # Save outputs
    events_path = os.path.join(args.outdir, f"pattern_events_h{args.horizon}.csv")
    summary_path = os.path.join(args.outdir, f"pattern_summary_h{args.horizon}.csv")
    events.to_csv(events_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved events:  {events_path}")
    print(f"Saved summary: {summary_path}")