# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from typing import Tuple

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(n, min_periods=n).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_sma = pd.Series(gain, index=series.index).rolling(n, min_periods=n).mean()
    loss_sma = pd.Series(loss, index=series.index).rolling(n, min_periods=n).mean()
    rs = gain_sma / (loss_sma.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.clip(0, 100)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    dif = ema_fast - ema_slow
    dea = ema(dif, signal)
    hist = dif - dea
    return dif, dea, hist

def bollinger(series: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = sma(series, n)
    std = series.rolling(n, min_periods=n).std()
    up = mid + k * std
    low = mid - k * std
    bw = (up - low) / mid
    return mid, up, low, bw

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (volume * direction).fillna(0).cumsum()

def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    ma = tp.rolling(n, min_periods=n).mean()
    md = (tp - ma).abs().rolling(n, min_periods=n).mean()
    return (tp - ma) / (0.015 * md)

def roc(series: pd.Series, r: int = 10) -> pd.Series:
    return series.pct_change(r)

def kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
    low_min = df['low'].rolling(n, min_periods=n).min()
    high_max = df['high'].rolling(n, min_periods=n).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    K = rsv.ewm(alpha=1/m1, adjust=False, min_periods=n).mean()
    D = K.ewm(alpha=1/m2, adjust=False, min_periods=n).mean()
    J = 3 * K - 2 * D
    return K, D, J

def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high_max = df['high'].rolling(n, min_periods=n).max()
    low_min = df['low'].rolling(n, min_periods=n).min()
    return (high_max - df['close']) / (high_max - low_min) * 100

def dmi_adx(df: pd.DataFrame, n: int = 14):
    high = df['high']; low = df['low']
    prev_high = high.shift(1); prev_low = low.shift(1)

    plus_dm = (high - prev_high).clip(lower=0.0)
    minus_dm = (prev_low - low).clip(lower=0.0)
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0.0)
    minus_dm = np.where(minus_dm > (high - prev_high).clip(lower=0.0), minus_dm, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = true_range(df)
    atr_n = tr.rolling(n, min_periods=n).sum()
    plus_di = 100 * plus_dm.rolling(n, min_periods=n).sum() / atr_n
    minus_di = 100 * minus_dm.rolling(n, min_periods=n).sum() / atr_n
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx = dx.rolling(n, min_periods=n).mean()
    return plus_di, minus_di, adx

def psar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    high = df['high'].values; low = df['low'].values
    length = len(df); psar = np.zeros(length)
    bull = True; af = step; ep = low[0]; psar[0] = low[0]
    for i in range(1, length):
        prior_psar = psar[i-1]
        if bull:
            psar[i] = min(prior_psar + af * (ep - prior_psar), low[i-1])
            if low[i] < psar[i]:
                bull = False; psar[i] = ep; af = step; ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]; af = min(af + step, max_step)
        else:
            psar[i] = max(prior_psar + af * (ep - prior_psar), high[i-1])
            if high[i] > psar[i]:
                bull = True; psar[i] = ep; af = step; ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]; af = min(af + step, max_step)
    return pd.Series(psar, index=df.index)

# ===========================
# V2 features for 2nd system
# ===========================

_EPS = 1e-12


def rolling_max_drawdown(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling max drawdown (negative number, closer to 0 is better).
    Compute drawdown vs rolling peak, then take rolling minimum of drawdown.

    DD_t = close_t / rolling_max(close, window) - 1
    MDD_window = rolling_min(DD, window)
    """
    peak = series.rolling(window, min_periods=window).max()
    dd = series / (peak.replace(0, np.nan)) - 1.0
    mdd = dd.rolling(window, min_periods=window).min()
    return mdd


def gap_atr(df: pd.DataFrame, atr_n: int = 14) -> pd.Series:
    """
    ATR-normalized overnight gap risk:
    |open_t - close_{t-1}| / ATR_n(t)
    """
    prev_close = df["close"].shift(1)
    atr_val = atr(df, atr_n)
    return (df["open"] - prev_close).abs() / (atr_val + _EPS)


def tr_pct(df: pd.DataFrame) -> pd.Series:
    """
    True Range normalized by prev close:
    TR% = TR / close_{t-1}
    """
    tr = true_range(df)
    prev_close = df["close"].shift(1)
    return tr / (prev_close.abs() + _EPS)


def rolling_percentile_rank(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling percentile rank of the last value within the window.
    Output range: [0, 1]. Larger means more extreme/high within recent history.

    Note: uses rolling.apply -> slower but clear and stable.
    """
    def _pct_rank(x: np.ndarray) -> float:
        last = x[-1]
        # percent of values <= last
        return float(np.mean(x <= last))

    return series.rolling(window, min_periods=window).apply(_pct_rank, raw=True)


def tr_pctile(df: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Percentile rank of TR% over rolling window.
    Useful to detect abnormal volatility days.
    """
    return rolling_percentile_rank(tr_pct(df), window=window)


def close_location_value(df: pd.DataFrame) -> pd.Series:
    """
    CLV in [-1, 1]:
    ((close-low) - (high-close)) / (high-low)
    close near high -> 1, close near low -> -1
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    rng = (high - low)
    clv = ((close - low) - (high - close)) / (rng.replace(0, np.nan))
    return clv.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)


def wick_ratios(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Candlestick structure ratios:
    upper_wick_ratio, lower_wick_ratio, body_ratio (all divided by daily range)

    upper = high - max(open, close)
    lower = min(open, close) - low
    body  = abs(close - open)
    ratio = part / (high - low)
    """
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    rng = (h - l).replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    body = (c - o).abs()

    upper_r = (upper / rng).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    lower_r = (lower / rng).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    body_r = (body / rng).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

    return upper_r, lower_r, body_r


def dist_to_high(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Distance to rolling high:
    close / rolling_max(close) - 1
    Close near rolling high -> ~0 (hot), more negative -> pulled back.
    """
    roll_high = series.rolling(window, min_periods=window).max()
    return series / (roll_high.replace(0, np.nan)) - 1.0


def breakout_strength_atr(df: pd.DataFrame, lookback: int = 20, atr_n: int = 14) -> pd.Series:
    """
    Breakout strength normalized by ATR:
    (close - max(high_{t-lookback..t-1})) / ATR_n

    >0 suggests breakout above recent highs.
    Large value suggests over-extended breakout.
    """
    ref = df["high"].shift(1).rolling(lookback, min_periods=lookback).max()
    atr_val = atr(df, atr_n)
    return (df["close"] - ref) / (atr_val + _EPS)


def rvol(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Relative volume: vol / SMA(vol, window)
    """
    base = sma(volume, window)
    return volume / (base + _EPS)


def vol_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Volume z-score: (vol - mean) / std over rolling window
    """
    mu = volume.rolling(window, min_periods=window).mean()
    sd = volume.rolling(window, min_periods=window).std()
    z = (volume - mu) / (sd + _EPS)
    return z.replace([np.inf, -np.inf], np.nan)


def range_adjusted_volume(df: pd.DataFrame) -> pd.Series:
    """
    Range-adjusted volume: volume / (high-low)
    Measures volume per unit daily range.
    """
    rng = (df["high"] - df["low"]).abs()
    return df["volume"] / (rng + _EPS)


def rav_relative(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Relative range-adjusted volume:
    RAV / SMA(RAV, window)
    """
    rav = range_adjusted_volume(df)
    return rav / (sma(rav, window) + _EPS)


def volume_price_divergence(
    df: pd.DataFrame,
    price_mom: int = 5,
    vol_short: int = 5,
    vol_long: int = 20
) -> pd.Series:
    """
    Simple volume-price divergence:
    Pmom = close/close_{t-price_mom} - 1
    Vmom = SMA(vol, vol_short)/SMA(vol, vol_long) - 1
    Div  = Pmom - Vmom

    Positive: price stronger than volume trend (possible weak breakout)
    Negative: volume stronger than price trend (possible accumulation or noise)
    """
    pmom = df["close"].pct_change(price_mom)
    vshort = sma(df["volume"], vol_short)
    vlong = sma(df["volume"], vol_long)
    vmom = vshort / (vlong + _EPS) - 1.0
    return pmom - vmom


def atr_ratio(df: pd.DataFrame, atr_short: int = 14, atr_long: int = 60) -> pd.Series:
    """
    Volatility regime: ATR_short / SMA(ATR_short, atr_long)
    >1: volatility expanding, <1: contracting.
    """
    a = atr(df, atr_short)
    base = sma(a, atr_long)
    return a / (base + _EPS)


def bollinger_bandwidth(series: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """
    Bollinger bandwidth:
    (up-low)/mid
    """
    mid = sma(series, n)
    std = series.rolling(n, min_periods=n).std()
    up = mid + k * std
    low = mid - k * std
    bw = (up - low) / (mid + _EPS)
    return bw


def bollinger_bw_quantile(series: pd.Series, n: int = 20, k: float = 2.0, q_window: int = 252) -> Tuple[pd.Series, pd.Series]:
    """
    Bollinger bandwidth + rolling percentile rank of bandwidth.
    Returns: (bw, bw_q) where bw_q in [0, 1]
    """
    bw = bollinger_bandwidth(series, n=n, k=k)
    bw_q = rolling_percentile_rank(bw, window=q_window)
    return bw, bw_q

def vol_change_rate(volume):
    return volume.pct_change()

def vol_ratio_short_long(volume, short=5, long=20):
    return volume.rolling(short).mean() / volume.rolling(long).mean()

def vol_slope(volume, window=5):
    import numpy as np
    return volume.rolling(window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0],
        raw=False
    )

import numpy as np
import pandas as pd

# =========================
# Technical indicators for GRU training
# Lookback <= 60 trading days
# =========================

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist

def _true_range(high: pd.Series, low: pd.Series, preclose: pd.Series) -> pd.Series:
    tr1 = high - low
    tr2 = (high - preclose).abs()
    tr3 = (low - preclose).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def _atr(tr: pd.Series, n: int = 14) -> pd.Series:
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _bollinger_bw(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper = mid + k * sd
    lower = mid - k * sd
    return (upper - lower) / (mid.abs() + 1e-12)

def _rolling_max_drawdown(close: pd.Series, window: int = 60) -> pd.Series:
    roll_max = close.rolling(window, min_periods=1).max()
    dd = close / (roll_max + 1e-12) - 1.0
    return dd.rolling(window, min_periods=1).min()

def _clv(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    denom = (high - low).replace(0, np.nan)
    return ((2*close - high - low) / denom).fillna(0.0)

def _wick_ratios(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series):
    body = (close - open_).abs()
    upper = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower = pd.concat([open_, close], axis=1).min(axis=1) - low
    rng = (high - low).replace(0, np.nan)
    upper_r = (upper / rng).fillna(0.0)
    lower_r = (lower / rng).fillna(0.0)
    body_r = (body / rng).fillna(0.0)
    return upper_r, lower_r, body_r

def _vol_slope(v: pd.Series, window: int = 5) -> pd.Series:
    x = np.arange(window, dtype=np.float32)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _s(arr):
        y = arr.astype(np.float32)
        y_mean = y.mean()
        return float(((x - x_mean) * (y - y_mean)).sum() / (denom + 1e-12))

    return v.rolling(window).apply(_s, raw=True)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 20+ technical indicators per symbol.

    Input columns must already be normalized by IO:
      code,date,open,high,low,close
    and optionally:
      preclose, volume, amount

    All lookbacks <= 60.
    """
    if df.empty:
        return df

    df = df.sort_values(["code", "date"]).copy()

    def _per_sym(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        o = g["open"].astype(float)
        h = g["high"].astype(float)
        l = g["low"].astype(float)
        c = g["close"].astype(float)

        pc = g["preclose"].astype(float) if "preclose" in g.columns else c.shift(1)
        v = g["volume"].astype(float) if "volume" in g.columns else pd.Series(np.nan, index=g.index)
        amt = g["amount"].astype(float) if "amount" in g.columns else pd.Series(np.nan, index=g.index)
        to = g["turnover"].astype(float) if "turnover" in g.columns else pd.Series(np.nan, index=g.index)

        # A) momentum / trend
        g["roc_10"] = c.pct_change(10, fill_method=None)
        g["ret_5"]  = c.pct_change(5,  fill_method=None)
        g["ret_20"] = c.pct_change(20, fill_method=None)
        g["ret_60"] = c.pct_change(60, fill_method=None)
        g["rsi_14"] = _rsi(c, 14)

        dif, dea, hist = _macd(c)
        g["macd_dif"] = dif
        g["macd_dea"] = dea
        g["macd_hist"] = hist

        # B) volatility / risk
        tr = _true_range(h, l, pc)
        g["tr"] = tr
        g["tr_pct"] = tr / (pc.abs() + 1e-12)
        g["tr_pct_z_60"] = (g["tr_pct"] - g["tr_pct"].rolling(60, min_periods=20).mean()) / (
            g["tr_pct"].rolling(60, min_periods=20).std() + 1e-12
        )
        # 21-day z-score variant for tr_pct
        g["tr_pct_z_21"] = (g["tr_pct"] - g["tr_pct"].rolling(21, min_periods=10).mean()) / (
            g["tr_pct"].rolling(21, min_periods=10).std() + 1e-12
        )

        atr14 = _atr(tr, 14)
        atr60 = _atr(tr, 60)
        g["atr_14"] = atr14
        g["atr_ratio_14_60"] = atr14 / (atr60 + 1e-12)

        # gap (normalized)
        g["gap_pct"] = (o - pc) / (pc.abs() + 1e-12)
        g["gap_atr_14"] = (o - pc).abs() / (atr14.abs() + 1e-12)

        # breakout strength vs prior 20D high, normalized by ATR
        ref_high_20 = h.shift(1).rolling(20, min_periods=20).max()
        g["breakout_atr_20"] = (c - ref_high_20) / (atr14.abs() + 1e-12)

        bw = _bollinger_bw(c, 20, 2.0)
        g["bb_bw"] = bw
        g["bb_bw_z_60"] = (bw - bw.rolling(60, min_periods=20).mean()) / (bw.rolling(60, min_periods=20).std() + 1e-12)
        # 21-day z-score variant for Bollinger bandwidth
        g["bb_bw_z_21"] = (bw - bw.rolling(21, min_periods=10).mean()) / (bw.rolling(21, min_periods=10).std() + 1e-12)

        # C) position / drawdown
        g["dist_to_high_20"] = c / (c.rolling(20, min_periods=1).max() + 1e-12) - 1.0
        g["mdd_60"] = _rolling_max_drawdown(c, 60)
        g["clv"] = _clv(c, h, l)

        # D) candlestick
        uw, lw, br = _wick_ratios(o, h, l, c)
        g["wick_upper"] = uw
        g["wick_lower"] = lw
        g["wick_body"] = br
        # simple candlestick pattern flags (0/1)
        # doji: tiny real body relative to range
        g["is_doji"] = (g["wick_body"] <= 0.10).astype(float)
        # long upper wick (potential rejection)
        g["is_long_upper"] = (g["wick_upper"] >= 0.60).astype(float)
        # long lower wick (potential support)
        g["is_long_lower"] = (g["wick_lower"] >= 0.60).astype(float)
        # close up/down day flags
        g["is_up_day"] = (c >= pc).astype(float)
        g["is_down_day"] = (c < pc).astype(float)

        # E) volume / flow
        g["rvol_20"] = v / (v.rolling(20, min_periods=5).mean() + 1e-12)
        g["vol_z_20"] = (v - v.rolling(20, min_periods=5).mean()) / (v.rolling(20, min_periods=5).std() + 1e-12)
        # turnover (if available)
        g["turnover_ratio_5_20"] = to.rolling(5, min_periods=3).mean() / (to.rolling(20, min_periods=5).mean() + 1e-12)
        g["turnover_z_20"] = (to - to.rolling(20, min_periods=5).mean()) / (to.rolling(20, min_periods=5).std() + 1e-12)
        g["vol_change_1"] = v.pct_change(1, fill_method=None)
        g["vol_ratio_5_20"] = v.rolling(5, min_periods=3).mean() / (v.rolling(20, min_periods=5).mean() + 1e-12)
        # 21-day volume z-score variant for L=21
        g["vol_z_21"] = (v - v.rolling(21, min_periods=10).mean()) / (v.rolling(21, min_periods=10).std() + 1e-12)
        g["vol_slope_5"] = _vol_slope(v, 5)
        g["vol_std_ratio_5_20"] = v.rolling(5, min_periods=3).std() / (v.rolling(20, min_periods=5).std() + 1e-12)

        # volume-price divergence
        g["vpd"] = c.pct_change(5, fill_method=None) - (
            v.rolling(5, min_periods=3).mean() / (v.rolling(20, min_periods=5).mean() + 1e-12) - 1.0
        )

        # amount relative
        g["rav_rel_20"] = amt / (amt.rolling(20, min_periods=5).mean() + 1e-12)

        return g

    return df.groupby("code", group_keys=False).apply(_per_sym)