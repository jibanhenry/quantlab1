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