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
