# -*- coding: utf-8 -*-
"""
quant_framework.py  (panel-friendly + tqdm)

支持两种数据组织方式：
1) 多文件：每个股票一个CSV（原版支持）
2) 单文件：一个CSV包含所有股票（必须含 code 列），本版新增支持

输入中文列：
  code, 日期, 开盘, 最高, 最低, 收盘, 前收, 成交量, 成交额, 换手率, 涨跌幅, pbMRQ, psTTM

输出：
  signals_daily.csv, trades_ledger.csv, strategy_summary.csv, candidates_YYYYMMDD.csv

仅依赖：pandas, numpy, pyyaml, tqdm
"""

import os
import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from tqdm import tqdm   # ✅ 进度条

# =========================
# 工具：中文列名 → 内部英文字段
# =========================
CHN2ENG = {
    "code": "code",
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "前收": "preclose",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
    "涨跌幅": "pct_chg",
    "pbMRQ": "pb_mrq",
    "psTTM": "ps_ttm",
}

# =========================
# 指标计算函数（仅 pandas / numpy）
# =========================
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

def dmi_adx(df: pd.DataFrame, n: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    high = df['high']
    low = df['low']
    prev_high = high.shift(1)
    prev_low = low.shift(1)

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
    high = df['high'].values
    low = df['low'].values
    length = len(df)
    psar = np.zeros(length)
    bull = True
    af = step
    ep = low[0]
    psar[0] = low[0]

    for i in range(1, length):
        prior_psar = psar[i-1]
        if bull:
            psar[i] = min(prior_psar + af * (ep - prior_psar), low[i-1])
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                af = step
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:
            psar[i] = max(prior_psar + af * (ep - prior_psar), high[i-1])
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                af = step
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)
    return pd.Series(psar, index=df.index)
# =========================
# 指标装配 & 状态判别
# =========================
def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    out['ema20'] = ema(out['close'], cfg['S1']['ema_fast'])
    out['ema50'] = ema(out['close'], cfg['S1']['ema_mid'])
    out['ema200'] = ema(out['close'], cfg['S1']['ema_slow'])
    out['macd_dif'], out['macd_dea'], out['macd_hist'] = macd(out['close'])
    out['rsi14'] = rsi(out['close'], 14)
    out['atr14'] = atr(out, 14)
    out['atr_pct'] = out['atr14'] / out['close']
    out['boll_mid'], out['boll_up'], out['boll_low'], out['boll_bw'] = bollinger(out['close'], cfg['S2']['boll_n'], cfg['S2']['boll_k'])
    out['obv'] = obv(out['close'], out['volume'])
    out['plus_di14'], out['minus_di14'], out['adx14'] = dmi_adx(out, 14)
    out['cci20'] = cci(out, 20)
    out['roc10'] = roc(out['close'], 10)
    K, D, J = kdj(out, 9, 3, 3)
    out['kdj_k'], out['kdj_d'], out['kdj_j'] = K, D, J
    out['wr14'] = williams_r(out, 14)
    out['psar'] = psar(out)
    return out

def judge_stock_state(stock_df: pd.DataFrame, cfg: dict) -> pd.Series:
    def decide_row(row):
        adx = row['adx14']
        if pd.isna(adx): return np.nan
        if adx >= cfg['stock_state']['adx_trend_th']: return 'trend'
        if adx < cfg['stock_state']['adx_range_th']: return 'range'
        return 'neutral'
    return stock_df.apply(decide_row, axis=1)

def build_index_state_from_panel(indicator_dict: Dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    frames = []
    for code, df in indicator_dict.items():
        frames.append(df[['date','adx14','boll_bw']])
    panel = pd.concat(frames, axis=0, ignore_index=True)
    panel = panel.dropna(subset=['date'])

    agg = panel.groupby('date').agg(median_adx=('adx14','median'),
                                    median_bw=('boll_bw','median')).reset_index()

    M = cfg['S2'].get('bw_quantile_window', 120)
    def last_rank_pct(s):
        s = pd.Series(s).dropna()
        if len(s)==0: return np.nan
        return s.rank(pct=True).iloc[-1]
    agg['idx_bw_q'] = agg['median_bw'].rolling(M, min_periods=M).apply(last_rank_pct, raw=False)

    def decide(row):
        if pd.isna(row['median_adx']) or pd.isna(row['idx_bw_q']): return 'neutral'
        if (row['median_adx'] >= cfg['index_gate']['adx_trend_th']) and (row['idx_bw_q'] > cfg['index_gate']['bw_low_quantile']):
            return 'trend_ok'
        if (row['median_adx'] < cfg['index_gate']['adx_range_th']) and (row['idx_bw_q'] <= cfg['index_gate']['bw_low_quantile']):
            return 'range_bias'
        return 'neutral'

    agg['market_state_index'] = agg.apply(decide, axis=1)
    agg = agg.rename(columns={'median_adx':'idx_adx14','median_bw':'idx_bw'})
    return agg[['date','idx_adx14','idx_bw','idx_bw_q','market_state_index']]

# =========================
# 策略信号（S1 / S2 / S3 / S4）
# =========================
def s1_signal(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['s1_entry'] = 0; out['s1_reason'] = ""
    cond_trend = (df['ema50'] > df['ema200']) & (df['adx14'] >= cfg['stock_state']['adx_trend_th'])
    delta_price = (df['close'] - df['ema20']).abs()
    cond_pullback = delta_price <= (cfg['S1']['atr_mul_pullback'] * df['atr14'])
    cond_macd = df['macd_hist'] > 0
    cond_rsi = df['rsi14'] >= cfg['S1']['rsi_bull_low']
    N = cfg['S1']['obv_lookback']
    obv_roll_max = df['obv'].rolling(N, min_periods=N).max()
    obv_slope = df['obv'].diff(N)
    cond_obv = (obv_slope > 0) | (df['obv'] >= obv_roll_max)

    cond = cond_trend & cond_pullback & cond_macd & cond_rsi
    if cfg['S1']['obv_confirm']: cond = cond & cond_obv

    out.loc[cond, 's1_entry'] = 1
    out.loc[cond, 's1_reason'] = "pullback_to_ema20 & macd_hist>0 & rsi_bull & obv_confirm"
    out['s1_stop'] = df['close'] - cfg['S1']['atr_mul'] * df['atr14']
    out['s1_pos'] = cfg['risk']['per_trade_risk_pct'] / (cfg['S1']['atr_mul'] * df['atr_pct'].replace(0, np.nan))
    out['s1_pos'] = out['s1_pos'].clip(upper=cfg['risk']['max_pos_per_stock'])
    return out

def s2_signal(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['s2_entry'] = 0; out['s2_reason'] = ""
    M = cfg['S2']['bw_quantile_window']
    def last_rank_pct(s):
        s = pd.Series(s).dropna()
        if len(s) == 0: return np.nan
        return s.rank(pct=True).iloc[-1]
    bw_q = df['boll_bw'].rolling(M, min_periods=M).apply(last_rank_pct, raw=False)
    cond_squeeze = bw_q <= cfg['S2']['bw_quantile']
    cond_break = df['close'] > df['boll_up']
    cond_adx_up = df['adx14'] > df['adx14'].shift(1)
    N = cfg['S1']['obv_lookback']
    obv_roll_max = df['obv'].rolling(N, min_periods=N).max()
    cond_obv = df['obv'] >= obv_roll_max
    cond = cond_squeeze & cond_break & cond_adx_up
    if cfg['S2']['obv_confirm']: cond = cond & cond_obv
    out.loc[cond, 's2_entry'] = 1
    out.loc[cond, 's2_reason'] = "squeeze_breakout & adx_rising & obv_confirm"
    out['s2_stop'] = df['boll_mid'] - cfg['S2']['atr_stop_mul'] * df['atr14']
    out['s2_pos'] = cfg['risk']['per_trade_risk_pct'] / (cfg['S2']['atr_stop_mul'] * df['atr_pct'].replace(0, np.nan))
    out['s2_pos'] = out['s2_pos'].clip(upper=cfg['risk']['max_pos_per_stock'])
    return out

def s3_signal(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['s3_long_entry'] = 0; out['s3_reason'] = ""
    cond_rsi = df['rsi14'] <= cfg['S3']['rsi_buy']
    cond_band = (df['close'] <= df['boll_low']) | (df['close'] <= (df['ema20'] - cfg['S3']['atr_n'] * df['atr14']))
    cond = cond_rsi & cond_band
    out.loc[cond, 's3_long_entry'] = 1
    out.loc[cond, 's3_reason'] = "mean_revert_long: rsi_low & near_lower_band"
    out['s3_stop'] = df['close'] - cfg['S3']['atr_stop_mul'] * df['atr14']
    out['s3_pos'] = cfg['risk']['per_trade_risk_pct'] / (cfg['S3']['atr_stop_mul'] * df['atr_pct'].replace(0, np.nan))
    out['s3_pos'] = out['s3_pos'].clip(upper=cfg['risk']['max_pos_per_stock'])
    return out

def s4_signal(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['s4_pyramid'] = 0; out['s4_reason'] = ""
    cond_trend_power = (df['plus_di14'] > df['minus_di14']) & (df['adx14'] > df['adx14'].shift(1))
    cond_price_momo = (df['cci20'] > cfg['S4']['cci_th']) | (df['roc10'] > 0)
    N = cfg['S1']['obv_lookback']
    obv_roll_max = df['obv'].rolling(N, min_periods=N).max()
    cond_obv = df['obv'] >= obv_roll_max
    cond = cond_trend_power & cond_price_momo & cond_obv
    out.loc[cond, 's4_pyramid'] = 1
    out.loc[cond, 's4_reason'] = "trend_momo_volume: +DI>-DI & ADX↑ & CCI/ROC & OBV_breakout"
    return out
# =========================
# 信号拼装
# =========================
def assemble_signals(stock_df: pd.DataFrame, idx_state: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = stock_df.copy()
    df = df.merge(idx_state[['date','market_state_index']], on='date', how='left')
    df['market_state_stock'] = judge_stock_state(df, cfg)

    s1 = s1_signal(df, cfg)
    s2 = s2_signal(df, cfg)
    s3 = s3_signal(df, cfg)
    s4 = s4_signal(df, cfg)

    out = pd.DataFrame(index=df.index)
    out['date'] = df['date'].values
    out['market_state_index'] = df['market_state_index'].values
    out['market_state_stock'] = df['market_state_stock'].values
    out['adx14'] = df['adx14'].values

    cond_trend_allowed = df['market_state_index'].isin(['trend_ok','neutral'])
    cond_stock_trend = df['market_state_stock'].eq('trend')
    cond_range = df['market_state_stock'].eq('range')

    out['s1_entry'] = np.where(cond_trend_allowed & cond_stock_trend, s1['s1_entry'], 0)
    out['s1_reason'] = np.where(out['s1_entry'] == 1, s1['s1_reason'], "")
    out['s1_stop'] = s1['s1_stop']
    out['s1_pos'] = s1['s1_pos']

    out['s2_entry'] = np.where(cond_trend_allowed, s2['s2_entry'], 0)
    out['s2_reason'] = np.where(out['s2_entry'] == 1, s2['s2_reason'], "")
    out['s2_stop'] = s2['s2_stop']
    out['s2_pos'] = s2['s2_pos']

    out['s3_long_entry'] = np.where(cond_range, s3['s3_long_entry'], 0)
    out['s3_reason'] = np.where(out['s3_long_entry'] == 1, s3['s3_reason'], "")
    out['s3_stop'] = s3['s3_stop']
    out['s3_pos'] = s3['s3_pos']

    out['s4_pyramid'] = np.where(cond_trend_allowed & cond_stock_trend, s4['s4_pyramid'], 0)
    out['s4_reason'] = np.where(out['s4_pyramid'] == 1, s4['s4_reason'], "")

    ref_cols = ['ema20','ema50','ema200','macd_dif','macd_dea','macd_hist','rsi14','atr14','obv',
                'boll_mid','boll_up','boll_low','boll_bw','plus_di14','minus_di14','psar','close','open']
    for c in ref_cols:
        out[c] = df[c].values
    return out.reset_index(drop=True)

# =========================
# 简化持仓与成交流水
# =========================
@dataclass
class Position:
    symbol: str
    strategy: str
    entry_date: pd.Timestamp
    entry_price: float
    position: float
    stop: float
    initial_stop: float
    reason: str
    holding: bool = True

def backtest_simple(data: Dict[str, pd.DataFrame],
                    idx_state_df: pd.DataFrame,
                    cfg: dict,
                    cost_bp: float = 2.0):
    signals_all = []
    trades = []

    for symbol, df in tqdm(data.items(), desc="回测信号拼装", unit="stock"):  # ✅ 加进度条
        sig = assemble_signals(df, idx_state_df, cfg)
        sig['symbol'] = symbol
        signals_all.append(sig)

        pos: Optional[Position] = None
        for i in range(len(sig) - 1):
            today = sig.iloc[i]; tomorrow = sig.iloc[i+1]
            if pos and pos.holding:
                exit_flag = False; exit_reason = ""; stop = pos.stop
                if today['close'] < stop:
                    exit_flag = True; exit_reason = "hit_stop"
                else:
                    if pos.strategy == 'S1':
                        macd_dead = (today['macd_dif'] < today['macd_dea']) and \
                                    (sig.iloc[i-1]['macd_dif'] >= sig.iloc[i-1]['macd_dea']) if i>0 else False
                        if (today['close'] < today['ema50']) or macd_dead:
                            exit_flag = True; exit_reason = "ema50_break or macd_dead"
                    elif pos.strategy == 'S2':
                        if today['close'] < today['boll_mid']:
                            exit_flag = True; exit_reason = "midband_fail"
                    elif pos.strategy == 'S3':
                        if today['close'] >= today['boll_mid']:
                            exit_flag = True; exit_reason = "mean_revert_tp"

                if exit_flag:
                    exit_price = tomorrow['open']
                    pnl_pct = (exit_price - pos.entry_price) / pos.entry_price - (2*cost_bp/10000.0)
                    trades.append({
                        "symbol": symbol, "strategy": pos.strategy,
                        "entry_date": pos.entry_date, "entry_price": pos.entry_price,
                        "exit_date": tomorrow['date'], "exit_price": exit_price,
                        "pnl_pct": pnl_pct, "entry_pos": pos.position, "exit_reason": exit_reason,
                        "initial_stop": pos.initial_stop, "stop_on_exit": stop
                    })
                    pos.holding = False; pos = None

            if (pos is None) and (today['market_state_index'] in ['trend_ok','neutral'] or today['market_state_stock'] == 'range'):
                if today['s2_entry'] == 1:
                    entry_price = tomorrow['open']; stop = today['s2_stop']
                    pos = Position(symbol, 'S2', today['date'], entry_price, today['s2_pos'], stop, stop, today['s2_reason'])
                elif today['s1_entry'] == 1 and today['market_state_stock'] == 'trend':
                    entry_price = tomorrow['open']; stop = today['s1_stop']
                    pos = Position(symbol, 'S1', today['date'], entry_price, today['s1_pos'], stop, stop, today['s1_reason'])
                elif (today['market_state_stock'] == 'range') and (today['s3_long_entry'] == 1):
                    entry_price = tomorrow['open']; stop = today['s3_stop']
                    pos = Position(symbol, 'S3', today['date'], entry_price, today['s3_pos'], stop, stop, today['s3_reason'])

            if pos and pos.holding and today['s4_pyramid'] == 1 and pos.strategy in ['S1','S2']:
                new_stop = max(pos.stop, today['psar'], today['ema20'])
                pos.stop = new_stop

        if pos and pos.holding:
            last = sig.iloc[-1]
            exit_price = last['close']
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price - (2*cost_bp/10000.0)
            trades.append({
                "symbol": symbol, "strategy": pos.strategy,
                "entry_date": pos.entry_date, "entry_price": pos.entry_price,
                "exit_date": last['date'], "exit_price": exit_price,
                "pnl_pct": pnl_pct, "entry_pos": pos.position, "exit_reason": "eod_close",
                "initial_stop": pos.initial_stop, "stop_on_exit": pos.stop
            })

    signals_daily = pd.concat(signals_all, ignore_index=True)

    last_day = signals_daily['date'].max()
    today_cand = signals_daily[signals_daily['date'] == last_day].copy()
    def to_confidence(row):
        score = 0
        score += min(100, max(0, (row['adx14'] or 0)))
        if row.get('s2_entry', 0) == 1: score += 20
        if row.get('s1_entry', 0) == 1: score += 10
        return int(min(100, score / 2))
    today_cand['strategy'] = np.where(today_cand['s2_entry']==1, 'S2',
                               np.where(today_cand['s1_entry']==1, 'S1',
                               np.where(today_cand['s3_long_entry']==1, 'S3', 'None')))
    today_cand = today_cand[(today_cand['strategy']!='None')]
    today_cand['confidence'] = today_cand.apply(to_confidence, axis=1)
    today_cand['entry_price_ref'] = np.nan
    today_cand['stop_ref'] = np.where(today_cand['strategy']=='S2', today_cand['s2_stop'],
                               np.where(today_cand['strategy']=='S1', today_cand['s1_stop'],
                                        today_cand['s3_stop']))
    today_cand['pos_ref'] = np.where(today_cand['strategy']=='S2', today_cand['s2_pos'],
                              np.where(today_cand['strategy']=='S1', today_cand['s1_pos'],
                                       today_cand['s3_pos']))
    today_cand['key_notes'] = np.where(today_cand['strategy']=='S2', today_cand['s2_reason'],
                                 np.where(today_cand['strategy']=='S1', today_cand['s1_reason'],
                                          today_cand['s3_reason']))

    trades_ledger = pd.DataFrame(trades)
    if not trades_ledger.empty:
        summary = (trades_ledger
                   .groupby('strategy')['pnl_pct']
                   .agg(trades='count',
                        win_rate=lambda s: np.mean(s>0),
                        avg_win=lambda s: np.mean(s[s>0]) if np.any(s>0) else 0.0,
                        avg_loss=lambda s: np.mean(s[s<=0]) if np.any(s<=0) else 0.0,
                        expectancy=lambda s: np.mean(s)))
        summary.reset_index(inplace=True)
        strategy_summary = summary
    else:
        strategy_summary = pd.DataFrame(columns=['strategy','trades','win_rate','avg_win','avg_loss','expectancy'])

    return signals_daily, trades_ledger, strategy_summary, today_cand

# =========================
# 配置（可外部 YAML 覆盖）
# =========================
DEFAULT_CONFIG = {
    'index_gate': {
        'adx_trend_th': 25,
        'adx_range_th': 20,
        'bw_low_quantile': 0.3,
    },
    'stock_state': {
        'adx_trend_th': 25,
        'adx_range_th': 20,
    },
    'S1': {
        'ema_fast': 20,
        'ema_mid': 50,
        'ema_slow': 200,
        'atr_mul': 1.8,
        'atr_mul_pullback': 0.8,
        'rsi_bull_low': 45,
        'obv_confirm': True,
        'obv_lookback': 20,
    },
    'S2': {
        'boll_n': 20,
        'boll_k': 2.0,
        'bw_quantile': 0.3,
        'bw_quantile_window': 120,
        'atr_stop_mul': 1.0,
        'obv_confirm': True,
    },
    'S3': {
        'rsi_buy': 35,
        'atr_n': 1.0,
        'atr_stop_mul': 1.2,
    },
    'S4': {
        'cci_th': 100,
    },
    'risk': {
        'per_trade_risk_pct': 0.005,
        'max_pos_per_stock': 0.2,
    }
}

# =========================
# I/O：加载CSV并标准化列
# =========================
def load_csv_cn(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {cn: CHN2ENG[cn] for cn in CHN2ENG if cn in df.columns}
    df = df.rename(columns=rename_map)
    need = ['date','open','high','low','close','preclose','volume','amount','turnover','pct_chg','pb_mrq','ps_ttm']
    for c in need:
        if c not in df.columns:
            df[c] = np.nan
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code','date']).reset_index(drop=True) if 'code' in df.columns else df.sort_values('date').reset_index(drop=True)
    return df

def save_outputs(signals_daily: pd.DataFrame,
                 trades_ledger: pd.DataFrame,
                 strategy_summary: pd.DataFrame,
                 candidates_today: pd.DataFrame,
                 outdir: str):
    os.makedirs(outdir, exist_ok=True)
    signals_daily.to_csv(os.path.join(outdir, "signals_daily.csv"), index=False, encoding='utf-8-sig')
    trades_ledger.to_csv(os.path.join(outdir, "trades_ledger.csv"), index=False, encoding='utf-8-sig')
    strategy_summary.to_csv(os.path.join(outdir, "strategy_summary.csv"), index=False, encoding='utf-8-sig')
    if not candidates_today.empty:
        last_day = pd.to_datetime(candidates_today['date'].max()).strftime("%Y%m%d")
        candidates_today.to_csv(os.path.join(outdir, f"candidates_{last_day}.csv"), index=False, encoding='utf-8-sig')

# =========================
# __main__
# =========================
if __name__ == "__main__":
    cfg_path = "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(DEFAULT_CONFIG, f, allow_unicode=True, sort_keys=False)

    all_in_one_csv = "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv"
    outdir = "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/output"

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            CONFIG = yaml.safe_load(f)
    except Exception:
        CONFIG = DEFAULT_CONFIG

    if os.path.exists(all_in_one_csv):
        df_all = load_csv_cn(all_in_one_csv)

        indicator_dict: Dict[str, pd.DataFrame] = {}
        for code, sub in tqdm(df_all.groupby('code'), desc="计算指标", unit="stock"):  # ✅ 加进度条
            ind = compute_indicators(sub, CONFIG)
            indicator_dict[str(code)] = ind

        idx_state_df = build_index_state_from_panel(indicator_dict, CONFIG)

        signals_all = {}
        for code, ind in indicator_dict.items():
            signals_all[code] = ind

        signals_daily, trades_ledger, strategy_summary, candidates_today = backtest_simple(
            data=signals_all, idx_state_df=idx_state_df, cfg=CONFIG, cost_bp=2.0
        )
        save_outputs(signals_daily, trades_ledger, strategy_summary, candidates_today, outdir)
        print(f"[单文件模式] 完成。输出已保存至: {outdir}")
    else:
        print("未检测到可用的数据文件，请检查路径。")