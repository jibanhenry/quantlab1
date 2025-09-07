# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from .indicators import ema, macd, rsi, atr, bollinger, obv, dmi_adx, cci, roc, kdj, williams_r, psar

def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    out['ema20'] = ema(out['close'], cfg['S1']['ema_fast'])
    out['ema50'] = ema(out['close'], cfg['S1']['ema_mid'])
    out['ema200'] = ema(out['close'], cfg['S1']['ema_slow'])
    out['macd_dif'], out['macd_dea'], out['macd_hist'] = macd(out['close'])
    out['rsi14'] = rsi(out['close'], 14)
    out['atr14'] = atr(out, 14); out['atr_pct'] = out['atr14'] / out['close']
    out['boll_mid'], out['boll_up'], out['boll_low'], out['boll_bw'] = bollinger(out['close'], cfg['S2']['boll_n'], cfg['S2']['boll_k'])
    out['obv'] = obv(out['close'], out['volume'])
    out['plus_di14'], out['minus_di14'], out['adx14'] = dmi_adx(out, 14)
    out['cci20'] = cci(out, 20); out['roc10'] = roc(out['close'], 10)
    K, D, J = kdj(out, 9, 3, 3); out['kdj_k'], out['kdj_d'], out['kdj_j'] = K, D, J
    out['wr14'] = williams_r(out, 14); out['psar'] = psar(out)
    return out

def judge_stock_state(stock_df: pd.DataFrame, cfg: dict) -> pd.Series:
    def decide_row(row):
        adx = row['adx14']
        if pd.isna(adx): return np.nan
        if adx >= cfg['stock_state']['adx_trend_th']: return 'trend'
        if adx < cfg['stock_state']['adx_range_th']: return 'range'
        return 'neutral'
    return stock_df.apply(decide_row, axis=1)

def s1_signal(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index); out['s1_entry']=0; out['s1_reason']=""
    cond_trend = (df['ema50'] > df['ema200']) & (df['adx14'] >= cfg['stock_state']['adx_trend_th'])
    delta_price = (df['close'] - df['ema20']).abs()
    cond_pullback = delta_price <= (cfg['S1']['atr_mul_pullback'] * df['atr14'])
    cond_macd = df['macd_hist'] > 0; cond_rsi = df['rsi14'] >= cfg['S1']['rsi_bull_low']
    N = cfg['S1']['obv_lookback']; obv_roll_max = df['obv'].rolling(N, min_periods=N).max()
    obv_slope = df['obv'].diff(N); cond_obv = (obv_slope > 0) | (df['obv'] >= obv_roll_max)
    cond = cond_trend & cond_pullback & cond_macd & cond_rsi
    if cfg['S1']['obv_confirm']: cond = cond & cond_obv
    out.loc[cond,'s1_entry']=1; out.loc[cond,'s1_reason']="pullback_to_ema20 & macd_hist>0 & rsi_bull & obv_confirm"
    out['s1_stop'] = df['close'] - cfg['S1']['atr_mul'] * df['atr14']
    out['s1_pos'] = cfg['risk']['per_trade_risk_pct'] / (cfg['S1']['atr_mul'] * df['atr_pct'].replace(0, np.nan))
    out['s1_pos'] = out['s1_pos'].clip(upper=cfg['risk']['max_pos_per_stock'])
    return out

def s2_signal(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index); out['s2_entry']=0; out['s2_reason']=""
    M = cfg['S2']['bw_quantile_window']
    def last_rank_pct(s):
        s = pd.Series(s).dropna()
        if len(s)==0: return np.nan
        return s.rank(pct=True).iloc[-1]
    bw_q = df['boll_bw'].rolling(M, min_periods=M).apply(last_rank_pct, raw=False)
    cond_squeeze = bw_q <= cfg['S2']['bw_quantile']; cond_break = df['close'] > df['boll_up']
    cond_adx_up = df['adx14'] > df['adx14'].shift(1)
    N = cfg['S1']['obv_lookback']; obv_roll_max = df['obv'].rolling(N, min_periods=N).max()
    cond_obv = df['obv'] >= obv_roll_max
    cond = cond_squeeze & cond_break & cond_adx_up
    if cfg['S2']['obv_confirm']: cond = cond & cond_obv
    out.loc[cond,'s2_entry']=1; out.loc[cond,'s2_reason']="squeeze_breakout & adx_rising & obv_confirm"
    out['s2_stop'] = df['boll_mid'] - cfg['S2']['atr_stop_mul'] * df['atr14']
    out['s2_pos'] = cfg['risk']['per_trade_risk_pct'] / (cfg['S2']['atr_stop_mul'] * df['atr_pct'].replace(0, np.nan))
    out['s2_pos'] = out['s2_pos'].clip(upper=cfg['risk']['max_pos_per_stock'])
    return out

def s3_signal(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index); out['s3_long_entry']=0; out['s3_reason']=""
    cond_rsi = df['rsi14'] <= cfg['S3']['rsi_buy']
    cond_band = (df['close'] <= df['boll_low']) | (df['close'] <= (df['ema20'] - cfg['S3']['atr_n'] * df['atr14']))
    cond = cond_rsi & cond_band
    out.loc[cond,'s3_long_entry']=1; out.loc[cond,'s3_reason']="mean_revert_long: rsi_low & near_lower_band"
    out['s3_stop'] = df['close'] - cfg['S3']['atr_stop_mul'] * df['atr14']
    out['s3_pos'] = cfg['risk']['per_trade_risk_pct'] / (cfg['S3']['atr_stop_mul'] * df['atr_pct'].replace(0, np.nan))
    out['s3_pos'] = out['s3_pos'].clip(upper=cfg['risk']['max_pos_per_stock'])
    return out

def s4_signal(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index); out['s4_pyramid']=0; out['s4_reason']=""
    cond_trend_power = (df['plus_di14'] > df['minus_di14']) & (df['adx14'] > df['adx14'].shift(1))
    cond_price_momo = (df['cci20'] > cfg['S4']['cci_th']) | (df['roc10'] > 0)
    N = cfg['S1']['obv_lookback']; obv_roll_max = df['obv'].rolling(N, min_periods=N).max()
    cond_obv = df['obv'] >= obv_roll_max
    cond = cond_trend_power & cond_price_momo & cond_obv
    out.loc[cond,'s4_pyramid']=1; out.loc[cond,'s4_reason']="trend_momo_volume: +DI>-DI & ADX↑ & CCI/ROC & OBV_breakout"
    return out

def assemble_signals(stock_df: pd.DataFrame, idx_state: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = stock_df.copy()
    if 'bucket_id' in df.columns and 'bucket_id' in idx_state.columns:
        df = df.merge(idx_state[['date','bucket_id','market_state_index']], on=['date','bucket_id'], how='left')
    else:
        df = df.merge(idx_state[['date','market_state_index']], on='date', how='left')
    df['market_state_stock'] = judge_stock_state(df, cfg)

    s1 = s1_signal(df, cfg); s2 = s2_signal(df, cfg); s3 = s3_signal(df, cfg); s4 = s4_signal(df, cfg)

    out = pd.DataFrame(index=df.index)
    out['date'] = df['date'].values
    out['market_state_index'] = df['market_state_index'].values
    out['market_state_stock'] = df['market_state_stock'].values
    out['adx14'] = df['adx14'].values

    cond_trend_allowed = df['market_state_index'].isin(['trend_ok','neutral'])
    cond_stock_trend = df['market_state_stock'].eq('trend')
    cond_range = df['market_state_stock'].eq('range')

    out['s1_entry'] = np.where(cond_trend_allowed & cond_stock_trend, s1['s1_entry'], 0)
    out['s1_reason'] = np.where(out['s1_entry']==1, s1['s1_reason'], "")
    out['s1_stop'] = s1['s1_stop']; out['s1_pos'] = s1['s1_pos']

    out['s2_entry'] = np.where(cond_trend_allowed, s2['s2_entry'], 0)
    out['s2_reason'] = np.where(out['s2_entry']==1, s2['s2_reason'], "")
    out['s2_stop'] = s2['s2_stop']; out['s2_pos'] = s2['s2_pos']

    out['s3_long_entry'] = np.where(cond_range, s3['s3_long_entry'], 0)
    out['s3_reason'] = np.where(out['s3_long_entry']==1, s3['s3_reason'], "")
    out['s3_stop'] = s3['s3_stop']; out['s3_pos'] = s3['s3_pos']

    out['s4_pyramid'] = np.where(cond_trend_allowed & cond_stock_trend, s4['s4_pyramid'], 0)
    out['s4_reason'] = np.where(out['s4_pyramid']==1, s4['s4_reason'], "")

    ref_cols = ['ema20','ema50','ema200','macd_dif','macd_dea','macd_hist','rsi14','atr14','obv','boll_mid','boll_up','boll_low','boll_bw','plus_di14','minus_di14','psar','close','open','code']
    for c in ref_cols:
        out[c] = df.get(c, np.nan).values if c in df.columns else np.nan
    if 'bucket_id' in df.columns:
        out['bucket_id'] = df['bucket_id'].values
    return out.reset_index(drop=True)
