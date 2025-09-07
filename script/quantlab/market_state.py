# -*- coding: utf-8 -*-
import numpy as np, pandas as pd

def build_index_state_from_panel(indicator_df: pd.DataFrame, cfg: dict, by_bucket: bool=False) -> pd.DataFrame:
    df = indicator_df[['date','adx14','boll_bw']].copy()
    if by_bucket and 'bucket_id' in indicator_df.columns:
        df['bucket_id'] = indicator_df['bucket_id'].values
        groups = ['date','bucket_id']
    else:
        indicator_df = indicator_df.copy()
        indicator_df['bucket_id'] = 'ALL'
        df['bucket_id'] = 'ALL'
        groups = ['date','bucket_id']

    agg = df.groupby(groups).agg(idx_adx14=('adx14','median'),
                                 idx_bw=('boll_bw','median')).reset_index()

    M = cfg['S2'].get('bw_quantile_window', 120)
    def last_rank_pct(s):
        s = pd.Series(s).dropna()
        if len(s)==0: return np.nan
        return s.rank(pct=True).iloc[-1]

    agg['idx_bw_q'] = np.nan
    if by_bucket:
        agg['idx_bw_q'] = agg.groupby('bucket_id')['idx_bw'].transform(lambda s: s.rolling(M, min_periods=M).apply(last_rank_pct, raw=False))
    else:
        agg['idx_bw_q'] = agg['idx_bw'].rolling(M, min_periods=M).apply(last_rank_pct, raw=False)

    def decide(row):
        if pd.isna(row['idx_adx14']) or pd.isna(row['idx_bw_q']): return 'neutral'
        if (row['idx_adx14'] >= cfg['index_gate']['adx_trend_th']) and (row['idx_bw_q'] > cfg['index_gate']['bw_low_quantile']):
            return 'trend_ok'
        if (row['idx_adx14'] < cfg['index_gate']['adx_range_th']) and (row['idx_bw_q'] <= cfg['index_gate']['bw_low_quantile']):
            return 'range_bias'
        return 'neutral'

    agg['market_state_index'] = agg.apply(decide, axis=1)
    return agg[['date','bucket_id','idx_adx14','idx_bw','idx_bw_q','market_state_index']]
