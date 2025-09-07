# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from typing import Optional

def _qcut_id(series: pd.Series, k: int) -> pd.Series:
    try:
        return pd.qcut(series.rank(method="first"), q=k, labels=[f"B{i+1}" for i in range(k)])
    except Exception:
        return pd.Series(["B1"]*len(series), index=series.index)

def monthly_freeze_bucket_map(df_all: pd.DataFrame, mode: str="size", k: int=3, code_industry: Optional[dict]=None) -> pd.DataFrame:
    df = df_all[['code','date','close','volume']].copy()
    df['ym'] = df['date'].dt.to_period('M')
    if mode == "size":
        feat = df.groupby(['code','ym'])['close'].last().reset_index().rename(columns={'close':'feat'})
    elif mode == "vol":
        r = df.groupby('code')['close'].pct_change()
        vol = r.groupby([df['code'], df['date'].dt.to_period('M')]).std().reset_index().rename(columns={'close':'feat','date':'ym'})
        feat = vol
    else:
        feat = df.groupby(['code','ym'])['close'].last().reset_index().rename(columns={'close':'feat'})
    snap = feat.groupby('code').apply(lambda x: x.iloc[-1:]).reset_index(drop=True)
    snap['bucket_id'] = _qcut_id(snap['feat'], k)
    return snap[['code','bucket_id']]
