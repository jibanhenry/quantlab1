# -*- coding: utf-8 -*-
import pandas as pd

def perf_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=['strategy','trades','win_rate','avg_win','avg_loss','expectancy'])
    summary = (trades
               .groupby('strategy')['pnl_pct']
               .agg(trades='count',
                    win_rate=lambda s: float((s>0).mean()),
                    avg_win=lambda s: float(s[s>0].mean()) if (s>0).any() else 0.0,
                    avg_loss=lambda s: float(s[s<=0].mean()) if (s<=0).any() else 0.0,
                    expectancy=lambda s: float(s.mean()))).reset_index()
    return summary
