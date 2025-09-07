# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np, pandas as pd
from tqdm.auto import tqdm
from .signals import assemble_signals

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

    for symbol in tqdm(list(data.keys()), desc="[回测] per-symbol"):
        df = data[symbol]
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
                        macd_dead = (today['macd_dif'] < today['macd_dea']) and (sig.iloc[i-1]['macd_dif'] >= sig.iloc[i-1]['macd_dea']) if i>0 else False
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

    signals_daily = pd.concat(signals_all, ignore_index=True) if signals_all else pd.DataFrame()
    last_day = signals_daily['date'].max() if not signals_daily.empty else None
    today_cand = signals_daily[signals_daily['date'] == last_day].copy() if last_day is not None else pd.DataFrame()

    def to_confidence(row):
        score = 0
        score += min(100, max(0, (row['adx14'] or 0)))
        if row.get('s2_entry', 0) == 1: score += 20
        if row.get('s1_entry', 0) == 1: score += 10
        return int(min(100, score / 2))
    if not today_cand.empty:
        import numpy as _np
        today_cand['strategy'] = _np.where(today_cand['s2_entry']==1, 'S2',
                                    _np.where(today_cand['s1_entry']==1, 'S1',
                                    _np.where(today_cand['s3_long_entry']==1, 'S3', 'None')))
        today_cand = today_cand[(today_cand['strategy']!='None')]
        today_cand['confidence'] = today_cand.apply(to_confidence, axis=1)
        today_cand['entry_price_ref'] = np.nan
        today_cand['stop_ref'] = _np.where(today_cand['strategy']=='S2', today_cand['s2_stop'],
                                    _np.where(today_cand['strategy']=='S1', today_cand['s1_stop'],
                                              today_cand['s3_stop']))
        today_cand['pos_ref'] = _np.where(today_cand['strategy']=='S2', today_cand['s2_pos'],
                                   _np.where(today_cand['strategy']=='S1', today_cand['s1_pos'],
                                             today_cand['s3_pos']))
        today_cand['key_notes'] = _np.where(today_cand['strategy']=='S2', today_cand['s2_reason'],
                                     _np.where(today_cand['strategy']=='S1', today_cand['s1_reason'],
                                               today_cand['s3_reason']))

    trades_ledger = pd.DataFrame(trades)
    if not trades_ledger.empty:
        summary = (trades_ledger
                   .groupby('strategy')['pnl_pct']
                   .agg(trades='count',
                        win_rate=lambda s: float((s>0).mean()),
                        avg_win=lambda s: float(s[s>0].mean()) if (s>0).any() else 0.0,
                        avg_loss=lambda s: float(s[s<=0].mean()) if (s<=0).any() else 0.0,
                        expectancy=lambda s: float(s.mean())))
        summary = summary.reset_index()
        strategy_summary = summary
    else:
        strategy_summary = pd.DataFrame(columns=['strategy','trades','win_rate','avg_win','avg_loss','expectancy'])

    return signals_daily, trades_ledger, strategy_summary, today_cand
