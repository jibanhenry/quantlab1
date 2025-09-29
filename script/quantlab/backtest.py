# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
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
    confidence: float  # <<< 新增：进场当日的置信度
    holding: bool = True


def backtest_simple(data: Dict[str, pd.DataFrame],
                    idx_state_df: pd.DataFrame,
                    cfg: dict,
                    cost_bp: float = 2.0):
    """
    简化版本回测：
      - data: {symbol: dataframe with indicators (需要 assemble_signals 里的字段)}
      - idx_state_df: 市场气候（会被 assemble_signals merge）
      - cfg: 配置
      - cost_bp: 单边成本 基点（默认2bp；来回4bp）
    返回:
      signals_daily, trades_ledger, strategy_summary, today_cand
    """

    # --- 置信度函数：放在函数顶部，便于建仓时调用 ---
    def to_confidence(row: pd.Series) -> int:
        score = 0
        # 趋势强度：adx14；做边界保护
        adx_val = row.get('adx14', np.nan)
        if pd.notna(adx_val):
            score += int(min(100, max(0, float(adx_val))))
        # 策略加分：S2突破 +20；S1回撤 +10
        if row.get('s2_entry', 0) == 1:
            score += 20
        if row.get('s1_entry', 0) == 1:
            score += 10
        # 缩放并截断到 [0,100]
        return int(min(100, score / 2))

    signals_all = []
    trades = []

    # === 逐标的 ===
    for symbol in tqdm(list(data.keys()), desc="[回测] per-symbol"):
        df = data[symbol]
        sig = assemble_signals(df, idx_state_df, cfg)
        sig['symbol'] = symbol
        signals_all.append(sig)

        pos: Optional[Position] = None

        # === 逐日 ===
        for i in range(len(sig) - 1):
            today = sig.iloc[i]
            tomorrow = sig.iloc[i + 1]

            # --- 管理持仓：是否触发离场 ---
            if pos and pos.holding:
                exit_flag = False
                exit_reason = ""
                stop = pos.stop

                if today['close'] < stop:
                    exit_flag = True
                    exit_reason = "hit_stop"
                else:
                    if pos.strategy == 'S1':
                        macd_dead = (
                            (today['macd_dif'] < today['macd_dea']) and
                            (sig.iloc[i - 1]['macd_dif'] >= sig.iloc[i - 1]['macd_dea'])
                        ) if i > 0 else False
                        if (today['close'] < today['ema50']) or macd_dead:
                            exit_flag = True
                            exit_reason = "ema50_break or macd_dead"
                    elif pos.strategy == 'S2':
                        if today['close'] < today['boll_mid']:
                            exit_flag = True
                            exit_reason = "midband_fail"
                    elif pos.strategy == 'S3':
                        if today['close'] >= today['boll_mid']:
                            exit_flag = True
                            exit_reason = "mean_revert_tp"

                if exit_flag:
                    exit_price = tomorrow['open']
                    pnl_pct = (exit_price - pos.entry_price) / pos.entry_price - (2 * cost_bp / 10000.0)
                    trades.append({
                        "symbol": symbol,
                        "strategy": pos.strategy,
                        "entry_date": pos.entry_date,
                        "entry_price": pos.entry_price,
                        "exit_date": tomorrow['date'],
                        "exit_price": exit_price,
                        "pnl_pct": float(pnl_pct),
                        "entry_pos": float(pos.position),
                        "exit_reason": exit_reason,
                        "initial_stop": float(pos.initial_stop),
                        "stop_on_exit": float(stop),
                        "confidence": float(pos.confidence),  # <<< 写入进场置信度
                    })
                    pos.holding = False
                    pos = None

            # --- 无持仓：寻找入场 ---
            if (pos is None) and (today['market_state_index'] in ['trend_ok', 'neutral'] or today['market_state_stock'] == 'range'):
                # 优先级：S2 > S1 > S3
                if today['s2_entry'] == 1:
                    entry_price = tomorrow['open']
                    stop = today['s2_stop']
                    pos = Position(
                        symbol=symbol,
                        strategy='S2',
                        entry_date=today['date'],
                        entry_price=float(entry_price),
                        position=float(today['s2_pos']),
                        stop=float(stop),
                        initial_stop=float(stop),
                        reason=str(today['s2_reason']),
                        confidence=to_confidence(today),
                    )
                elif (today['s1_entry'] == 1) and (today['market_state_stock'] == 'trend'):
                    entry_price = tomorrow['open']
                    stop = today['s1_stop']
                    pos = Position(
                        symbol=symbol,
                        strategy='S1',
                        entry_date=today['date'],
                        entry_price=float(entry_price),
                        position=float(today['s1_pos']),
                        stop=float(stop),
                        initial_stop=float(stop),
                        reason=str(today['s1_reason']),
                        confidence=to_confidence(today),
                    )
                elif (today['market_state_stock'] == 'range') and (today['s3_long_entry'] == 1):
                    entry_price = tomorrow['open']
                    stop = today['s3_stop']
                    pos = Position(
                        symbol=symbol,
                        strategy='S3',
                        entry_date=today['date'],
                        entry_price=float(entry_price),
                        position=float(today['s3_pos']),
                        stop=float(stop),
                        initial_stop=float(stop),
                        reason=str(today['s3_reason']),
                        confidence=to_confidence(today),
                    )

            # --- 趋势加仓 / 移动止损 ---
            if pos and pos.holding and (today['s4_pyramid'] == 1) and (pos.strategy in ['S1', 'S2']):
                new_stop = max(pos.stop, today['psar'], today['ema20'])
                pos.stop = float(new_stop)

        # --- 数据末尾强制平仓 ---
        if pos and pos.holding:
            last = sig.iloc[-1]
            exit_price = last['close']
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price - (2 * cost_bp / 10000.0)
            trades.append({
                "symbol": symbol,
                "strategy": pos.strategy,
                "entry_date": pos.entry_date,
                "entry_price": pos.entry_price,
                "exit_date": last['date'],
                "exit_price": float(exit_price),
                "pnl_pct": float(pnl_pct),
                "entry_pos": float(pos.position),
                "exit_reason": "eod_close",
                "initial_stop": float(pos.initial_stop),
                "stop_on_exit": float(pos.stop),
                "confidence": float(pos.confidence),  # <<< 写入进场置信度
            })

    # === 汇总 ===
    signals_daily = pd.concat(signals_all, ignore_index=True) if signals_all else pd.DataFrame()
    last_day = signals_daily['date'].max() if not signals_daily.empty else None
    today_cand = signals_daily[signals_daily['date'] == last_day].copy() if last_day is not None else pd.DataFrame()

    # 末日候选（用于当日选股）
    if not today_cand.empty:
        today_cand['strategy'] = np.where(today_cand['s2_entry'] == 1, 'S2',
                                   np.where(today_cand['s1_entry'] == 1, 'S1',
                                            np.where(today_cand['s3_long_entry'] == 1, 'S3', 'None')))
        today_cand = today_cand[(today_cand['strategy'] != 'None')]
        today_cand['confidence'] = today_cand.apply(to_confidence, axis=1)
        today_cand['entry_price_ref'] = np.nan
        today_cand['stop_ref'] = np.where(today_cand['strategy'] == 'S2', today_cand['s2_stop'],
                                   np.where(today_cand['strategy'] == 'S1', today_cand['s1_stop'],
                                            today_cand['s3_stop']))
        today_cand['pos_ref'] = np.where(today_cand['strategy'] == 'S2', today_cand['s2_pos'],
                                  np.where(today_cand['strategy'] == 'S1', today_cand['s1_pos'],
                                           today_cand['s3_pos']))
        today_cand['key_notes'] = np.where(today_cand['strategy'] == 'S2', today_cand['s2_reason'],
                                    np.where(today_cand['strategy'] == 'S1', today_cand['s1_reason'],
                                             today_cand['s3_reason']))

    # 生成交易汇总
    trades_ledger = pd.DataFrame(trades)
    if not trades_ledger.empty:
        summary = (trades_ledger
                   .groupby('strategy')['pnl_pct']
                   .agg(trades='count',
                        win_rate=lambda s: float((s > 0).mean()),
                        avg_win=lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0,
                        avg_loss=lambda s: float(s[s <= 0].mean()) if (s <= 0).any() else 0.0,
                        expectancy=lambda s: float(s.mean())))
        summary = summary.reset_index()
        strategy_summary = summary
    else:
        strategy_summary = pd.DataFrame(columns=['strategy', 'trades', 'win_rate', 'avg_win', 'avg_loss', 'expectancy'])

    return signals_daily, trades_ledger, strategy_summary, today_cand

# === 新增：把“所有候选”按与 ledger 相同的退出规则，生成“虚拟交易” ===
import numpy as np
import pandas as pd
from typing import Dict, List

from tqdm.auto import tqdm

def all_signals_virtual_trades(indicator_dict: Dict[str, pd.DataFrame],
                               idx_state_df: pd.DataFrame,
                               cfg: dict,
                               cost_bp: float = 2.0) -> pd.DataFrame:
    """
    目标：输出与 trades_ledger 完全同构的“虚拟交易”明细表：
      - 每一个命中 S1/S2/S3 的【候选信号日】都独立开一笔（不受是否已有持仓限制）；
      - 入场/出场/止损/移动止损 与 backtest_simple 保持一致；
      - 列结构一致：symbol,strategy,entry_date,entry_price,exit_date,exit_price,pnl_pct,
                    entry_pos,exit_reason,initial_stop,stop_on_exit,confidence
    """

    def to_confidence(row: pd.Series) -> int:
        score = 0
        adx_val = row.get('adx14', np.nan)
        if pd.notna(adx_val):
            score += int(min(100, max(0, float(adx_val))))
        if row.get('s2_entry', 0) == 1:
            score += 20
        if row.get('s1_entry', 0) == 1:
            score += 10
        return int(min(100, score / 2))

    out: List[dict] = []

    ALLOW_IDX = {'trend_ok', 'neutral'}  # 与 backtest_simple 保持一致
    stop_bp = 2.0 * 2 / 10000.0  # 双边成本：来回各 2bp（与 backtest_simple 同口径）

    # === 逐标的 ===
    for symbol, df in tqdm(indicator_dict.items(), desc="[virtual] assemble & simulate", total=len(indicator_dict)):
        g = df.sort_values("date").reset_index(drop=True).copy()
        # 关键：先拼接信号（得到 s1_entry/s2_entry/s3_long_entry & market_state_*）
        sig = assemble_signals(g, idx_state_df, cfg)
        sig = sig.sort_values("date").reset_index(drop=True)

        s1_flag = (sig.get("s1_entry", 0) == 1)
        s2_flag = (sig.get("s2_entry", 0) == 1)
        s3_flag = (sig.get("s3_long_entry", 0) == 1)

        # === 逐日（候选即开仓；不限制“已持仓”）===
        for i in range(len(sig) - 1):  # 需要 next_open
            today = sig.iloc[i]
            tomorrow = sig.iloc[i + 1]

            # 市场状态门槛（与 backtest_simple 相同）
            idx_ok = (str(today.get('market_state_index', '')) in ALLOW_IDX) or \
                     (str(today.get('market_state_stock', '')) == 'range')
            if not idx_ok:
                continue

            # 当天可能多个策略命中：每个策略各开一笔（与 ledger 的“只开一笔”不同，这是“候选就开”）
            cands = []
            if bool(s2_flag.iloc[i]): cands.append('S2')
            if bool(s1_flag.iloc[i]) and str(today.get('market_state_stock','')) == 'trend': cands.append('S1')
            if bool(s3_flag.iloc[i]) and str(today.get('market_state_stock','')) == 'range': cands.append('S3')
            if not cands:
                continue

            for strat in cands:
                entry_price = float(tomorrow['open'])
                entry_dt    = tomorrow['date']
                # 与 backtest_simple 一致的初始止损与仓位
                if strat == 'S2':
                    stop0 = float(today.get('s2_stop', np.nan))
                    pos_w = float(today.get('s2_pos', 0.0))
                    reason = str(today.get('s2_reason', ''))
                elif strat == 'S1':
                    stop0 = float(today.get('s1_stop', np.nan))
                    pos_w = float(today.get('s1_pos', 0.0))
                    reason = str(today.get('s1_reason', ''))
                else:  # S3
                    stop0 = float(today.get('s3_stop', np.nan))
                    pos_w = float(today.get('s3_pos', 0.0))
                    reason = str(today.get('s3_reason', ''))

                trailing_stop = stop0
                initial_stop  = stop0
                confidence    = float(to_confidence(today))

                exit_reason = None
                exit_price  = None
                stop_on_exit= None
                exit_dt     = None

                # === 推进直到退出（完全复刻 backtest_simple 的口径）===
                for j in range(i, len(sig) - 1):
                    row      = sig.iloc[j]
                    next_row = sig.iloc[j + 1]
                    close    = float(row['close'])

                    # a) 止损触发：收盘 < 止损；次日开盘成交
                    if np.isfinite(trailing_stop) and (close < trailing_stop):
                        exit_reason = "hit_stop"
                        exit_price  = float(next_row['open'])
                        exit_dt     = next_row['date']
                        stop_on_exit= float(trailing_stop)
                        break

                    # b) 策略型退出
                    if strat == 'S1':
                        macd_dead = ((row.get('macd_dif', 0.0) < row.get('macd_dea', 0.0)) and
                                     (sig.iloc[j - 1].get('macd_dif', 0.0) >= sig.iloc[j - 1].get('macd_dea', 0.0))) if j > 0 else False
                        if (close < float(row.get('ema50', close))) or macd_dead:
                            exit_reason = "ema50_break or macd_dead"
                            exit_price  = float(next_row['open'])
                            exit_dt     = next_row['date']
                            break
                    elif strat == 'S2':
                        if close < float(row.get('boll_mid', close)):
                            exit_reason = "midband_fail"
                            exit_price  = float(next_row['open'])
                            exit_dt     = next_row['date']
                            break
                    else:  # S3
                        if close >= float(row.get('boll_mid', close)):
                            exit_reason = "mean_revert_tp"
                            exit_price  = float(next_row['open'])
                            exit_dt     = next_row['date']
                            break

                    # c) 趋势加仓/移动止损（与 backtest_simple 一致）
                    if int(row.get('s4_pyramid', 0)) == 1 and strat in ['S1','S2']:
                        psar  = float(row.get('psar', trailing_stop if np.isfinite(trailing_stop) else close))
                        ema20 = float(row.get('ema20', trailing_stop if np.isfinite(trailing_stop) else close))
                        trailing_stop = float(max(trailing_stop, psar, ema20)) if np.isfinite(trailing_stop) else float(max(psar, ema20))

                # d) 末日强平（兜底，按 backtest_simple 的“最后一根用收盘价”）
                if exit_reason is None:
                    last = sig.iloc[-1]
                    exit_reason = "eod_close" if last.name == (len(sig) - 1) else "last_bar_exit"
                    exit_price  = float(last['close'])
                    exit_dt     = last['date']
                    stop_on_exit= float(trailing_stop) if np.isfinite(trailing_stop) else np.nan

                pnl_pct = (exit_price - entry_price) / entry_price - (2 * cost_bp / 10000.0)

                out.append({
                    "symbol": symbol,
                    "code": str(symbol),  # <<< 新增：对齐预测键
                    "strategy": strat,
                    "signal_date": today['date'],  # <<< 新增：信号发生日（预测用）
                    "entry_date": entry_dt,
                    "entry_price": float(entry_price),
                    "exit_date": exit_dt,
                    "exit_price": float(exit_price),
                    "pnl_pct": float(pnl_pct),
                    "entry_pos": float(pos_w),
                    "exit_reason": exit_reason,
                    "initial_stop": float(initial_stop) if np.isfinite(initial_stop) else np.nan,
                    "stop_on_exit": float(stop_on_exit) if stop_on_exit is not None else np.nan,
                    "confidence": float(confidence),
                })

    return pd.DataFrame(out)