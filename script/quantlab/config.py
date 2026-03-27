# -*- coding: utf-8 -*-
import yaml, os
from typing import Optional

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
    },
    'valuation': {
        'enabled': False,
        'mode': 'rank_only',
        'expensive_cut': 0.8,
        'pb_weight': 0.5,
        'ps_weight': 0.5,
        'tech_weight': 0.7,
        'value_weight': 0.3,
        'ml_weight': 0.0,
        'rolling_window': 120,
    },
    'portfolio': {
        'top_n': 3,
        'strategy_profile': 'blended',
        'min_score': 55.0,
        'min_hold_days': 7,
        'max_hold_days': 20,
        'exit_rank_mult': 4.0,
        'turnover_buffer': 6.0,
        'min_total_exposure': 0.15,
        'base_total_exposure': 0.45,
        'max_total_exposure': 0.70,
        'min_position_weight': 0.10,
        'max_position_weight': 0.35,
        'trend_weight': 0.24,
        'momentum_weight': 0.16,
        'signal_weight': 0.14,
        'medium_term_weight': 0.20,
        'quality_weight': 0.12,
        'value_weight': 0.08,
        'risk_weight': 0.04,
        'liquidity_weight': 0.02,
    }
}

def load_config(path=None):
    if path is None or not os.path.exists(path):
        return DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # shallow merge dicts
    out = {}
    for k,v in DEFAULT_CONFIG.items():
        out[k] = v.copy() if isinstance(v, dict) else v
    for k,v in cfg.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            merged = out[k].copy()
            merged.update(v)
            out[k] = merged
        else:
            out[k] = v
    return out


def merge_config(base: dict, overrides: Optional[dict] = None) -> dict:
    if not overrides:
        return base
    out = {}
    for k, v in base.items():
        out[k] = v.copy() if isinstance(v, dict) else v
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            merged = out[k].copy()
            merged.update(v)
            out[k] = merged
        else:
            out[k] = v
    return out
