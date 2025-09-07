# -*- coding: utf-8 -*-
import yaml, os

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
