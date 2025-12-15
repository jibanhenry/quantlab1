# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

import indicators as ind
from io_utils import load_market_csv_multi


# =========================
# 路径配置
# =========================
HISTORY_CSV = "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2020-2025_all.csv"
DAILY_CSV   = "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv"

MODEL_DIR = "model_outputs_v2"
RET_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_ret_q0.7_N20.json")
MAE_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_mae_q0.1_N20.json")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_cols.txt")

OUTDIR = "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/output/quantlab"
os.makedirs(OUTDIR, exist_ok=True)

# 实时输出（每天收盘后）
PRED_LATEST_OUT   = os.path.join(OUTDIR, "latest_preds_N20.csv")
ORDERS_LATEST_OUT = os.path.join(OUTDIR, "latest_orders_next_open.csv")
POSITIONS_PATH    = os.path.join(OUTDIR, "positions_state.csv")

# 回测/监控输出（覆盖 DAILY_CSV 的日期区间）
BT_TRADES_OUT     = os.path.join(OUTDIR, "backtest_trades.csv")
BT_DAILY_OUT      = os.path.join(OUTDIR, "backtest_daily_monitor.csv")
BT_EQUITY_OUT     = os.path.join(OUTDIR, "backtest_equity_curve.csv")


# =========================
# 策略/执行参数
# =========================
DATE_COL = "date"
SYM_COL = "code"

EPS = 1e-12
SCORE_EPS = 1e-12

N = 20  # 与训练一致

# 风险硬止损（基于 pred_mae 的分位数下界，通常为负）
RISK_HARD_STOP = -0.08   # 你容忍 8% 回撤：这里用 pred_mae < -0.08 视为硬风控

# 防抖退出：连续 OUT_CONFIRM_DAYS 天游离 topcap 才退出
OUT_CONFIRM_DAYS = 2

# 组合风险预算：用 abs(pred_mae) 作为“每只标的风险占用”近似
RISK_BUDGET = 0.12

# 是否要求预测收益为正
REQUIRE_POSITIVE_RET = True

# 允许空仓
ALLOW_FLAT = True

# 长窗特征开关（实盘更稳建议 False）
USE_LONG_WINDOW = False


# =========================
# 数据结构
# =========================
@dataclass
class Position:
    code: str
    status: str                # "PENDING_ENTRY" / "OPEN"
    entry_date: str            # "YYYY-MM-DD" 或 "NEXT_OPEN"
    entry_price: float         # 实盘收盘后未知可为 NaN
    out_counter: int = 0
    last_signal_date: str = "" # 记录最近一次对它做出决策的信号日


# =========================
# IO & 清理
# =========================
def read_market_csv(path: str) -> pd.DataFrame:
    df = load_market_csv_multi([path])

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if SYM_COL in df.columns:
        df[SYM_COL] = df[SYM_COL].astype(str)

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[DATE_COL, SYM_COL, "open", "high", "low", "close", "volume"]).copy()
    df = df.sort_values([SYM_COL, DATE_COL]).reset_index(drop=True)
    return df


def load_feature_list(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"feature list not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cols = [line.strip() for line in f if line.strip()]
    if not cols:
        raise ValueError("feature_cols.txt is empty")
    return cols


def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    return out


def fill_features_by_train_stats(train_like_df: pd.DataFrame, infer_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    用“全量历史（训练风格）统计”填充推理日缺失特征。
    """
    med = train_like_df[feature_cols].median(numeric_only=True)
    out = infer_df.copy()
    out[feature_cols] = out[feature_cols].fillna(med)
    return out


def load_booster(path: str) -> xgb.Booster:
    if not os.path.exists(path):
        raise FileNotFoundError(f"model not found: {path}")
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


# =========================
# 指标计算（与你训练一致）
# =========================
def add_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values([SYM_COL, DATE_COL]).copy()

    def _calc_one(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(DATE_COL).copy()
        close = g["close"]
        vol = g["volume"]

        # 基础
        g["ema_10"] = ind.ema(close, 10)
        g["ema_20"] = ind.ema(close, 20)
        g["ema_60"] = ind.ema(close, 60)

        g["sma_10"] = ind.sma(close, 10)
        g["sma_20"] = ind.sma(close, 20)
        g["sma_60"] = ind.sma(close, 60)

        g["tr"] = ind.true_range(g[["high", "low", "close"]])
        g["atr_14"] = ind.atr(g[["high", "low", "close"]], 14)
        g["atr_pct_14"] = g["atr_14"] / (close.abs() + EPS)

        g["rsi_14"] = ind.rsi(close, 14)

        dif, dea, hist = ind.macd(close, 12, 26, 9)
        g["macd_dif"] = dif
        g["macd_dea"] = dea
        g["macd_hist"] = hist

        mid, up, low, bw = ind.bollinger(close, 20, 2.0)
        g["boll_mid"] = mid
        g["boll_up"] = up
        g["boll_low"] = low
        g["boll_bw"] = bw

        g["obv"] = ind.obv(close, vol)
        g["cci_20"] = ind.cci(g[["high", "low", "close"]], 20)
        g["roc_10"] = ind.roc(close, 10)

        k, d, j = ind.kdj(g[["high", "low", "close"]], 9, 3, 3)
        g["kdj_k"] = k
        g["kdj_d"] = d
        g["kdj_j"] = j

        g["willr_14"] = ind.williams_r(g[["high", "low", "close"]], 14)

        pdi, mdi, adx = ind.dmi_adx(g[["high", "low", "close"]], 14)
        g["pdi_14"] = pdi
        g["mdi_14"] = mdi
        g["adx_14"] = adx

        g["psar"] = ind.psar(g[["high", "low"]], 0.02, 0.2)

        # V2
        g["mdd_60"] = ind.rolling_max_drawdown(close, window=60)
        g["gap_atr_14"] = ind.gap_atr(g[["open", "high", "low", "close"]], atr_n=14)

        g["tr_pct"] = ind.tr_pct(g[["high", "low", "close"]])
        if USE_LONG_WINDOW:
            g["tr_pctile_252"] = ind.tr_pctile(g[["high", "low", "close"]], window=252)
        else:
            g["tr_pctile_252"] = np.nan

        g["clv"] = ind.close_location_value(g[["open", "high", "low", "close"]])
        uw, lw, br = ind.wick_ratios(g[["open", "high", "low", "close"]])
        g["upper_wick_r"] = uw
        g["lower_wick_r"] = lw
        g["body_r"] = br

        g["dist_to_high_20"] = ind.dist_to_high(close, window=20)
        g["breakout_atr_20_14"] = ind.breakout_strength_atr(
            g[["open", "high", "low", "close"]], lookback=20, atr_n=14
        )
        g["rvol_20"] = ind.rvol(vol, window=20)
        g["vol_z20"] = ind.vol_zscore(vol, window=20)

        g["rav"] = ind.range_adjusted_volume(g)
        g["rav_rel_20"] = ind.rav_relative(g, window=20)
        g["vp_div_5_5_20"] = ind.volume_price_divergence(g, price_mom=5, vol_short=5, vol_long=20)
        g["atr_ratio_14_60"] = ind.atr_ratio(g, atr_short=14, atr_long=60)

        g["boll_bw2"] = ind.bollinger_bandwidth(close, n=20, k=2.0)
        if USE_LONG_WINDOW:
            _, bw_q = ind.bollinger_bw_quantile(close, n=20, k=2.0, q_window=252)
            g["boll_bw_q252"] = bw_q
        else:
            g["boll_bw_q252"] = np.nan

        return g

    parts = []
    for _, g in out.groupby(SYM_COL, sort=False):
        parts.append(_calc_one(g))
    out2 = pd.concat(parts, axis=0).sort_values([SYM_COL, DATE_COL]).reset_index(drop=True)
    return sanitize_numeric(out2)


def add_entry_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values([SYM_COL, DATE_COL]).copy()
    out["entry"] = out.groupby(SYM_COL)["open"].shift(-1)          # 次日开盘
    out["next_date"] = out.groupby(SYM_COL)[DATE_COL].shift(-1)    # 次日日期（交易日）
    return out


# =========================
# 推理：生成某一天的预测表
# =========================
def predict_for_day(
    feat_all: pd.DataFrame,
    signal_day: pd.Timestamp,
    feature_cols: List[str],
    model_ret: xgb.Booster,
    model_mae: xgb.Booster
) -> pd.DataFrame:
    day_df = feat_all[feat_all[DATE_COL] == signal_day].copy()
    if len(day_df) == 0:
        return day_df

    # 用全历史统计填特征缺失（允许 entry 为空：实时收盘后就会这样）
    day_df = fill_features_by_train_stats(feat_all, day_df, feature_cols)

    X = day_df[feature_cols]
    dmat = xgb.DMatrix(X)

    out = day_df[[DATE_COL, SYM_COL, "entry", "next_date"]].copy()
    out["pred_ret"] = model_ret.predict(dmat)
    out["pred_mae"] = model_mae.predict(dmat)
    out["risk"] = out["pred_mae"].abs()
    out["score"] = out["pred_ret"] / (out["risk"] + SCORE_EPS)
    return out.reset_index(drop=True)


# =========================
# 策略：候选池、topcap
# =========================
def build_candidate_pool(day_pred: pd.DataFrame) -> pd.DataFrame:
    cand = day_pred[day_pred["pred_mae"] >= RISK_HARD_STOP].copy()
    if REQUIRE_POSITIVE_RET:
        cand = cand[cand["pred_ret"] > 0].copy()
    cand = cand[np.isfinite(cand["score"].values)]
    return cand


def compute_cap_and_topset(cand: pd.DataFrame, risk_budget: float) -> Tuple[int, List[str]]:
    if len(cand) == 0:
        return 0, []
    cand = cand.sort_values("score", ascending=False)
    cum = 0.0
    picked: List[str] = []
    for code, r in zip(cand[SYM_COL].astype(str).values, cand["risk"].values):
        if not np.isfinite(r) or r <= 0:
            continue
        if cum + float(r) <= risk_budget + 1e-15:
            picked.append(code)
            cum += float(r)
        else:
            break
    return len(picked), picked


def risk_parity_weights(day_pred: pd.DataFrame, codes: List[str]) -> Dict[str, float]:
    """
    给持仓分配权重：w_i ∝ 1/risk_i，归一化到 sum=1
    """
    if not codes:
        return {}
    sub = day_pred.set_index(SYM_COL).loc[codes]
    r = sub["risk"].astype(float).values
    r = np.clip(r, 1e-6, None)
    inv = 1.0 / r
    w = inv / (inv.sum() + 1e-12)
    return {c: float(wi) for c, wi in zip(codes, w)}


# =========================
# 持仓文件（实时）
# =========================
def load_positions(path: str) -> Dict[str, Position]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if len(df) == 0:
        return {}
    pos = {}
    for _, r in df.iterrows():
        pos[str(r["code"])] = Position(
            code=str(r["code"]),
            status=str(r.get("status", "OPEN")),
            entry_date=str(r["entry_date"]),
            entry_price=float(r["entry_price"]) if pd.notna(r["entry_price"]) else float("nan"),
            out_counter=int(r.get("out_counter", 0)),
            last_signal_date=str(r.get("last_signal_date", "")),
        )
    return pos


def save_positions(path: str, positions: Dict[str, Position]):
    rows = []
    for p in positions.values():
        rows.append({
            "code": p.code,
            "status": p.status,
            "entry_date": p.entry_date,
            "entry_price": p.entry_price,
            "out_counter": p.out_counter,
            "last_signal_date": p.last_signal_date,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# =========================
# 实时：收盘后出名单（不强制需要 entry）
# =========================
def run_latest_signal_and_orders(
    feat_all: pd.DataFrame,
    day_pred: pd.DataFrame,
    signal_day: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    输出：
    - latest_preds_N20.csv：最新信号日全市场预测
    - latest_orders_next_open.csv：基于最新信号日生成的明日开盘订单
    """
    positions = load_positions(POSITIONS_PATH)

    cand = build_candidate_pool(day_pred)
    cap, top_codes = compute_cap_and_topset(cand, risk_budget=RISK_BUDGET)
    top_set = set(top_codes)

    # ====== 决策：卖出 ======
    exits: List[Tuple[str, str]] = []
    for code, pos in list(positions.items()):
        row = day_pred.loc[day_pred[SYM_COL] == code]
        if len(row) == 0:
            pos.out_counter += 1
        else:
            pred_mae = float(row["pred_mae"].values[0])
            if pred_mae < RISK_HARD_STOP:
                exits.append((code, "risk_hard_stop"))
                continue

            if code in top_set:
                pos.out_counter = 0
            else:
                pos.out_counter += 1

        if pos.out_counter >= OUT_CONFIRM_DAYS:
            exits.append((code, "out_of_topcap"))

    exit_codes = set([c for c, _ in exits])
    will_hold = [c for c in positions.keys() if c not in exit_codes]

    # ====== 决策：买入（填满 topcap） ======
    slots = max(0, cap - len(will_hold))
    buys: List[str] = []
    if slots > 0:
        for code in top_codes:
            if code in will_hold or code in exit_codes:
                continue
            buys.append(code)
            if len(buys) >= slots:
                break

    execute_date = "NEXT_OPEN"
    orders = []

    # 卖单：实时模式下 price 可能未知（如果 signal_day 是最新一天，entry 可能 NaN）
    for code, reason in exits:
        row = day_pred.loc[day_pred[SYM_COL] == code]
        px = float(row["entry"].values[0]) if len(row) and pd.notna(row["entry"].values[0]) else np.nan
        orders.append({
            "signal_date": signal_day.strftime("%Y-%m-%d"),
            "execute_date": execute_date,
            "code": code,
            "side": "SELL",
            "price": px,
            "reason": reason,
            "pred_ret": float(row["pred_ret"].values[0]) if len(row) else np.nan,
            "pred_mae": float(row["pred_mae"].values[0]) if len(row) else np.nan,
            "score": float(row["score"].values[0]) if len(row) else np.nan,
        })

    # 买单
    for code in buys:
        row = day_pred.loc[day_pred[SYM_COL] == code]
        px = float(row["entry"].values[0]) if len(row) and pd.notna(row["entry"].values[0]) else np.nan
        orders.append({
            "signal_date": signal_day.strftime("%Y-%m-%d"),
            "execute_date": execute_date,
            "code": code,
            "side": "BUY",
            "price": px,
            "reason": "enter_topcap",
            "pred_ret": float(row["pred_ret"].values[0]) if len(row) else np.nan,
            "pred_mae": float(row["pred_mae"].values[0]) if len(row) else np.nan,
            "score": float(row["score"].values[0]) if len(row) else np.nan,
        })

    # ====== 更新持仓文件（实时状态） ======
    for code, _ in exits:
        if code in positions:
            del positions[code]

    for code in buys:
        if code in positions:
            continue
        row = day_pred.loc[day_pred[SYM_COL] == code]
        px = float(row["entry"].values[0]) if len(row) and pd.notna(row["entry"].values[0]) else float("nan")
        positions[code] = Position(
            code=code,
            status="PENDING_ENTRY",          # 收盘后下单，等待次日开盘补成交
            entry_date="NEXT_OPEN",
            entry_price=px,
            out_counter=0,
            last_signal_date=signal_day.strftime("%Y-%m-%d"),
        )

    save_positions(POSITIONS_PATH, positions)

    print("\n========== Latest Daily Decision ==========")
    print("signal_day:", signal_day.strftime("%Y-%m-%d"))
    print("candidate_count:", int(len(cand)))
    print("cap:", int(cap))
    print("top_codes_sample:", top_codes[:10])
    print("sell_count:", len(exits))
    print("buy_count:", len(buys))
    print("positions_after_update:", len(positions))

    return day_pred, pd.DataFrame(orders)


# =========================
# 回测：覆盖 DAILY_CSV 的日期区间，生成组合监控表
# =========================
def run_backtest_on_test_range(
    feat_all: pd.DataFrame,
    feature_cols: List[str],
    model_ret: xgb.Booster,
    model_mae: xgb.Booster,
    test_dates: List[pd.Timestamp],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    规则：
    - 信号日 t：收盘后生成 BUY/SELL 名单
    - 成交日：t 的 next_date（也就是每只股票的 shift(-1) 那天）开盘价成交
    - 每天在 t 收盘后做一次 rebalance：先根据 topcap/防抖/硬风控决定 t+1 开盘买卖
    - 组合收益用“开盘到开盘”的持有收益近似（因为交易也在开盘执行）
    """
    # 用 dict 保存回测持仓（只存 OPEN 的）
    positions: Dict[str, Dict] = {}  # code -> {"entry_open":float, "entry_dt":Timestamp, "out_counter":int}
    equity = 1.0
    peak = 1.0
    max_dd = 0.0

    trades_rows = []
    daily_rows = []
    equity_rows = []

    # 预先做一个价格表索引：用于取 open
    open_px = feat_all[[DATE_COL, SYM_COL, "open"]].copy()
    open_px[DATE_COL] = pd.to_datetime(open_px[DATE_COL])
    open_px.set_index([DATE_COL, SYM_COL], inplace=True)

    # 注意：最后一天无法在 t+1 开盘成交，所以回测只跑到倒数第二个信号日
    for i in range(len(test_dates) - 1):
        t = pd.to_datetime(test_dates[i])
        t_next = pd.to_datetime(test_dates[i + 1])  # 交易日意义上的“下一天”（来自 test_dates 序列）

        # 1) 在 t 收盘后生成预测
        day_pred = predict_for_day(feat_all, t, feature_cols, model_ret, model_mae)
        if len(day_pred) == 0:
            continue

        # 2) 候选池 + topcap
        cand = build_candidate_pool(day_pred)
        cap, top_codes = compute_cap_and_topset(cand, risk_budget=RISK_BUDGET)
        top_set = set(top_codes)

        # 3) 先决定哪些要卖（在 t_next 开盘卖）
        exits = []
        for code in list(positions.keys()):
            row = day_pred.loc[day_pred[SYM_COL] == code]
            if len(row) == 0:
                positions[code]["out_counter"] += 1
                pred_ret = np.nan; pred_mae = np.nan; score = np.nan
            else:
                pred_ret = float(row["pred_ret"].values[0])
                pred_mae = float(row["pred_mae"].values[0])
                score = float(row["score"].values[0])

                # 硬风控
                if pred_mae < RISK_HARD_STOP:
                    exits.append((code, "risk_hard_stop", pred_ret, pred_mae, score))
                    continue

                # topcap 防抖
                if code in top_set:
                    positions[code]["out_counter"] = 0
                else:
                    positions[code]["out_counter"] += 1

            if positions[code]["out_counter"] >= OUT_CONFIRM_DAYS:
                exits.append((code, "out_of_topcap", pred_ret, pred_mae, score))

        exit_codes = set([x[0] for x in exits])
        will_hold = [c for c in positions.keys() if c not in exit_codes]

        # 4) 再决定要买哪些（填满 cap）
        slots = max(0, cap - len(will_hold))
        buys = []
        if slots > 0:
            for code in top_codes:
                if code in will_hold or code in exit_codes:
                    continue
                buys.append(code)
                if len(buys) >= slots:
                    break

        # 5) 在 t_next 开盘执行交易：先卖后买
        # 5.1 卖出并结算收益
        for code, reason, pred_ret, pred_mae, score in exits:
            if code not in positions:
                continue
            try:
                sell_px = float(open_px.loc[(t_next, code), "open"])
            except Exception:
                # 没有价格则跳过（极少数缺失）
                continue

            entry_px = float(positions[code]["entry_open"])
            pnl = sell_px / (entry_px + EPS) - 1.0
            hold_days = (t_next - positions[code]["entry_dt"]).days

            trades_rows.append({
                "signal_date": t.strftime("%Y-%m-%d"),
                "execute_date": t_next.strftime("%Y-%m-%d"),
                "code": code,
                "side": "SELL",
                "execute_price": sell_px,
                "entry_price": entry_px,
                "pnl": pnl,
                "hold_days": hold_days,
                "reason": reason,
                "pred_ret": pred_ret,
                "pred_mae": pred_mae,
                "score": score,
            })

            del positions[code]

        # 5.2 买入（分配权重：risk parity）
        # 注意：这里“组合层面”的资金分配在监控表里体现，不在 trades 里做复杂仓位规模字段
        buy_weights = risk_parity_weights(day_pred, buys)

        for code in buys:
            try:
                buy_px = float(open_px.loc[(t_next, code), "open"])
            except Exception:
                continue

            row = day_pred.loc[day_pred[SYM_COL] == code]
            pred_ret = float(row["pred_ret"].values[0]) if len(row) else np.nan
            pred_mae = float(row["pred_mae"].values[0]) if len(row) else np.nan
            score = float(row["score"].values[0]) if len(row) else np.nan

            positions[code] = {
                "entry_open": buy_px,
                "entry_dt": t_next,
                "out_counter": 0,
                "weight": float(buy_weights.get(code, 0.0)),
            }

            trades_rows.append({
                "signal_date": t.strftime("%Y-%m-%d"),
                "execute_date": t_next.strftime("%Y-%m-%d"),
                "code": code,
                "side": "BUY",
                "execute_price": buy_px,
                "entry_price": buy_px,
                "pnl": 0.0,
                "hold_days": 0,
                "reason": "enter_topcap",
                "pred_ret": pred_ret,
                "pred_mae": pred_mae,
                "score": score,
            })

        # 6) 计算组合从 t_next 开盘到 (t_next 的下一开盘) 的收益，用于 equity 曲线
        # 为了严格对齐开盘成交，这里用“开盘到开盘”的组合回报：
        # return_{t_next} = sum_i w_i * (open_{t_next+1}/open_{t_next} - 1)
        # 因为我们只跑到 test_dates[-2]，所以 t_next 至少还有下一天价格
        if i + 2 < len(test_dates):
            t_next2 = pd.to_datetime(test_dates[i + 2])
            port_ret = 0.0
            wsum = 0.0
            for code, p in positions.items():
                try:
                    o1 = float(open_px.loc[(t_next, code), "open"])
                    o2 = float(open_px.loc[(t_next2, code), "open"])
                except Exception:
                    continue
                w = float(p.get("weight", 0.0))
                if w <= 0:
                    continue
                port_ret += w * (o2 / (o1 + EPS) - 1.0)
                wsum += w

            # 如果权重不满 1（例如因为缺价跳过），按实际 wsum 归一
            if wsum > 1e-12:
                port_ret = port_ret / wsum

            equity *= (1.0 + port_ret)
            peak = max(peak, equity)
            dd = equity / (peak + EPS) - 1.0
            max_dd = min(max_dd, dd)

            equity_rows.append({
                "date_open": t_next.strftime("%Y-%m-%d"),
                "next_date_open": t_next2.strftime("%Y-%m-%d"),
                "portfolio_open_to_open_ret": port_ret,
                "equity": equity,
                "drawdown": dd,
                "max_drawdown_so_far": max_dd,
                "n_positions": len(positions),
            })

        # 7) 每日监控行：记录当日信号、cap、成交计划、仓位、风险等
        daily_rows.append({
            "signal_date": t.strftime("%Y-%m-%d"),
            "execute_date": t_next.strftime("%Y-%m-%d"),
            "candidate_count": int(len(cand)),
            "cap": int(cap),
            "buy_count": int(len(buys)),
            "sell_count": int(len(exits)),
            "positions_after": int(len(positions)),
            "top_codes_sample": ",".join(top_codes[:10]),
        })

    trades_df = pd.DataFrame(trades_rows)
    daily_df = pd.DataFrame(daily_rows)
    equity_df = pd.DataFrame(equity_rows)
    return trades_df, daily_df, equity_df


# =========================
# 主入口
# =========================
def main():
    print("OUTDIR:", OUTDIR)
    print("Loading models + feature list ...")
    feature_cols = load_feature_list(FEATURE_LIST_PATH)
    model_ret = load_booster(RET_MODEL_PATH)
    model_mae = load_booster(MAE_MODEL_PATH)

    print("Reading data ...")
    hist = read_market_csv(HISTORY_CSV)
    daily = read_market_csv(DAILY_CSV)

    # test_dates = DAILY_CSV 覆盖的交易日序列（用于回测范围）
    test_dates = sorted(pd.to_datetime(daily[DATE_COL].drop_duplicates()).tolist())
    if len(test_dates) < 3:
        raise ValueError("DAILY_CSV has too few dates for backtest. Need at least 3 trading days.")

    print("Merging history + daily ...")
    merged = pd.concat([hist, daily], ignore_index=True)
    merged = merged.sort_values([SYM_COL, DATE_COL])
    merged = merged.drop_duplicates(subset=[SYM_COL, DATE_COL], keep="last").reset_index(drop=True)

    print("Computing indicators (this is the heavy step) ...")
    feat_all = add_indicator_features(merged)
    feat_all = add_entry_column(feat_all)
    feat_all = sanitize_numeric(feat_all)

    # ========= 1) 实时：最新信号日（最新日期，不强制 entry） =========
    latest_day = pd.to_datetime(feat_all[DATE_COL].max())
    latest_pred = predict_for_day(feat_all, latest_day, feature_cols, model_ret, model_mae)

    latest_pred.to_csv(PRED_LATEST_OUT, index=False)
    print("Saved latest preds:", PRED_LATEST_OUT)

    _, latest_orders = run_latest_signal_and_orders(feat_all, latest_pred, latest_day)
    latest_orders.to_csv(ORDERS_LATEST_OUT, index=False)
    print("Saved latest orders:", ORDERS_LATEST_OUT)

    # ========= 2) 回测：测试集范围监控表 =========
    print("\nRunning backtest on DAILY_CSV date range ...")
    trades_df, daily_df, equity_df = run_backtest_on_test_range(
        feat_all=feat_all,
        feature_cols=feature_cols,
        model_ret=model_ret,
        model_mae=model_mae,
        test_dates=test_dates,
    )

    trades_df.to_csv(BT_TRADES_OUT, index=False)
    daily_df.to_csv(BT_DAILY_OUT, index=False)
    equity_df.to_csv(BT_EQUITY_OUT, index=False)

    print("Saved backtest trades :", BT_TRADES_OUT)
    print("Saved daily monitor   :", BT_DAILY_OUT)
    print("Saved equity curve    :", BT_EQUITY_OUT)

    # 简单汇总
    if len(equity_df):
        final_eq = float(equity_df["equity"].iloc[-1])
        max_dd = float(equity_df["max_drawdown_so_far"].iloc[-1])
        print("\n========== Backtest Summary ==========")
        print("Final equity:", round(final_eq, 6))
        print("Max drawdown:", round(max_dd, 6))

    print("\nDone.")


if __name__ == "__main__":
    main()