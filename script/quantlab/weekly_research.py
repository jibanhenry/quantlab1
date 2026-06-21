# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .io_utils import load_market_csv_multi


WEEKLY_FEATURE_COLS = [
    "ret_1w",
    "ret_4w",
    "ret_8w",
    "ret_12w",
    "ret_20w",
    "vol_20w",
    "downside_vol_20w",
    "mdd_20w",
    "atr_pct_14w",
    "amount_20w",
    "amount_chg_4w",
    "rvol_20w",
    "turnover_20w",
    "turnover_sum_20w",
    "dist_high_20w",
    "breakout_20w_atr",
    "clv_20w",
    "bp",
    "sp",
    "market_ret_1w",
    "market_ret_20w",
    "market_vol_20w",
    "market_above_ma20",
    "momentum_score_w",
    "lowvol_score_w",
    "liquidity_score_w",
    "value_score_w",
    "quality_price_score_w",
    "classic_factor_score",
]

DIAGNOSTIC_SCORE_COLS = [
    "classic_factor_score",
    "pure_momentum_score",
    "pure_lowvol_score",
    "value_score_w",
    "liquidity_score_w",
]


def _ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def _write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _safe_numeric(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _rank_pct_by_date(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    vals = pd.to_numeric(df[col], errors="coerce")
    return vals.groupby(df["date"]).rank(pct=True, method="average", ascending=ascending) * 100.0


def _rolling_mdd(close: pd.Series, window: int) -> pd.Series:
    peak = close.rolling(window, min_periods=window).max()
    dd = close / peak.replace(0, np.nan) - 1.0
    return dd.rolling(window, min_periods=window).min()


def _infer_next_open_limit_flags(weekly: pd.DataFrame) -> pd.DataFrame:
    out = weekly.sort_values(["code", "date"]).copy()
    g = out.groupby("code", sort=False)
    out["next_date"] = g["date"].shift(-1)
    out["next_date_2"] = g["date"].shift(-2)
    out["next_date_6"] = g["date"].shift(-6)
    out["next_open"] = g["open"].shift(-1)
    out["next_open_2"] = g["open"].shift(-2)
    out["next_open_6"] = g["open"].shift(-6)
    return out


def aggregate_daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized daily A-share panel into per-stock weekly bars."""
    need = ["code", "date", "open", "high", "low", "close", "volume", "amount", "turnover", "pb_mrq", "ps_ttm", "pct_chg"]
    work = df[[c for c in need if c in df.columns]].copy()
    work["code"] = work["code"].astype(str).str.split(".").str[-1].str.zfill(6)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["code", "date", "open", "high", "low", "close"]).sort_values(["code", "date"])
    for col in ["open", "high", "low", "close", "volume", "amount", "turnover", "pb_mrq", "ps_ttm", "pct_chg"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["week"] = work["date"].dt.to_period("W-FRI")
    agg = {
        "date": "last",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "amount": "sum",
        "turnover": "mean",
        "pb_mrq": "last",
        "ps_ttm": "last",
        "pct_chg": "sum",
    }
    agg = {k: v for k, v in agg.items() if k in work.columns}
    weekly = work.groupby(["code", "week"], sort=False).agg(agg).reset_index(drop=False)
    if "turnover" in work.columns:
        turnover_sum = work.groupby(["code", "week"], sort=False)["turnover"].sum().reset_index(name="turnover_sum")
        weekly = weekly.merge(turnover_sum, on=["code", "week"], how="left")
    weekly = weekly.drop(columns=["week"]).sort_values(["code", "date"]).reset_index(drop=True)
    weekly["list_age_weeks"] = weekly.groupby("code", sort=False).cumcount() + 1
    return weekly


def build_weekly_feature_panel(daily_df: pd.DataFrame, min_amount_20w: float = 20_000_000.0) -> pd.DataFrame:
    weekly = aggregate_daily_to_weekly(daily_df)
    weekly = weekly.sort_values(["code", "date"]).reset_index(drop=True)
    g = weekly.groupby("code", group_keys=False, sort=False)
    close = _safe_numeric(weekly, "close")
    amount = _safe_numeric(weekly, "amount")
    turnover = _safe_numeric(weekly, "turnover")
    ret1 = g["close"].pct_change(fill_method=None)

    weekly["ret_1w"] = ret1
    for n in [4, 8, 12, 20]:
        weekly[f"ret_{n}w"] = g["close"].pct_change(n, fill_method=None)
    weekly["vol_20w"] = ret1.groupby(weekly["code"]).rolling(20, min_periods=16).std().reset_index(level=0, drop=True)
    weekly["downside_vol_20w"] = ret1.where(ret1 < 0, 0.0).groupby(weekly["code"]).rolling(20, min_periods=16).std().reset_index(level=0, drop=True)
    weekly["mdd_20w"] = g["close"].transform(lambda s: _rolling_mdd(s, 20))
    prev_close = g["close"].shift(1)
    tr = pd.concat(
        [
            weekly["high"] - weekly["low"],
            (weekly["high"] - prev_close).abs(),
            (prev_close - weekly["low"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    weekly["atr_14w"] = tr.groupby(weekly["code"]).rolling(14, min_periods=10).mean().reset_index(level=0, drop=True)
    weekly["atr_pct_14w"] = weekly["atr_14w"] / close.replace(0, np.nan)
    weekly["amount_20w"] = amount.groupby(weekly["code"]).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    weekly["amount_4w"] = amount.groupby(weekly["code"]).rolling(4, min_periods=3).mean().reset_index(level=0, drop=True)
    weekly["amount_chg_4w"] = weekly["amount_4w"] / weekly["amount_20w"].replace(0, np.nan) - 1.0
    weekly["rvol_20w"] = amount / weekly["amount_20w"].replace(0, np.nan)
    weekly["turnover_20w"] = turnover.groupby(weekly["code"]).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    weekly["turnover_sum_20w"] = turnover.groupby(weekly["code"]).rolling(20, min_periods=10).sum().reset_index(level=0, drop=True)
    high20 = g["high"].transform(lambda s: s.shift(1).rolling(20, min_periods=16).max())
    weekly["dist_high_20w"] = close / high20.replace(0, np.nan) - 1.0
    weekly["breakout_20w_atr"] = (close - high20) / weekly["atr_14w"].replace(0, np.nan)
    weekly["clv_20w"] = ((close - weekly["low"]) - (weekly["high"] - close)) / (weekly["high"] - weekly["low"]).replace(0, np.nan)
    weekly["bp"] = 1.0 / _safe_numeric(weekly, "pb_mrq").where(_safe_numeric(weekly, "pb_mrq") > 0)
    weekly["sp"] = 1.0 / _safe_numeric(weekly, "ps_ttm").where(_safe_numeric(weekly, "ps_ttm") > 0)

    weekly["fwd_ret_5w"] = g["close"].shift(-5) / close - 1.0
    weekly["fwd_ret_1w_open"] = g["open"].shift(-2) / g["open"].shift(-1) - 1.0
    weekly["fwd_ret_5w_open"] = g["open"].shift(-6) / g["open"].shift(-1) - 1.0

    weekly = add_market_regime_features(weekly)
    weekly = add_weekly_scores_and_labels(weekly, min_amount_20w=min_amount_20w)
    weekly = _infer_next_open_limit_flags(weekly)
    return weekly


def add_market_regime_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    market = (
        out.groupby("date", as_index=False)
        .agg(market_ret_1w=("ret_1w", "mean"), market_close=("close", "mean"))
        .sort_values("date")
    )
    market["market_ret_20w"] = market["market_close"].pct_change(20, fill_method=None)
    market["market_vol_20w"] = market["market_ret_1w"].rolling(20, min_periods=16).std()
    market["market_ma20"] = market["market_close"].rolling(20, min_periods=16).mean()
    market["market_above_ma20"] = (market["market_close"] > market["market_ma20"]).astype(float)
    return out.merge(market[["date", "market_ret_1w", "market_ret_20w", "market_vol_20w", "market_above_ma20"]], on="date", how="left")


def add_weekly_scores_and_labels(panel: pd.DataFrame, min_amount_20w: float) -> pd.DataFrame:
    out = panel.copy()
    out["is_limit_up_like"] = _safe_numeric(out, "pct_chg") >= 9.5
    out["is_limit_down_like"] = _safe_numeric(out, "pct_chg") <= -9.5
    out["tradable"] = (
        out["list_age_weeks"].ge(26)
        & _safe_numeric(out, "amount_20w").ge(float(min_amount_20w))
        & _safe_numeric(out, "open").notna()
        & _safe_numeric(out, "close").notna()
        & ~out["code"].astype(str).str.startswith("399")
        & ~out["is_limit_up_like"].fillna(False)
        & ~out["is_limit_down_like"].fillna(False)
    )
    out["ret_4w_score"] = _rank_pct_by_date(out, "ret_4w")
    out["ret_12w_score"] = _rank_pct_by_date(out, "ret_12w")
    out["ret_20w_score"] = _rank_pct_by_date(out, "ret_20w")
    out["breakout_score"] = _rank_pct_by_date(out, "breakout_20w_atr")
    out["low_vol_score"] = _rank_pct_by_date(out, "vol_20w", ascending=False)
    out["low_downside_score"] = _rank_pct_by_date(out, "downside_vol_20w", ascending=False)
    out["low_mdd_score"] = _rank_pct_by_date(out, "mdd_20w")
    out["low_atr_score"] = _rank_pct_by_date(out, "atr_pct_14w", ascending=False)
    out["amount_score"] = _rank_pct_by_date(out, "amount_20w")
    out["turnover_score"] = _rank_pct_by_date(out, "turnover_20w")
    out["bp_score"] = _rank_pct_by_date(out, "bp")
    out["sp_score"] = _rank_pct_by_date(out, "sp")
    out["clv_score"] = _rank_pct_by_date(out, "clv_20w")
    out["rvol_quality_score"] = _rank_pct_by_date(out, "rvol_20w")
    out["momentum_score_w"] = out[["ret_4w_score", "ret_12w_score", "ret_20w_score", "breakout_score"]].mean(axis=1)
    out["lowvol_score_w"] = out[["low_vol_score", "low_downside_score", "low_mdd_score", "low_atr_score"]].mean(axis=1)
    out["liquidity_score_w"] = out[["amount_score", "turnover_score"]].mean(axis=1)
    out["value_score_w"] = out[["bp_score", "sp_score"]].mean(axis=1)
    out["quality_price_score_w"] = out[["clv_score", "rvol_quality_score"]].mean(axis=1)
    out["pure_momentum_score"] = out["momentum_score_w"]
    out["pure_lowvol_score"] = out["lowvol_score_w"]
    out["classic_factor_score"] = (
        out["momentum_score_w"] * 0.30
        + out["lowvol_score_w"] * 0.25
        + out["quality_price_score_w"] * 0.15
        + out["value_score_w"] * 0.15
        + out["liquidity_score_w"] * 0.15
    )

    def _label_date(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        valid = g["tradable"].fillna(False) & pd.to_numeric(g["fwd_ret_5w"], errors="coerce").notna()
        if int(valid.sum()) < 80:
            g["y_cls"] = np.nan
            return g
        fwd = pd.to_numeric(g.loc[valid, "fwd_ret_5w"], errors="coerce")
        up = float(fwd.quantile(0.80))
        dn = float(fwd.quantile(0.20))
        y = pd.Series(np.nan, index=g.index, dtype=float)
        y.loc[valid & (pd.to_numeric(g["fwd_ret_5w"], errors="coerce") >= up)] = 2.0
        y.loc[valid & (pd.to_numeric(g["fwd_ret_5w"], errors="coerce") <= dn)] = 0.0
        y.loc[valid & y.isna()] = 1.0
        g["y_cls"] = y
        return g

    labeled = [_label_date(sub) for _, sub in out.groupby("date", sort=False)]
    return pd.concat(labeled, ignore_index=True) if labeled else out


def generate_week_windows(dates: Iterable[pd.Timestamp], train_weeks: int, val_weeks: int, test_weeks: int, step_weeks: int) -> List[Tuple[int, List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]]]:
    uniq = sorted(pd.to_datetime(pd.Series(list(dates))).dropna().unique())
    windows = []
    total = train_weeks + val_weeks + test_weeks
    start = 0
    wid = 1
    while start + total <= len(uniq):
        train = list(uniq[start : start + train_weeks])
        val = list(uniq[start + train_weeks : start + train_weeks + val_weeks])
        test = list(uniq[start + train_weeks + val_weeks : start + total])
        windows.append((wid, train, val, test))
        wid += 1
        start += step_weeks
    return windows


def _model_frame(panel: pd.DataFrame, dates: List[pd.Timestamp], feature_cols: List[str]) -> pd.DataFrame:
    sub = panel[panel["date"].isin(dates) & panel["tradable"].fillna(False) & panel["y_cls"].notna()].copy()
    sub = sub.dropna(subset=["fwd_ret_5w"])
    keep = ["date", "code", "y_cls", "fwd_ret_5w", "fwd_ret_1w_open", "next_open", "next_open_2"] + feature_cols
    keep = [c for c in keep if c in sub.columns]
    return sub[keep].replace([np.inf, -np.inf], np.nan)


def _build_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=0.05,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=42,
    )


def _build_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, C=0.3, class_weight="balanced", random_state=42)),
        ]
    )


def _score_with_model(model, data: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = data[["date", "code", "fwd_ret_5w", "fwd_ret_1w_open"]].copy()
    X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    probs = model.predict_proba(X)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf"), "classes_"):
        classes = list(model.named_steps["clf"].classes_)
    else:
        classes = [0, 1, 2]
    up_idx = classes.index(2.0) if 2.0 in classes else classes.index(2) if 2 in classes else int(np.argmax(classes))
    dn_idx = classes.index(0.0) if 0.0 in classes else classes.index(0) if 0 in classes else int(np.argmin(classes))
    out["model_score"] = probs[:, up_idx] - probs[:, dn_idx]
    out["p_up"] = probs[:, up_idx]
    out["p_down"] = probs[:, dn_idx]
    return out


def _evaluate_score(data: pd.DataFrame, score_col: str, ret_col: str = "fwd_ret_5w", top_frac: float = 0.10) -> Dict[str, float]:
    rows = []
    for dt, sub in data.groupby("date", sort=False):
        score = pd.to_numeric(sub[score_col], errors="coerce")
        ret = pd.to_numeric(sub[ret_col], errors="coerce")
        m = score.notna() & ret.notna()
        if int(m.sum()) < 50:
            continue
        tmp = sub.loc[m, [score_col, ret_col]].sort_values(score_col, ascending=False)
        k = max(int(math.floor(len(tmp) * top_frac)), 1)
        top = tmp.head(k)
        bottom = tmp.tail(k)
        rank_ic = tmp[score_col].rank(method="average").corr(tmp[ret_col].rank(method="average"))
        rows.append(
            {
                "date": dt,
                "top_ret": float(top[ret_col].mean()),
                "bottom_ret": float(bottom[ret_col].mean()),
                "spread": float(top[ret_col].mean() - bottom[ret_col].mean()),
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                "n": int(len(tmp)),
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return {"days": 0, "top_ret": np.nan, "bottom_ret": np.nan, "spread": np.nan, "rank_ic": np.nan, "positive_spread_rate": np.nan}
    return {
        "days": int(len(daily)),
        "avg_n": float(daily["n"].mean()),
        "top_ret": float(daily["top_ret"].mean()),
        "bottom_ret": float(daily["bottom_ret"].mean()),
        "spread": float(daily["spread"].mean()),
        "rank_ic": float(daily["rank_ic"].mean()),
        "positive_spread_rate": float((daily["spread"] > 0).mean()),
    }


def train_and_score_walkforward(
    panel: pd.DataFrame,
    feature_cols: List[str],
    train_weeks: int,
    val_weeks: int,
    test_weeks: int,
    step_weeks: int,
    max_train_rows: int = 300_000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    windows = generate_week_windows(panel["date"].drop_duplicates(), train_weeks, val_weeks, test_weeks, step_weeks)
    score_frames = []
    metrics = []
    for window_id, train_dates, val_dates, test_dates in windows:
        train_df = _model_frame(panel, train_dates, feature_cols)
        val_df = _model_frame(panel, val_dates, feature_cols)
        test_df = _model_frame(panel, test_dates, feature_cols)
        if train_df.empty or val_df.empty or test_df.empty:
            continue
        if max_train_rows and len(train_df) > max_train_rows:
            train_df = train_df.sample(n=int(max_train_rows), random_state=42).sort_values(["date", "code"])
        candidates = {}

        hgb = _build_hgb()
        hgb.fit(train_df[feature_cols], train_df["y_cls"].astype(int))
        hgb_val = _score_with_model(hgb, val_df, feature_cols)
        candidates["hgb"] = (hgb, _evaluate_score(hgb_val, "model_score"))

        logit = _build_logistic()
        logit.fit(train_df[feature_cols], train_df["y_cls"].astype(int))
        logit_val = _score_with_model(logit, val_df, feature_cols)
        candidates["logistic"] = (logit, _evaluate_score(logit_val, "model_score"))

        classic_val = val_df[["date", "code", "fwd_ret_5w", "fwd_ret_1w_open", "classic_factor_score"]].rename(columns={"classic_factor_score": "model_score"})
        candidates["classic_factor"] = (None, _evaluate_score(classic_val, "model_score"))
        selected_name = max(candidates.keys(), key=lambda k: (-np.inf if pd.isna(candidates[k][1].get("spread")) else candidates[k][1].get("spread")))

        train_val = pd.concat([train_df, val_df], ignore_index=True)
        if max_train_rows and len(train_val) > max_train_rows:
            train_val = train_val.sample(n=int(max_train_rows), random_state=100 + window_id).sort_values(["date", "code"])
        if selected_name == "hgb":
            final_model = _build_hgb()
            final_model.fit(train_val[feature_cols], train_val["y_cls"].astype(int))
            test_score = _score_with_model(final_model, test_df, feature_cols)
        elif selected_name == "logistic":
            final_model = _build_logistic()
            final_model.fit(train_val[feature_cols], train_val["y_cls"].astype(int))
            test_score = _score_with_model(final_model, test_df, feature_cols)
        else:
            test_score = test_df[["date", "code", "fwd_ret_5w", "fwd_ret_1w_open", "classic_factor_score"]].rename(columns={"classic_factor_score": "model_score"})
            test_score["p_up"] = np.nan
            test_score["p_down"] = np.nan
        test_eval = _evaluate_score(test_score, "model_score")
        test_score["window_id"] = window_id
        test_score["selected_model"] = selected_name
        score_frames.append(test_score)

        row = {
            "window_id": window_id,
            "train_start": min(train_dates),
            "train_end": max(train_dates),
            "val_start": min(val_dates),
            "val_end": max(val_dates),
            "test_start": min(test_dates),
            "test_end": max(test_dates),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "selected_model": selected_name,
        }
        for name, (_, val_eval) in candidates.items():
            row[f"val_{name}_spread"] = val_eval.get("spread")
            row[f"val_{name}_rank_ic"] = val_eval.get("rank_ic")
        for key, value in test_eval.items():
            row[f"test_{key}"] = value
        metrics.append(row)
    scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    return scores, pd.DataFrame(metrics)


def score_latest_with_final_model(panel: pd.DataFrame, feature_cols: List[str], max_train_rows: int = 300_000, top_n: int = 10) -> pd.DataFrame:
    train_df = panel[panel["tradable"].fillna(False) & panel["y_cls"].notna()].copy()
    latest_date = panel["date"].max()
    latest = panel[panel["date"].eq(latest_date) & panel["tradable"].fillna(False)].copy()
    if train_df.empty or latest.empty:
        return pd.DataFrame()
    train_df = train_df.dropna(subset=["fwd_ret_5w"])
    if max_train_rows and len(train_df) > max_train_rows:
        train_df = train_df.sample(n=int(max_train_rows), random_state=2026).sort_values(["date", "code"])
    model = _build_hgb()
    model.fit(train_df[feature_cols], train_df["y_cls"].astype(int))
    scored = _score_with_model(model, latest, feature_cols)
    scored = latest.drop(columns=["model_score", "p_up", "p_down"], errors="ignore").merge(
        scored[["date", "code", "model_score", "p_up", "p_down"]],
        on=["date", "code"],
        how="left",
    )
    scored["strategy"] = "weekly_model_latest"
    scored["signal_date"] = latest_date
    return scored.sort_values(["model_score", "amount_20w"], ascending=[False, False]).head(top_n)


def factor_ic_report(panel: pd.DataFrame, score_cols: List[str], ret_col: str = "fwd_ret_5w") -> pd.DataFrame:
    rows = []
    work = panel[panel["tradable"].fillna(False)].copy()
    for score_col in score_cols:
        daily = []
        if score_col not in work.columns:
            continue
        for dt, sub in work.groupby("date", sort=False):
            x = pd.to_numeric(sub[score_col], errors="coerce")
            y = pd.to_numeric(sub[ret_col], errors="coerce")
            m = x.notna() & y.notna()
            if int(m.sum()) < 50:
                continue
            ic = x[m].rank(method="average").corr(y[m].rank(method="average"))
            daily.append({"date": dt, "rank_ic": float(ic) if pd.notna(ic) else np.nan, "n": int(m.sum())})
        d = pd.DataFrame(daily).dropna(subset=["rank_ic"])
        if d.empty:
            rows.append({"factor": score_col, "days": 0, "rank_ic_mean": np.nan, "rank_ic_ir": np.nan, "positive_rate": np.nan})
            continue
        std = float(d["rank_ic"].std(ddof=1))
        rows.append(
            {
                "factor": score_col,
                "days": int(len(d)),
                "avg_n": float(d["n"].mean()),
                "rank_ic_mean": float(d["rank_ic"].mean()),
                "rank_ic_ir": float(d["rank_ic"].mean() / std * math.sqrt(52)) if std > 0 else np.nan,
                "positive_rate": float((d["rank_ic"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def layer_return_report(panel: pd.DataFrame, score_cols: List[str], ret_col: str = "fwd_ret_5w", bins: int = 5) -> pd.DataFrame:
    rows = []
    work = panel[panel["tradable"].fillna(False)].copy()
    for score_col in score_cols:
        if score_col not in work.columns:
            continue
        for dt, sub in work.groupby("date", sort=False):
            x = pd.to_numeric(sub[score_col], errors="coerce")
            y = pd.to_numeric(sub[ret_col], errors="coerce")
            m = x.notna() & y.notna()
            if int(m.sum()) < bins * 20:
                continue
            try:
                layer = pd.qcut(x[m].rank(method="first"), bins, labels=False) + 1
            except Exception:
                continue
            tmp = pd.DataFrame({"layer": layer.astype(int), "ret": y[m].values})
            for layer_id, value in tmp.groupby("layer")["ret"].mean().items():
                rows.append({"date": dt, "factor": score_col, "layer": int(layer_id), "mean_return": float(value)})
    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(columns=["factor", "layer", "mean_return", "days"])
    out = raw.groupby(["factor", "layer"])["mean_return"].agg(["mean", "count"]).reset_index()
    out = out.rename(columns={"mean": "mean_return", "count": "days"})
    spreads = []
    for factor, sub in out.groupby("factor"):
        top = sub[sub["layer"] == bins]["mean_return"]
        bottom = sub[sub["layer"] == 1]["mean_return"]
        if not top.empty and not bottom.empty:
            spreads.append({"factor": factor, "layer": "top_minus_bottom", "mean_return": float(top.iloc[0] - bottom.iloc[0]), "days": int(sub["days"].max())})
    return pd.concat([out, pd.DataFrame(spreads)], ignore_index=True)


def simulate_weekly_portfolio(
    panel: pd.DataFrame,
    score_col: str,
    top_n: int,
    min_score_quantile: float = 0.70,
    hold_weeks: int = 5,
    total_exposure: float = 0.45,
    cost_bp: float = 2.0,
    label: str = "model_top3",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = panel.sort_values(["date", "code"]).copy()
    exit_open_col = f"next_open_{hold_weeks + 1}"
    exit_date_col = f"next_date_{hold_weeks + 1}"
    if hold_weeks != 5 or exit_open_col not in work.columns or exit_date_col not in work.columns:
        exit_open_col = "next_open_6"
        exit_date_col = "next_date_6"
    work = work[
        work["tradable"].fillna(False)
        & pd.to_numeric(work[score_col], errors="coerce").notna()
        & pd.to_numeric(work["next_open"], errors="coerce").notna()
        & pd.to_numeric(work[exit_open_col], errors="coerce").notna()
    ].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    work["rank_today"] = work.groupby("date")[score_col].rank(method="first", ascending=False)
    work["score_quantile"] = work.groupby("date")[score_col].rank(pct=True, method="average")
    trades = []
    equity_rows = []
    weekly_picks = []
    equity = 1.0
    cost = cost_bp / 10000.0
    cohort_exposure = total_exposure / max(int(hold_weeks), 1)
    weight = cohort_exposure / max(top_n, 1)
    for dt, day in work.groupby("date", sort=False):
        candidates = day[day["score_quantile"].ge(float(min_score_quantile))].sort_values([score_col, "amount_20w"], ascending=[False, False])
        if candidates.empty:
            equity_rows.append({"date": dt, "strategy": label, "equity": float(equity), "positions": 0, "cash": float(equity)})
            continue
        top = candidates.head(top_n).copy()
        weekly_picks.append(top.assign(strategy=label, signal_date=dt))
        realized = []
        for _, row in top.iterrows():
            entry_px = float(pd.to_numeric(row.get("next_open"), errors="coerce"))
            exit_px = float(pd.to_numeric(row.get(exit_open_col), errors="coerce"))
            if not np.isfinite(entry_px) or not np.isfinite(exit_px):
                continue
            pnl = exit_px / entry_px - 1.0 - 2 * cost
            realized.append(pnl)
            trades.append(
                {
                    "strategy": label,
                    "code": str(row["code"]),
                    "entry_signal_date": dt,
                    "entry_date": row.get("next_date"),
                    "entry_price": entry_px,
                    "exit_signal_date": dt,
                    "exit_date": row.get(exit_date_col),
                    "exit_price": exit_px,
                    "pnl_pct": float(pnl),
                    "weeks_held": int(hold_weeks),
                    "entry_score": float(pd.to_numeric(row.get(score_col), errors="coerce")),
                    "exit_score": np.nan,
                    "weight": weight,
                    "exit_reason": "weekly_rebalance",
                }
            )
        if realized:
            basket_ret = float(np.mean(realized))
            equity *= 1.0 + cohort_exposure * basket_ret
        exit_date = pd.to_datetime(top[exit_date_col].dropna().max()) if top[exit_date_col].notna().any() else pd.Timestamp(dt)
        equity_rows.append({"date": exit_date, "strategy": label, "equity": float(equity), "positions": int(len(realized)), "cash": float(equity * (1.0 - total_exposure))})

    latest_date = work["date"].max()
    latest = work[work["date"].eq(latest_date)].sort_values([score_col, "amount_20w"], ascending=[False, False]).head(top_n).copy()
    latest["strategy"] = label
    latest["signal_date"] = latest_date
    picks = pd.concat(weekly_picks, ignore_index=True) if weekly_picks else pd.DataFrame()
    return pd.DataFrame(trades), pd.DataFrame(equity_rows), latest


def _portfolio_metrics(trades: pd.DataFrame, equity: pd.DataFrame, strategy: str) -> Dict[str, float]:
    if equity is None or equity.empty:
        return {"strategy": strategy, "annual_return": np.nan, "max_drawdown": np.nan, "return_drawdown": np.nan, "win_rate": np.nan, "trade_count": 0}
    eq = equity.sort_values("date").copy()
    eq["date"] = pd.to_datetime(eq["date"], errors="coerce")
    start, end = eq["date"].min(), eq["date"].max()
    years = max((end - start).days / 365.25, 1e-9)
    final_eq = float(eq["equity"].iloc[-1])
    ann = final_eq ** (1.0 / years) - 1.0 if final_eq > 0 else np.nan
    peak = eq["equity"].cummax()
    dd = eq["equity"] / peak.replace(0, np.nan) - 1.0
    max_dd = float(dd.min()) if len(dd) else np.nan
    pnl = pd.to_numeric(trades.get("pnl_pct"), errors="coerce") if trades is not None and not trades.empty else pd.Series(dtype=float)
    return {
        "strategy": strategy,
        "start": start,
        "end": end,
        "final_equity": final_eq,
        "annual_return": float(ann),
        "max_drawdown": max_dd,
        "return_drawdown": float(ann / abs(max_dd)) if max_dd and max_dd < 0 else np.nan,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "expectancy": float(pnl.mean()) if len(pnl) else np.nan,
        "trade_count": int(len(pnl)),
        "avg_hold_weeks": float(pd.to_numeric(trades.get("weeks_held"), errors="coerce").mean()) if trades is not None and not trades.empty else np.nan,
    }


def run_weekly_research(
    csvs: List[str],
    outdir: str,
    train_weeks: int = 156,
    val_weeks: int = 26,
    test_weeks: int = 26,
    step_weeks: int = 13,
    min_amount_20w: float = 20_000_000.0,
    max_train_rows: int = 300_000,
    save_panel: bool = False,
) -> Dict[str, pd.DataFrame]:
    _ensure_outdir(outdir)
    daily = load_market_csv_multi(csvs)
    weekly = build_weekly_feature_panel(daily, min_amount_20w=min_amount_20w)
    feature_cols = [c for c in WEEKLY_FEATURE_COLS if c in weekly.columns]
    score_cols = [c for c in DIAGNOSTIC_SCORE_COLS if c in weekly.columns]

    ic = factor_ic_report(weekly, score_cols)
    layers = layer_return_report(weekly, score_cols)
    model_scores, model_metrics = train_and_score_walkforward(
        weekly,
        feature_cols=feature_cols,
        train_weeks=train_weeks,
        val_weeks=val_weeks,
        test_weeks=test_weeks,
        step_weeks=step_weeks,
        max_train_rows=max_train_rows,
    )
    panel_for_bt = weekly.merge(
        model_scores[["date", "code", "model_score", "p_up", "p_down", "window_id", "selected_model"]],
        on=["date", "code"],
        how="left",
    )
    model_ic = factor_ic_report(panel_for_bt[panel_for_bt["model_score"].notna()].copy(), ["model_score"])
    if not model_ic.empty:
        ic = pd.concat([ic, model_ic.assign(factor="model_score")], ignore_index=True)
    model_layers = layer_return_report(panel_for_bt[panel_for_bt["model_score"].notna()].copy(), ["model_score"])
    if not model_layers.empty:
        layers = pd.concat([layers, model_layers], ignore_index=True)

    strategies = [
        ("weekly_model_top3", "model_score", 3),
        ("weekly_model_top5", "model_score", 5),
        ("classic_factor_top3", "classic_factor_score", 3),
        ("pure_momentum_top3", "pure_momentum_score", 3),
        ("pure_lowvol_top3", "pure_lowvol_score", 3),
    ]
    oos_dates = set(pd.to_datetime(model_scores["date"]).dropna().unique()) if not model_scores.empty else set()
    all_trades = []
    all_equity = []
    latest_frames = []
    summary_rows = []
    for name, score_col, top_n in strategies:
        bt_panel = panel_for_bt[panel_for_bt[score_col].notna()].copy()
        if oos_dates:
            bt_panel = bt_panel[bt_panel["date"].isin(oos_dates)].copy()
        if score_col == "model_score":
            bt_panel = bt_panel[bt_panel["window_id"].notna()].copy()
        trades, equity, latest = simulate_weekly_portfolio(bt_panel, score_col=score_col, top_n=top_n, label=name)
        all_trades.append(trades)
        all_equity.append(equity)
        latest_frames.append(latest)
        summary_rows.append(_portfolio_metrics(trades, equity, name))

    latest_model = score_latest_with_final_model(weekly, feature_cols, max_train_rows=max_train_rows, top_n=10)
    if not latest_model.empty:
        latest_frames.append(latest_model)

    trades_out = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_out = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    latest_out = pd.concat(latest_frames, ignore_index=True) if latest_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    _write_csv(model_metrics, os.path.join(outdir, "weekly_model_metrics.csv"))
    _write_csv(ic, os.path.join(outdir, "weekly_factor_ic_report.csv"))
    _write_csv(layers, os.path.join(outdir, "weekly_layer_return_report.csv"))
    _write_csv(model_metrics.copy(), os.path.join(outdir, "weekly_walkforward_metrics.csv"))
    _write_csv(equity_out, os.path.join(outdir, "weekly_equity_curve.csv"))
    _write_csv(latest_out, os.path.join(outdir, "weekly_latest_candidates.csv"))
    _write_csv(summary, os.path.join(outdir, "weekly_strategy_summary.csv"))
    _write_csv(trades_out, os.path.join(outdir, "weekly_trades.csv"))
    if save_panel:
        _write_csv(panel_for_bt, os.path.join(outdir, "weekly_signal_panel.csv"))

    return {
        "weekly_panel": panel_for_bt,
        "weekly_model_metrics": model_metrics,
        "weekly_factor_ic_report": ic,
        "weekly_layer_return_report": layers,
        "weekly_walkforward_metrics": model_metrics,
        "weekly_equity_curve": equity_out,
        "weekly_latest_candidates": latest_out,
        "weekly_strategy_summary": summary,
        "weekly_trades": trades_out,
    }
