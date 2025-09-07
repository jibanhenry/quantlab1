# -*- coding: utf-8 -*-
"""
One-Pass Streaming Multi-Target Window Similarity (Interactive, English UI)
- Overlay (top): shows target window (solid) + target future (dotted, with connecting segment),
                 and ALL matches window (dashed) + ALL matches future (dotted, with connecting segment).
  => Overlay is NOT interactive; click only controls bottom-right candles.
- Bottom-left: target candles (window + future), trading days, colored by up/down (red/green).
- Bottom-right: selected match candles (window + future), trading days, colored by up/down (red/green).
- Window & future both normalized by SAME base0 (first close of the window).
- One figure per target (not mixed). Robust CSV mapping & code normalization.

Dependencies:
  pip install numpy pandas matplotlib tqdm
"""

from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import pandas as pd
import heapq
import os
import re
from tqdm import tqdm

# ========= Ensure interactive backend BEFORE pyplot =========
import sys
import matplotlib
def _ensure_interactive_backend():
    try:
        current = matplotlib.get_backend().lower()
    except Exception:
        current = ""
    if any(k in current for k in ("agg", "inline", "interagg", "nbagg")):
        try:
            if sys.platform == "darwin":
                matplotlib.use("MacOSX", force=True)
            else:
                try:
                    matplotlib.use("Qt5Agg", force=True)
                except Exception:
                    matplotlib.use("TkAgg", force=True)
        except Exception:
            pass

_ensure_interactive_backend()
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---- pandas display ----
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

# =========================
# Utilities
# =========================
def _auto_pick(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    for c in candidates:
        for real in cols:
            if c.lower() in real.lower():
                return real
    return None

def _norm_code(x: Union[str, int, float]) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    s = s.replace(".SH", "").replace(".SZ", "").replace("SH", "").replace("SZ", "")
    s = re.sub(r"\D", "", s)
    if not s:
        return None
    return s.zfill(6)

def load_and_concat_csvs(
    file_paths: List[str],
    column_map: Dict[str, Optional[str]] = None,
    auto_detect: bool = True
) -> pd.DataFrame:
    print(f"[Step 1/3] Reading {len(file_paths)} CSVs and concatenating…")
    dfs = []
    for p in tqdm(file_paths, desc="Read CSVs"):
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")
        df = pd.read_csv(p)
        dfs.append(df)
    raw = pd.concat(dfs, axis=0, ignore_index=True)
    raw.columns = [str(c).strip() for c in raw.columns]
    print(f"  ✓ Concat done. Initial rows: {len(raw):,}")

    if column_map is None:
        column_map = {}

    def pick(name_in_map: str, candidates: List[str]) -> Optional[str]:
        v = (column_map.get(name_in_map) if column_map else None)
        if v:
            v_clean = str(v).strip()
            if v_clean in raw.columns:
                return v_clean
            got = _auto_pick(list(raw.columns), [v_clean] + candidates)
            if got:
                print(f"  · Hint: column_map['{name_in_map}']='{v}' not found; fallback to '{got}'")
                return got
            return None
        return _auto_pick(list(raw.columns), candidates) if auto_detect else None

    code_col     = pick("code",     ["code","ts_code","symbol","证券代码","股票代码","代码"])
    date_col     = pick("date",     ["date","trade_date","交易日期","日期"])
    open_col     = pick("open",     ["open","开盘"])
    high_col     = pick("high",     ["high","最高"])
    low_col      = pick("low",      ["low","最低"])
    close_col    = pick("close",    ["close","收盘","收盘价"])
    preclose_col = pick("preclose", ["preclose","pre_close","昨收","前收"])
    turnover_col = pick("turnover", ["turnover","turnover_rate","换手率","换手"])
    pct_chg_col  = pick("pct_chg",  ["pct_chg","涨跌幅","涨幅"])

    required = [code_col, date_col, open_col, high_col, low_col, close_col, preclose_col]
    req_names = ["code","date","open","high","low","close","preclose"]
    if any(x is None for x in required):
        missing = [n for x,n in zip(required, req_names) if x is None]
        preview_cols = ", ".join(list(raw.columns)[:15]) + ("..." if len(raw.columns) > 15 else "")
        raise ValueError(f"Missing required columns (before rename): {missing}; columns detected: [{preview_cols}]")

    out = raw.rename(columns={
        code_col: "code",
        date_col: "date",
        open_col: "open",
        high_col: "high",
        low_col: "low",
        close_col: "close",
        preclose_col: "preclose",
        **({turnover_col: "turnover"} if turnover_col else {}),
        **({pct_chg_col: "pct_chg"} if pct_chg_col else {})
    }).copy()

    required_after = ["code","date","open","high","low","close","preclose"]
    missing_after = [c for c in required_after if c not in out.columns]
    if missing_after:
        raise ValueError(
            f"Missing columns after rename: {missing_after}. "
            f"Original columns: {list(raw.columns)}. "
            f"Check COLUMN_MAP or enable auto_detect."
        )

    out["code"] = out["code"].map(_norm_code)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["code","date"]).copy()

    for c in ["open","high","low","close","preclose"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "turnover" in out.columns:
        out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce")
    if "pct_chg" in out.columns:
        out["pct_chg"] = pd.to_numeric(out["pct_chg"], errors="coerce")

    out.sort_values(["code","date"], kind="mergesort", inplace=True)
    before = len(out)
    out = out.drop_duplicates(subset=["code","date"], keep="last").reset_index(drop=True)
    print(f"  ✓ Cleaned & deduped: {before:,} → {len(out):,} rows")
    return out

# =========================
# Features
# =========================
def build_features(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    print("[Step 2/3] Building features by code…")
    need = ["code","date","open","high","low","close","preclose"]
    for n in need:
        if n not in df.columns:
            raise ValueError(f"Missing required column: {n}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values(["code","date"]).reset_index(drop=True)

    if "turnover" not in out.columns:
        out["turnover"] = 0.0
    else:
        mx = out["turnover"].abs().max()
        if pd.notna(mx) and mx > 1.5:
            out["turnover"] = out["turnover"] / 100.0

    if "pct_chg" in out.columns:
        mx = out["pct_chg"].abs().max()
        if pd.notna(mx) and mx > 1.5:
            out["pct_chg"] = out["pct_chg"] / 100.0
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            base = out["preclose"].replace(0, np.nan)
            out["pct_chg"] = (out["close"] / base - 1.0).fillna(0.0)

    blocks: Dict[str, pd.DataFrame] = {}
    for code, g in tqdm(out.groupby("code", sort=False), desc="Group by code"):
        gg = g[["date","open","high","low","close","preclose","turnover","pct_chg"]].copy()
        for c in ["open","high","low","close","preclose","turnover","pct_chg"]:
            gg[c] = pd.to_numeric(gg[c], errors="coerce")
        blocks[str(code)] = gg.reset_index(drop=True)

    print(f"  ✓ Built features for {len(blocks):,} codes")
    return blocks

# =========================
# Streaming similarity
# =========================
def _target_flatten(df_t: pd.DataFrame, start_idx: int, end_idx: int) -> np.ndarray:
    win = df_t.iloc[start_idx:end_idx+1]
    base0 = float(win["close"].iloc[0]) if float(win["close"].iloc[0]) != 0 else 1.0
    t_segs = [
        win["open"    ].to_numpy(np.float32) / base0,
        win["high"    ].to_numpy(np.float32) / base0,
        win["low"     ].to_numpy(np.float32) / base0,
        win["close"   ].to_numpy(np.float32) / base0,
        win["preclose"].to_numpy(np.float32) / base0,
        win["turnover"].to_numpy(np.float32),
        win["pct_chg" ].to_numpy(np.float32),
    ]
    return np.concatenate(t_segs, axis=0)

def _sim_streaming_segments(segs: List[np.ndarray], tgt_splits: List[np.ndarray], metric: str) -> float:
    if metric == "euclidean":
        acc = 0.0
        for seg, tseg in zip(segs, tgt_splits):
            d = seg - tseg
            acc += float(np.dot(d, d))
        return 1.0 / (1.0 + np.sqrt(acc))
    elif metric == "cosine":
        dot = 0.0
        normx = 0.0
        for seg, tseg in zip(segs, tgt_splits):
            dot   += float(np.dot(seg, tseg))
            normx += float(np.dot(seg, seg))
        return dot / (np.sqrt(normx) + 1e-12)
    else:
        raise ValueError("metric must be 'euclidean' or 'cosine'")

def _make_target_payload_for_plot(fdf: pd.DataFrame, start_idx: int, end_idx: int, future_horizon: int) -> Dict:
    win = fdf.iloc[start_idx:end_idx+1].copy()
    base0 = float(win["close"].iloc[0]) if float(win["close"].iloc[0]) != 0 else 1.0

    # Window (all / base0)
    rel_open  = win["open"].to_numpy(dtype=float)     / base0
    rel_high  = win["high"].to_numpy(dtype=float)     / base0
    rel_low   = win["low"].to_numpy(dtype=float)      / base0
    rel_close = win["close"].to_numpy(dtype=float)    / base0
    rel_prec  = win["preclose"].to_numpy(dtype=float) / base0

    # Future (trading days, / base0)
    fut_slice = fdf.iloc[end_idx+1 : end_idx+1+future_horizon].copy()
    future_days = len(fut_slice)
    if future_days > 0:
        fut_o = fut_slice["open" ].to_numpy(dtype=float)  / base0
        fut_h = fut_slice["high" ].to_numpy(dtype=float)  / base0
        fut_l = fut_slice["low"  ].to_numpy(dtype=float)  / base0
        fut_c = fut_slice["close"].to_numpy(dtype=float)  / base0
        fut_dates = fut_slice["date"].tolist()
    else:
        fut_o = fut_h = fut_l = fut_c = np.array([], dtype=float)
        fut_dates = []

    return {
        "win_dates": win["date"].tolist(),
        "rel_open": rel_open,
        "rel_high": rel_high,
        "rel_low": rel_low,
        "rel_close_vec": rel_close,
        "rel_preclose": rel_prec,
        "fut_dates": fut_dates,
        "fut_rel_open":  fut_o,
        "fut_rel_high":  fut_h,
        "fut_rel_low":   fut_l,
        "fut_rel_close": fut_c,
        "future_days": future_days,
        "future_horizon": future_horizon
    }

def _prepare_targets(feat_by_code: Dict[str, pd.DataFrame],
                     targets: List[Tuple[str, Union[str, pd.Timestamp]]],
                     window_len: int,
                     future_horizon: int,
                     metric: str) -> List[Dict]:
    prepped = []
    for code_raw, end_date in targets:
        code = _norm_code(code_raw)
        if code not in feat_by_code:
            raise KeyError(f"Target code {code_raw}->{code} not found. Sample keys: {list(feat_by_code.keys())[:10]}")
        df_t = feat_by_code[code]
        end_date = pd.to_datetime(end_date)
        idx_end = df_t["date"].searchsorted(end_date, side="right") - 1
        if idx_end < 0:
            raise ValueError(f"{code}: target end date is earlier than first record.")
        start_idx = idx_end - window_len + 1
        if start_idx < 0:
            raise ValueError(f"{code}: not enough data for window_len={window_len}.")
        T_flat = _target_flatten(df_t, start_idx, idx_end)
        T_splits = np.split(T_flat, 7)
        prepped.append({
            "code": code,
            "df": df_t,
            "idx_s": start_idx,
            "idx_e": idx_end,
            "T_splits": T_splits,
            "future_horizon": future_horizon,
            "metric": metric
        })
    return prepped

def find_similar_for_targets_streaming_one_pass(
    feat_by_code: Dict[str, pd.DataFrame],
    targets: List[Tuple[str, Union[str, pd.Timestamp]]],
    window_len: int,
    future_horizon: int = 5,
    search_recent_days: Optional[int] = None,
    top_k_each: int = 10,
    min_similarity: Optional[float] = None,
    metric: str = "euclidean"
) -> List[Dict]:
    T_list = _prepare_targets(feat_by_code, targets, window_len, future_horizon, metric)
    heaps: Dict[str, List[Tuple[float, str, int, int]]] = {t["code"]: [] for t in T_list}

    # cutoff by trading date (optional)
    global_max_date = max(block["date"].max() for block in feat_by_code.values())
    cutoff_date = None
    if search_recent_days is not None and search_recent_days > 0:
        cutoff_date = pd.to_datetime(global_max_date) - pd.Timedelta(days=search_recent_days)
        print(f"  · Candidate window end >= {cutoff_date.date()} (last {search_recent_days} days)")
    else:
        print("  · Candidate range: full history")

    # one pass over all candidates
    for code_cand, fdf in tqdm(feat_by_code.items(), desc="Scan stocks", total=len(feat_by_code)):
        N = len(fdf)
        if N < window_len:
            continue
        dates = fdf["date"].to_numpy()
        o  = fdf["open"    ].to_numpy(np.float32)
        h  = fdf["high"    ].to_numpy(np.float32)
        l  = fdf["low"     ].to_numpy(np.float32)
        c  = fdf["close"   ].to_numpy(np.float32)
        pc = fdf["preclose"].to_numpy(np.float32)
        tr = fdf["turnover"].to_numpy(np.float32)
        pg = fdf["pct_chg" ].to_numpy(np.float32)

        for s in range(0, N - window_len + 1):
            e = s + window_len
            if cutoff_date is not None and dates[e-1] < np.datetime64(cutoff_date):
                continue

            base0 = c[s] if c[s] != 0 else 1.0
            segs = [o[s:e]/base0, h[s:e]/base0, l[s:e]/base0, c[s:e]/base0, pc[s:e]/base0, tr[s:e], pg[s:e]]

            for tgt in T_list:
                if tgt["code"] == code_cand and s == tgt["idx_s"]:
                    continue  # exclude identical window
                score = _sim_streaming_segments(segs, tgt["T_splits"], tgt["metric"])
                if (min_similarity is not None) and (score < min_similarity):
                    continue
                heap = heaps[tgt["code"]]
                if len(heap) < top_k_each:
                    heapq.heappush(heap, (score, code_cand, s, e))
                else:
                    if score > heap[0][0]:
                        heapq.heapreplace(heap, (score, code_cand, s, e))

    # finalize per-target
    outputs = []
    for tgt in T_list:
        code = tgt["code"]
        df_t = tgt["df"]
        idx_s, idx_e = tgt["idx_s"], tgt["idx_e"]
        heap = heaps[code]
        if not heap:
            raise ValueError(f"Target {code}: no candidates found.")
        heap.sort(reverse=True)

        rows = []
        matches_payload: Dict[int, Dict] = {}
        internal_id = 0

        for score, code_cand, s, e in heap:
            fdf = feat_by_code[code_cand]
            win_dates = fdf["date"].iloc[s:e].tolist()

            base0 = float(fdf["close"].iloc[s]) if float(fdf["close"].iloc[s]) != 0 else 1.0
            rel_open  = (fdf["open"    ].iloc[s:e].to_numpy(np.float32)) / base0
            rel_high  = (fdf["high"    ].iloc[s:e].to_numpy(np.float32)) / base0
            rel_low   = (fdf["low"     ].iloc[s:e].to_numpy(np.float32)) / base0
            rel_close = (fdf["close"   ].iloc[s:e].to_numpy(np.float32)) / base0
            rel_prec  = (fdf["preclose"].iloc[s:e].to_numpy(np.float32)) / base0

            fut_start = e
            fut_end   = min(len(fdf), fut_start + tgt["future_horizon"])
            fut_slice = fdf.iloc[fut_start:fut_end]
            if len(fut_slice) > 0:
                fut_o = fut_slice["open" ].to_numpy(np.float32) / base0
                fut_h = fut_slice["high" ].to_numpy(np.float32) / base0
                fut_l = fut_slice["low"  ].to_numpy(np.float32) / base0
                fut_c = fut_slice["close"].to_numpy(np.float32) / base0
                fut_dates = fut_slice["date"].tolist()
            else:
                fut_o = fut_h = fut_l = fut_c = np.array([], dtype=np.float32)
                fut_dates = []

            rows.append({
                "internal_id": internal_id,
                "target_code": code,
                "target_start": df_t["date"].iloc[idx_s],
                "target_end":   df_t["date"].iloc[idx_e],
                "window_len":   (idx_e - idx_s + 1),
                "future_horizon": tgt["future_horizon"],
                "code": str(code_cand),
                "start_date": win_dates[0],
                "end_date":   win_dates[-1],
                "similarity": float(score),
                "future_days_available": int(fut_end - fut_start)
            })
            matches_payload[internal_id] = {
                "win_dates": win_dates,
                "rel_open":  rel_open,
                "rel_high":  rel_high,
                "rel_low":   rel_low,
                "rel_close": rel_close,          # overlay window
                "rel_close_vec": rel_close,      # candles window
                "rel_preclose":  rel_prec,
                "fut_dates": fut_dates,
                "fut_rel_open":  fut_o,          # overlay & candles future (same base0)
                "fut_rel_high":  fut_h,
                "fut_rel_low":   fut_l,
                "fut_rel_close": fut_c,
                "future_days": len(fut_slice),
                "future_horizon": tgt["future_horizon"],
            }
            internal_id += 1

        res = pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)
        tgt_payload = _make_target_payload_for_plot(df_t, idx_s, idx_e, tgt["future_horizon"])
        outputs.append({"target_code": code, "res": res, "bundle": {"target": tgt_payload, "matches": matches_payload}})
    return outputs

# =========================
# Visualization
# =========================
def _build_ohlc_from_rel(rel_open, rel_high, rel_low, rel_close):
    return (np.asarray(rel_open, dtype=float),
            np.asarray(rel_high, dtype=float),
            np.asarray(rel_low,  dtype=float),
            np.asarray(rel_close,dtype=float))

def _plot_candles_updown(ax, dates, o, h, l, c, title: str = ""):
    """
    Red for up (close >= open), green for down (close < open).
    Wicks and bodies both follow the same color per candle.
    """
    x = np.arange(len(dates))
    width = 0.6
    for i in range(len(x)):
        up = c[i] >= o[i]
        color = "#D62728" if up else "#2CA02C"  # red / green
        # wick
        ax.vlines(x[i], l[i], h[i], linewidth=1.0, color=color, alpha=0.95, zorder=2)
        # body
        lower = min(o[i], c[i]); height = abs(c[i] - o[i])
        rect = patches.Rectangle((x[i] - width/2, lower),
                                 width, height if height > 0 else 1e-8,
                                 facecolor=color, edgecolor=color, alpha=0.85, zorder=3)
        ax.add_patch(rect)
    ax.set_xlim(-1, len(x))
    ax.set_title(title)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)

class SingleTargetViewer:
    """
    One window per target:
      Top: shows target (solid) + target future (dotted, with bridge),
           ALL matches (dashed) + ALL matches future (dotted, with bridge). Click does NOT change overlay.
      Bottom-left: target candles (window + future), red/green by up/down.
      Bottom-right: selected match candles (window + future), red/green by up/down. Click dashed line to select.
    """
    def __init__(self, item: Dict, limit_to_topk: Optional[int] = None):
        self.item = item
        self.limit_to_topk = limit_to_topk

        self.fig = plt.figure(figsize=(13, 9))
        self.ax_overlay   = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        self.ax_k_target  = plt.subplot2grid((2, 2), (1, 0))
        self.ax_k_match   = plt.subplot2grid((2, 2), (1, 1))

        self._line_to_mid: Dict[plt.Line2D, int] = {}
        self.active_mid: Optional[int] = None

        # ---------------- 关键新增：颜色一致性分配 ----------------
        res_df_for_color = self.item["res"].copy()
        if self.limit_to_topk is not None:
            res_df_for_color = res_df_for_color.head(self.limit_to_topk)

        # 使用 Matplotlib 的默认调色板；若缺失则给一个兜底列表
        palette = plt.rcParams.get("axes.prop_cycle", None)
        palette = (palette.by_key().get("color", []) if palette else [])
        if not palette:
            palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
                       "#9467bd","#8c564b","#e377c2","#7f7f7f",
                       "#bcbd22","#17becf"]

        # 目标序列保留第一种颜色
        self._color_target = palette[0]
        self._color_for_mid: Dict[int, str] = {}
        for i, (_, row) in enumerate(res_df_for_color.iterrows()):
            mid = int(row["internal_id"])
            self._color_for_mid[mid] = palette[(i + 1) % len(palette)]
        # -------------------------------------------------------

        self._draw_overlay()
        self._draw_target_candles()
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)
        plt.tight_layout()

    def _bridge(self, ax, x_last, y_last, x_first, y_first, **kw):
        """Draw a short segment to visually bridge window -> future."""
        ax.plot([x_last, x_first], [y_last, y_first], **kw)

    def _draw_overlay(self):
        self.ax_overlay.clear()
        self._line_to_mid.clear()
        tgt = self.item["bundle"]["target"]
        y = tgt["rel_close_vec"]
        L = len(y); x = np.arange(L)

        # 目标窗口（实线）使用专属颜色
        (line_target,) = self.ax_overlay.plot(
            x, y, linewidth=2.8, linestyle='-',
            color=self._color_target,
            label=f"Target {self.item['target_code']}"
        )
        line_target.set_zorder(5)

        # 目标未来（点线）+ 桥接，颜色一致
        fut = tgt.get("fut_rel_close", None)
        if fut is not None and len(fut) > 0:
            xf = np.arange(L, L + len(fut))
            self.ax_overlay.plot(
                xf, fut, linewidth=2.0, linestyle=':',
                color=self._color_target, alpha=0.95, zorder=4,
                label="Target future"
            )
            self._bridge(
                self.ax_overlay, L-1, y[-1], L, fut[0],
                linewidth=1.6, linestyle=":", color=self._color_target, alpha=0.95
            )

        # 匹配（虚线）+ 其未来（点线）均使用为该匹配分配的固定颜色
        res_df = self.item["res"]
        if self.limit_to_topk is not None:
            res_df = res_df.head(self.limit_to_topk)
        matches = self.item["bundle"]["matches"]

        for _, row in res_df.iterrows():
            mid = int(row["internal_id"])
            payload = matches[mid]
            yy = payload["rel_close"]
            if len(yy) != len(y):
                continue
            color = self._color_for_mid.get(mid, "#7f7f7f")

            # 窗口（可点击选择）
            line, = self.ax_overlay.plot(
                x, yy, linestyle='--', linewidth=1.3, alpha=0.95,
                color=color, label=f"{row['code']}"
            )
            line.set_picker(True)
            line.set_pickradius(8)
            self._line_to_mid[line] = mid

            # 未来 + 桥接，颜色一致
            fut_m = payload.get("fut_rel_close", [])
            if fut_m is not None and len(fut_m) > 0:
                xf = np.arange(L, L + len(fut_m))
                self.ax_overlay.plot(
                    xf, fut_m, linewidth=1.2, alpha=0.9,
                    linestyle=':', color=color
                )
                self._bridge(
                    self.ax_overlay, L-1, yy[-1], L, fut_m[0],
                    linewidth=1.2, linestyle=':', color=color, alpha=0.9
                )

        self.ax_overlay.set_title(f"Overlay — {self.item['target_code']} (click a dashed line to show its candles)")
        self.ax_overlay.set_xlabel("Index within window")
        self.ax_overlay.set_ylabel("Relative close (norm)")
        self.ax_overlay.grid(True, alpha=0.3)

        # 图例去重
        handles, labels = self.ax_overlay.get_legend_handles_labels()
        seen = {}
        dedup_handles, dedup_labels = [], []
        for h, lb in zip(handles, labels):
            if lb not in seen:
                seen[lb] = True
                dedup_handles.append(h)
                dedup_labels.append(lb)
        self.ax_overlay.legend(dedup_handles, dedup_labels, loc="upper left", fontsize=9, ncol=2)

    def _draw_target_candles(self):
        self.ax_k_target.clear()
        tgt = self.item["bundle"]["target"]

        # Window candles (red/green)
        o, h, l, c = _build_ohlc_from_rel(
            tgt["rel_open"], tgt["rel_high"], tgt["rel_low"], tgt["rel_close_vec"]
        )
        _plot_candles_updown(self.ax_k_target, tgt["win_dates"], o, h, l, c,
                             f"Target {self.item['target_code']} (window)")

        # Future candles (red/green), docked right after window
        if len(tgt["fut_rel_close"]) > 0:
            L = len(tgt["win_dates"])
            fo, fh, fl, fc = _build_ohlc_from_rel(
                tgt["fut_rel_open"], tgt["fut_rel_high"], tgt["fut_rel_low"], tgt["fut_rel_close"]
            )
            # 与窗口连在一条时间轴：future 的 x 从 L 开始
            x_offset = L
            for i in range(len(fc)):
                up = fc[i] >= fo[i]
                color = "#D62728" if up else "#2CA02C"
                # wick
                self.ax_k_target.vlines(x_offset + i, fl[i], fh[i], linewidth=1.0, color=color, alpha=0.95, zorder=2)
                # body
                lower = min(fo[i], fc[i]); height = abs(fc[i] - fo[i])
                rect = patches.Rectangle((x_offset + i - 0.3, lower), 0.6,
                                         height if height > 0 else 1e-8,
                                         facecolor=color, edgecolor=color, alpha=0.85, zorder=3)
                self.ax_k_target.add_patch(rect)

        fd = int(self.item["bundle"]["target"].get("future_days", 0))
        fh = int(self.item["bundle"]["target"].get("future_horizon", fd))
        self.ax_k_target.set_title(f"Target candles: window + future  [future {fd}/{fh} trading days]")
        self.ax_k_target.grid(True, alpha=0.3)

        # 右下角：根据选择绘制匹配蜡烛图
        if self.active_mid is not None:
            self._draw_match_candles(self.active_mid)
        else:
            self.ax_k_match.clear()
            self.ax_k_match.set_title("Match candles (click a dashed line above)")
            self.ax_k_match.grid(True, alpha=0.3)

    def _draw_match_candles(self, mid: int):
        self.ax_k_match.clear()
        row = self.item["res"].loc[self.item["res"]["internal_id"] == mid].iloc[0]
        payload = self.item["bundle"]["matches"][mid]

        o, h, l, c = _build_ohlc_from_rel(
            payload["rel_open"], payload["rel_high"], payload["rel_low"], payload["rel_close_vec"]
        )

        fd = int(payload.get("future_days", 0))
        fh = int(payload.get("future_horizon", fd))
        title = (
            f"Match {row['code']} "
            f"[{pd.to_datetime(row['start_date']).date()}–{pd.to_datetime(row['end_date']).date()}]  "
            f"sim={row['similarity']:.3f}  "
            f"(future {fd}/{fh} trading days)"
        )

        # window candles
        _plot_candles_updown(self.ax_k_match, payload["win_dates"], o, h, l, c, title)

        # future candles
        if len(payload["fut_rel_close"]) > 0:
            L = len(payload["win_dates"])
            fo, fhh, fll, fcc = _build_ohlc_from_rel(
                payload["fut_rel_open"], payload["fut_rel_high"], payload["fut_rel_low"], payload["fut_rel_close"]
            )
            x_offset = L
            for i in range(len(fcc)):
                up = fcc[i] >= fo[i]
                color = "#D62728" if up else "#2CA02C"
                self.ax_k_match.vlines(x_offset + i, fll[i], fhh[i], linewidth=1.0, color=color, alpha=0.95, zorder=2)
                lower = min(fo[i], fcc[i]); height = abs(fcc[i] - fo[i])
                rect = patches.Rectangle((x_offset + i - 0.3, lower), 0.6,
                                         height if height > 0 else 1e-8,
                                         facecolor=color, edgecolor=color, alpha=0.85, zorder=3)
                self.ax_k_match.add_patch(rect)

        self.ax_k_match.grid(True, alpha=0.3)

    def _on_pick(self, event):
        artist = event.artist
        if artist in self._line_to_mid:
            self.active_mid = self._line_to_mid[artist]
            # 仅更新下方两块；上方 overlay 保持不变
            self._draw_target_candles()
            self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

# ======================
# Example
# ======================
if __name__ == "__main__":
    FILE_PATHS = [
        "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2020-2025_all.csv",
        "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv",
    ]
    COLUMN_MAP = {
        "code": "code",
        "date": "日期",       # If CSV uses English, set to "date"
        "open": "开盘",
        "high": "最高",
        "low": "最低",
        "close": "收盘",
        "preclose": "前收",
        "turnover": "换手率",
        "pct_chg": "涨跌幅"
    }

    df_all = load_and_concat_csvs(FILE_PATHS, column_map=COLUMN_MAP, auto_detect=True)
    feat_by_code = build_features(df_all)

    TARGETS = [
        ("002815", "2025-08-28"),
        ("605255", "2025-08-28")
    ]
    window_len = 7
    future_horizon = 5
    search_recent_days = None
    top_k_each = 10
    min_similarity = None
    metric = "euclidean"

    results = find_similar_for_targets_streaming_one_pass(
        feat_by_code=feat_by_code,
        targets=TARGETS,
        window_len=window_len,
        future_horizon=future_horizon,
        search_recent_days=search_recent_days,
        top_k_each=top_k_each,
        min_similarity=min_similarity,
        metric=metric
    )

    # 打印每个目标的结果表
    for item in results:
        print(f"\n=== Target {item['target_code']} ===")
        print(item["res"])

    # 为每个目标单独展示一个交互窗口
    for item in results:
        viewer = SingleTargetViewer(item, limit_to_topk=top_k_each)
        viewer.show()