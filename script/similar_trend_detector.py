# -*- coding: utf-8 -*-
"""
Multi-file Stock Window Similarity (NO z-score) + Interactive Highlight + Candlestick (with OPEN) + Progress Bars
----------------------------------------------------------------------------------------------------------------

依赖：
  pip install numpy pandas matplotlib tqdm

功能要点：
- 读取多个 CSV（两份或更多），合并后按 (code, date) 去重；
- 特征：open, high, low, close, preclose, turnover(小数), pct_chg(小数)
- 窗口相似检索：以 “结束日期 + 窗口长度 L”
- 可选仅在最近 N 天范围内搜索
- 相似度：欧氏（默认）或余弦
- 进度展示：tqdm 进度条 + 阶段性 print
- 交互可视化：叠加曲线点击高亮 + 双 K 线详情（目标 / 被选匹配）+ 未来 X 天延伸

使用方式：见文件底部 __main__ 示例
"""

from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from datetime import timedelta
import matplotlib.patches as patches
import os
from tqdm import tqdm  # <<< 新增：进度条


# =========================
# 0) 工具：读多个文件并合并
# =========================
def _auto_pick(cols: List[str], candidates: List[str]) -> Optional[str]:
    """从候选别名中挑选实际列名（精确/大小写/包含 3 重策略）"""
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


def load_and_concat_csvs(
    file_paths: List[str],
    column_map: Dict[str, Optional[str]] = None,
    auto_detect: bool = True
) -> pd.DataFrame:
    """
    读取多份 CSV 并纵向合并（列名对齐），最后按 (code, date) 去重。
    参数：
      - file_paths: CSV 路径列表
      - column_map: 映射到统一名字：
            {
              "code": "...", "date": "...",
              "open": "...", "high": "...", "low": "...",
              "close": "...", "preclose": "...",
              "turnover": "...", "pct_chg": "..."
            }
        若某项为 None 或缺省，且 auto_detect=True，会尝试自动识别常见别名。
      - auto_detect: 是否启用自动识别
    返回：
      - df 统一列名的整合 DataFrame
    """
    print(f"[Step 1/3] 正在读取 {len(file_paths)} 个文件并合并…")
    dfs = []
    for p in tqdm(file_paths, desc="读取CSV文件"):
        if not os.path.exists(p):
            raise FileNotFoundError(f"文件不存在: {p}")
        df = pd.read_csv(p)
        dfs.append(df)

    raw = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"  ✓ 合并完成，初始总行数：{len(raw):,}")

    # 列名映射
    if column_map is None:
        column_map = {}

    cols = list(raw.columns)
    code_col     = column_map.get("code")     or (auto_detect and _auto_pick(cols, ["code","ts_code","symbol","证券代码","股票代码","代码"]))
    date_col     = column_map.get("date")     or (auto_detect and _auto_pick(cols, ["date","trade_date","交易日期","日期"]))
    open_col     = column_map.get("open")     or (auto_detect and _auto_pick(cols, ["open","开盘"]))
    high_col     = column_map.get("high")     or (auto_detect and _auto_pick(cols, ["high","最高"]))
    low_col      = column_map.get("low")      or (auto_detect and _auto_pick(cols, ["low","最低"]))
    close_col    = column_map.get("close")    or (auto_detect and _auto_pick(cols, ["close","收盘","收盘价"]))
    preclose_col = column_map.get("preclose") or (auto_detect and _auto_pick(cols, ["preclose","pre_close","昨收","前收"]))
    turnover_col = column_map.get("turnover") or (auto_detect and _auto_pick(cols, ["turnover","turnover_rate","换手率","换手"]))
    pct_chg_col  = column_map.get("pct_chg")  or (auto_detect and _auto_pick(cols, ["pct_chg","涨跌幅","涨幅"]))

    required = [code_col, date_col, open_col, high_col, low_col, close_col, preclose_col]
    req_names = ["code","date","open","high","low","close","preclose"]
    if any(x is None for x in required):
        missing = [n for x,n in zip(required, req_names) if x is None]
        preview_cols = ", ".join(cols[:15]) + ("..." if len(cols) > 15 else "")
        raise ValueError(f"缺少必要列：{missing}；文件中的列名示例：[{preview_cols}]")

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

    # 基础清洗
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["code","date"]).copy()

    # 类型转换
    for c in ["open","high","low","close","preclose"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "turnover" in out.columns:
        out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce")
    if "pct_chg" in out.columns:
        out["pct_chg"] = pd.to_numeric(out["pct_chg"], errors="coerce")

    # 按 (code,date) 排序去重：若同一对重复，优先保留“后读入”的记录（concat 后的靠后行）
    out.sort_values(["code","date"], kind="mergesort", inplace=True)
    before = len(out)
    out = out.drop_duplicates(subset=["code","date"], keep="last").reset_index(drop=True)
    print(f"  ✓ 清洗&去重完成：{before:,} → {len(out):,} 行")

    return out


# =========================
# 1) 特征构造（含开盘）
# =========================
def build_features(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    输入：统一列名的 df（必须包含：
      code, date, open, high, low, close, preclose；可选：turnover, pct_chg）
    输出：按代码分组后的特征表，不做 z-score、不相对化价格；百分比→小数。
    """
    print("[Step 2/3] 按股票代码构造特征…")
    need = ["code","date","open","high","low","close","preclose"]
    for n in need:
        if n not in df.columns:
            raise ValueError(f"缺少必要列：{n}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values(["code","date"]).reset_index(drop=True)

    # 百分比类转小数
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
    # tqdm 显示每只股票的处理进度
    for code, g in tqdm(out.groupby("code", sort=False), desc="构造特征（逐代码）"):
        gg = g[["date","open","high","low","close","preclose","turnover","pct_chg"]].copy()
        for c in ["open","high","low","close","preclose","turnover","pct_chg"]:
            gg[c] = pd.to_numeric(gg[c], errors="coerce")
        blocks[str(code)] = gg.reset_index(drop=True)

    print(f"  ✓ 已构造 {len(blocks):,} 只股票的特征")
    return blocks


# ========================================
# 2) 相似检索（结束日 + L；可限定最近 N 天）
# ========================================
def find_similar_windows(
    feat_by_code: Dict[str, pd.DataFrame],
    target_code: str,
    window_len: int,
    target_end_date: Union[str, pd.Timestamp],
    future_horizon: int = 5,
    search_recent_days: Optional[int] = None,   # 仅在最近 N 天内搜索；None = 全量
    top_k: Optional[int] = 10,
    min_similarity: Optional[float] = None,     # 例如 0.9；与 top_k 可并用
    exclude_same_window: bool = True,
    metric: str = "euclidean"                   # "euclidean"（保幅度）或 "cosine"
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    使用 7 特征（窗口阶段相对化 / 小数原值）：
      [rel_open, rel_high, rel_low, rel_close, rel_preclose, turnover, pct_chg]
    """
    print(f"[Step 3/3] 搜索相似窗口中… 目标代码={target_code} 窗口长度={window_len} 预测X={future_horizon}")
    assert target_code in feat_by_code, f"代码 {target_code} 不在数据中。"
    target_end_date = pd.to_datetime(target_end_date)

    df_t = feat_by_code[target_code].copy()
    idx_end = df_t["date"].searchsorted(target_end_date, side="right") - 1
    if idx_end < 0:
        raise ValueError("目标结束日早于该代码的最早交易日。")
    if idx_end - window_len + 1 < 0:
        raise ValueError(f"{target_code} 数据不足以构造窗口长度 {window_len}。")

    start_idx = idx_end - window_len + 1
    tgt = df_t.iloc[start_idx:idx_end+1].copy()

    # 目标窗口相对化：/ 窗口首日 close
    base_close = float(tgt["close"].iloc[0]) if float(tgt["close"].iloc[0]) != 0 else 1.0
    t = {
        "rel_open":     tgt["open"].to_numpy(dtype=float)     / base_close,
        "rel_high":     tgt["high"].to_numpy(dtype=float)     / base_close,
        "rel_low":      tgt["low"].to_numpy(dtype=float)      / base_close,
        "rel_close":    tgt["close"].to_numpy(dtype=float)    / base_close,
        "rel_preclose": tgt["preclose"].to_numpy(dtype=float) / base_close,
        "turnover":     tgt["turnover"].to_numpy(dtype=float),
        "pct_chg":      tgt["pct_chg"].to_numpy(dtype=float),
    }
    T = np.vstack([t["rel_open"], t["rel_high"], t["rel_low"], t["rel_close"], t["rel_preclose"], t["turnover"], t["pct_chg"]]).T
    T_flat = T.reshape(-1)
    T_norm = np.linalg.norm(T_flat) + 1e-12

    # 目标未来段
    tgt_future = df_t.iloc[idx_end+1: idx_end+1+future_horizon].copy()

    # 最近 N 天截止线
    global_max_date = max(block["date"].max() for block in feat_by_code.values())
    cutoff_date = None
    if search_recent_days is not None and search_recent_days > 0:
        cutoff_date = pd.to_datetime(global_max_date) - timedelta(days=search_recent_days)
        print(f"  · 限制候选窗口结束日 ≥ {cutoff_date.date()} (最近 {search_recent_days} 天)")
    else:
        print("  · 候选范围：全量数据")

    rows = []
    matches_payload: Dict[int, Dict] = {}
    internal_id = 0

    # tqdm：按股票扫描
    for code, fdf in tqdm(feat_by_code.items(), desc="扫描股票", total=len(feat_by_code)):
        N = len(fdf)
        if N < window_len:
            continue
        if cutoff_date is not None and fdf["date"].iloc[-1] < cutoff_date:
            continue

        date_arr = fdf["date"].to_numpy()
        win_open  = sliding_window_view(fdf["open"].to_numpy(dtype=float),     window_len)
        win_high  = sliding_window_view(fdf["high"].to_numpy(dtype=float),     window_len)
        win_low   = sliding_window_view(fdf["low"].to_numpy(dtype=float),      window_len)
        win_close = sliding_window_view(fdf["close"].to_numpy(dtype=float),    window_len)
        win_prec  = sliding_window_view(fdf["preclose"].to_numpy(dtype=float), window_len)
        win_turn  = sliding_window_view(fdf["turnover"].to_numpy(dtype=float), window_len)
        win_pct   = sliding_window_view(fdf["pct_chg"].to_numpy(dtype=float),  window_len)
        win_dates = sliding_window_view(date_arr, window_len)

        # 仅保留最近 N 天内的窗口（按窗口结束日）
        if cutoff_date is not None:
            keep_recent = (win_dates[:, -1] >= np.datetime64(cutoff_date))
        else:
            keep_recent = np.ones(win_dates.shape[0], dtype=bool)

        # 排除“同代码同窗口”
        if exclude_same_window and code == target_code:
            ex_mask = np.ones(win_dates.shape[0], dtype=bool)
            if 0 <= start_idx < ex_mask.size:
                ex_mask[start_idx] = False
            keep_recent = keep_recent & ex_mask

        if not keep_recent.any():
            continue

        # 价格类相对化
        base = win_close[:, [0]]
        base = np.where(base == 0, 1.0, base)
        rel_open  = win_open  / base
        rel_high  = win_high  / base
        rel_low   = win_low   / base
        rel_close = win_close / base
        rel_prec  = win_prec  / base

        # 拼接
        W = np.stack([rel_open, rel_high, rel_low, rel_close, rel_prec, win_turn, win_pct], axis=2)
        Wf = W.reshape(W.shape[0], -1)

        # 相似度
        if metric.lower() == "euclidean":
            diff = Wf - T_flat
            dist = np.sqrt(np.sum(diff * diff, axis=1))
            sim = 1.0 / (1.0 + dist)
        elif metric.lower() == "cosine":
            norms = np.linalg.norm(Wf, axis=1) + 1e-12
            sim = (Wf @ T_flat) / (norms * T_norm)
        else:
            raise ValueError("metric 仅支持 'euclidean' 或 'cosine'。")

        keep = np.ones_like(sim, dtype=bool)
        if min_similarity is not None:
            keep &= (sim >= min_similarity)
        if not keep.any():
            continue

        idxs = np.where(keep)[0]
        for w in idxs:
            s_dt = pd.to_datetime(win_dates[w, 0])
            e_dt = pd.to_datetime(win_dates[w, -1])

            # 未来 X 天
            end_pos = int(fdf["date"].searchsorted(e_dt, side="right") - 1)
            fut_start = end_pos + 1
            fut_end   = min(N, fut_start + future_horizon)
            fut_len   = max(fut_end - fut_start, 0)

            # 叠加浏览用：相对收盘
            base_close_vec = win_close[w]
            base0 = base_close_vec[0] if base_close_vec[0] != 0 else 1.0
            base_rel_close = base_close_vec / base0

            # 未来段（相对窗口末日 close）
            fut_slice = fdf.iloc[fut_start:fut_end].copy()
            if fut_len > 0:
                fut_vals = np.concatenate([[base_close_vec[-1]], fut_slice["close"].to_numpy(dtype=float)])
                fut_rel  = fut_vals / base_close_vec[-1]
            else:
                fut_rel = np.array([1.0])

            rows.append({
                "internal_id": internal_id,  # 交互高亮用
                "target_code": target_code,
                "target_start": pd.to_datetime(tgt["date"].iloc[0]),
                "target_end":   pd.to_datetime(tgt["date"].iloc[-1]),
                "window_len":   window_len,
                "future_horizon": future_horizon,
                "code": str(code),
                "start_date": s_dt,
                "end_date":   e_dt,
                "similarity": float(sim[w]),
                "future_days_available": int(fut_len)
            })

            matches_payload[internal_id] = {
                "win_dates": win_dates[w, :].tolist(),
                "rel_close": base_rel_close,   # 叠加浏览
                "rel_open":  rel_open[w],
                "rel_high":  rel_high[w],
                "rel_low":   rel_low[w],
                "rel_close_vec": rel_close[w],
                "rel_preclose":  rel_prec[w],
                "fut_dates": fut_slice["date"].tolist(),
                "fut_rel_close": fut_rel
            }
            internal_id += 1

    if len(rows) == 0:
        raise ValueError("没有找到候选（请放宽阈值、增大搜索范围或延长窗口）。")

    res = pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)
    if top_k is not None:
        res = res.head(top_k).copy()

    tgt_payload = _make_target_payload_for_plot(tgt, tgt_future)
    print(f"  ✓ 搜索完成。候选数量（排序后）={len(res)}，top_k={top_k}")
    return res, {"target": tgt_payload, "matches": matches_payload}


def _make_target_payload_for_plot(tgt_slice: pd.DataFrame, tgt_future: pd.DataFrame) -> Dict:
    base0 = float(tgt_slice["close"].iloc[0]) if float(tgt_slice["close"].iloc[0]) != 0 else 1.0
    rel_open  = tgt_slice["open"].to_numpy(dtype=float)     / base0
    rel_high  = tgt_slice["high"].to_numpy(dtype=float)     / base0
    rel_low   = tgt_slice["low"].to_numpy(dtype=float)      / base0
    rel_close = tgt_slice["close"].to_numpy(dtype=float)    / base0
    rel_prec  = tgt_slice["preclose"].to_numpy(dtype=float) / base0

    if len(tgt_future) > 0:
        fut_vals = np.concatenate([[tgt_slice["close"].iloc[-1]], tgt_future["close"].to_numpy(dtype=float)])
        fut_rel  = fut_vals / tgt_slice["close"].iloc[-1]
    else:
        fut_rel = np.array([1.0])

    return {
        "win_dates": tgt_slice["date"].tolist(),
        "rel_open": rel_open,
        "rel_high": rel_high,
        "rel_low": rel_low,
        "rel_close_vec": rel_close,
        "rel_preclose": rel_prec,
        "fut_dates": tgt_future["date"].tolist(),
        "fut_rel_close": fut_rel
    }


# ============================
# 3) 蜡烛图与交互可视化
# ============================
def _build_ohlc_from_rel(rel_open, rel_high, rel_low, rel_close):
    return (np.asarray(rel_open, dtype=float),
            np.asarray(rel_high, dtype=float),
            np.asarray(rel_low,  dtype=float),
            np.asarray(rel_close,dtype=float))

def _plot_candles(ax, dates, o, h, l, c, title: str = ""):
    x = np.arange(len(dates))
    width = 0.6
    for i in range(len(x)):
        ax.vlines(x[i], l[i], h[i])
    for i in range(len(x)):
        open_i = o[i]
        close_i = c[i]
        lower = min(open_i, close_i)
        height = abs(close_i - open_i)
        rect = patches.Rectangle((x[i] - width/2, lower), width, height if height > 0 else 1e-8)
        ax.add_patch(rect)
    ax.set_xlim(-1, len(x))
    ax.set_title(title)
    ax.set_xticks(x)
    ax.grid(True)

class InteractiveMatcherViewer:
    def __init__(self, res_df: pd.DataFrame, bundle: Dict, limit_to_topk: Optional[int] = None):
        self.res = res_df.copy()
        if limit_to_topk is not None:
            self.res = self.res.head(limit_to_topk).copy()

        self.target = bundle["target"]
        ids = set(self.res["internal_id"].tolist())
        self.matches = {mid: payload for mid, payload in bundle["matches"].items() if mid in ids}

        self.fig = plt.figure(figsize=(12, 9))
        self.ax_overlay   = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        self.ax_k_target  = plt.subplot2grid((2, 2), (1, 0))
        self.ax_k_match   = plt.subplot2grid((2, 2), (1, 1))

        self._line_to_id: Dict[plt.Line2D, int] = {}
        self.selected_id: Optional[int] = None

        self._draw_overlay()
        self._draw_target_candles()
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)
        plt.tight_layout()

    def _draw_overlay(self):
        self.ax_overlay.clear()
        t_y = self.target["rel_close_vec"]
        L = len(t_y)
        x = np.arange(L)
        self.ax_overlay.plot(x, t_y, linewidth=2.5, linestyle='-', label="Target")

        self._line_to_id.clear()
        for mid, payload in self.matches.items():
            y = payload["rel_close"]
            if len(y) != L:
                continue
            line, = self.ax_overlay.plot(x, y, linestyle='--', linewidth=1.0, alpha=0.7, picker=True, pickradius=5)
            self._line_to_id[line] = mid

        if self.selected_id is not None and self.selected_id in self.matches:
            sel = self.matches[self.selected_id]
            self.ax_overlay.plot(x, sel["rel_close"], linewidth=3.0, linestyle='-', marker='o')
            if len(sel["fut_rel_close"]) > 1:
                xf = np.arange(L-1, L-1 + len(sel["fut_rel_close"]))
                self.ax_overlay.plot(xf, sel["fut_rel_close"], linewidth=2.0)

        self.ax_overlay.set_title("Overlay — click a match to highlight")
        self.ax_overlay.set_xlabel("Index within window")
        self.ax_overlay.set_ylabel("Relative close (norm)")
        self.ax_overlay.grid(True)

    def _draw_target_candles(self):
        self.ax_k_target.clear()
        o, h, l, c = _build_ohlc_from_rel(
            self.target["rel_open"],
            self.target["rel_high"],
            self.target["rel_low"],
            self.target["rel_close_vec"],
        )
        _plot_candles(self.ax_k_target, self.target["win_dates"], o, h, l, c, "Target (candles)")
        if len(self.target["fut_rel_close"]) > 1:
            L = len(self.target["win_dates"])
            xf = np.arange(L-1, L-1 + len(self.target["fut_rel_close"]))
            self.ax_k_target.plot(xf, self.target["fut_rel_close"])
        self.ax_k_target.grid(True)

        self.ax_k_match.clear()
        self.ax_k_match.set_title("Selected match (candles)")
        self.ax_k_match.grid(True)

    def _draw_match_candles(self, match_payload: Dict, subtitle: str = ""):
        self.ax_k_match.clear()
        o, h, l, c = _build_ohlc_from_rel(
            match_payload["rel_open"],
            match_payload["rel_high"],
            match_payload["rel_low"],
            match_payload["rel_close_vec"],
        )
        _plot_candles(self.ax_k_match, match_payload["win_dates"], o, h, l, c, f"Match {subtitle}")
        if len(match_payload["fut_rel_close"]) > 1:
            L = len(match_payload["win_dates"])
            xf = np.arange(L-1, L-1 + len(match_payload["fut_rel_close"]))
            self.ax_k_match.plot(xf, match_payload["fut_rel_close"])
        self.ax_k_match.grid(True)

    def _on_pick(self, event):
        artist = event.artist
        if artist in self._line_to_id:
            mid = self._line_to_id[artist]
            self.selected_id = mid

            self._draw_overlay()
            row = self.res.loc[self.res["internal_id"] == mid]
            subtitle = ""
            if not row.empty:
                r = row.iloc[0]
                subtitle = f"{r['code']} [{pd.to_datetime(r['start_date']).date()}–{pd.to_datetime(r['end_date']).date()}]  sim={r['similarity']:.3f}"
            self._draw_match_candles(self.matches[mid], subtitle)
            self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ======================
# 4) 使用示例（按需修改）
# ======================
if __name__ == "__main__":
    # --- 1) 多文件路径（这里示例两个，你可以放更多） ---
    FILE_PATHS = [
        "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2020-2025_all.csv",
        "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv",
    ]

    # --- 2) 列名映射（按你的文件列名修改；留空可自动识别常见别名） ---
    COLUMN_MAP = {
        "code": "code",          # 例如 "证券代码"
        "date": "日期",          # 例如 "交易日期"
        "open": "开盘",           # 开盘价列名
        "high": "最高",           # 最高价列名
        "low": "最低",            # 最低价列名
        "close": "收盘",          # 收盘价列名
        "preclose": "前收",       # 前收价列名
        "turnover": "换手率",     # 换手率列名（如果没有可以写 None）
        "pct_chg": "涨跌幅"       # 涨跌幅列名（如果没有可以写 None）
    }

    # --- 3) 读并合并 ---
    df_all = load_and_concat_csvs(FILE_PATHS, column_map=COLUMN_MAP, auto_detect=True)

    # --- 4) 构造特征 ---
    feat_by_code = build_features(df_all)

    # --- 5) 搜索参数 ---
    target_code = "600410"
    window_len = 7
    target_end_date = "2025-08-12"
    future_horizon = 5
    search_recent_days = None   # None 表示全量
    top_k = 10
    min_similarity = None
    metric = "euclidean"       # or "cosine"

    # --- 6) 相似搜索 ---
    res, bundle = find_similar_windows(
        feat_by_code=feat_by_code,
        target_code=target_code,
        window_len=window_len,
        target_end_date=target_end_date,
        future_horizon=future_horizon,
        search_recent_days=search_recent_days,
        top_k=top_k,
        min_similarity=min_similarity,
        exclude_same_window=True,
        metric=metric
    )
    print("\nTop results:")
    print(res.head(10))

    # --- 7) 交互可视化（Top-K） ---
    viewer = InteractiveMatcherViewer(res_df=res, bundle=bundle, limit_to_topk=top_k)
    viewer.show()