import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(
    "once",
    message=r"The default fill_method='pad' in Series.pct_change is deprecated.*",
    category=FutureWarning,
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# from gru_model import GRUQuantileModel
from io_utils import load_market_csv_multi
from indicators import add_indicators


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    seq_len: int = 21              # 约一个月交易日
    horizon: int = 21              # 未来一个月窗口
    q_up: float = 0.80
    q_dn: float = 0.20
    min_universe: int = 200
    batch_size: int = 256
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 15
    grad_clip: float = 1.0
    num_workers: int = 0
    device: str = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Utils
# -------------------------

# --- Numeric sanitation helper ---
def _sanitize_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Replace inf/-inf with NaN for robust stats/dataloader.

    Note: we intentionally do not fill NaNs here; downstream dropna decides what becomes a valid sample.
    """
    if not cols:
        return df
    df = df.copy()
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    return df



# --- Triple-barrier labeling for channel outcome classification ---
def compute_triple_barrier_labels(
    df: pd.DataFrame,
    sym_col: str,
    date_col: str,
    close_col: str,
    high_col: str,
    low_col: str,
    horizon: int,
    up_barrier: float = 0.10,
    dn_barrier: float = -0.08,
) -> pd.DataFrame:
    """Triple-barrier labels within a fixed horizon.

    For each t (per symbol), define reference price p0 = close[t].
    Look ahead j=1..horizon:
      up_hit  if high[t+j]/p0 - 1 >= up_barrier
      dn_hit  if low[t+j] /p0 - 1 <= dn_barrier
    Label by first hit time:
      2 = Up, 0 = Down, 1 = Sideways (no hit)
    If both hit on the same future day, mark as Sideways (ambiguous).

    Output column: y_cls (int64)
    """
    df = df.sort_values([sym_col, date_col]).copy()

    def _per_symbol(x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        c = x[close_col].astype(float).to_numpy()
        h = x[high_col].astype(float).to_numpy()
        l = x[low_col].astype(float).to_numpy()
        n = len(x)
        y = np.full(n, np.nan, dtype=np.float32)

        for i in range(n):
            p0 = c[i]
            if not np.isfinite(p0) or p0 == 0:
                continue
            up_idx = None
            dn_idx = None
            max_j = min(horizon, n - i - 1)
            if max_j <= 0:
                continue

            # scan forward to find first hit
            for j in range(1, max_j + 1):
                if up_idx is None:
                    if (h[i + j] / (p0 + 1e-12) - 1.0) >= up_barrier:
                        up_idx = j
                if dn_idx is None:
                    if (l[i + j] / (p0 + 1e-12) - 1.0) <= dn_barrier:
                        dn_idx = j
                if up_idx is not None or dn_idx is not None:
                    # if one has hit, we can early stop only if the other cannot hit earlier than j
                    # but since we are scanning increasing j, the first time we see a hit is earliest for that barrier.
                    # we still need to know if the other barrier hits on the same j.
                    if up_idx is not None and dn_idx is not None:
                        break

            if up_idx is None and dn_idx is None:
                y[i] = 1.0  # Sideways
            elif up_idx is not None and dn_idx is None:
                y[i] = 2.0
            elif dn_idx is not None and up_idx is None:
                y[i] = 0.0
            else:
                # both hit
                if up_idx == dn_idx:
                    y[i] = 1.0  # ambiguous -> Sideways
                elif up_idx < dn_idx:
                    y[i] = 2.0
                else:
                    y[i] = 0.0

        x["y_cls"] = y
        return x

    out = df.groupby(sym_col, group_keys=False).apply(_per_symbol)
    out["y_cls"] = out["y_cls"].astype("float")
    return out


# --- Cross-sectional forward-return quantile labeling (Scheme A) ---
def compute_cs_fwdret_quantile_labels(
    df: pd.DataFrame,
    sym_col: str,
    date_col: str,
    close_col: str,
    horizon: int,
    q_up: float = 0.80,
    q_dn: float = 0.20,
    min_universe: int = 200,
) -> pd.DataFrame:
    """Cross-sectional labeling by future close-to-close return quantiles (Scheme A).

    Steps:
      1) For each symbol, compute fwd_ret = close[t+horizon]/close[t] - 1
      2) For each date, compute cross-sectional quantiles of fwd_ret among all symbols on that date
      3) Label:
           y_cls = 2 (Up)       if fwd_ret >= q_up_quantile
           y_cls = 0 (Down)     if fwd_ret <= q_dn_quantile
           y_cls = 1 (Sideways) otherwise
    Notes:
      - Uses ONLY close prices (no high/low).
      - Drops rows where fwd_ret is NaN (e.g., tail where horizon not available).
      - Dates with too-small universe (<min_universe) are labeled as NaN and will be dropped later.
    """
    d = df.sort_values([sym_col, date_col]).copy()
    # future close-to-close return within each symbol
    d["fwd_ret"] = d.groupby(sym_col, sort=False)[close_col].transform(lambda s: s.shift(-horizon) / (s + 1e-12) - 1.0)

    def _per_date(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        v = g["fwd_ret"].astype(float)
        v = v.replace([np.inf, -np.inf], np.nan).dropna()
        if int(v.shape[0]) < int(min_universe):
            g["y_cls"] = np.nan
            return g
        up_th = float(v.quantile(q_up))
        dn_th = float(v.quantile(q_dn))
        fr = g["fwd_ret"].astype(float)
        y = np.full(len(g), np.nan, dtype=np.float32)
        m = np.isfinite(fr.to_numpy())
        frv = fr.to_numpy()
        y[m & (frv >= up_th)] = 2.0
        y[m & (frv <= dn_th)] = 0.0
        y[m & (frv > dn_th) & (frv < up_th)] = 1.0
        g["y_cls"] = y
        return g

    d = d.groupby(date_col, group_keys=False).apply(_per_date)
    d["y_cls"] = d["y_cls"].astype("float")
    return d




def pick_feature_cols(df: pd.DataFrame, sym_col: str, date_col: str) -> List[str]:
    """
    取“所有指标”作为输入：默认把非数值列、价格列、标签列排除，其余数值列全部当特征。
    你加的新增成交量变化特征也会自动包含进来。
    """
    exclude = {
        sym_col, date_col,
        "open", "high", "low", "close", "adj_close", "volume",
        "y_cls",
        "preclose", "amount", "turnover", "pct_chg", "adjclose",
        "fwd_ret"
    }
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric feature columns found. Check your CSV includes indicator columns.")
    return numeric_cols



class SeqDataset(Dataset):
    """Memory- and speed-efficient sliding-window dataset.

    Stores per-symbol numpy arrays and uses cumulative counts to map a global index
    to (symbol_id, local_start). Avoids pandas .loc in __getitem__.
    """

    def __init__(self, df: pd.DataFrame, sym_col: str, date_col: str,
                 feature_cols: List[str], seq_len: int):
        self.seq_len = int(seq_len)
        self.feature_cols = list(feature_cols)

        df = df.sort_values([sym_col, date_col]).copy()
        need_cols = self.feature_cols + ["y_cls"]
        df = _sanitize_numeric(df, need_cols)
        df = df.dropna(subset=need_cols)

        self.X_list = []  # list of (n_i, F)
        self.y_list = []  # list of (n_i, 1)
        self.counts = []  # samples per symbol

        for _, g in df.groupby(sym_col, sort=False):
            x = g[self.feature_cols].to_numpy(dtype=np.float16, copy=True)
            y = g[["y_cls"]].to_numpy(dtype=np.int8, copy=True)
            n = x.shape[0]
            if n >= self.seq_len:
                self.X_list.append(x)
                self.y_list.append(y)
                self.counts.append(n - self.seq_len + 1)

        if not self.counts:
            raise ValueError("No valid sequences found. Check seq_len and missing values.")

        self.cum = np.cumsum(np.array(self.counts, dtype=np.int64))

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, idx: int):
        idx = int(idx)
        sym_i = int(np.searchsorted(self.cum, idx, side="right"))
        prev = int(self.cum[sym_i - 1]) if sym_i > 0 else 0
        local = idx - prev
        start = local
        end = local + self.seq_len

        x = self.X_list[sym_i][start:end]  # (T,F)
        y = int(self.y_list[sym_i][end - 1][0])  # scalar int
        x_tensor = torch.from_numpy(x.astype(np.float32, copy=False))
        y_tensor_long = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor_long


def fit_standardizer(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, np.ndarray]:
    tmp = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    mu = tmp.mean(axis=0).to_numpy(dtype=np.float32)
    sd = tmp.std(axis=0).replace(0, np.nan).to_numpy(dtype=np.float32)
    sd = np.where(np.isfinite(sd), sd, 1.0).astype(np.float32)
    return {"mean": mu, "std": sd}


def apply_standardizer(df: pd.DataFrame, feature_cols: List[str], stats: Dict[str, np.ndarray]) -> pd.DataFrame:
    df = df.copy()
    mu, sd = stats["mean"], stats["std"]
    tmp = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    vals = tmp.to_numpy(dtype=np.float32)
    vals = (vals - mu) / sd
    df[feature_cols] = vals
    return df

# -------------------------
# Evaluation helpers
# -------------------------

# --- Helper to build eligible last-step rows in the same way as SeqDataset ---
def build_laststep_frame_like_dataset(
    df: pd.DataFrame,
    sym_col: str,
    date_col: str,
    feature_cols: List[str],
    seq_len: int,
) -> pd.DataFrame:
    """Build the validation rows that correspond 1:1 to SeqDataset samples.

    SeqDataset does:
      - sort by (sym, date)
      - sanitize numeric
      - dropna on feature_cols + targets
      - for each symbol with n>=seq_len, create (n-seq_len+1) samples
      - label is y at the last step of each window (rows at positions seq_len-1 .. n-1)

    This function returns those last-step rows concatenated in the same groupby order.
    """
    d = df.sort_values([sym_col, date_col]).copy()
    # Try both possible target cols for backward compat
    for tcol in [["y_cls"], ["y_up90", "y_dn10"]]:
        need_cols = list(feature_cols) + tcol
        if all(c in d.columns for c in tcol):
            break
    d = _sanitize_numeric(d, need_cols)
    d = d.dropna(subset=need_cols)

    pieces = []
    for _, g in d.groupby(sym_col, sort=False):
        n = len(g)
        if n >= seq_len:
            # last-step rows for each sliding window
            pieces.append(g.iloc[seq_len - 1 :].copy())

    if not pieces:
        return d.iloc[0:0].copy()

    out = pd.concat(pieces, ignore_index=True)
    return out










# -------------------------
# GRU Channel Classifier
# -------------------------

class GRUChannelClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2, num_classes: int = 3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (B,T,F)
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return self.head(h)


# -------------------------
# Train
# -------------------------
def train_one_epoch(model, loader, optimizer, device, grad_clip: float, class_weights: Optional[torch.Tensor] = None, show_pbar: bool = False):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    total_loss = 0.0
    n = 0
    it = loader
    if show_pbar:
        it = tqdm(loader, desc="train", leave=True, dynamic_ncols=True)
    for x, y in it:
        x = x.to(device)  # (B,T,F)
        y = y.to(device)  # (B,)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)  # (B,3)
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            # Print minimal diagnostics then raise to stop training early
            print("[NaN/Inf loss detected]")
            print("  x finite:", torch.isfinite(x).float().mean().item())
            print("  y finite:", torch.isfinite(y).float().mean().item())
            print("  logits finite:", torch.isfinite(logits).float().mean().item())
            raise RuntimeError("Loss became NaN/Inf. Check feature sanitation and label generation.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
        if show_pbar:
            it.set_postfix(loss=float(loss.item()))
    return total_loss / max(n, 1)




@torch.no_grad()
def eval_epoch_cls(model, loader, device, show_pbar: bool = False):
    """Return (ce_loss, accuracy, balanced_accuracy, f1_up) over samples.
    f1_up: F1 score for class y==2 (Up).
    """
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    n = 0
    correct = 0
    cls_correct = np.zeros(3, dtype=np.int64)
    cls_total = np.zeros(3, dtype=np.int64)
    tp_up = 0
    fp_up = 0
    fn_up = 0

    it = loader
    if show_pbar:
        it = tqdm(loader, desc="val", leave=True, dynamic_ncols=True)

    for x, y in it:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item())

        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        n += int(y.numel())

        for k in (0, 1, 2):
            mask = (y == k)
            cls_total[k] += int(mask.sum().item())
            if mask.any():
                cls_correct[k] += int((pred[mask] == k).sum().item())

        # F1 for class 2 ("Up")
        tp_up += int(((pred == 2) & (y == 2)).sum().item())
        fp_up += int(((pred == 2) & (y != 2)).sum().item())
        fn_up += int(((pred != 2) & (y == 2)).sum().item())

        if show_pbar:
            it.set_postfix(loss=float(loss.item()) / max(int(y.numel()), 1))

    if n <= 0:
        return math.inf, 0.0, 0.0, 0.0

    acc = correct / n
    per_cls_acc = np.where(cls_total > 0, cls_correct / np.maximum(cls_total, 1), 0.0)
    bal_acc = float(per_cls_acc.mean())
    prec_up = tp_up / max(tp_up + fp_up, 1)
    rec_up  = tp_up / max(tp_up + fn_up, 1)
    f1_up = (2 * prec_up * rec_up) / max(prec_up + rec_up, 1e-12)
    return total_loss / n, float(acc), float(bal_acc), float(f1_up)


@torch.no_grad()
def eval_val_topfrac_hit(
    model,
    loader,
    device,
    eligible_val: pd.DataFrame,
    date_col: str,
    top_frac: float = 0.10,
    ret_threshold: float = 0.10,
    show_pbar: bool = False,
) -> Tuple[float, int, float]:
    """
    Compute a practical validation metric:
      - score = p_up - p_down
      - for each date, take top `top_frac` by score
      - compute hit-rate = fraction with fwd_ret >= ret_threshold
      - return mean hit-rate across dates, number of evaluated dates, mean universe/day

    Returns: (mean_hit_rate, days_evaluated, mean_universe)
    """
    model.eval()

    probs_list = []
    for x, _y in (tqdm(loader, desc="val_score", leave=False, dynamic_ncols=True) if show_pbar else loader):
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs_list.append(probs)

    if not probs_list:
        return 0.0, 0, 0.0

    probs_all = np.concatenate(probs_list, axis=0)
    if len(eligible_val) != probs_all.shape[0]:
        print(f"[WARN] alignment mismatch for top-frac metric: eligible_val={len(eligible_val)} preds={probs_all.shape[0]}")
        return 0.0, 0, 0.0

    ev = eligible_val.copy()
    ev["p_up"] = probs_all[:, 2]
    ev["p_down"] = probs_all[:, 0]
    ev["score"] = ev["p_up"] - ev["p_down"]

    # require fwd_ret and score
    ev = ev.dropna(subset=[date_col, "score", "fwd_ret"]).copy()
    if len(ev) == 0:
        return 0.0, 0, 0.0

    def _per_date(g: pd.DataFrame) -> pd.Series:
        g = g.dropna(subset=["score", "fwd_ret"])
        n = len(g)
        if n < 50:
            return pd.Series({"hit": np.nan, "universe": n})
        g = g.sort_values("score", ascending=False)
        k = max(int(math.floor(n * float(top_frac))), 1)
        top = g.iloc[:k]
        hit = float((top["fwd_ret"] >= float(ret_threshold)).mean())
        return pd.Series({"hit": hit, "universe": n})

    daily = ev.groupby(date_col, group_keys=False).apply(_per_date)
    daily = daily.dropna(subset=["hit"])
    if len(daily) == 0:
        return 0.0, 0, 0.0

    mean_hit = float(daily["hit"].mean())
    days = int(daily.shape[0])
    mean_uni = float(daily["universe"].mean())
    return mean_hit, days, mean_uni


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--train_csv",
        default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2020-2025_all.csv",
        help="Path to training CSV"
    )
    ap.add_argument(
        "--val_csv",
        default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv",
        help="Path to validation CSV"
    )
    ap.add_argument(
        "--outdir",
        default="./gru_output",
        help="Directory to save model and metadata"
    )
    ap.add_argument(
        "--price_col",
        default="close",
        help="Price column name AFTER IO normalization (usually 'close')"
    )
    ap.add_argument(
        "--high_col",
        default="high",
        help="High column name AFTER IO normalization (usually 'high')"
    )
    ap.add_argument(
        "--low_col",
        default="low",
        help="Low column name AFTER IO normalization (usually 'low')"
    )
    # --- Scheme A quantile labeling args ---
    ap.add_argument(
        "--q_up",
        type=float,
        default=0.9,
        help="Upper quantile for Up label (default: 0.80)"
    )
    ap.add_argument(
        "--q_dn",
        type=float,
        default=0.1,
        help="Lower quantile for Down label (default: 0.20)"
    )
    ap.add_argument(
        "--min_universe",
        type=int,
        default=200,
        help="Minimum universe size per date for labeling (default: 200)"
    )
    ap.add_argument(
        "--top_frac",
        type=float,
        default=0.10,
        help="Top fraction of universe by model score for validation selection (default: 0.10)"
    )
    ap.add_argument(
        "--ret_threshold",
        type=float,
        default=0.10,
        help="Return threshold on fwd_ret used to compute hit-rate within selected set (default: 0.10)"
    )
    ap.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs (early stopping may stop earlier)"
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience (stop after this many epochs without val improvement)"
    )
    ap.add_argument(
        "--min_delta",
        type=float,
        default=1e-4,
        help="Minimum val_loss improvement to be considered as an improvement"
    )
    ap.add_argument(
        "--no_pbar",
        action="store_true",
        help="Disable tqdm progress bars"
    )

    ap.add_argument("--batch_size", type=int, default=256, help="Batch size (reduce if out-of-memory)")
    ap.add_argument("--hidden_dim", type=int, default=128, help="GRU hidden size")
    ap.add_argument("--num_layers", type=int, default=2, help="GRU layers")
    ap.add_argument("--dropout", type=float, default=0.2, help="Dropout")

    args = ap.parse_args()

    print("\n=== Running GRU training with parameters ===")
    print(f"train_csv: {args.train_csv}")
    print(f"val_csv:   {args.val_csv}")
    print(f"outdir:    {args.outdir}")
    print(f"price_col: {args.price_col}")
    print(f"q_up:     {args.q_up}")
    print(f"q_dn:     {args.q_dn}")
    print(f"top_frac: {args.top_frac}")
    print(f"ret_threshold: {args.ret_threshold}")
    print(f"max_epochs:{args.max_epochs}")
    print(f"patience:  {args.patience}")
    print(f"min_delta: {args.min_delta}")
    print(f"no_pbar:   {args.no_pbar}")
    print("===========================================\n")

    cfg = TrainConfig(
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        q_up=args.q_up,
        q_dn=args.q_dn,
        min_universe=args.min_universe,
    )

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(cfg.device)

    # Use IO normalization so Chinese/variant columns are mapped to a consistent schema
    # Expected output columns include: code, date, open, high, low, close, volume, ...
    df_tr = load_market_csv_multi([args.train_csv])
    df_va = load_market_csv_multi([args.val_csv])

    # After normalization, we use contract-based column names
    sym_col = "code"
    date_col = "date"

    # Basic schema checks
    for need in [sym_col, date_col, args.price_col]:
        if need not in df_tr.columns:
            raise ValueError(f"train data missing required column after IO normalization: {need}")
        if need not in df_va.columns:
            raise ValueError(f"val data missing required column after IO normalization: {need}")

    # Ensure datetime
    df_tr[date_col] = pd.to_datetime(df_tr[date_col])
    df_va[date_col] = pd.to_datetime(df_va[date_col])

    # Quick schema peek
    print(f"Normalized train columns (head): {list(df_tr.columns)[:30]}")
    print(f"Normalized val columns (head):   {list(df_va.columns)[:30]}")

    # 0) 计算技术指标（20+ 特征）。由于 train/val 在时间上连续，
    #    这里把它们先拼接起来计算“只依赖过去”的指标，避免 val 起始段因为缺历史而产生额外 NaN。
    df_all = pd.concat(
        [df_tr.assign(__split="tr"), df_va.assign(__split="va")],
        ignore_index=True
    )
    df_all = add_indicators(df_all)

    df_tr = df_all[df_all["__split"] == "tr"].drop(columns=["__split"])
    df_va = df_all[df_all["__split"] == "va"].drop(columns=["__split"])

    # 1) 计算 Scheme A 标签：按“日期横截面”的未来收益分位数做 3 分类（Up / Sideways / Down）
    #    注意：train / val 分开计算，避免跨集合阈值泄露（虽然这是横截面阈值，但仍保持严格划分）
    df_tr = compute_cs_fwdret_quantile_labels(
        df_tr, sym_col, date_col,
        close_col=args.price_col,
        horizon=cfg.horizon,
        q_up=float(args.q_up),
        q_dn=float(args.q_dn),
        min_universe=int(args.min_universe),
    )
    df_va = compute_cs_fwdret_quantile_labels(
        df_va, sym_col, date_col,
        close_col=args.price_col,
        horizon=cfg.horizon,
        q_up=float(args.q_up),
        q_dn=float(args.q_dn),
        min_universe=int(args.min_universe),
    )

    # 3) 选特征列：以训练集为准（自动包含你新增的成交量变化特征）
    feature_cols = pick_feature_cols(df_tr, sym_col, date_col)

    # val 必须包含同名特征列
    missing = [c for c in feature_cols if c not in df_va.columns]
    if missing:
        raise ValueError(f"val_csv missing feature columns (present in train): {missing[:10]} ... total={len(missing)}")

    # Keep only minimal columns to reduce RAM
    keep_cols = [sym_col, date_col, args.price_col, args.high_col, args.low_col, "y_cls", "fwd_ret"] + feature_cols
    df_tr = df_tr[keep_cols].copy()
    df_va = df_va[keep_cols].copy()

    # 4) 清理 inf/-inf（否则会导致标准化和 BCE loss 产生 NaN）
    df_tr = _sanitize_numeric(df_tr, feature_cols + ["y_cls", "fwd_ret"])
    df_va = _sanitize_numeric(df_va, feature_cols + ["y_cls", "fwd_ret"])

    # 5) 标准化：只用训练集拟合，然后同一套参数应用到 train / val
    stats = fit_standardizer(df_tr.dropna(subset=feature_cols), feature_cols)
    df_tr = apply_standardizer(df_tr, feature_cols, stats)
    df_va = apply_standardizer(df_va, feature_cols, stats)

    # Build eligible last-step frames (needed for class weights + post-eval alignment)
    eligible_tr = build_laststep_frame_like_dataset(df_tr, sym_col, date_col, feature_cols, cfg.seq_len)
    eligible_va = build_laststep_frame_like_dataset(df_va, sym_col, date_col, feature_cols, cfg.seq_len)

    # 5) Dataset / Loader：train_csv 全部作为训练集，val_csv 全部作为验证集
    train_ds = SeqDataset(df_tr, sym_col, date_col, feature_cols, cfg.seq_len)
    val_ds   = SeqDataset(df_va, sym_col, date_col, feature_cols, cfg.seq_len)

    # Free large frames to reduce peak memory
    del df_all
    del df_tr
    del df_va
    import gc
    gc.collect()

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, drop_last=False)

    print(f"Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,} | Features: {len(feature_cols)}")

    # class weights (based on usable last-step rows after dropna alignment)
    y_counts = eligible_tr["y_cls"].astype(int).value_counts().to_dict()
    total = sum(y_counts.get(k, 0) for k in (0, 1, 2))
    w = []
    for k in (0, 1, 2):
        ck = max(int(y_counts.get(k, 0)), 1)
        w.append(total / (3.0 * ck))
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"Class counts (train eligible): {y_counts} | class_weights: {[round(x,4) for x in w]}")
    print(f"Early-stopping metric: val_top10_hit10 (top_frac={args.top_frac}, ret_threshold={args.ret_threshold})")

    # 6) Model
    model = GRUChannelClassifier(
        input_dim=len(feature_cols),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        num_classes=3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_hit = -math.inf
    best_val_loss = math.inf
    best_path = os.path.join(args.outdir, "gru_channel_best.pt")

    bad_epochs = 0
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, cfg.grad_clip, class_weights=class_weights, show_pbar=(not args.no_pbar))
        va_loss, va_acc, va_bal, va_f1_up = eval_epoch_cls(model, val_loader, device, show_pbar=(not args.no_pbar))
        va_hit, va_days, va_mean_uni = eval_val_topfrac_hit(
            model, val_loader, device, eligible_va, date_col,
            top_frac=args.top_frac, ret_threshold=args.ret_threshold, show_pbar=False
        )
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.5f} | val_loss={va_loss:.5f} | val_acc={va_acc:.5f} | val_bal_acc={va_bal:.5f} | val_f1_up={va_f1_up:.5f} | val_top10_hit10={va_hit:.5f} (days={va_days}, mean_uni={va_mean_uni:.1f})")

        # Early stopping check (maximize val_top10_hit10)
        improved = (va_hit - best_val_hit) > float(args.min_delta)
        if improved:
            best_val_hit = va_hit
            best_val_loss = va_loss
            best_epoch = epoch
            bad_epochs = 0
            ckpt = {
                "target_type": "cs_fwdret_quantile_classification",
                "model_state": model.state_dict(),
                "feature_cols": feature_cols,
                "standardizer": {"mean": stats["mean"].tolist(), "std": stats["std"].tolist()},
                "cfg": cfg.__dict__,
                "sym_col": sym_col,
                "date_col": date_col,
                "price_col": args.price_col,
                "early_stopping": {
                    "max_epochs": int(args.max_epochs),
                    "patience": int(args.patience),
                    "min_delta": float(args.min_delta),
                    "best_val_loss": float(best_val_loss),
                    "best_val_top10_hit10": float(best_val_hit),
                    "best_epoch": int(best_epoch),
                },
            }
            torch.save(ckpt, best_path)
            print(f"  saved: {best_path}")
        else:
            bad_epochs += 1
            print(f"  no improvement (bad_epochs={bad_epochs}/{args.patience})")

        if bad_epochs >= int(args.patience):
            print(f"Early stopping triggered at epoch {epoch}. Best val_top10_hit10={best_val_hit:.5f}")
            break

    # =============================
    # Post-train evaluation on VAL
    # =============================
    if os.path.exists(best_path):
        ckpt_best = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt_best["model_state"])  # type: ignore
        model.eval()

    # Predict class probabilities on VAL in order
    probs_list = []
    y_list = []
    for x, y in (tqdm(val_loader, desc="val_predict", leave=True, dynamic_ncols=True) if (not args.no_pbar) else val_loader):
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()  # (B,3)
        probs_list.append(probs)
        y_list.append(y.numpy())

    probs_all = np.concatenate(probs_list, axis=0) if probs_list else np.zeros((0, 3), dtype=np.float32)
    y_true = np.concatenate(y_list, axis=0) if y_list else np.zeros((0,), dtype=np.int64)

    p_up = probs_all[:, 2] if probs_all.size else np.zeros((0,), dtype=np.float32)
    p_max = probs_all.max(axis=1) if probs_all.size else np.zeros((0,), dtype=np.float32)
    y_pred = probs_all.argmax(axis=1) if probs_all.size else np.zeros((0,), dtype=np.int64)

    # confidence metrics
    # 1) max-prob confidence
    conf_maxprob = p_max
    # 2) margin confidence: top1 - top2
    if probs_all.size:
        part = np.partition(probs_all, -2, axis=1)
        top2 = part[:, -2:]
        conf_margin = top2[:, 1] - top2[:, 0]
    else:
        conf_margin = np.zeros((0,), dtype=np.float32)

    # 3) normalized entropy (0=confident, 1=uncertain)
    if probs_all.size:
        ent = -(probs_all * np.log(probs_all + 1e-12)).sum(axis=1)
        ent_norm = ent / np.log(3.0)
    else:
        ent_norm = np.zeros((0,), dtype=np.float32)

    print("\n=== VAL confidence summary (global) ===")
    if len(conf_maxprob):
        print(f"mean max-prob: {conf_maxprob.mean():.4f} | p50: {np.median(conf_maxprob):.4f} | p90: {np.quantile(conf_maxprob, 0.90):.4f}")
        print(f"mean margin:   {conf_margin.mean():.4f} | p50: {np.median(conf_margin):.4f} | p90: {np.quantile(conf_margin, 0.90):.4f}")
        print(f"mean ent_norm: {ent_norm.mean():.4f} | p50: {np.median(ent_norm):.4f} | p90: {np.quantile(ent_norm, 0.90):.4f} (lower is more confident)")

    eligible_val = eligible_va.copy()
    if len(eligible_val) != int(p_up.shape[0]):
        print(f"[WARN] alignment mismatch in post-eval: eligible_val={len(eligible_val)} preds={p_up.shape[0]}")
    else:
        eligible_val["p_up"] = probs_all[:, 2]
        eligible_val["p_down"] = probs_all[:, 0]
        eligible_val["score"] = eligible_val["p_up"] - eligible_val["p_down"]
        eligible_val["p_max"] = conf_maxprob
        eligible_val["conf_margin"] = conf_margin
        eligible_val["ent_norm"] = ent_norm
        eligible_val["y_pred"] = y_pred
        eligible_val["y_cls"] = eligible_val["y_cls"].astype(int)

        # --- By-date cross-sectional backtest (top/bottom 10%) ---
        top_frac = 0.10
        bottom_frac = 0.10

        def _per_date_cs(g: pd.DataFrame) -> pd.DataFrame:
            g = g.dropna(subset=["score", "fwd_ret", "y_cls"]).copy()
            n = len(g)
            if n < 50:
                return pd.DataFrame()
            g = g.sort_values("score", ascending=False)
            k_top = max(int(math.floor(n * top_frac)), 1)
            k_bot = max(int(math.floor(n * bottom_frac)), 1)

            top = g.iloc[:k_top]
            bot = g.iloc[-k_bot:]

            return pd.DataFrame({
                "date": [g[date_col].iloc[0]],
                "universe": [n],
                "top_up_rate": [float((top["y_cls"] == 2).mean())],
                "bot_up_rate": [float((bot["y_cls"] == 2).mean())],
                "top_fwd_ret": [float(top["fwd_ret"].mean())],
                "bot_fwd_ret": [float(bot["fwd_ret"].mean())],
            })

        daily = eligible_val.groupby(date_col, group_keys=False).apply(_per_date_cs)
        if len(daily) == 0:
            print("\n[WARN] No daily cross-sectional groups available for evaluation.")
        else:
            days_eval = int(daily.shape[0])
            mean_uni = float(daily["universe"].mean())
            top_up = float(daily["top_up_rate"].mean())
            bot_up = float(daily["bot_up_rate"].mean())
            top_ret = float(daily["top_fwd_ret"].mean())
            bot_ret = float(daily["bot_fwd_ret"].mean())

            print("\n=== VAL Cross-sectional (by-date) backtest (Scheme A) ===")
            print(f"score: p_up - p_down | top_frac: {top_frac:.2f} | bottom_frac: {bottom_frac:.2f}")
            print(f"days_evaluated: {days_eval} | mean universe/day: {mean_uni:.1f}")
            print(f"top Up-rate: {top_up:.4f} | bottom Up-rate: {bot_up:.4f} | spread: {top_up - bot_up:.4f}")
            print(f"top fwd_ret: {top_ret:.4f} | bottom fwd_ret: {bot_ret:.4f} | spread: {top_ret - bot_ret:.4f}")

    # Save meta
    meta_path = os.path.join(args.outdir, "gru_channel_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "cs_fwdret_quantile_classification",
                "label_map": {"0": "Down", "1": "Sideways", "2": "Up"},
                "labeling": {
                    "type": "by_date_quantiles",
                    "q_up": args.q_up,
                    "q_dn": args.q_dn,
                    "horizon": cfg.horizon,
                    "min_universe": args.min_universe,
                },
                "confidence": {
                    "primary": "max_softmax_probability",
                    "also_report": ["margin", "normalized_entropy"],
                },
                "feature_dim": len(feature_cols),
                "feature_cols": feature_cols,
                "cfg": cfg.__dict__,
                "early_stopping": {
                    "max_epochs": int(args.max_epochs),
                    "patience": int(args.patience),
                    "min_delta": float(args.min_delta),
                    "best_val_loss": float(best_val_loss),
                    "best_val_top10_hit10": float(best_val_hit),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"meta saved: {meta_path}")


if __name__ == "__main__":
    main()