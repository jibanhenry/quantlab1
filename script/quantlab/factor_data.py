# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


SNAPSHOT_COL_MAP = {
    "代码": "code",
    "名称": "name",
    "最新价": "last_price",
    "市盈率-动态": "pe_dynamic",
    "市净率": "pb_lf",
    "总市值": "total_mv",
    "流通市值": "circ_mv",
    "成交额": "spot_amount",
    "换手率": "spot_turnover",
}

INDUSTRY_COL_MAP = {
    "代码": "code",
    "名称": "name",
    "板块名称": "industry",
    "行业": "industry",
}

FINANCIAL_INDICATOR_COL_MAP = {
    "ROEJQ": "roe",
    "XSMLL": "gross_margin",
    "XSJLL": "net_margin",
    "ZCFZL": "debt_to_assets",
    "TOTALOPERATEREVETZ": "revenue_yoy",
    "PARENTNETPROFITTZ": "net_profit_yoy",
    "NCO_NETPROFIT": "ocf_to_profit",
}


def normalize_code(code) -> str:
    s = str(code).strip()
    m = re.search(r"(\d{6})", s)
    if m:
        return m.group(1)
    if "." in s:
        s = s.split(".")[-1]
    return s.zfill(6)


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_cache(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _load_akshare():
    try:
        import akshare as ak  # type: ignore
        return ak
    except Exception:
        return None


def _eastmoney_indicator_symbol(code: str) -> str:
    normalized = normalize_code(code)
    suffix = "SH" if normalized.startswith(("5", "6", "9")) else ("BJ" if normalized.startswith(("4", "8")) else "SZ")
    return f"{normalized}.{suffix}"


def _financial_code_cache_path(cache_dir: str, code: str) -> str:
    return os.path.join(cache_dir, "financial_indicator_by_code", f"{normalize_code(code)}.csv")


def _normalize_financial_indicator(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    required = ["REPORT_DATE", "NOTICE_DATE"]
    if raw is None or raw.empty or not all(c in raw.columns for c in required):
        return pd.DataFrame()
    report_period = pd.to_datetime(raw["REPORT_DATE"], errors="coerce")
    notice_date = pd.to_datetime(raw["NOTICE_DATE"], errors="coerce")
    effective_date = notice_date.where(notice_date.ge(report_period), report_period + pd.Timedelta(days=90))
    out = pd.DataFrame(
        {
            "code": normalize_code(code),
            "report_period": report_period,
            "effective_date": effective_date,
            "update_date": pd.to_datetime(raw.get("UPDATE_DATE"), errors="coerce"),
        }
    )
    for source, target in FINANCIAL_INDICATOR_COL_MAP.items():
        out[target] = pd.to_numeric(raw[source], errors="coerce") if source in raw.columns else np.nan
    out = out.dropna(subset=["report_period", "effective_date"])
    return (
        out.sort_values(["report_period", "effective_date", "update_date"])
        .drop_duplicates(["code", "report_period"], keep="last")
        .sort_values("effective_date")
        .reset_index(drop=True)
    )


def _assemble_financial_cache(cache_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(cache_dir, "financial_indicator_by_code", "*.csv")))
    frames = [_safe_read_csv(p) for p in files]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame(columns=["code", "effective_date"])
    out = pd.concat(frames, ignore_index=True)
    out["code"] = out["code"].map(normalize_code)
    out["effective_date"] = pd.to_datetime(out["effective_date"], errors="coerce")
    out["report_period"] = pd.to_datetime(out["report_period"], errors="coerce")
    fallback = out["report_period"] + pd.Timedelta(days=90)
    out["effective_date"] = out["effective_date"].where(out["effective_date"].ge(out["report_period"]), fallback)
    out = out.dropna(subset=["effective_date"]).drop_duplicates(["code", "report_period"], keep="last")
    _write_cache(out, os.path.join(cache_dir, "akshare_financial_indicators.csv"))
    return out


def load_akshare_financial_cache(cache_dir: str) -> pd.DataFrame:
    path = os.path.join(cache_dir, "akshare_financial_indicators.csv")
    cached = _safe_read_csv(path)
    if cached.empty:
        return _assemble_financial_cache(cache_dir)
    cached["code"] = cached["code"].map(normalize_code)
    cached["effective_date"] = pd.to_datetime(cached["effective_date"], errors="coerce")
    if "report_period" in cached.columns:
        cached["report_period"] = pd.to_datetime(cached["report_period"], errors="coerce")
    return cached


def fetch_akshare_financial_indicators(
    codes: Sequence[str],
    cache_dir: str,
    refresh: bool = False,
    workers: int = 4,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch announcement-date financial indicators with per-code resumable cache."""
    os.makedirs(os.path.join(cache_dir, "financial_indicator_by_code"), exist_ok=True)
    requested = list(dict.fromkeys(normalize_code(c) for c in codes if str(c).strip()))
    if limit is not None and limit > 0:
        requested = requested[:limit]
    pending = [c for c in requested if refresh or not os.path.exists(_financial_code_cache_path(cache_dir, c))]
    ak = _load_akshare()
    if ak is None:
        raise ImportError("akshare is required for financial indicator download")

    def _fetch_one(code: str):
        try:
            raw = ak.stock_financial_analysis_indicator_em(
                symbol=_eastmoney_indicator_symbol(code),
                indicator="按报告期",
            )
            normalized = _normalize_financial_indicator(raw, code)
            if normalized.empty:
                return code, "empty", 0, ""
            _write_cache(normalized, _financial_code_cache_path(cache_dir, code))
            return code, "ok", len(normalized), ""
        except Exception as exc:
            return code, "error", 0, str(exc)[:200]

    status_rows = []
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {pool.submit(_fetch_one, code): code for code in pending}
            for done, future in enumerate(as_completed(futures), start=1):
                code, status, rows, error = future.result()
                status_rows.append({"code": code, "status": status, "rows": rows, "error": error, "fetched_at": datetime.now().isoformat(timespec="seconds")})
                if done % 25 == 0 or done == len(pending):
                    print(f"[factor_finance] fetched={done}/{len(pending)} cached_before={len(requested) - len(pending)}", flush=True)
                time.sleep(0.01)
    status_path = os.path.join(cache_dir, "akshare_financial_fetch_status.csv")
    prior = _safe_read_csv(status_path)
    if status_rows or not prior.empty:
        status = pd.concat([prior, pd.DataFrame(status_rows)], ignore_index=True) if not prior.empty else pd.DataFrame(status_rows)
        status["code"] = status["code"].map(normalize_code)
        status = status.drop_duplicates("code", keep="last")
        non_equity = status["code"].str.startswith("399") & status["status"].eq("error")
        status.loc[non_equity, "status"] = "unsupported_non_equity"
        _write_cache(status, status_path)
    out = _assemble_financial_cache(cache_dir)
    print(f"[factor_finance] cache rows={len(out):,}, codes={out['code'].nunique() if not out.empty else 0:,}", flush=True)
    return out


def fetch_akshare_snapshot(cache_dir: str, refresh: bool = False) -> pd.DataFrame:
    """Fetch/cache a current A-share snapshot. Falls back to an empty frame."""
    today = datetime.now().strftime("%Y%m%d")
    path = os.path.join(cache_dir, f"akshare_spot_{today}.csv")
    if not refresh:
        cached = _safe_read_csv(path)
        if not cached.empty:
            cached["code"] = cached["code"].map(normalize_code)
            return cached
        return pd.DataFrame()

    ak = _load_akshare()
    if ak is None:
        return pd.DataFrame()
    try:
        raw = ak.stock_zh_a_spot_em()
    except Exception:
        return pd.DataFrame()

    df = raw.rename(columns={k: v for k, v in SNAPSHOT_COL_MAP.items() if k in raw.columns}).copy()
    if "code" not in df.columns:
        return pd.DataFrame()
    df["code"] = df["code"].map(normalize_code)
    keep = [c for c in SNAPSHOT_COL_MAP.values() if c in df.columns]
    df = df[keep].drop_duplicates("code", keep="last")
    _write_cache(df, path)
    return df


def fetch_akshare_industry(cache_dir: str, refresh: bool = False) -> pd.DataFrame:
    """Best-effort industry membership cache from AKShare Eastmoney board APIs."""
    today = datetime.now().strftime("%Y%m%d")
    path = os.path.join(cache_dir, f"akshare_industry_{today}.csv")
    if not refresh:
        cached = _safe_read_csv(path)
        if not cached.empty:
            cached["code"] = cached["code"].map(normalize_code)
            return cached
        return pd.DataFrame()

    ak = _load_akshare()
    if ak is None:
        return pd.DataFrame()

    rows = []
    try:
        boards = ak.stock_board_industry_name_em()
        name_col = "板块名称" if "板块名称" in boards.columns else boards.columns[0]
        for industry in boards[name_col].dropna().astype(str).unique().tolist():
            try:
                cons = ak.stock_board_industry_cons_em(symbol=industry)
            except Exception:
                continue
            if cons is None or cons.empty:
                continue
            code_col = "代码" if "代码" in cons.columns else None
            name_col2 = "名称" if "名称" in cons.columns else None
            if code_col is None:
                continue
            tmp = pd.DataFrame(
                {
                    "code": cons[code_col].map(normalize_code),
                    "industry": industry,
                    "name": cons[name_col2].astype(str) if name_col2 else np.nan,
                }
            )
            rows.append(tmp)
    except Exception:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True).drop_duplicates("code", keep="last")
    _write_cache(df, path)
    return df


def load_external_factor_data(
    cache_dir: Optional[str] = None,
    refresh: bool = False,
    extra_csvs: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Load optional non-price data keyed by code.

    This is intentionally best-effort: if AKShare is unavailable or an endpoint
    changes, the factor pipeline degrades to price/PB/PS factors.
    """
    point_in_time_frames = []
    static_frames = []
    cache_dir = cache_dir or os.path.join(os.getcwd(), "output", "factor_cache")
    financial = load_akshare_financial_cache(cache_dir)
    if not financial.empty:
        point_in_time_frames.append(financial)
    snapshot = fetch_akshare_snapshot(cache_dir, refresh=refresh)
    if not snapshot.empty:
        static_frames.append(snapshot)
    industry = fetch_akshare_industry(cache_dir, refresh=refresh)
    if not industry.empty:
        static_frames.append(industry)

    for p in extra_csvs or []:
        df = _safe_read_csv(p)
        if df.empty or "code" not in df.columns:
            continue
        df = df.copy()
        df["code"] = df["code"].map(normalize_code)
        if "effective_date" not in df.columns:
            announcement_col = next((c for c in ["announcement_date", "ann_date", "公告日期"] if c in df.columns), None)
            report_col = next((c for c in ["report_period", "报告期", "报告日期"] if c in df.columns), None)
            if announcement_col:
                df["effective_date"] = pd.to_datetime(df[announcement_col], errors="coerce")
            elif report_col:
                df["effective_date"] = pd.to_datetime(df[report_col], errors="coerce") + pd.Timedelta(days=90)
        if "effective_date" in df.columns:
            point_in_time_frames.append(df)
        else:
            static_frames.append(df)

    if not point_in_time_frames and not static_frames:
        return pd.DataFrame(columns=["code"])

    if point_in_time_frames:
        out = point_in_time_frames[0].copy()
        for df in point_in_time_frames[1:]:
            overlap = [c for c in df.columns if c in out.columns and c not in {"code", "effective_date"}]
            df2 = df.drop(columns=overlap)
            out = out.merge(df2, on=["code", "effective_date"], how="outer")
    else:
        out = static_frames.pop(0).copy()

    for df in static_frames:
        overlap = [c for c in df.columns if c in out.columns and c != "code"]
        df2 = df.drop(columns=overlap)
        out = out.merge(df2, on="code", how="outer")
    if "effective_date" in out.columns:
        out["effective_date"] = pd.to_datetime(out["effective_date"], errors="coerce")
        return out.sort_values(["code", "effective_date"]).drop_duplicates(["code", "effective_date"], keep="last")
    return out.drop_duplicates("code", keep="last")


def merge_external_data(panel: pd.DataFrame, external: Optional[pd.DataFrame]) -> pd.DataFrame:
    if external is None or external.empty or "code" not in external.columns:
        return panel
    out = panel.copy()
    out["code"] = out["code"].map(normalize_code)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    ext = external.copy()
    ext["code"] = ext["code"].map(normalize_code)
    overlap = [c for c in ext.columns if c in out.columns and c not in {"code", "date"}]
    ext = ext.drop(columns=overlap)
    effective_col = "effective_date" if "effective_date" in ext.columns else ("date" if "date" in ext.columns else None)
    if effective_col:
        ext["effective_date"] = pd.to_datetime(ext[effective_col], errors="coerce")
        ext = ext.drop(columns=["date"], errors="ignore").dropna(subset=["effective_date"])
        left = out.sort_values(["date", "code"])
        right = ext.sort_values(["effective_date", "code"])
        merged = pd.merge_asof(
            left,
            right,
            left_on="date",
            right_on="effective_date",
            by="code",
            direction="backward",
            allow_exact_matches=True,
        )
        snapshot_cols = [c for c in SNAPSHOT_COL_MAP.values() if c not in {"code", "name"} and c in merged.columns]
        if snapshot_cols:
            old_rows = merged["date"] < merged["date"].max()
            merged.loc[old_rows, snapshot_cols] = np.nan
        return merged.sort_values(["code", "date"]).reset_index(drop=True)

    # Point-in-time snapshots are useful for latest candidates, not historical tests.
    ext_cols = [c for c in ext.columns if c != "code"]
    merged = out.merge(ext, on="code", how="left")
    old_rows = merged["date"] < merged["date"].max()
    merged.loc[old_rows, ext_cols] = np.nan
    return merged
