# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional

import pandas as pd

from .industry_weekly import (
    _call_with_retries,
    _empty_fund_flow_daily,
    _empty_fund_flow_weekly,
    _filter_start,
    _load_akshare,
    _merge_unique,
    _normalize_fund_flow_daily,
    _normalize_industries,
    _normalize_ths_daily_to_weekly,
    _parse_yyyymmdd,
    _safe_read_csv,
    _write_csv,
    aggregate_fund_flow_weekly,
    build_board_weekly_viewer,
)


def _normalize_concept_list(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["industry", "board_code", "data_source"])
    out = raw.rename(columns={"name": "industry", "code": "board_code", "板块名称": "industry", "板块代码": "board_code"}).copy()
    keep = []
    for col in ["industry", "board_code"]:
        if col in out.columns and col not in keep:
            keep.append(col)
    out = out[keep].copy()
    out["industry"] = out["industry"].astype(str).str.strip()
    out["board_code"] = out["board_code"].astype(str).str.strip()
    out["data_source"] = "ths"
    return out.drop_duplicates("industry", keep="last").sort_values("industry").reset_index(drop=True)


def _last_date_for_concept(df: pd.DataFrame, concept: str) -> Optional[pd.Timestamp]:
    if df is None or df.empty or "industry" not in df.columns or "date" not in df.columns:
        return None
    dates = pd.to_datetime(df.loc[df["industry"].eq(concept), "date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max()


def build_concept_weekly_viewer(outdir: str) -> dict:
    return build_board_weekly_viewer(
        outdir=outdir,
        kline_filename="concept_weekly_kline.csv",
        flow_weekly_filename="concept_fund_flow_weekly.csv",
        json_filename="concept_weekly_data.json",
        html_filename="concept_weekly_viewer.html",
        title="A股概念周线查看器",
        board_label="概念",
    )


def run_concept_weekly_update(
    outdir: str,
    start_date: str = "20200101",
    end_date: Optional[str] = None,
    refresh: bool = False,
    concepts: Optional[str] = None,
    with_fund_flow: bool = False,
    sleep_seconds: float = 0.05,
    generate_viewer: bool = True,
) -> dict[str, pd.DataFrame]:
    ak = _load_akshare()
    os.makedirs(outdir, exist_ok=True)
    start_yyyymmdd = _parse_yyyymmdd(start_date, "20200101")
    end_yyyymmdd = _parse_yyyymmdd(end_date)
    selected = _normalize_industries(concepts)

    list_path = os.path.join(outdir, "concept_list.csv")
    kline_path = os.path.join(outdir, "concept_weekly_kline.csv")
    status_path = os.path.join(outdir, "concept_update_status.csv")
    flow_daily_path = os.path.join(outdir, "concept_fund_flow_daily.csv")
    flow_weekly_path = os.path.join(outdir, "concept_fund_flow_weekly.csv")

    try:
        concept_list = _normalize_concept_list(
            _call_with_retries(lambda: ak.stock_board_concept_name_ths(), retries=3, sleep_seconds=1.0)
        )
    except Exception:
        concept_list = _safe_read_csv(list_path)
        if concept_list.empty:
            raise
    if selected is not None:
        concept_list = concept_list[concept_list["industry"].isin(selected)].reset_index(drop=True)
    _write_csv(concept_list, list_path)

    existing_kline = pd.DataFrame() if refresh else _safe_read_csv(kline_path)
    existing_flow = pd.DataFrame() if refresh else _safe_read_csv(flow_daily_path)
    frames = []
    flow_frames = []
    status_rows = []
    for _, row in concept_list.iterrows():
        concept = str(row["industry"])
        board_code = str(row.get("board_code", ""))
        last_dt = _last_date_for_concept(existing_kline, concept)
        fetch_start = start_yyyymmdd
        if last_dt is not None:
            fetch_start = (last_dt + pd.Timedelta(days=1)).strftime("%Y%m%d")
        if pd.to_datetime(fetch_start, format="%Y%m%d") > pd.to_datetime(end_yyyymmdd, format="%Y%m%d"):
            status_rows.append({"industry": concept, "board_code": board_code, "target": "kline", "status": "up_to_date", "rows": 0, "message": ""})
            continue
        try:
            raw = _call_with_retries(
                lambda: ak.stock_board_concept_index_ths(
                    symbol=concept,
                    start_date=fetch_start,
                    end_date=end_yyyymmdd,
                ),
                retries=3,
                sleep_seconds=1.0,
            )
            kline = _normalize_ths_daily_to_weekly(raw, concept, board_code, fetch_start, end_yyyymmdd)
            frames.append(kline)
            status_rows.append({"industry": concept, "board_code": board_code, "target": "kline_ths_weekly", "status": "ok", "rows": len(kline), "message": ""})
        except Exception as exc:
            status_rows.append({"industry": concept, "board_code": board_code, "target": "kline_ths_weekly", "status": "error", "rows": 0, "message": str(exc)[:500]})
        if sleep_seconds:
            time.sleep(float(sleep_seconds))

        if with_fund_flow:
            try:
                raw_flow = _call_with_retries(
                    lambda: ak.stock_concept_fund_flow_hist(symbol=concept),
                    retries=3,
                    sleep_seconds=1.0,
                )
                flow = _normalize_fund_flow_daily(raw_flow, concept, board_code)
                flow = _filter_start(flow, start_yyyymmdd)
                if end_yyyymmdd:
                    end_dt = pd.to_datetime(end_yyyymmdd, format="%Y%m%d", errors="coerce")
                    flow = flow[pd.to_datetime(flow["date"], errors="coerce").le(end_dt)].reset_index(drop=True)
                flow_frames.append(flow)
                status_rows.append({"industry": concept, "board_code": board_code, "target": "fund_flow", "status": "ok", "rows": len(flow), "message": ""})
            except Exception as exc:
                status_rows.append({"industry": concept, "board_code": board_code, "target": "fund_flow", "status": "error", "rows": 0, "message": str(exc)[:500]})
            if sleep_seconds:
                time.sleep(float(sleep_seconds))

    new_kline = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    kline_all = _merge_unique(existing_kline, new_kline, ["industry", "date"])
    if not kline_all.empty:
        kline_all = kline_all[kline_all["date"].notna()].copy()
        _write_csv(kline_all, kline_path)

    if with_fund_flow:
        new_flow = pd.concat(flow_frames, ignore_index=True) if flow_frames else pd.DataFrame()
        flow_all = _merge_unique(existing_flow, new_flow, ["industry", "date"])
        if not flow_all.empty:
            flow_all = flow_all[flow_all["date"].notna()].copy()
            _write_csv(flow_all, flow_daily_path)
            flow_weekly = aggregate_fund_flow_weekly(flow_all)
            _write_csv(flow_weekly, flow_weekly_path)
        else:
            flow_all = _empty_fund_flow_daily()
            flow_weekly = _empty_fund_flow_weekly()
            _write_csv(flow_all, flow_daily_path)
            _write_csv(flow_weekly, flow_weekly_path)
    else:
        flow_all = existing_flow if not existing_flow.empty else _empty_fund_flow_daily()
        flow_weekly = _safe_read_csv(flow_weekly_path)
        if flow_weekly.empty:
            flow_weekly = _empty_fund_flow_weekly()
        if not os.path.exists(flow_daily_path):
            _write_csv(flow_all, flow_daily_path)
        if not os.path.exists(flow_weekly_path):
            _write_csv(flow_weekly, flow_weekly_path)

    status = pd.DataFrame(status_rows)
    if not status.empty:
        status.insert(0, "run_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        _write_csv(status, status_path)

    viewer_info = build_concept_weekly_viewer(outdir) if generate_viewer else {}
    return {
        "concept_list": concept_list,
        "concept_weekly_kline": kline_all,
        "concept_fund_flow_daily": flow_all,
        "concept_fund_flow_weekly": flow_weekly,
        "concept_update_status": status,
        "viewer_info": viewer_info,
    }
