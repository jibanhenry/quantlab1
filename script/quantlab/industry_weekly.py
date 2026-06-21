# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import time
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _load_akshare():
    try:
        import akshare as ak  # type: ignore
        return ak
    except Exception as exc:
        raise ImportError("akshare is required for industry weekly download") from exc


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _parse_yyyymmdd(value: Optional[str], default: Optional[str] = None) -> str:
    raw = value or default
    if not raw:
        return datetime.now().strftime("%Y%m%d")
    text = str(raw).strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"date must be YYYYMMDD or YYYY-MM-DD, got: {value}")
    return text


def _normalize_industries(value: Optional[str]) -> Optional[set[str]]:
    if not value:
        return None
    items = [x.strip() for x in str(value).replace("，", ",").split(",") if x.strip()]
    return set(items) if items else None


def _normalize_industry_list(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["industry", "board_code"])
    out = raw.copy()
    rename = {
        "name": "industry",
        "code": "board_code",
        "板块名称": "industry",
        "板块代码": "board_code",
        "最新价": "last_price",
        "涨跌额": "change",
        "涨跌幅": "pct_chg",
        "总市值": "total_mv",
        "换手率": "turnover",
        "上涨家数": "up_count",
        "下跌家数": "down_count",
        "领涨股票": "leading_stock",
        "领涨股票-涨跌幅": "leading_stock_pct_chg",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    keep = []
    for col in rename.values():
        if col in out.columns and col not in keep:
            keep.append(col)
    out = out[keep].copy()
    out["industry"] = out["industry"].astype(str).str.strip()
    if "board_code" in out.columns:
        out["board_code"] = out["board_code"].astype(str).str.strip()
    for col in [c for c in out.columns if c not in {"industry", "board_code", "leading_stock"}]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.drop_duplicates("industry", keep="last").sort_values("industry").reset_index(drop=True)


def _fetch_industry_list(ak) -> pd.DataFrame:
    try:
        out = _normalize_industry_list(
            _call_with_retries(lambda: ak.stock_board_industry_name_em(), retries=3, sleep_seconds=1.0)
        )
        if not out.empty:
            out["data_source"] = "eastmoney"
            return out
    except Exception:
        pass
    out = _normalize_industry_list(
        _call_with_retries(lambda: ak.stock_board_industry_name_ths(), retries=3, sleep_seconds=1.0)
    )
    if not out.empty:
        out["data_source"] = "ths"
    return out


def _normalize_weekly_kline(raw: pd.DataFrame, industry: str, board_code: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    rename = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "换手率": "turnover",
    }
    out = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns}).copy()
    keep = ["date", "industry", "board_code", "open", "high", "low", "close", "pct_chg", "change", "volume", "amount", "amplitude", "turnover"]
    out["industry"] = industry
    out["board_code"] = board_code
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in [c for c in keep if c not in {"date", "industry", "board_code"}]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    out = out[keep].dropna(subset=["date", "industry"])
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def _normalize_fund_flow_daily(raw: pd.DataFrame, industry: str, board_code: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    rename = {
        "日期": "date",
        "主力净流入-净额": "main_net_inflow",
        "主力净流入-净占比": "main_net_inflow_pct",
        "超大单净流入-净额": "super_net_inflow",
        "超大单净流入-净占比": "super_net_inflow_pct",
        "大单净流入-净额": "large_net_inflow",
        "大单净流入-净占比": "large_net_inflow_pct",
        "中单净流入-净额": "medium_net_inflow",
        "中单净流入-净占比": "medium_net_inflow_pct",
        "小单净流入-净额": "small_net_inflow",
        "小单净流入-净占比": "small_net_inflow_pct",
    }
    out = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns}).copy()
    keep = ["date", "industry", "board_code"] + [v for k, v in rename.items() if v != "date"]
    out["industry"] = industry
    out["board_code"] = board_code
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in [c for c in keep if c not in {"date", "industry", "board_code"}]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    out = out[keep].dropna(subset=["date", "industry"])
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def _normalize_ths_daily_to_weekly(raw: pd.DataFrame, industry: str, board_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    rename = {
        "日期": "date",
        "开盘价": "open",
        "最高价": "high",
        "最低价": "low",
        "收盘价": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    daily = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns}).copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        daily[col] = pd.to_numeric(daily.get(col), errors="coerce")
    start = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce")
    end = pd.to_datetime(end_date, format="%Y%m%d", errors="coerce")
    daily = daily[daily["date"].between(start, end)].dropna(subset=["date", "open", "high", "low", "close"])
    if daily.empty:
        return pd.DataFrame()
    daily["week_end"] = daily["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    weekly = daily.groupby("week_end", as_index=False).agg(
        actual_date=("date", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    )
    prev_close = weekly["close"].shift(1)
    weekly["change"] = weekly["close"] - prev_close
    weekly["pct_chg"] = weekly["change"] / prev_close * 100.0
    weekly["amplitude"] = (weekly["high"] - weekly["low"]) / prev_close * 100.0
    weekly["turnover"] = np.nan
    weekly["industry"] = industry
    weekly["board_code"] = board_code
    weekly = weekly.rename(columns={"actual_date": "date"})
    keep = ["date", "industry", "board_code", "open", "high", "low", "close", "pct_chg", "change", "volume", "amount", "amplitude", "turnover"]
    weekly["date"] = pd.to_datetime(weekly["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return weekly[keep].sort_values("date").reset_index(drop=True)


def aggregate_fund_flow_weekly(flow_daily: pd.DataFrame) -> pd.DataFrame:
    if flow_daily is None or flow_daily.empty:
        return pd.DataFrame(columns=["date", "industry", "board_code"])
    work = flow_daily.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date", "industry"])
    work["week_end"] = work["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    amount_cols = [c for c in work.columns if c.endswith("_inflow") and not c.endswith("_pct")]
    pct_cols = [c for c in work.columns if c.endswith("_pct")]
    agg = {col: "sum" for col in amount_cols}
    agg.update({col: "mean" for col in pct_cols})
    out = work.groupby(["industry", "board_code", "week_end"], as_index=False).agg(agg)
    out = out.rename(columns={"week_end": "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.sort_values(["industry", "date"]).reset_index(drop=True)


def _empty_fund_flow_daily() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "industry",
            "board_code",
            "main_net_inflow",
            "main_net_inflow_pct",
            "super_net_inflow",
            "super_net_inflow_pct",
            "large_net_inflow",
            "large_net_inflow_pct",
            "medium_net_inflow",
            "medium_net_inflow_pct",
            "small_net_inflow",
            "small_net_inflow_pct",
        ]
    )


def _empty_fund_flow_weekly() -> pd.DataFrame:
    return _empty_fund_flow_daily()


def _merge_unique(existing: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    frames = [df for df in [existing, new] if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def _last_date_for_industry(df: pd.DataFrame, industry: str) -> Optional[pd.Timestamp]:
    if df is None or df.empty or "industry" not in df.columns or "date" not in df.columns:
        return None
    dates = pd.to_datetime(df.loc[df["industry"].eq(industry), "date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max()


def _filter_start(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return df
    start = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce")
    out = df.copy()
    dt = pd.to_datetime(out["date"], errors="coerce")
    return out[dt.ge(start)].reset_index(drop=True)


def _call_with_retries(func, retries: int = 3, sleep_seconds: float = 1.0):
    last_exc = None
    for i in range(max(1, int(retries))):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if i < retries - 1 and sleep_seconds:
                time.sleep(float(sleep_seconds) * (i + 1))
    raise last_exc


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _build_viewer_payload(kline: pd.DataFrame, flow_weekly: pd.DataFrame) -> dict:
    if kline is None or kline.empty:
        return {"industries": [], "records": [], "has_fund_flow": False, "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    work = kline.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date", "industry", "close"]).sort_values(["industry", "date"]).reset_index(drop=True)
    for col in ["open", "high", "low", "close", "pct_chg", "volume", "amount"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    for window in [5, 10, 20]:
        work[f"ma{window}"] = work.groupby("industry")["close"].transform(lambda s: s.rolling(window, min_periods=1).mean())

    flow_cols = ["main_net_inflow", "main_net_inflow_pct"]
    if flow_weekly is not None and not flow_weekly.empty and all(c in flow_weekly.columns for c in ["date", "industry"]):
        flow = flow_weekly.copy()
        flow["date"] = pd.to_datetime(flow["date"], errors="coerce")
        for col in flow_cols:
            flow[col] = pd.to_numeric(flow.get(col), errors="coerce")
        work = work.merge(flow[["date", "industry"] + flow_cols], on=["date", "industry"], how="left")
    else:
        for col in flow_cols:
            work[col] = np.nan

    records = []
    for row in work.itertuples(index=False):
        records.append(
            {
                "d": row.date.strftime("%Y-%m-%d"),
                "i": row.industry,
                "o": _json_value(getattr(row, "open", np.nan)),
                "h": _json_value(getattr(row, "high", np.nan)),
                "l": _json_value(getattr(row, "low", np.nan)),
                "c": _json_value(getattr(row, "close", np.nan)),
                "p": _json_value(getattr(row, "pct_chg", np.nan)),
                "v": _json_value(getattr(row, "volume", np.nan)),
                "a": _json_value(getattr(row, "amount", np.nan)),
                "ma5": _json_value(getattr(row, "ma5", np.nan)),
                "ma10": _json_value(getattr(row, "ma10", np.nan)),
                "ma20": _json_value(getattr(row, "ma20", np.nan)),
                "mf": _json_value(getattr(row, "main_net_inflow", np.nan)),
                "mfp": _json_value(getattr(row, "main_net_inflow_pct", np.nan)),
            }
        )
    industries = sorted(work["industry"].dropna().astype(str).unique().tolist())
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date_min": work["date"].min().strftime("%Y-%m-%d"),
        "date_max": work["date"].max().strftime("%Y-%m-%d"),
        "industries": industries,
        "records": records,
        "has_fund_flow": bool(pd.to_numeric(work["main_net_inflow"], errors="coerce").notna().any()),
    }


def build_board_weekly_viewer(
    outdir: str,
    kline_filename: str,
    flow_weekly_filename: str,
    json_filename: str,
    html_filename: str,
    title: str,
    board_label: str,
) -> dict:
    kline_path = os.path.join(outdir, kline_filename)
    flow_weekly_path = os.path.join(outdir, flow_weekly_filename)
    json_path = os.path.join(outdir, json_filename)
    html_path = os.path.join(outdir, html_filename)

    kline = _safe_read_csv(kline_path)
    flow_weekly = _safe_read_csv(flow_weekly_path)
    payload = _build_viewer_payload(kline, flow_weekly)
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    _write_text(payload_json, json_path)
    _write_text(_viewer_html(payload_json, title=title, board_label=board_label), html_path)
    return {"json_path": json_path, "html_path": html_path, "records": len(payload.get("records", [])), "industries": len(payload.get("industries", []))}


def build_industry_weekly_viewer(outdir: str) -> dict:
    return build_board_weekly_viewer(
        outdir=outdir,
        kline_filename="industry_weekly_kline.csv",
        flow_weekly_filename="industry_fund_flow_weekly.csv",
        json_filename="industry_weekly_data.json",
        html_filename="industry_weekly_viewer.html",
        title="A股行业周线查看器",
        board_label="行业",
    )


def _viewer_html(payload_json: str, title: str = "A股行业周线查看器", board_label: str = "行业") -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>""" + title + """</title>
<style>
:root { color-scheme: light; --text:#172033; --muted:#687386; --line:#d9dee8; --soft:#f5f7fb; --blue:#2563eb; --red:#dc2626; --green:#16a34a; }
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: #fff; }
header { padding: 18px 24px 10px; border-bottom: 1px solid var(--line); }
h1 { margin: 0 0 6px; font-size: 20px; font-weight: 700; letter-spacing: 0; }
.meta { color: var(--muted); font-size: 13px; }
.layout { display: grid; grid-template-columns: 280px 1fr; min-height: calc(100vh - 74px); }
aside { border-right: 1px solid var(--line); padding: 16px; background: var(--soft); }
main { padding: 16px 20px 24px; min-width: 0; }
label { display: block; margin: 12px 0 6px; color: #334155; font-size: 13px; font-weight: 600; }
select, input { width: 100%; min-height: 34px; border: 1px solid #cbd5e1; border-radius: 6px; background: #fff; color: var(--text); padding: 6px 8px; font-size: 13px; }
select[multiple] { height: 260px; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.checks label { display: flex; align-items: center; gap: 8px; margin: 10px 0; font-weight: 500; }
.checks input { width: 16px; min-height: 16px; }
button { min-height: 34px; border: 1px solid #cbd5e1; border-radius: 6px; background: #fff; color: var(--text); padding: 6px 10px; cursor: pointer; }
button.primary { background: var(--blue); border-color: var(--blue); color: #fff; }
.btns { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
.chart-wrap { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; margin-bottom: 14px; background: #fff; }
.chart-title { display:flex; justify-content:space-between; gap:12px; align-items:center; padding: 10px 12px; border-bottom: 1px solid var(--line); font-size: 13px; font-weight: 700; }
.chart-title span { color: var(--muted); font-weight: 500; }
svg { display: block; width: 100%; height: 430px; }
#volumeChart, #flowChart { height: 190px; }
.empty { padding: 36px; text-align: center; color: var(--muted); }
.legend { display:flex; flex-wrap:wrap; gap:8px 14px; padding: 0 2px 12px; color: var(--muted); font-size: 12px; }
.legend i { display:inline-block; width: 16px; height: 3px; vertical-align: middle; margin-right: 5px; }
.tip { position: fixed; pointer-events:none; display:none; padding:8px 10px; background:#111827; color:#fff; border-radius:6px; font-size:12px; max-width:280px; z-index:3; }
@media (max-width: 880px) { .layout { grid-template-columns: 1fr; } aside { border-right:0; border-bottom:1px solid var(--line); } select[multiple] { height: 180px; } }
</style>
</head>
<body>
<header>
  <h1>""" + title + """</h1>
  <div class="meta" id="meta"></div>
</header>
<div class="layout">
  <aside>
    <label for="industrySelect">""" + board_label + """</label>
    <select id="industrySelect" multiple></select>
    <div class="btns">
      <button id="allBtn">全选</button>
      <button id="clearBtn">清空</button>
    </div>
    <div class="row">
      <div><label for="startDate">开始</label><input id="startDate" type="date" /></div>
      <div><label for="endDate">结束</label><input id="endDate" type="date" /></div>
    </div>
    <div class="checks">
      <label><input id="ma5" type="checkbox" checked /> 5周线</label>
      <label><input id="ma10" type="checkbox" checked /> 10周线</label>
      <label><input id="ma20" type="checkbox" checked /> 20周线</label>
      <label><input id="volumeToggle" type="checkbox" checked /> 成交量</label>
      <label><input id="flowToggle" type="checkbox" checked /> 主力流入</label>
    </div>
    <button class="primary" id="applyBtn" style="width:100%;margin-top:8px;">更新图表</button>
  </aside>
  <main>
    <div class="legend" id="legend"></div>
    <section class="chart-wrap">
      <div class="chart-title">走势 <span id="priceNote"></span></div>
      <div id="priceBox"><svg id="priceChart"></svg></div>
    </section>
    <section class="chart-wrap" id="volumeWrap">
      <div class="chart-title">成交量 <span>所选行业合计</span></div>
      <svg id="volumeChart"></svg>
    </section>
    <section class="chart-wrap" id="flowWrap">
      <div class="chart-title">主力净流入 <span id="flowNote"></span></div>
      <svg id="flowChart"></svg>
    </section>
  </main>
</div>
<div class="tip" id="tip"></div>
<script id="payload" type="application/json">""" + payload_json.replace("</", "<\\/") + """</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const colors = ["#2563eb","#dc2626","#16a34a","#9333ea","#ea580c","#0891b2","#4f46e5","#be123c","#15803d","#a16207","#0f766e","#7c3aed"];
const $ = id => document.getElementById(id);
const fmt = n => n == null || Number.isNaN(n) ? "-" : Number(n).toLocaleString("zh-CN", {maximumFractionDigits: 2});
const fmtBig = n => n == null || Number.isNaN(n) ? "-" : (Math.abs(n) >= 1e8 ? (n/1e8).toFixed(2)+"亿" : fmt(n));
const byIndustry = new Map();
for (const r of DATA.records) {
  if (!byIndustry.has(r.i)) byIndustry.set(r.i, []);
  byIndustry.get(r.i).push(r);
}

function init() {
  $("meta").textContent = `生成时间 ${DATA.generated_at} ｜ ${DATA.industries.length} 个行业 ｜ ${DATA.date_min} 至 ${DATA.date_max}`;
  $("startDate").value = DATA.date_min;
  $("endDate").value = DATA.date_max;
  $("industrySelect").innerHTML = DATA.industries.map(x => `<option value="${x}" selected>${x}</option>`).join("");
  $("flowNote").textContent = DATA.has_fund_flow ? "所选行业合计" : "暂无资金流数据";
  for (const id of ["applyBtn","ma5","ma10","ma20","volumeToggle","flowToggle","startDate","endDate"]) $(id).addEventListener("change", render);
  $("applyBtn").addEventListener("click", render);
  $("allBtn").addEventListener("click", () => { for (const o of $("industrySelect").options) o.selected = true; render(); });
  $("clearBtn").addEventListener("click", () => { for (const o of $("industrySelect").options) o.selected = false; render(); });
  $("industrySelect").addEventListener("change", render);
  window.addEventListener("resize", render);
  render();
}

function selectedIndustries() {
  return Array.from($("industrySelect").selectedOptions).map(o => o.value);
}

function filtered() {
  const start = $("startDate").value || DATA.date_min;
  const end = $("endDate").value || DATA.date_max;
  return selectedIndustries().map((name, idx) => ({
    name,
    color: colors[idx % colors.length],
    rows: (byIndustry.get(name) || []).filter(r => r.d >= start && r.d <= end)
  })).filter(s => s.rows.length);
}

function bounds(values) {
  const nums = values.filter(v => v != null && Number.isFinite(v));
  if (!nums.length) return [0, 1];
  let min = Math.min(...nums), max = Math.max(...nums);
  if (min === max) { min *= 0.98; max *= 1.02; }
  const pad = (max - min) * 0.06;
  return [min - pad, max + pad];
}

function clearSvg(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }
function el(name, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [k,v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  return node;
}

function drawAxes(svg, width, height, minY, maxY, dates) {
  const pad = {l:54,r:18,t:18,b:28};
  const plotW = Math.max(10, width-pad.l-pad.r), plotH = Math.max(10, height-pad.t-pad.b);
  for (let i=0;i<=4;i++) {
    const y = pad.t + plotH * i / 4;
    svg.appendChild(el("line", {x1:pad.l, y1:y, x2:width-pad.r, y2:y, stroke:"#e5e7eb", "stroke-width":1}));
    const val = maxY - (maxY-minY)*i/4;
    svg.appendChild(el("text", {x:8, y:y+4, fill:"#64748b", "font-size":11}, ));
    svg.lastChild.textContent = fmtBig(val);
  }
  const ticks = dates.length > 1 ? [0, Math.floor(dates.length/3), Math.floor(dates.length*2/3), dates.length-1] : [0];
  for (const idx of [...new Set(ticks)]) {
    const x = pad.l + plotW * idx / Math.max(1, dates.length-1);
    svg.appendChild(el("text", {x:x, y:height-8, fill:"#64748b", "font-size":11, "text-anchor":"middle"}));
    svg.lastChild.textContent = dates[idx] || "";
  }
  return {pad, plotW, plotH, x:i => pad.l + plotW * i / Math.max(1, dates.length-1), y:v => pad.t + (maxY-v)/(maxY-minY)*plotH};
}

function linePath(rows, yKey, scale, normalizeBase=null) {
  let d = "";
  rows.forEach((r, i) => {
    let val = r[yKey];
    if (val == null) return;
    if (normalizeBase) val = val / normalizeBase * 100;
    const cmd = d ? "L" : "M";
    d += `${cmd}${scale.x(i).toFixed(1)},${scale.y(val).toFixed(1)}`;
  });
  return d;
}

function renderPrice(series) {
  const svg = $("priceChart"); clearSvg(svg);
  const box = svg.getBoundingClientRect(), width = Math.max(600, box.width), height = 430;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  if (!series.length) { $("priceBox").innerHTML = '<div class="empty">请选择至少一个行业</div>'; return; }
  if (!$("priceChart")) return;
  const single = series.length === 1;
  $("priceNote").textContent = single ? "单行业周K线，可显示均线" : "多行业按区间首周收盘价归一为 100";
  const dates = series[0].rows.map(r => r.d);
  let vals = [];
  for (const s of series) {
    const base = single ? null : s.rows.find(r => r.c != null)?.c;
    for (const r of s.rows) {
      if (single) {
        if (r.h != null) vals.push(r.h);
        if (r.l != null) vals.push(r.l);
      } else if (r.c != null) {
        vals.push(base ? r.c/base*100 : r.c);
      }
      if (single && $("ma5").checked && r.ma5 != null) vals.push(r.ma5);
      if (single && $("ma10").checked && r.ma10 != null) vals.push(r.ma10);
      if (single && $("ma20").checked && r.ma20 != null) vals.push(r.ma20);
    }
  }
  const [minY,maxY] = bounds(vals);
  const scale = drawAxes(svg, width, height, minY, maxY, dates);
  if (single) {
    const s = series[0];
    drawCandles(svg, s.rows, scale);
    if ($("ma5").checked) svg.appendChild(el("path", {d:linePath(s.rows, "ma5", scale), fill:"none", stroke:"#f59e0b", "stroke-width":1.5}));
    if ($("ma10").checked) svg.appendChild(el("path", {d:linePath(s.rows, "ma10", scale), fill:"none", stroke:"#7c3aed", "stroke-width":1.5}));
    if ($("ma20").checked) svg.appendChild(el("path", {d:linePath(s.rows, "ma20", scale), fill:"none", stroke:"#0f766e", "stroke-width":1.5}));
  } else {
    series.forEach((s) => {
      const base = s.rows.find(r => r.c != null)?.c;
      svg.appendChild(el("path", {d:linePath(s.rows, "c", scale, base), fill:"none", stroke:s.color, "stroke-width":2}));
    });
  }
}

function drawCandles(svg, rows, scale) {
  const candleW = Math.max(2, Math.min(12, scale.plotW / Math.max(1, rows.length) * 0.62));
  rows.forEach((r, i) => {
    if ([r.o, r.h, r.l, r.c].some(v => v == null || !Number.isFinite(v))) return;
    const x = scale.x(i);
    const up = r.c >= r.o;
    const color = up ? "#dc2626" : "#16a34a";
    const yHigh = scale.y(r.h), yLow = scale.y(r.l), yOpen = scale.y(r.o), yClose = scale.y(r.c);
    svg.appendChild(el("line", {x1:x, y1:yHigh, x2:x, y2:yLow, stroke:color, "stroke-width":1.2}));
    const bodyY = Math.min(yOpen, yClose);
    const bodyH = Math.max(1, Math.abs(yClose - yOpen));
    svg.appendChild(el("rect", {x:x-candleW/2, y:bodyY, width:candleW, height:bodyH, fill:up ? "rgba(220,38,38,0.14)" : color, stroke:color, "stroke-width":1}));
  });
}

function aggregateByDate(series, key) {
  const map = new Map();
  for (const s of series) for (const r of s.rows) {
    if (!map.has(r.d)) map.set(r.d, 0);
    map.set(r.d, map.get(r.d) + (Number(r[key]) || 0));
  }
  return Array.from(map.entries()).sort((a,b)=>a[0].localeCompare(b[0])).map(([d,v]) => ({d,v}));
}

function renderBars(id, rows, posColor, negColor=null) {
  const wrap = $(id === "volumeChart" ? "volumeWrap" : "flowWrap");
  wrap.style.display = (id === "volumeChart" ? $("volumeToggle").checked : $("flowToggle").checked) ? "block" : "none";
  const svg = $(id); clearSvg(svg);
  if (wrap.style.display === "none") return;
  const box = svg.getBoundingClientRect(), width = Math.max(600, box.width), height = 190;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  if (!rows.length || rows.every(r => !r.v)) {
    svg.appendChild(el("text", {x:width/2, y:height/2, fill:"#64748b", "font-size":13, "text-anchor":"middle"}));
    svg.lastChild.textContent = id === "flowChart" ? "暂无资金流数据" : "暂无成交量数据";
    return;
  }
  const dates = rows.map(r => r.d);
  const vals = rows.map(r => r.v || 0);
  const minY = Math.min(0, ...vals), maxY = Math.max(0, ...vals);
  const scale = drawAxes(svg, width, height, minY, maxY || 1, dates);
  const barW = Math.max(1, scale.plotW / Math.max(1, rows.length) * 0.72);
  const zeroY = scale.y(0);
  rows.forEach((r, i) => {
    const x = scale.x(i) - barW/2, y = scale.y(Math.max(0, r.v || 0));
    const h = Math.abs(scale.y(r.v || 0) - zeroY);
    svg.appendChild(el("rect", {x, y: (r.v || 0) >= 0 ? y : zeroY, width:barW, height:Math.max(1,h), fill:(r.v || 0) >= 0 ? posColor : (negColor || posColor), opacity:0.72}));
  });
}

function renderLegend(series) {
  let html = series.slice(0, 36).map(s => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("");
  if (series.length > 36) html += `<span>其余 ${series.length-36} 个行业已显示在图中</span>`;
  if (series.length === 1) html += '<span><i style="background:#f59e0b"></i>MA5</span><span><i style="background:#7c3aed"></i>MA10</span><span><i style="background:#0f766e"></i>MA20</span>';
  $("legend").innerHTML = html;
}

function render() {
  const box = $("priceBox");
  if (!box.querySelector("svg")) box.innerHTML = '<svg id="priceChart"></svg>';
  const series = filtered();
  renderLegend(series);
  renderPrice(series);
  renderBars("volumeChart", aggregateByDate(series, "v"), "#64748b");
  renderBars("flowChart", aggregateByDate(series, "mf"), "#dc2626", "#16a34a");
}
init();
</script>
</body>
</html>
"""


def run_industry_weekly_update(
    outdir: str,
    start_date: str = "20200101",
    end_date: Optional[str] = None,
    refresh: bool = False,
    industries: Optional[str] = None,
    with_fund_flow: bool = True,
    sleep_seconds: float = 0.15,
    generate_viewer: bool = True,
) -> dict[str, pd.DataFrame]:
    ak = _load_akshare()
    os.makedirs(outdir, exist_ok=True)
    start_yyyymmdd = _parse_yyyymmdd(start_date, "20200101")
    end_yyyymmdd = _parse_yyyymmdd(end_date)
    selected = _normalize_industries(industries)

    list_path = os.path.join(outdir, "industry_list.csv")
    kline_path = os.path.join(outdir, "industry_weekly_kline.csv")
    flow_daily_path = os.path.join(outdir, "industry_fund_flow_daily.csv")
    flow_weekly_path = os.path.join(outdir, "industry_fund_flow_weekly.csv")
    status_path = os.path.join(outdir, "industry_update_status.csv")

    try:
        industry_list = _fetch_industry_list(ak)
    except Exception:
        industry_list = _safe_read_csv(list_path)
        if industry_list.empty:
            raise
    if selected is not None:
        industry_list = industry_list[industry_list["industry"].isin(selected)].reset_index(drop=True)
    _write_csv(industry_list, list_path)

    existing_kline = pd.DataFrame() if refresh else _safe_read_csv(kline_path)
    existing_flow = pd.DataFrame() if refresh else _safe_read_csv(flow_daily_path)
    kline_frames = []
    flow_frames = []
    status_rows = []

    for _, row in industry_list.iterrows():
        industry = str(row["industry"])
        board_code = str(row.get("board_code", ""))
        data_source = str(row.get("data_source", "eastmoney"))
        last_dt = _last_date_for_industry(existing_kline, industry)
        fetch_start = start_yyyymmdd
        if last_dt is not None:
            fetch_start = (last_dt + pd.Timedelta(days=1)).strftime("%Y%m%d")
        if pd.to_datetime(fetch_start, format="%Y%m%d") > pd.to_datetime(end_yyyymmdd, format="%Y%m%d"):
            status_rows.append({"industry": industry, "board_code": board_code, "target": "kline", "status": "up_to_date", "rows": 0, "message": ""})
        else:
            try:
                if data_source == "ths":
                    raw = _call_with_retries(
                        lambda: ak.stock_board_industry_index_ths(
                            symbol=industry,
                            start_date=fetch_start,
                            end_date=end_yyyymmdd,
                        ),
                        retries=3,
                        sleep_seconds=1.0,
                    )
                    kline = _normalize_ths_daily_to_weekly(raw, industry, board_code, fetch_start, end_yyyymmdd)
                    target = "kline_ths_weekly"
                else:
                    raw = _call_with_retries(
                        lambda: ak.stock_board_industry_hist_em(
                            symbol=industry,
                            start_date=fetch_start,
                            end_date=end_yyyymmdd,
                            period="周k",
                            adjust="",
                        ),
                        retries=3,
                        sleep_seconds=1.0,
                    )
                    kline = _normalize_weekly_kline(raw, industry, board_code)
                    target = "kline"
                kline_frames.append(kline)
                status_rows.append({"industry": industry, "board_code": board_code, "target": target, "status": "ok", "rows": len(kline), "message": ""})
            except Exception as exc:
                status_rows.append({"industry": industry, "board_code": board_code, "target": "kline", "status": "error", "rows": 0, "message": str(exc)[:500]})
        if sleep_seconds:
            time.sleep(float(sleep_seconds))

        if with_fund_flow:
            try:
                raw_flow = _call_with_retries(
                    lambda: ak.stock_sector_fund_flow_hist(symbol=industry),
                    retries=3,
                    sleep_seconds=1.0,
                )
                flow = _normalize_fund_flow_daily(raw_flow, industry, board_code)
                flow = _filter_start(flow, start_yyyymmdd)
                if end_yyyymmdd:
                    end_dt = pd.to_datetime(end_yyyymmdd, format="%Y%m%d", errors="coerce")
                    flow = flow[pd.to_datetime(flow["date"], errors="coerce").le(end_dt)].reset_index(drop=True)
                flow_frames.append(flow)
                status_rows.append({"industry": industry, "board_code": board_code, "target": "fund_flow", "status": "ok", "rows": len(flow), "message": ""})
            except Exception as exc:
                status_rows.append({"industry": industry, "board_code": board_code, "target": "fund_flow", "status": "error", "rows": 0, "message": str(exc)[:500]})
            if sleep_seconds:
                time.sleep(float(sleep_seconds))

    new_kline = pd.concat(kline_frames, ignore_index=True) if kline_frames else pd.DataFrame()
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
        flow_all = existing_flow
        flow_weekly = _safe_read_csv(flow_weekly_path)

    status = pd.DataFrame(status_rows)
    if not status.empty:
        status.insert(0, "run_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        _write_csv(status, status_path)

    viewer_info = build_industry_weekly_viewer(outdir) if generate_viewer else {}

    result = {
        "industry_list": industry_list,
        "industry_weekly_kline": kline_all,
        "industry_fund_flow_daily": flow_all,
        "industry_fund_flow_weekly": flow_weekly,
        "industry_update_status": status,
    }
    result["viewer_info"] = viewer_info
    return result
