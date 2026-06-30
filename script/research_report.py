#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified daily/weekly research report.

This script is intentionally a reporting layer: it updates market data, runs the
existing research lines, saves compact intermediate CSVs, and renders static HTML.
It does not change the underlying strategy logic.
"""
from __future__ import annotations

import argparse
import html
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEGACY_DAILY_CSV = Path("/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv")
DEFAULT_DAILY_CSV = str(ROOT / "output" / "market_data" / "2025_06_daily.csv")
DEFAULT_BASE_CSV = "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2020-2025_all.csv"
DEFAULT_CFG = str(ROOT / "output" / "tuned_config_quarterly_20260225.json")
DEFAULT_MODEL = str(ROOT / "model" / "xgb_20250831.joblib")
DEFAULT_BUCKET_MAP = str(ROOT / "output" / "quantlab" / "bucket_map_202602.csv")
DEFAULT_UPDATE_WORKERS = 1


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_code(value) -> str:
    text = str(value).split(".")[-1].strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def _recent_weekday(today: Optional[datetime] = None) -> str:
    day = today or datetime.now()
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day.strftime("%Y-%m-%d")


def _csvs_arg(base_csv: str, daily_csv: str) -> List[str]:
    paths = []
    if base_csv and Path(base_csv).exists():
        paths.append(base_csv)
    paths.append(daily_csv)
    return paths


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _safe_mkdir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _run_cmd(args: List[str], cwd: Path = ROOT) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=str(cwd), check=True)


def _load_daily_stats(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {
            "rows": 0,
            "codes": 0,
            "latest_date": "",
            "codes_at_latest": 0,
            "lagging_codes": 0,
        }
    df = pd.read_csv(path, usecols=lambda c: c in {"code", "日期"}, dtype={"code": str}, parse_dates=["日期"])
    df["code"] = df["code"].map(_normalize_code)
    latest_by_code = df.groupby("code")["日期"].max()
    latest_date = df["日期"].max()
    codes_at_latest = int(latest_by_code.eq(latest_date).sum()) if pd.notna(latest_date) else 0
    return {
        "rows": int(len(df)),
        "codes": int(df["code"].nunique()),
        "latest_date": latest_date.date().isoformat() if pd.notna(latest_date) else "",
        "codes_at_latest": codes_at_latest,
        "lagging_codes": int(len(latest_by_code) - codes_at_latest),
    }


def _seed_default_daily_csv(daily_csv: Path) -> None:
    if daily_csv.exists():
        return
    if daily_csv.resolve() != Path(DEFAULT_DAILY_CSV).resolve():
        return
    if not LEGACY_DAILY_CSV.exists():
        return
    _safe_mkdir(daily_csv.parent)
    print(f"初始化项目内日线缓存：{LEGACY_DAILY_CSV} -> {daily_csv}", flush=True)
    shutil.copy2(LEGACY_DAILY_CSV, daily_csv)


def update_market_data(
    daily_csv: Path,
    target_date: str,
    skip_update: bool,
    allow_stale_on_update_fail: bool,
    update_workers: int,
    update_batch_size: int,
    update_flush_every: int,
    update_max_codes: int,
) -> Dict[str, object]:
    _seed_default_daily_csv(daily_csv)
    before = _load_daily_stats(daily_csv)
    update_status = "跳过更新" if skip_update else "成功"
    update_error = ""
    if not skip_update:
        try:
            from script.data_updater import update_data

            update_data(
                output_file=str(daily_csv),
                end_date=target_date,
                include_new_codes=True,
                sleep=0.01,
                workers=update_workers,
                batch_size=update_batch_size,
                flush_every=update_flush_every,
                max_update_codes=update_max_codes if update_max_codes > 0 else None,
            )
        except Exception as exc:
            update_error = str(exc)
            if not allow_stale_on_update_fail or int(before["rows"]) <= 0:
                raise RuntimeError(
                    "行情更新失败，且未允许使用本地缓存继续生成报告。"
                    f" daily_csv={daily_csv} target_date={target_date} error={update_error}"
                ) from exc
            update_status = "更新失败，使用本地缓存"
            print(
                "[WARN] 行情更新失败，使用现有本地日线继续生成报告："
                f"{update_error}",
                flush=True,
            )
    after = _load_daily_stats(daily_csv)
    return {
        **after,
        "target_date": target_date,
        "rows_before": before["rows"],
        "rows_added_est": int(after["rows"]) - int(before["rows"]),
        "update_status": update_status,
        "update_error": update_error,
        "update_workers": update_workers,
    }


def latest_matching_file(outdir: Path, prefix: str, suffix: str = ".csv") -> Optional[Path]:
    files = sorted(outdir.glob(f"{prefix}*{suffix}"))
    return files[-1] if files else None


def _latest_date_from_csv(path: Path, date_columns: Iterable[str]) -> pd.Timestamp:
    if not path.exists():
        return pd.NaT
    try:
        header = pd.read_csv(path, nrows=0)
        date_col = next((col for col in date_columns if col in header.columns), None)
        if date_col is None:
            return pd.NaT
        values = pd.read_csv(path, usecols=[date_col])[date_col]
        return pd.to_datetime(values, errors="coerce").max()
    except Exception:
        return pd.NaT


def run_line1_daily(
    csvs: List[str],
    cfg: str,
    bucket_map_csv: str,
    workdir: Path,
    threshold: float,
    reuse_existing: bool,
) -> pd.DataFrame:
    _safe_mkdir(workdir)
    if not reuse_existing:
        from script.quantlab.pipeline import daily_run

        daily_run(
            csvs,
            cfg_path=cfg if cfg and Path(cfg).exists() else None,
            outdir=str(workdir),
            bucket_map_csv=bucket_map_csv if bucket_map_csv and Path(bucket_map_csv).exists() else None,
            save_signals=False,
            save_trades=True,
            save_summary=True,
            save_candidates=True,
            export_virtual_trades=False,
        )
    cand_file = latest_matching_file(workdir, "candidates_")
    if cand_file is None and reuse_existing:
        cand_file = latest_matching_file(ROOT / "output" / "quantlab", "candidates_")
    df = _read_csv(cand_file) if cand_file else pd.DataFrame()
    if df.empty:
        return df
    if "predicted_return" not in df.columns:
        try:
            from script.quantlab.model import add_predictions_to_candidates

            df = add_predictions_to_candidates(df)
            if cand_file is not None and "predicted_return" in df.columns:
                _write_csv(df, cand_file)
        except Exception as exc:
            print(f"[line1] failed to add predicted_return: {exc}", file=sys.stderr, flush=True)
    if "predicted_return" in df.columns:
        df = df[pd.to_numeric(df["predicted_return"], errors="coerce") > threshold].copy()
        df = df.sort_values("predicted_return", ascending=False)
    else:
        print(
            "[line1] missing predicted_return; returning empty line1 instead of unfiltered candidates.",
            file=sys.stderr,
            flush=True,
        )
        return pd.DataFrame()
    return df


def run_line2_portfolio(
    csvs: List[str],
    cfg: str,
    workdir: Path,
    reuse_existing: bool,
    top_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _safe_mkdir(workdir)
    if not reuse_existing:
        from script.quantlab.portfolio import run_portfolio_regime_daily

        run_portfolio_regime_daily(
            csvs,
            cfg_path=cfg if cfg and Path(cfg).exists() else None,
            outdir=str(workdir),
            regime_lookback_months=3,
            action_recent_days=10,
            save_signal_panel=False,
        )
    elif not (workdir / "portfolio_latest_candidates.csv").exists():
        fallback = ROOT / "output" / "quantlab_regime_daily_latest"
        if fallback.exists():
            workdir = fallback
    candidates = _read_csv(workdir / "portfolio_latest_candidates.csv")
    actions = _read_csv(workdir / "portfolio_daily_actions.csv")
    regime = _read_csv(workdir / "portfolio_daily_regime.csv")

    if not candidates.empty:
        if "recommended_action" in candidates.columns:
            best = candidates[candidates["recommended_action"].astype(str).str.upper().eq("BUY")].copy()
            if best.empty:
                best = candidates.copy()
        else:
            best = candidates.copy()
        sort_cols = [c for c in ["strategy_score", "candidate_score", "confidence"] if c in best.columns]
        if sort_cols:
            best = best.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        candidates = best.head(top_n).copy()
    return candidates, actions, regime


def run_line3_probability(
    daily_csv: Path,
    model_path: str,
    threshold: float,
    workdir: Path,
    reuse_existing: bool,
) -> pd.DataFrame:
    _safe_mkdir(workdir)
    out_path = workdir / "line3_all_predictions.csv"
    daily_latest = _latest_date_from_csv(daily_csv, ("日期", "date"))

    def _read_current_predictions(path: Optional[Path]) -> pd.DataFrame:
        if path is None:
            return pd.DataFrame()
        pred = _read_csv(path, dtype={"code": str})
        if pred.empty:
            return pred
        if pd.notna(daily_latest):
            if "event_date" not in pred.columns:
                return pd.DataFrame()
            pred_latest = pd.to_datetime(pred["event_date"], errors="coerce").max()
            if pd.notna(pred_latest) and pred_latest < daily_latest:
                print(
                    "[line3] cached predictions are stale: "
                    f"{path} latest_event_date={pred_latest.date()} daily_latest={daily_latest.date()}; regenerating.",
                    file=sys.stderr,
                    flush=True,
                )
                return pd.DataFrame()
        return pred

    pred_df = pd.DataFrame()
    if reuse_existing and out_path.exists():
        pred_df = _read_current_predictions(out_path)

    if pred_df.empty and reuse_existing:
        fallback = latest_matching_file(ROOT / "output", "predict_")
        cached = _read_current_predictions(fallback)
        if not cached.empty and "y_prob" in cached.columns:
            pred_df = cached
            _write_csv(pred_df, out_path)

    if pred_df.empty:
        import script.predictor as pred

        try:
            df = pd.read_csv(daily_csv, low_memory=False)
            df = pred._normalize_daily_columns(df)
            if "code" in df.columns:
                df["code"] = df["code"].astype(str).map(_normalize_code)
            payload = pred._load_model_payload(model_path)
            pred_df = pred.predict_windows(df, payload, latest_only=True)
            _write_csv(pred_df, out_path)
        except Exception as exc:
            print(
                f"[line3] failed to run probability model: {exc}; "
                "using existing predictions if available.",
                file=sys.stderr,
                flush=True,
            )
            pred_df = _read_current_predictions(out_path)
            if pred_df.empty:
                return pd.DataFrame()

    if pred_df.empty or "y_prob" not in pred_df.columns:
        return pd.DataFrame()
    pred_df["code"] = pred_df["code"].map(_normalize_code) if "code" in pred_df.columns else ""
    pred_df["event_date"] = pd.to_datetime(pred_df["event_date"], errors="coerce")
    latest_date = pred_df["event_date"].max()
    high = pred_df[pred_df["event_date"].eq(latest_date) & (pd.to_numeric(pred_df["y_prob"], errors="coerce") > threshold)].copy()
    return high.sort_values("y_prob", ascending=False)


def run_line4_weekly_breakout(
    csvs: List[str],
    workdir: Path,
    reuse_existing: bool,
) -> pd.DataFrame:
    _safe_mkdir(workdir)
    if not reuse_existing:
        from script.quantlab.weekly_breakout import run_weekly_breakout_experiment

        result = run_weekly_breakout_experiment(
            csvs,
            outdir=str(workdir),
            min_amount_20w=20_000_000.0,
            total_exposure=0.45,
            cost_bp=2.0,
        )
        panel = result.get("panel", pd.DataFrame())
        if not panel.empty:
            latest = panel[pd.to_datetime(panel["date"]).eq(pd.to_datetime(panel["date"]).max())].copy()
            buy_signal = latest["buy_signal"].fillna(False) if "buy_signal" in latest.columns else pd.Series(False, index=latest.index)
            active_position = latest["active_position"].fillna(False) if "active_position" in latest.columns else pd.Series(False, index=latest.index)
            latest = latest[buy_signal | active_position]
            sort_cols = [c for c in ["buy_signal", "active_position", "amount_20w"] if c in latest.columns]
            if sort_cols:
                latest = latest.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            _write_csv(latest, workdir / "weekly_line4_latest.csv")
    latest = _read_csv(workdir / "weekly_line4_latest.csv", dtype={"code": str})
    if latest.empty and reuse_existing:
        latest = _read_csv(ROOT / "output" / "ma20_breakouts" / "latest_ma20_breakouts.csv", dtype={"code": str})
    if not latest.empty and "code" in latest.columns:
        latest["code"] = latest["code"].map(_normalize_code)
    return latest


def run_line5_mainline(
    industry_kline: str,
    concept_kline: str,
    workdir: Path,
    reuse_existing: bool,
) -> pd.DataFrame:
    _safe_mkdir(workdir)
    if not reuse_existing:
        from script.quantlab.mainline_radar import run_mainline_radar

        run_mainline_radar(
            industry_kline_path=industry_kline,
            concept_kline_path=concept_kline,
            outdir=str(workdir),
            top_n=30,
            min_weeks=30,
        )
    latest = _read_csv(workdir / "mainline_latest.csv")
    if latest.empty and reuse_existing:
        latest = _read_csv(ROOT / "output" / "mainline_radar" / "mainline_latest.csv")
    if latest.empty:
        return latest
    if "signal_grade" in latest.columns:
        latest = latest[latest["signal_grade"].isin(["A", "B"])].copy()
    if "mainline_score" in latest.columns:
        latest = latest.sort_values(["signal_grade", "mainline_score"], ascending=[True, False])
    return latest


def _fmt_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if abs(float(value)) >= 1_000_000:
            return f"{float(value):,.0f}"
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    return str(value)


def _table_html(df: pd.DataFrame, columns: List[str], limit: int = 30) -> str:
    if df is None or df.empty:
        return '<div class="empty">无符合条件标的</div>'
    use_cols = [c for c in columns if c in df.columns]
    if not use_cols:
        use_cols = list(df.columns[:12])
    view = df[use_cols].head(limit).copy()
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in use_cols)
    rows = []
    for _, row in view.iterrows():
        cells = "".join(f"<td>{html.escape(_fmt_value(row[c]))}</td>" for c in use_cols)
        rows.append(f"<tr>{cells}</tr>")
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def _metric_card(label: str, value: object) -> str:
    return f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'


def _render_page(title: str, subtitle: str, metrics: Dict[str, object], sections: List[Dict[str, object]]) -> str:
    metric_html = "".join(_metric_card(k, v) for k, v in metrics.items())
    section_html = []
    for sec in sections:
        section_html.append(
            f"""
            <section>
              <div class="section-head">
                <div>
                  <h2>{html.escape(str(sec.get('title', '')))}</h2>
                  <p>{html.escape(str(sec.get('desc', '')))}</p>
                </div>
                <span class="count">{html.escape(str(sec.get('count', 0)))} 条</span>
              </div>
              {sec.get('table', '')}
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #657080;
      --line: #d7dde5;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --accent: #176b87;
      --accent2: #8a5a12;
      --good: #147a4d;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; background: var(--bg); color: var(--ink); }}
    header {{ padding: 28px 32px 18px; background: var(--panel); border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    header p {{ margin: 0; color: var(--muted); }}
    main {{ padding: 22px 32px 40px; max-width: 1480px; margin: 0 auto; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-bottom: 18px; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 6px; }}
    .metric strong {{ font-size: 20px; }}
    section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; margin: 14px 0; overflow: hidden; }}
    .section-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; padding: 16px 18px; border-bottom: 1px solid var(--line); }}
    h2 {{ margin: 0 0 6px; font-size: 18px; }}
    .section-head p {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .count {{ color: var(--accent); border: 1px solid #b9d7e1; padding: 4px 8px; border-radius: 999px; white-space: nowrap; font-size: 12px; }}
    .table-wrap {{ overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #edf0f4; padding: 9px 10px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ position: sticky; top: 0; background: #f1f5f9; color: #334155; font-weight: 650; }}
    tr:hover td {{ background: #f8fbfd; }}
    .empty {{ padding: 18px; color: var(--muted); }}
    nav {{ margin-top: 12px; }}
    nav a {{ color: var(--accent); margin-right: 14px; text-decoration: none; font-weight: 600; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>{html.escape(subtitle)}</p>
    <nav><a href="index.html">报告索引</a></nav>
  </header>
  <main>
    <div class="metrics">{metric_html}</div>
    {''.join(section_html)}
  </main>
</body>
</html>
"""


def render_index(outdir: Path, daily_name: Optional[str], weekly_name: Optional[str]) -> None:
    links = []
    if daily_name:
        links.append(f'<a href="{html.escape(daily_name)}">最新日报</a>')
    if weekly_name:
        links.append(f'<a href="{html.escape(weekly_name)}">最新周报</a>')
    body = "<br>".join(links) if links else "暂无报告"
    content = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>研究报告索引</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;margin:40px;color:#17202a}}a{{display:block;margin:12px 0;color:#176b87;font-weight:700;text-decoration:none}}</style>
</head><body><h1>研究报告索引</h1>{body}</body></html>"""
    (outdir / "index.html").write_text(content, encoding="utf-8")


def render_failure_report(outdir: Path, stamp: str, daily_csv: Path, error: Exception) -> Path:
    stats = _load_daily_stats(daily_csv)
    metrics = {
        "状态": "行情更新失败，未生成有效日报",
        "本地行情最新日": stats["latest_date"],
        "股票数": stats["codes"],
        "错误": str(error),
    }
    sections = [
        {
            "title": "失败原因",
            "desc": "脚本未完成最新日线更新，因此没有继续生成日报研究线。",
            "count": 0,
            "table": '<div class="empty">请先恢复行情源网络连接，或手动确认允许使用缓存后再运行。</div>',
        }
    ]
    failure_path = outdir / f"daily_{stamp}.html"
    failure_path.write_text(
        _render_page("量化研究日报生成失败", f"生成时间 {datetime.now():%Y-%m-%d %H:%M:%S}", metrics, sections),
        encoding="utf-8",
    )
    render_index(outdir, failure_path.name, None)
    return failure_path


def build_reports(args) -> Tuple[Optional[Path], Optional[Path]]:
    outdir = Path(args.outdir)
    data_dir = outdir / "data"
    work_dir = outdir / "work"
    _safe_mkdir(data_dir)
    _safe_mkdir(work_dir)

    stamp = _now_stamp()
    target_date = args.end_date or _recent_weekday()
    try:
        market = update_market_data(
            Path(args.daily_csv),
            target_date,
            bool(args.skip_update),
            bool(args.allow_stale_on_update_fail),
            int(args.update_workers),
            int(args.update_batch_size),
            int(args.update_flush_every),
            int(args.update_max_codes),
        )
    except Exception as exc:
        if bool(args.run_daily):
            render_failure_report(outdir, stamp, Path(args.daily_csv), exc)
        raise
    csvs = _csvs_arg(args.base_csv, args.daily_csv)

    daily_path: Optional[Path] = None
    weekly_path: Optional[Path] = None

    if bool(args.run_daily):
        line1 = run_line1_daily(
            csvs,
            args.cfg,
            args.bucket_map_csv,
            work_dir / "line1_daily",
            float(args.line1_threshold),
            bool(args.reuse_existing),
        )
        line2, actions, regime = run_line2_portfolio(
            csvs,
            args.cfg,
            work_dir / "line2_portfolio",
            bool(args.reuse_existing),
            int(args.line2_top_n),
        )
        line3 = run_line3_probability(
            Path(args.daily_csv),
            args.model,
            float(args.line3_threshold),
            work_dir / "line3_probability",
            bool(args.reuse_existing),
        )
        _write_csv(line1, data_dir / "daily_line1_predicted_return.csv")
        _write_csv(line2, data_dir / "daily_line2_portfolio_best.csv")
        _write_csv(actions, data_dir / "daily_line2_actions.csv")
        _write_csv(regime, data_dir / "daily_line2_regime.csv")
        _write_csv(line3, data_dir / "daily_line3_high_prob.csv")

        sections = [
            {
                "title": "研究线1：predicted_return > 0.1",
                "desc": "旧版规则信号候选叠加回归预测收益，按 predicted_return 降序。",
                "count": len(line1),
                "table": _table_html(line1, ["date", "code", "strategy", "predicted_return", "predicted_bin_abs", "predicted_bin_rel", "candidate_score", "technical_score", "value_score_100"], 40),
            },
            {
                "title": "研究线2：组合做优候选",
                "desc": "portfolio_regime_daily 当前窗口下的 BUY 候选，按 strategy_score 优先。",
                "count": len(line2),
                "table": _table_html(line2, ["date", "code", "recommended_action", "strategy", "strategy_profile", "strategy_score", "confidence", "target_weight", "composite_reason"], 20),
            },
            {
                "title": "研究线2：动作摘要",
                "desc": "当前持仓/买卖/观望动作，用来避免只看候选 Top 排名。",
                "count": len(actions),
                "table": _table_html(actions, ["action_date", "code", "action", "strategy", "score", "weight", "reason", "pnl_pct"], 30),
            },
            {
                "title": "研究线3：y_prob > 0.85",
                "desc": "分类概率模型最高置信候选，只展示最新事件日。",
                "count": len(line3),
                "table": _table_html(line3, ["event_date", "code", "y_prob", "y_pred"], 40),
            },
        ]
        metrics = {
            "行情最新日": market["latest_date"],
            "目标更新日": market["target_date"],
            "行情更新状态": market["update_status"],
            "更新并发数": market["update_workers"],
            "新增行数估算": market["rows_added_est"],
            "股票数": market["codes"],
            "落后代码数": market["lagging_codes"],
            "线1数量": len(line1),
            "线2数量": len(line2),
            "线3数量": len(line3),
        }
        if market.get("update_error"):
            metrics["更新错误"] = market["update_error"]
        daily_path = outdir / f"daily_{stamp}.html"
        daily_path.write_text(
            _render_page("量化研究日报", f"生成时间 {datetime.now():%Y-%m-%d %H:%M:%S}", metrics, sections),
            encoding="utf-8",
        )

    if bool(args.run_weekly):
        line4 = run_line4_weekly_breakout(
            csvs,
            work_dir / "line4_weekly_breakout",
            bool(args.reuse_existing),
        )
        line5 = run_line5_mainline(
            args.mainline_industry_kline,
            args.mainline_concept_kline,
            work_dir / "line5_mainline",
            bool(args.reuse_existing),
        )
        _write_csv(line4, data_dir / "weekly_line4_ma20_breakout.csv")
        _write_csv(line5, data_dir / "weekly_line5_mainline_radar.csv")
        sections = [
            {
                "title": "研究线4：周线 MA20 突破/持仓",
                "desc": "最新周线维度的 MA20 突破与 active_position 样本。",
                "count": len(line4),
                "table": _table_html(line4, ["date", "code", "close", "ma20", "buy_signal", "active_position", "amount_20w", "next_open"], 50),
            },
            {
                "title": "研究线5：行业/概念主线雷达",
                "desc": "A/B 级主线，按 mainline_score 和等级排序。",
                "count": len(line5),
                "table": _table_html(line5, ["date", "board_type", "board", "signal_grade", "mainline_score", "rs_score", "trend_score", "volume_score", "stability_score", "signal_reason"], 40),
            },
        ]
        metrics = {
            "行情最新日": market["latest_date"],
            "目标更新日": market["target_date"],
            "行情更新状态": market["update_status"],
            "更新并发数": market["update_workers"],
            "股票数": market["codes"],
            "线4数量": len(line4),
            "线5数量": len(line5),
        }
        if market.get("update_error"):
            metrics["更新错误"] = market["update_error"]
        weekly_path = outdir / f"weekly_{stamp}.html"
        weekly_path.write_text(
            _render_page("量化研究周报", f"生成时间 {datetime.now():%Y-%m-%d %H:%M:%S}", metrics, sections),
            encoding="utf-8",
        )

    render_index(
        outdir,
        daily_path.name if daily_path else None,
        weekly_path.name if weekly_path else None,
    )
    return daily_path, weekly_path


def parse_args(argv: Optional[Iterable[str]] = None):
    ap = argparse.ArgumentParser(description="生成日报/周报可视化研究报告")
    ap.add_argument("--daily-csv", default=DEFAULT_DAILY_CSV)
    ap.add_argument("--base-csv", default=DEFAULT_BASE_CSV)
    ap.add_argument("--outdir", default=str(ROOT / "output" / "reports"))
    ap.add_argument("--cfg", default=DEFAULT_CFG)
    ap.add_argument("--bucket-map-csv", default=DEFAULT_BUCKET_MAP)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--run-daily", type=int, choices=[0, 1], default=1)
    ap.add_argument("--run-weekly", type=int, choices=[0, 1], default=1)
    ap.add_argument("--skip-update", type=int, choices=[0, 1], default=0)
    ap.add_argument("--allow-stale-on-update-fail", type=int, choices=[0, 1], default=0)
    ap.add_argument("--update-workers", type=int, default=DEFAULT_UPDATE_WORKERS)
    ap.add_argument("--update-batch-size", type=int, default=25)
    ap.add_argument("--update-flush-every", type=int, default=250)
    ap.add_argument("--update-max-codes", type=int, default=0)
    ap.add_argument("--reuse-existing", type=int, choices=[0, 1], default=0)
    ap.add_argument("--end-date", default=None, help="行情更新目标日，默认最近工作日")
    ap.add_argument("--line1-threshold", type=float, default=0.1)
    ap.add_argument("--line2-top-n", type=int, default=10)
    ap.add_argument("--line3-threshold", type=float, default=0.85)
    ap.add_argument("--mainline-industry-kline", default=str(ROOT / "output" / "industry_weekly" / "industry_weekly_kline.csv"))
    ap.add_argument("--mainline-concept-kline", default=str(ROOT / "output" / "concept_weekly" / "concept_weekly_kline.csv"))
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    daily_path, weekly_path = build_reports(args)
    if daily_path:
        print(f"日报: {daily_path}")
    if weekly_path:
        print(f"周报: {weekly_path}")
    print(f"索引: {Path(args.outdir) / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
