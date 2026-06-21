# -*- coding: utf-8 -*-
# 允许直接运行 main.py：自动把父目录塞进 sys.path，让 "from quantlab..." 可用
if __name__ == "__main__" and __package__ is None:
    import sys, pathlib
    pkg_parent = pathlib.Path(__file__).resolve().parent.parent
    if str(pkg_parent) not in sys.path:
        sys.path.insert(0, str(pkg_parent))
    __package__ = "quantlab"

import argparse, os
from typing import List
from .pipeline import daily_run
from .io_utils import load_market_csv_multi
from .buckets import monthly_freeze_bucket_map
from .portfolio import run_portfolio_daily, run_portfolio_walkforward, run_portfolio_regime_analysis, run_portfolio_regime_daily
from .factor_data import fetch_akshare_financial_indicators, normalize_code
from .factor_research import run_factor_lowvol_daily, run_factor_research
from .industry_weekly import run_industry_weekly_update
from .concept_weekly import run_concept_weekly_update
from .weekly_research import run_weekly_research
from .weekly_breakout import run_weekly_breakout_experiment
from .board_weekly_breakout import run_board_weekly_breakout_experiment
from .mainline_radar import run_mainline_radar

def _parse_csvs(arg: str) -> List[str]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return parts

def main():
    ap = argparse.ArgumentParser(description="quantlab runner (daily / monthly / quarterly / portfolio_daily / portfolio_quarterly / portfolio_regime_analysis / portfolio_regime_daily / factor_fetch_finance / factor_lowvol_daily / factor_research / weekly_research / weekly_breakout / board_weekly_breakout / mainline_radar / industry_weekly / concept_weekly)")
    ap.add_argument("--mode", choices=["daily","monthly","quarterly","portfolio_daily","portfolio_quarterly","portfolio_regime_analysis","portfolio_regime_daily","factor_fetch_finance","factor_lowvol_daily","factor_research","weekly_research","weekly_breakout","board_weekly_breakout","mainline_radar","industry_weekly","concept_weekly"], default="portfolio_regime_daily")
    ap.add_argument(
        "--csv",
        default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2020-2025_all.csv,/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv",
        help="多个CSV用逗号分隔，例如：a.csv,b.csv"
    )
    ap.add_argument("--outdir", default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/output/quantlab")
    ap.add_argument("--bucket_map_csv",
                    default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/output/quantlab/bucket_map_202602.csv",
                    help="daily 模式使用的 bucket_map 路径（默认 bucket_map_202602.csv）")
    ap.add_argument("--cfg", default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/output/tuned_config_quarterly_20260225.json", help="外部 YAML 配置路径（可选）")
    ap.add_argument("--bucket_mode", choices=["size","vol"], default="vol", help="monthly/quarterly 用的分桶维度")
    # quarterly tuning params
    ap.add_argument("--train_months", type=int, default=48)
    ap.add_argument("--val_months", type=int, default=12)
    ap.add_argument("--step_months", type=int, default=3)
    ap.add_argument("--test_months", type=int, default=12)
    ap.add_argument("--trials", type=int, default=50)
    # selective saves
    ap.add_argument("--save_signals", type=int, choices=[0,1], default=0)
    ap.add_argument("--save_trades", type=int, choices=[0,1], default=1)
    ap.add_argument("--save_summary", type=int, choices=[0,1], default=1)
    ap.add_argument("--save_candidates", type=int, choices=[0,1], default=1)
    ap.add_argument("--save_portfolio_signal_panel", type=int, choices=[0,1], default=0,
                    help="是否保存 portfolio_signal_panel.csv（体积很大，默认 0）")
    ap.add_argument("--export_virtual_trades", type=int, choices=[0, 1], default=1,
                    help="是否生成 all_signals_trades.csv（默认 1）")
    ap.add_argument("--valuation_enabled", type=int, choices=[0, 1], default=0)
    ap.add_argument("--valuation_mode", choices=["rank_only", "soft_filter"], default="rank_only")
    ap.add_argument("--expensive_cut", type=float, default=0.8)
    ap.add_argument("--tech_weight", type=float, default=0.7)
    ap.add_argument("--value_weight", type=float, default=0.3)
    ap.add_argument("--ml_weight", type=float, default=0.0)
    ap.add_argument("--portfolio_top_n", type=int, default=3)
    ap.add_argument("--portfolio_min_score", type=float, default=58.0)
    ap.add_argument("--portfolio_max_hold_days", type=int, default=20)
    ap.add_argument("--portfolio_variant_set", default="full")
    ap.add_argument("--regime_lookback_months", type=int, default=3)
    ap.add_argument("--action_recent_days", type=int, default=10)
    ap.add_argument("--action_start", default=None)
    ap.add_argument("--action_end", default=None)
    ap.add_argument("--factor_cache_dir", default=None)
    ap.add_argument("--factor_refresh_external", type=int, choices=[0, 1], default=0)
    ap.add_argument("--factor_extra_csv", default="", help="外部因子CSV，多个用逗号分隔，需包含 code 列")
    ap.add_argument("--factor_min_amount_20d", type=float, default=20_000_000.0)
    ap.add_argument("--factor_skip_diagnostics", type=int, choices=[0, 1], default=0,
                    help="复用 outdir 已有因子诊断报告，只重跑组合回测")
    ap.add_argument("--factor_finance_codes", default="",
                    help="财务补数指定股票代码，多个用逗号分隔；留空则使用 --csv 中全部股票")
    ap.add_argument("--factor_finance_workers", type=int, default=4,
                    help="AKShare 财务补数并发数")
    ap.add_argument("--factor_finance_limit", type=int, default=0,
                    help="财务补数最多股票数，0 表示不限制")
    ap.add_argument("--weekly_train_weeks", type=int, default=156,
                    help="周频模型 walk-forward 训练窗口周数")
    ap.add_argument("--weekly_val_weeks", type=int, default=26,
                    help="周频模型 walk-forward 验证窗口周数")
    ap.add_argument("--weekly_test_weeks", type=int, default=26,
                    help="周频模型 walk-forward 测试窗口周数")
    ap.add_argument("--weekly_step_weeks", type=int, default=13,
                    help="周频模型 walk-forward 滚动步长周数")
    ap.add_argument("--weekly_min_amount_20w", type=float, default=20_000_000.0,
                    help="周频股票池近20周平均成交额下限")
    ap.add_argument("--weekly_max_train_rows", type=int, default=300_000,
                    help="每个窗口最多训练样本数，0 表示不抽样")
    ap.add_argument("--weekly_save_panel", type=int, choices=[0, 1], default=0,
                    help="是否保存 weekly_signal_panel.csv（较大，默认 0）")
    ap.add_argument("--breakout_min_amount_20w", type=float, default=20_000_000.0,
                    help="20周线突破实验的近20周平均成交额下限")
    ap.add_argument("--breakout_total_exposure", type=float, default=0.45,
                    help="20周线突破实验组合总仓位")
    ap.add_argument("--breakout_cost_bp", type=float, default=2.0,
                    help="20周线突破实验单边交易成本，单位 bp")
    ap.add_argument("--board_breakout_industry_kline", default="output/industry_weekly/industry_weekly_kline.csv",
                    help="行业板块周K缓存路径")
    ap.add_argument("--board_breakout_concept_kline", default="output/concept_weekly/concept_weekly_kline.csv",
                    help="概念板块周K缓存路径")
    ap.add_argument("--board_breakout_min_amount_20w", type=float, default=0.0,
                    help="板块20周线突破实验的近20周平均成交额下限")
    ap.add_argument("--mainline_industry_kline", default="output/industry_weekly/industry_weekly_kline.csv",
                    help="主线雷达使用的行业周K缓存路径")
    ap.add_argument("--mainline_concept_kline", default="output/concept_weekly/concept_weekly_kline.csv",
                    help="主线雷达使用的概念周K缓存路径")
    ap.add_argument("--mainline_top_n", type=int, default=20,
                    help="主线雷达最新候选输出数量")
    ap.add_argument("--mainline_min_weeks", type=int, default=30,
                    help="主线雷达要求板块至少有多少周历史")
    ap.add_argument("--industry_start_date", default="20200101",
                    help="行业周K下载开始日期，格式 YYYYMMDD 或 YYYY-MM-DD")
    ap.add_argument("--industry_end_date", default=None,
                    help="行业周K下载结束日期，默认今天")
    ap.add_argument("--industry_refresh", type=int, choices=[0, 1], default=0,
                    help="是否重建行业周K/资金流缓存")
    ap.add_argument("--industries", default="",
                    help="指定行业名称，多个用逗号分隔；留空则更新全部行业")
    ap.add_argument("--industry_with_fund_flow", type=int, choices=[0, 1], default=1,
                    help="是否同步下载行业主力资金流并聚合周频")
    ap.add_argument("--industry_sleep_seconds", type=float, default=0.15,
                    help="行业接口请求间隔秒数")
    ap.add_argument("--industry_generate_viewer", type=int, choices=[0, 1], default=1,
                    help="是否在行业数据更新后生成本地 HTML 查看器")
    ap.add_argument("--concept_start_date", default="20200101",
                    help="概念周K下载开始日期，格式 YYYYMMDD 或 YYYY-MM-DD")
    ap.add_argument("--concept_end_date", default=None,
                    help="概念周K下载结束日期，默认今天")
    ap.add_argument("--concept_refresh", type=int, choices=[0, 1], default=0,
                    help="是否重建概念周K缓存")
    ap.add_argument("--concepts", default="",
                    help="指定概念名称，多个用逗号分隔；留空则更新全部概念")
    ap.add_argument("--concept_with_fund_flow", type=int, choices=[0, 1], default=0,
                    help="是否同步下载概念主力资金流并聚合周频；该接口覆盖不完整，默认 0")
    ap.add_argument("--concept_sleep_seconds", type=float, default=0.05,
                    help="概念接口请求间隔秒数")
    ap.add_argument("--concept_generate_viewer", type=int, choices=[0, 1], default=1,
                    help="是否在概念数据更新后生成本地 HTML 查看器")
    args = ap.parse_args()

    csvs = _parse_csvs(args.csv)
    cfg_overrides = {
        "valuation": {
            "enabled": bool(args.valuation_enabled),
            "mode": args.valuation_mode,
            "expensive_cut": args.expensive_cut,
            "tech_weight": args.tech_weight,
            "value_weight": args.value_weight,
            "ml_weight": args.ml_weight,
        },
        "portfolio": {
            "top_n": args.portfolio_top_n,
            "min_score": args.portfolio_min_score,
            "max_hold_days": args.portfolio_max_hold_days,
            "variant_set": args.portfolio_variant_set,
        }
    }

    if args.mode=="monthly":
        df = load_market_csv_multi(csvs)
        bm = monthly_freeze_bucket_map(df, mode=args.bucket_mode, k=3, code_industry=None)
        out = os.path.join(args.outdir, f"bucket_map_{df['date'].max():%Y%m}.csv")
        os.makedirs(args.outdir, exist_ok=True)
        bm.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[monthly] bucket_map 写出：{out}")
    elif args.mode=="quarterly":
        from . import tuning
        tuning.run_quarterly_tuning(
            all_in_one_csv=",".join(csvs),
            outdir=args.outdir,
            cfg_path=args.cfg,
            cfg_overrides=cfg_overrides,
            bucket_mode=args.bucket_mode,
            train_months=args.train_months,
            val_months=args.val_months,
            step_months=args.step_months,
            trials=args.trials
        )
        print("[quarterly] 调参完成。")
    elif args.mode == "portfolio_daily":
        run_portfolio_daily(
            csvs,
            cfg_path=args.cfg,
            cfg_overrides=cfg_overrides,
            outdir=args.outdir,
            save_signal_panel=bool(args.save_portfolio_signal_panel),
        )
        print("[portfolio_daily] 组合策略回测完成。")
    elif args.mode == "portfolio_quarterly":
        run_portfolio_walkforward(
            csvs,
            cfg_path=args.cfg,
            cfg_overrides=cfg_overrides,
            outdir=args.outdir,
            train_months=args.train_months,
            val_months=args.val_months,
            test_months=args.test_months,
            step_months=args.step_months,
        )
        print("[portfolio_quarterly] 组合策略样本外回测完成。")
    elif args.mode == "portfolio_regime_analysis":
        run_portfolio_regime_analysis(
            csvs,
            cfg_path=args.cfg,
            cfg_overrides=cfg_overrides,
            outdir=args.outdir,
            train_months=args.train_months,
            val_months=args.val_months,
            test_months=args.test_months,
            step_months=args.step_months,
        )
        print("[portfolio_regime_analysis] 窗口类型稳定性分析完成。")
    elif args.mode == "portfolio_regime_daily":
        run_portfolio_regime_daily(
            csvs,
            cfg_path=args.cfg,
            cfg_overrides=cfg_overrides,
            outdir=args.outdir,
            regime_lookback_months=args.regime_lookback_months,
            action_recent_days=args.action_recent_days,
            action_start=args.action_start,
            action_end=args.action_end,
            save_signal_panel=bool(args.save_portfolio_signal_panel),
        )
        print("[portfolio_regime_daily] 今日窗口风格与操作建议已生成。")
    elif args.mode == "factor_fetch_finance":
        if args.factor_finance_codes:
            codes = [normalize_code(c) for c in _parse_csvs(args.factor_finance_codes)]
        else:
            market = load_market_csv_multi(csvs)
            codes = sorted(market["code"].map(normalize_code).dropna().unique().tolist())
        codes = [code for code in codes if not code.startswith("399")]
        cache_dir = args.factor_cache_dir or os.path.join(args.outdir, "factor_cache")
        financial = fetch_akshare_financial_indicators(
            codes,
            cache_dir=cache_dir,
            refresh=bool(args.factor_refresh_external),
            workers=args.factor_finance_workers,
            limit=args.factor_finance_limit or None,
        )
        print(f"[factor_fetch_finance] 财务补数完成：rows={len(financial):,}, codes={financial['code'].nunique() if not financial.empty else 0:,}，cache={cache_dir}")
    elif args.mode == "factor_lowvol_daily":
        run_factor_lowvol_daily(
            csvs,
            cfg_path=args.cfg,
            cfg_overrides=cfg_overrides,
            outdir=args.outdir,
            cache_dir=args.factor_cache_dir,
            refresh_external=bool(args.factor_refresh_external),
            min_amount_20d=args.factor_min_amount_20d,
            action_recent_days=args.action_recent_days,
            action_start=args.action_start,
            action_end=args.action_end,
        )
        print("[factor_lowvol_daily] pure_lowvol 每日持仓与操作建议已生成。")
    elif args.mode == "factor_research":
        run_factor_research(
            csvs,
            cfg_path=args.cfg,
            cfg_overrides=cfg_overrides,
            outdir=args.outdir,
            cache_dir=args.factor_cache_dir,
            refresh_external=bool(args.factor_refresh_external),
            extra_factor_csvs=_parse_csvs(args.factor_extra_csv) if args.factor_extra_csv else None,
            train_months=args.train_months,
            val_months=args.val_months,
            test_months=args.test_months,
            step_months=args.step_months,
            min_amount_20d=args.factor_min_amount_20d,
            skip_diagnostics=bool(args.factor_skip_diagnostics),
        )
        print("[factor_research] 因子诊断与稳健多因子样本外回测完成。")
    elif args.mode == "weekly_research":
        outdir = args.outdir
        if os.path.basename(os.path.normpath(outdir)) == "quantlab":
            outdir = os.path.join(os.path.dirname(os.path.normpath(outdir)), "quantlab_weekly_research")
        result = run_weekly_research(
            csvs,
            outdir=outdir,
            train_weeks=args.weekly_train_weeks,
            val_weeks=args.weekly_val_weeks,
            test_weeks=args.weekly_test_weeks,
            step_weeks=args.weekly_step_weeks,
            min_amount_20w=args.weekly_min_amount_20w,
            max_train_rows=args.weekly_max_train_rows,
            save_panel=bool(args.weekly_save_panel),
        )
        summary = result.get("weekly_strategy_summary")
        metrics = result.get("weekly_model_metrics")
        print(
            "[weekly_research] 周频选股排序与样本外回测完成："
            f"windows={len(metrics) if metrics is not None else 0}, "
            f"strategies={len(summary) if summary is not None else 0}, "
            f"outdir={outdir}"
        )
    elif args.mode == "weekly_breakout":
        outdir = args.outdir
        if os.path.basename(os.path.normpath(outdir)) == "quantlab":
            outdir = os.path.join(os.path.dirname(os.path.normpath(outdir)), "quantlab_weekly_breakout")
        result = run_weekly_breakout_experiment(
            csvs,
            outdir=outdir,
            min_amount_20w=args.breakout_min_amount_20w,
            total_exposure=args.breakout_total_exposure,
            cost_bp=args.breakout_cost_bp,
        )
        summary = result.get("summary")
        row = summary.iloc[0].to_dict() if summary is not None and not summary.empty else {}
        print(
            "[weekly_breakout] 20周线突破实验完成："
            f"win_rate={row.get('win_rate', float('nan')):.2%}, "
            f"annual_return={row.get('annual_return', float('nan')):.2%}, "
            f"max_drawdown={row.get('max_drawdown', float('nan')):.2%}, "
            f"sharpe={row.get('sharpe', float('nan')):.2f}, "
            f"outdir={outdir}"
        )
    elif args.mode == "board_weekly_breakout":
        outdir = args.outdir
        if os.path.basename(os.path.normpath(outdir)) == "quantlab":
            outdir = os.path.join(os.path.dirname(os.path.normpath(outdir)), "board_weekly_breakout")
        result = run_board_weekly_breakout_experiment(
            industry_kline_path=args.board_breakout_industry_kline,
            concept_kline_path=args.board_breakout_concept_kline,
            outdir=outdir,
            min_amount_20w=args.board_breakout_min_amount_20w,
            total_exposure=args.breakout_total_exposure,
            cost_bp=args.breakout_cost_bp,
        )
        summary = result.get("summary")
        print(
            "[board_weekly_breakout] 行业/概念20周线突破实验完成："
            f"strategies={len(summary) if summary is not None else 0}, "
            f"outdir={outdir}"
        )
    elif args.mode == "mainline_radar":
        outdir = args.outdir
        if os.path.basename(os.path.normpath(outdir)) == "quantlab":
            outdir = os.path.join(os.path.dirname(os.path.normpath(outdir)), "mainline_radar")
        result = run_mainline_radar(
            industry_kline_path=args.mainline_industry_kline,
            concept_kline_path=args.mainline_concept_kline,
            outdir=outdir,
            top_n=args.mainline_top_n,
            min_weeks=args.mainline_min_weeks,
        )
        latest = result.get("mainline_latest")
        history = result.get("mainline_rank_history")
        print(
            "[mainline_radar] 稳健启动主线雷达完成："
            f"latest={len(latest) if latest is not None else 0}, "
            f"history_rows={len(history) if history is not None else 0}, "
            f"outdir={outdir}"
        )
    elif args.mode == "industry_weekly":
        outdir = args.outdir
        if os.path.basename(os.path.normpath(outdir)) == "quantlab":
            outdir = os.path.join(os.path.dirname(os.path.normpath(outdir)), "industry_weekly")
        result = run_industry_weekly_update(
            outdir=outdir,
            start_date=args.industry_start_date,
            end_date=args.industry_end_date,
            refresh=bool(args.industry_refresh),
            industries=args.industries or None,
            with_fund_flow=bool(args.industry_with_fund_flow),
            sleep_seconds=args.industry_sleep_seconds,
            generate_viewer=bool(args.industry_generate_viewer),
        )
        kline = result.get("industry_weekly_kline")
        flow = result.get("industry_fund_flow_weekly")
        viewer = result.get("viewer_info") or {}
        print(
            "[industry_weekly] 行业周K更新完成："
            f"industries={len(result.get('industry_list', []))}, "
            f"kline_rows={len(kline) if kline is not None else 0}, "
            f"fund_flow_weekly_rows={len(flow) if flow is not None else 0}, "
            f"outdir={outdir}, "
            f"viewer={viewer.get('html_path', '')}"
        )
    elif args.mode == "concept_weekly":
        outdir = args.outdir
        if os.path.basename(os.path.normpath(outdir)) == "quantlab":
            outdir = os.path.join(os.path.dirname(os.path.normpath(outdir)), "concept_weekly")
        result = run_concept_weekly_update(
            outdir=outdir,
            start_date=args.concept_start_date,
            end_date=args.concept_end_date,
            refresh=bool(args.concept_refresh),
            concepts=args.concepts or None,
            with_fund_flow=bool(args.concept_with_fund_flow),
            sleep_seconds=args.concept_sleep_seconds,
            generate_viewer=bool(args.concept_generate_viewer),
        )
        kline = result.get("concept_weekly_kline")
        viewer = result.get("viewer_info") or {}
        print(
            "[concept_weekly] 概念周K更新完成："
            f"concepts={len(result.get('concept_list', []))}, "
            f"kline_rows={len(kline) if kline is not None else 0}, "
            f"outdir={outdir}, "
            f"viewer={viewer.get('html_path', '')}"
        )
    else:
        bucket_map_csv = args.bucket_map_csv
        print(f"[daily] 使用 bucket_map：{bucket_map_csv}")
        daily_run(csvs,
                  cfg_path=args.cfg,
                  cfg_overrides=cfg_overrides,
                  outdir=args.outdir,
                  bucket_map_csv=bucket_map_csv,
                  save_signals=bool(args.save_signals),
                  save_trades=bool(args.save_trades),
                  save_summary=bool(args.save_summary),
                  save_candidates=bool(args.save_candidates),
                  export_virtual_trades=bool(args.export_virtual_trades))

if __name__ == "__main__":
    main()
