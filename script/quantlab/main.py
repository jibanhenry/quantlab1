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

def _parse_csvs(arg: str) -> List[str]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return parts

def main():
    ap = argparse.ArgumentParser(description="quantlab runner (daily / monthly / quarterly / portfolio_daily / portfolio_quarterly / portfolio_regime_analysis / portfolio_regime_daily)")
    ap.add_argument("--mode", choices=["daily","monthly","quarterly","portfolio_daily","portfolio_quarterly","portfolio_regime_analysis","portfolio_regime_daily"], default="daily")
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
        )
        print("[portfolio_regime_daily] 今日窗口风格与操作建议已生成。")
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
