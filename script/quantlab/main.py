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

def _parse_csvs(arg: str) -> List[str]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return parts

def main():
    ap = argparse.ArgumentParser(description="quantlab runner (daily / monthly / quarterly)")
    ap.add_argument("--mode", choices=["daily","monthly","quarterly"], default="daily")
    ap.add_argument(
        "--csv",
        default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2020-2025_all.csv,/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv",
        help="多个CSV用逗号分隔，例如：a.csv,b.csv"
    )
    ap.add_argument("--outdir", default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/output/quantlab")
    ap.add_argument("--cfg", default="/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/output/tuned_config_quarterly_20250904.json", help="外部 YAML 配置路径（可选）")
    ap.add_argument("--bucket_mode", choices=["size","vol"], default="vol", help="monthly/quarterly 用的分桶维度")
    # quarterly tuning params
    ap.add_argument("--train_months", type=int, default=48)
    ap.add_argument("--val_months", type=int, default=12)
    ap.add_argument("--step_months", type=int, default=3)
    ap.add_argument("--trials", type=int, default=50)
    # selective saves
    ap.add_argument("--save_signals", type=int, choices=[0,1], default=0)
    ap.add_argument("--save_trades", type=int, choices=[0,1], default=1)
    ap.add_argument("--save_summary", type=int, choices=[0,1], default=1)
    ap.add_argument("--save_candidates", type=int, choices=[0,1], default=1)
    args = ap.parse_args()

    csvs = _parse_csvs(args.csv)

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
            bucket_mode=args.bucket_mode,
            train_months=args.train_months,
            val_months=args.val_months,
            step_months=args.step_months,
            trials=args.trials
        )
        print("[quarterly] 调参完成。")
    else:
        import datetime as _dt
        ym = _dt.datetime.today().strftime("%Y%m")
        bucket_map_csv = os.path.join(args.outdir, f"bucket_map_{ym}.csv")
        daily_run(csvs, cfg_path=args.cfg, outdir=args.outdir, bucket_map_csv=bucket_map_csv,
                  save_signals=bool(args.save_signals),
                  save_trades=bool(args.save_trades),
                  save_summary=bool(args.save_summary),
                  save_candidates=bool(args.save_candidates))

if __name__ == "__main__":
    main()
