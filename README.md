# QuantLab

A modular research & backtesting framework for quantitative trading strategies.

一个用于量化交易策略的模块化研究与回测框架。

---

## 📂 Project Structure | 项目结构

```
quantlab/
├── __init__.py
├── config.py          # Config loading/merging/validation | 配置加载/合并/校验
├── io_utils.py        # IO utils for CSV/Parquet | CSV/Parquet 读写工具
├── indicators.py      # Technical indicators (EMA/MACD/RSI...) | 技术指标
├── signals.py         # S1/S2/S3/S4 strategies | 策略信号
├── market_state.py    # Market climate recognition | 市场气候识别
├── buckets.py         # Buckets (size/vol/industry) | 分桶映射
├── backtest.py        # Backtest engine | 回测引擎
├── risk.py            # Risk and position sizing | 风控与仓位管理
├── tuning.py          # Walk-forward tuning | 滚动调参
├── bandit.py          # Contextual bandit allocation | 多臂老虎机分配
├── reporting.py       # Reports & visualization | 报告与可视化
├── pipeline.py        # End-to-end orchestration | 全流程编排
└── main.py            # CLI entrypoint | 命令行入口
```

---

## ⚙️ Installation | 安装

Requires **Python 3.9+**. | 需要 **Python 3.9+**。

```bash
pip install -r requirements.txt
```

---

## 🗂 Data | 数据格式

Input CSV must contain these columns:  
输入的 CSV 必须包含以下列：

| 中文列名 | 内部字段 |
|----------|----------|
| `code`   | 股票代码 |
| `日期`   | date |
| `开盘`   | open |
| `最高`   | high |
| `最低`   | low |
| `收盘`   | close |
| `前收`   | preclose |
| `成交量` | volume |
| `成交额` | amount |
| `换手率` | turnover |
| `涨跌幅` | pct_chg |
| `pbMRQ`  | pb_mrq |
| `psTTM`  | ps_ttm |

---

## 🚀 Usage | 使用方法

### CLI

```bash
python -m quantlab.main --mode daily --csv data/all.csv --outdir output
```

### Modes | 模式

- `daily` → Run signals & backtest | 日常运行，输出信号和回测  
- `monthly` → Re-map buckets | 月度更新分桶  
- `quarterly` → Walk-forward tuning | 季度滚动调参  

---

## 📝 CLI Arguments | 命令行参数

| 参数 | 说明 |
|------|------|
| `--mode` | 运行模式：`daily` / `monthly` / `quarterly` |
| `--csv` | 输入行情 CSV 文件（多个用逗号分隔） |
| `--outdir` | 输出目录 |
| `--cfg` | 外部 YAML 配置文件 |
| `--bucket_mode` | 分桶方式：`size` / `vol` |
| `--train_months` | 训练窗口（月） |
| `--val_months` | 验证窗口（月） |
| `--step_months` | 滚动步长（月） |
| `--trials` | 搜索次数（Optuna） |
| `--save_signals` | 保存 signals_daily.csv (1/0) |
| `--save_trades` | 保存 trades_ledger.csv (1/0) |
| `--save_summary` | 保存 strategy_summary.csv (1/0) |
| `--save_candidates` | 保存 candidates_xxx.csv (1/0) |

---

## 📊 Strategies | 策略

### S1 — Trend-follow Pullback | 趋势回撤买入
- Entry: EMA50>EMA200 & ADX≥25, 回撤至EMA20, MACD>0, RSI≥45, OBV确认  
- Exit: 跌破EMA50或MACD死叉  
- Stop: 1.5–2× ATR

### S2 — Bollinger Breakout | 布林突破
- Entry: 布林收缩 + 突破上轨 + ADX上升 + OBV确认  
- Exit: 回落至中轨下  
- Stop: 中轨 − ATR

### S3 — Mean Reversion | 区间均值回归
- Entry: ADX<20，价格接近下轨 + RSI≤35  
- Exit: 反弹至中轨  
- Stop: 1.2× ATR

### S4 — Trend Momentum Pyramid | 趋势加仓
- Entry: +DI > −DI, ADX上升, CCI>100 或 ROC>0, OBV创新高  
- Exit: 加仓策略，无单独退出条件  

---

## 🧪 Typical Workflow | 工作流

1. **Quarterly 调参**  
   ```bash
   python -m quantlab.main --mode quarterly --csv data/all.csv        --train_months 48 --val_months 6 --step_months 3 --trials 30
   ```

2. **Monthly 分桶冻结**  
   ```bash
   python -m quantlab.main --mode monthly --csv data/all.csv --bucket_mode size
   ```

3. **Daily 日常运行**  
   ```bash
   python -m quantlab.main --mode daily --csv data/all.csv
   ```

---

## 📈 Outputs | 输出

- `signals_daily.csv` → 每日信号  
- `trades_ledger.csv` → 交易流水  
- `strategy_summary.csv` → 策略汇总表现  
- `candidates_YYYYMMDD.csv` → 当日候选股票  

---

## 📌 Notes | 注意事项

- 本项目用于研究，不构成投资建议。  
- 确保输入数据质量（连续无缺失）。  
- 建议使用 quarterly 模式优化参数，提升适应性。  
