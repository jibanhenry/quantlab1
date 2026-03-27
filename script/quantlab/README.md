# QuantLab

A modular research and execution framework for A-share quantitative trading.

当前主线已经从单纯规则信号回测，升级为：

- `Top 3` 组合约束
- 不强制满仓
- 长持有波段风格
- 按窗口风格在不同策略之间切换
- 日常输出 `BUY / HOLD / SELL` 建议

---

## Current Strategy

当前日常实盘主线，不是“每天选新的 Top 3 全替换”，而是：

- 最多持有 `3` 只
- 可以只持有 `1` 只或 `2` 只
- 没有足够好的新机会时不补满
- 已有持仓优先 `HOLD`
- 只有出现空位或真实退出时才新增 `BUY`

当前主要使用的策略池有三条：

- `pullback_longhold`
- `pullback_quality_combo`
- `quality_trend_longhold`

实际日常使用时，主要以这两个为主：

- `pullback_longhold`
- `pullback_quality_combo`

---

## Regime Mapping

根据滚动窗口稳定性分析，目前的经验映射是：

- `pullback_regime` -> `pullback_longhold`
- `weak_range_regime` -> `pullback_quality_combo`
- `mixed_transition` -> 默认 `pullback_longhold`，激进时可参考 `pullback_quality_combo`

可以简单理解为：

- 趋势还在、但更像回踩轮动：用 `pullback_longhold`
- 弱势、区间、风格混合：用 `pullback_quality_combo`
- 很强的连续主升环境：`quality_trend_longhold` 可作为二级备选，但不是当前主线

---

## Expected Level

截至目前这批滚动回测，能得到的大致结论是：

- 最好的单窗，`pullback_longhold` 单笔 `expectancy` 可到约 `2%`
- 更密的滚动窗口下，最近阶段 `pullback_quality_combo` 在部分窗口更强
- 最新实盘风格更像：
  - 总仓位常见在 `30% - 60%`
  - 单票常见在 `15% - 26%`
  - 平均持有大约 `10 - 15` 天

这套系统已经到了“可以做日常辅助决策”的阶段，但还不应视为已经锁定稳定 alpha。

---

## Project Structure

```text
quantlab/
├── __init__.py
├── config.py
├── io_utils.py
├── indicators.py
├── signals.py
├── market_state.py
├── buckets.py
├── backtest.py
├── risk.py
├── tuning.py
├── pipeline.py
├── valuation.py
├── portfolio.py
└── main.py
```

---

## Installation

Requires Python `3.9+`.

```bash
pip install -r requirements.txt
```

---

## Data

输入 CSV 至少应包含这些列：

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

## CLI Modes

- `daily`：旧版规则信号与回测
- `monthly`：分桶更新
- `quarterly`：旧版滚动调参与回测
- `portfolio_daily`：综合因子 Top-N 组合回测
- `portfolio_quarterly`：组合样本外回测
- `portfolio_regime_analysis`：窗口风格稳定性分析
- `portfolio_regime_daily`：日常 regime 判断与操作建议

---

## Daily Usage

日常使用时，一般只需要先更新数据，再运行 `portfolio_regime_daily`。

```bash
./.venv/bin/python -m script.quantlab.main \
  --mode portfolio_regime_daily \
  --csv "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2020-2025_all.csv,/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv" \
  --outdir "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/output/quantlab_regime_daily_latest" \
  --cfg "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/output/tuned_config_quarterly_20260225.json" \
  --regime_lookback_months 3 \
  --action_recent_days 10
```

如果想看指定时间段的操作历史：

```bash
./.venv/bin/python -m script.quantlab.main \
  --mode portfolio_regime_daily \
  --csv "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2020-2025_all.csv,/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv" \
  --outdir "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/output/quantlab_regime_daily_latest" \
  --cfg "/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/system project/pythonProject/output/tuned_config_quarterly_20260225.json" \
  --regime_lookback_months 3 \
  --action_start 2026-01-13 \
  --action_end 2026-03-25
```

---

## Daily Outputs

日常最重要的是下面这些文件：

- `portfolio_daily_regime.csv`
  - 今天属于什么窗口风格
  - 当前应使用哪条策略
- `portfolio_positions_state.csv`
  - 当前实际持仓状态
- `portfolio_latest_candidates.csv`
  - 今天横截面最强候选
- `portfolio_orders_next_open.csv`
  - 历史真实下单记录
- `portfolio_daily_actions.csv`
  - 系统原始动作输出
- `portfolio_action_history.csv`
  - 最近若干天历史动作

为了更贴近实盘，本仓库当前还额外生成两张“渐进持仓版”文件：

- `portfolio_daily_actions_progressive.csv`
  - 优先 `HOLD`
  - 只有有空位时才 `BUY`
  - 不做“每天 Top 3 整组替换”
- `portfolio_action_history_progressive.csv`
  - 默认最近 `10` 天动作
  - `SELL` 行带 `pnl_pct`

---

## Daily Reading Order

每天建议按这个顺序看：

1. `portfolio_daily_regime.csv`
2. `portfolio_daily_actions_progressive.csv`
3. `portfolio_positions_state.csv`
4. `portfolio_action_history_progressive.csv`

实际操作原则：

- 如果今天只有 `HOLD`，就不动
- 如果持仓未满 `3` 只，才考虑新的 `BUY`
- 如果出现 `SELL`，先看收益率和退出原因
- 不因为“今日候选 Top 3 改了”就把现有 3 只全部换掉

---

## Research Workflow

如果要继续研究，而不是日常执行，推荐这样做：

1. `portfolio_regime_analysis`
   - 找不同窗口类型对应的优势策略
2. `portfolio_quarterly`
   - 做滚动样本外评估
3. 必要时再做季度级策略复核

这类研究模式不需要每天跑。

---

## Notes

- 本项目用于研究，不构成投资建议。
- 数据必须先更新，daily 输出才有意义。
- 当前更适合“每天跑 daily，季度再复核策略”，而不是每天调参。
- `pbMRQ` 和 `psTTM` 已接入估值特征层，但不是唯一主因子。
