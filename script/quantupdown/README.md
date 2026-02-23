

# K 线形态扫描与未来 30 日收盘统计（pattern_scan.py + main.py）

本项目用于在日线数据中扫描 4 类常见形态，并在形态“确认入场日”（entry day）之后，统计未来最多 30 个交易日内的收盘表现，输出可直接用于回测或人工筛选。

当前保留的 4 个形态：

- `HIGH_TIGHT_FLAG`（高位紧旗形）
- `PULLBACK_TO_MA30`（缩量回调到 MA30）
- `FALSE_BREAKDOWN`（假跌破洗盘）
- `SMALL_UP_DAYS_GENTLE_VOL`（连续小阳线且量能温和）

---

## 1. 输入数据要求

默认需要以下字段（列名可通过参数指定，或通过 `io_utils.load_market_csv_multi` 统一）：

- `code`：股票代码
- `date`：交易日（能被 `pd.to_datetime` 转换）
- `open, high, low, close`：日线 OHLC
- `volume`：成交量

如果项目中存在 `io_utils.py` 且提供 `load_market_csv_multi`，脚本会优先用它做字段规范化（与训练 GRU 时的数据字段映射逻辑保持一致）。否则使用 `_normalize_columns_for_patterns()` 做通用字段名归一。

---

## 2. 两个脚本分别做什么

### 2.1 `pattern_scan.py`

- 提供形态扫描函数（4 个 `detect_*`）
- 提供 `scan_patterns_and_summarize()`：扫描全部样本并输出 events + summary

其中 `events` 是逐条形态触发记录（包含 `entry_date` 等），`summary` 是按形态聚合后的统计摘要。

### 2.2 `main.py`

你另起的 `main.py` 用于“实盘筛选式输出”，只读取指定 CSV（例如：
`/Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv`），扫描 4 个形态后输出：

- `code`
- `entry_date`
- `pattern`
- 以及未来（最多 30 天）收盘统计字段（见第 5 节）

并且只保留 `entry_date >= 2026-01-01` 的记录。

---

## 3. 核心原则：以“确认后入场”为基准统计收益

所有形态都会输出：

- `date`：形态触发或关键日（通常是 Day0 或窗口末日）
- `entry_date`：形态确认后的入场日（更贴近实盘：看到确认才买）
- `entry_close`：入场日收盘价（默认以入场日收盘价作为买入基准）

**所有未来表现统计都基于 `entry_date`，并且从 `entry_date` 的下一交易日开始计算。**

---

## 4. 四个形态的详细定义与参数

下面所有逻辑均是“按单只股票、按日期排序后扫描”。

### 4.1 `HIGH_TIGHT_FLAG`（高位紧旗形）

目标：找到“平稳基座 → 放量大涨 → 次日十字星/小实体 → 窄幅缩量整理 → 向上突破并站稳”的强势形态。

**触发与入场：**
- `date`：Day0（放量大涨日）
- `entry_date`：突破确认并次日站稳后的入场日（更符合实盘）

**主要参数（函数 `detect_high_tight_flag`）：**

- Day0 条件
  - `day0_ret=0.055`：Day0 当日涨幅阈值（默认 5.5%）
  - `day0_vol_mult=1.5`：Day0 放量倍数（`volume >= vol_ma20 * 1.5`）

- Day1 条件（十字星/小实体代理）
  - `day1_range_pct=0.02`：Day1 振幅阈值（`(high-low)/close <= 2%`）
  - `day1_body_ratio=0.25`：Day1 实体占比阈值（`body/range <= 0.25`）

- 基座期（形态之前的“平稳”背景）
  - `base_days=20`：基座长度
  - `base_range_pct=0.035`：基座期平均振幅占比上限
  - `base_vol_cv=0.6`：基座期量能稳定性（成交量 CV = std/mean 上限）

- 旗形整理期（Day1 后的窄幅整理）
  - `flag_days=(3,10)`：整理期长度范围（3 到 10 天）
  - `band_buffer=0.01`：允许整理期间略微越界的容忍（区间上下沿 1%）
  - `vol_contract=0.8`：整理期均量相对 20 日均量缩量要求
  - `vol_contract_vs_day0=0.7`：整理期均量相对 Day0 成交量缩量要求

- 入场确认（突破与站稳）
  - `breakout_confirm_days=10`：整理结束后最多等待 10 天出现突破
  - `breakout_buffer=0.01`：突破阈值（`close >= band_high*(1+0.01)`）
  - `breakout_vol_mult=1.2`：突破日放量要求（`volume >= vol_ma20*1.2`）
  - `hold_buffer=0.003`：次日站稳容忍（次日 `close >= band_high*(1-0.003)`）

**实现逻辑概述：**
1) 先检查 Day0 前 `base_days` 是否“平稳”（振幅低、量能波动低）
2) Day0 满足大涨 + 放量
3) Day1 满足小振幅 + 小实体（十字星代理）
4) Day2..Day(1+N) 在 Day0/Day1 区间内窄幅波动，并且缩量
5) 之后等待向上突破区间上沿，并放量，且次日站稳确认后，`entry_date` 取站稳那一天

---

### 4.2 `PULLBACK_TO_MA30`（缩量回调到 MA30）

目标：趋势中回调到 MA30 附近、且缩量，随后站稳确认的“再加速”形态。

**触发与入场：**
- `date`：回踩触碰 MA30 的当日（Day0）
- `entry_date`：次日站稳 MA30 的确认日（Day1）

**主要参数（函数 `detect_pullback_to_ma30`）：**
- `touch_buffer=0.005`：触碰容忍（`low <= MA30*(1+0.5%)`）
- `hold_buffer=0.003`：站稳容忍（次日 `close >= MA30*(1-0.3%)`）
- `vol_contract=0.8`：缩量条件（`volume <= vol_ma20*0.8`）

**实现逻辑概述：**
1) 需要 MA30 与 vol_ma20 可用
2) Day0：`close > MA30`（趋势背景）且 `low` 触碰 MA30 附近，且缩量
3) Day1：次日收盘仍在 MA30 附近之上（站稳），确认入场

---

### 4.3 `FALSE_BREAKDOWN`（假跌破洗盘）

目标：短期跌破“前低/平台低点”但量能不大，随后快速拉回区间内的“洗盘/赶人下车”。

**触发与入场：**
- `date`：跌破日 Day0
- `entry_date`：随后第一个“收盘重新站回前低之上”的那一天

**主要参数（函数 `detect_false_breakdown`）：**
- `lookback=20`：前低参考窗口（过去 20 天的最低价，且不含当日）
- `vol_silent=0.8`：无量跌破（`volume <= vol_ma20*0.8`）
- `reclaim_days=3`：最多允许 3 天内收回（收盘站回前低之上）

**实现逻辑概述：**
1) 计算 `prior_low`：过去 lookback 天（不含当日）的最低价
2) Day0：`low < prior_low`（真跌破）且缩量（无量）
3) 在未来 `reclaim_days` 天内，如果出现 `close > prior_low`，则视为“假跌破”成立
4) `entry_date` 取第一次收盘站回的那一天（确认后入场）

---

### 4.4 `SMALL_UP_DAYS_GENTLE_VOL`（连续小阳线、量能温和）

目标：识别“连续多日小幅上涨”并伴随量能不走弱的趋势延续信号。

**触发与入场：**
- `date`：窗口最后一天（形态确认日）
- `entry_date`：下一交易日（默认确认后次日入场）

**主要参数（函数 `detect_rising_three_like`）：**
- `n_days=4`：观察窗口长度
- `small_body_pct=0.012`：小阳线定义：`0 < ret1 < 1.2%`
- `min_pos_days=3`：窗口内至少 3 天满足“小阳线”
- `vol_slope=1.0`：量能不走弱（窗口末端 `vol_ma5 >= 窗口起点 vol_ma5 * 1.0`）

**实现逻辑概述：**
1) 在每只股票上计算日收益 ret1
2) 统计过去 n_days 内符合“小阳线”的天数
3) 要求数量 ≥ min_pos_days
4) 量能均线 vol_ma5 不走弱
5) `entry_date` = 下一交易日（确认后入场）

---

## 5. 未来（最多 30 天）统计字段解释（按 entry_date 计算）

这些字段都是从 `entry_date` 的下一交易日开始，向后最多看 `horizon=30` 天得到。

- `entry_close`：入场日收盘价（买入基准）

- `fut_max_close_30d`：未来窗口内最高收盘价
- `fwd_max_close_ret_30d`：最高收盘涨幅 = `fut_max_close_30d / entry_close - 1`

- `fut_min_close_30d`：未来窗口内最低收盘价
- `fwd_min_close_ret_30d`：最低收盘跌幅 = `fut_min_close_30d / entry_close - 1`

- `fut_mean_close_30d`：未来窗口内平均收盘价
- `fwd_mean_close_ret_30d`：均价涨跌幅 = `fut_mean_close_30d / entry_close - 1`

### 5.1 未来不足 30 天时的口径（关键）

当 `entry_date` 距离数据最后一天不足 30 个交易日时：

- 直接用 `entry_date` 到 **数据最新交易日** 之间实际存在的未来收盘价序列计算
- 不会强行凑满 30 天

也就是说：

- `fut_max_close_30d` = 从 `entry_date` 下一天到“最新日”为止的最高收盘价
- `fut_min_close_30d` = 同窗口最低收盘价
- `fut_mean_close_30d` = 同窗口平均收盘价

如果 `entry_date` 本身就是数据最后一天，则未来窗口为空：

- `fut_max_close_30d / fut_min_close_30d / fut_mean_close_30d` 为 NaN
- 对应的 `fwd_*_ret_30d` 也为 NaN

---

## 6. 输出说明

### 6.1 main.py 输出（信号表）

建议输出列（最常用）：

- `code`
- `entry_date`
- `pattern`
- `entry_close`
- `fut_max_close_30d`, `fwd_max_close_ret_30d`
- `fut_min_close_30d`, `fwd_min_close_ret_30d`
- `fut_mean_close_30d`, `fwd_mean_close_ret_30d`

### 6.2 去重建议

如果你看到同一个 `code + entry_date + pattern` 出现重复行，通常来自：

- 同一形态在不同扫描路径下重复命中
- 或者数据源重复（CSV 合并时同一天记录重复）

可在输出前做：

- `drop_duplicates(subset=["code","entry_date","pattern"])`

---

## 7. 使用示例

### 7.1 扫描并输出 summary（pattern_scan.py）

```bash
python pattern_scan.py \
  --csv1 /Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2020-2025_all.csv \
  --csv2 /Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv \
  --horizon 30 \
  --outdir ./pattern_output
```

### 7.2 只输出 2026 年以来的 entry 信号（main.py）

```bash
python main.py \
  --csv /Users/wuguanhe/Desktop/吴冠鹤/业余/stock/output/2025_06_daily.csv \
  --from_date 2026-01-01 \
  --horizon 30 \
  --outdir ./pattern_output
```

---

## 8. 常见问题

### 8.1 为什么有的 entry_date 不是最新一天？

`entry_date` 的含义是“形态确认后可以入场的日期”，它由形态规则决定，不是“最新交易日”。

例如：
- `FALSE_BREAKDOWN` 需要“跌破后在 reclaim_days 内收盘站回前低”，entry_date 就是第一次站回那天
- `SMALL_UP_DAYS_GENTLE_VOL` 默认 entry_date 是确认后的下一交易日

### 8.2 为什么会出现未来统计 NaN？

最常见原因：
- entry_date 已经接近数据尾部，未来窗口为空或太短
- 或该 code 在 entry_date 当天缺失 close 数据（数据源不完整）

---

如果你希望把“买入价”从 entry_date 的收盘改为 `entry_date+1` 的开盘，也可以在 main.py 里将 `entry_close` 替换为“下一交易日 open”，并将未来窗口相应后移。