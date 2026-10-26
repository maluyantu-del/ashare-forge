# 第二期 截面引擎 · 设计

目标:把"一个因子想法"变成"一份可辩护的裁定"。原则是规则由代码执行、判据事前写定、多重检验计价,并按 A 股的交易制度落地。

> 本文为第二期设计稿。零假设校准与门槛两节已按实测修正后的现行实现改写,修正过程见各节末尾与提交记录。

## 截面场景的设计要点

| 维度 | 本平台口径 |
| --- | --- |
| 研究对象 | 全市场股票的横截面排序 |
| 核心指标 | **RankIC / ICIR / 多头超额** |
| 可交易性 | **T+1、涨跌停、停牌、ST/次新/微盘** |
| 做空 | **不假设可做空(融券 2024 后基本不可用)→ 多头超额为主** |
| 门槛 | **零假设校准**(见下)+ 绝对门槛 + PSR/DSR |

## 零假设校准(本期最重要的设计)

A股方向评审的反方实测:对"剩余天数"去趋势后估 AR(1) 半衰期,**纯随机游走有 82% 概率通过"半衰期≤15日"的固定判据**(Nelson-Kang 伪回归)。固定数值门槛对零假设几乎不设防。

因此本引擎**不以固定阈值单独作为成立判据**,而是:

1. 取候选因子,跑完整流水线得到真实指标(RankIC、ICIR、多头超额夏普等)。
2. 生成 N 份**零假设因子**(现行构造见下):摧毁"哪只股票拿哪个值"的截面信息,同时保留每条序列的时序路径与持续性,使零因子的换手与真实因子同量级。
3. 每份零因子穿过**同一条流水线**(同样的可交易性过滤、同样的成本、同样的调仓规则),得到零分布。
4. 逐指标计算经验单尾 p 值 (1 + #{零样本不劣于真实值}) / (n + 1)(Phipson & Smyth 2010),**p ≤ 0.05** 为通过;核心四项——IC 族(ic_mean、icir)与收益族(excess_sharpe、sharpe)——须全部通过,IC 族缺失即判不通过。另四项(t_stat、monotonicity、ann_return、excess_ann)只作披露,裁定记录中的"零校准 k/8"即八项合计。
5. **零分布有效性闸门**:零样本日均换手中位数 / 真实换手须落在 [0.8, 1.25],否则零因子与真实因子的成本不可比,主判据按不通过处理。

这同时校准了样本长度、股票数、换手成本带来的偏差——固定阈值做不到这点。

**零因子构造的三次迭代**(本节原稿写的是第一版"逐日截面打乱",实测证伪后两次修正):

| 版本 | 构造 | 实测问题 | 状态 |
| --- | --- | --- | --- |
| 第一版 `daily` | 每个交易日把因子值在横截面上独立打乱 | 同时摧毁时序路径:零样本日换手 0.85,真实因子 0.009,比较的是手续费而非选股能力 | `1dffd28` 弃用,保留作对照 |
| 第二版 `fixed_repack_legacy` | 全样本一个固定股票置换;某行缺失布局变化时按顺序重新装填 | 上市、退市、停牌几乎每天改变缺失布局,股票与供值股票的对应逐日滑动,退化为近似逐日打乱;2019–2021 实测零样本换手为真实的 2.2–13.4 倍 | `a15a404` 弃用,保留作对照 |
| 第三版 `fixed`(现行) | 按首/末有效日排序,切成不少于 20 只的组,组内循环错位(无自映射)指派固定供值股票(donor),整条序列照搬;本股或 donor 当日缺值即缺值 | 2019–2021 验证零样本换手为真实的 0.84–1.15 倍;donor 生命周期相近,零因子残留少量真实信息,判据略偏保守 | `a15a404` 起执行 |

## 模块

### `engine/metrics.py`
```python
def rank_ic(factor, fwd_ret, mask) -> pd.Series           # 逐日截面 Spearman
def ic_stats(ic: pd.Series) -> dict                       # mean/std/ICIR/t/正比例
def decile_returns(factor, fwd_ret, mask, n=10) -> pd.DataFrame   # 十分组日收益
def monotonicity(decile_ret: pd.DataFrame) -> float       # 分组序号与平均收益的 Spearman
def turnover(weights: pd.DataFrame) -> pd.Series          # 逐期换手(单边)
```

### `engine/backtest.py` — A股规则由代码执行
```python
def forward_returns(adj_close, horizon=1) -> pd.DataFrame  # t→t+horizon 的收益(用后复权价)
def tradable_returns(adj_open) -> pd.DataFrame             # t 日决策可吃到的收益 = t+1 开盘→t+2 开盘;截面指标一律对齐此段
def to_weights(factor, in_universe, top_pct=0.1, min_names=5) -> pd.DataFrame
    # 每日按因子排序取前 top_pct,等权;可选不足时维持前一日目标
def run_backtest(factor, panel, masks, bench=None, top_pct=0.1, ..., bench_open=None) -> dict
    # T+1:t 日收盘算因子 → t+1 开盘调仓;现金账户;漂移再平衡计成本;成本按合同(印花税按日期+佣金+冲击)
    # 基准传 bench_open 时同为"开盘→次日开盘";退市持仓按合同 delisting_haircut 清算
```

**T+1 的执行口径**(写死在代码里):
- t 日收盘后算出因子 → 目标权重
- t+1 开盘按 `can_buy`/`can_sell` 调仓;买不进的保持原权重、卖不掉的强制持有
- 收益从 t+1 开盘算起(不能用 t 日收盘到 t+1 收盘,那是偷看)
- 基准收益取同一时段(开盘→次日开盘);错位时段会虚增超额方差(`4ab2a8c` 实测中证1000 对中证500 跟踪误差 16.3% → 7.0%)

### `engine/null_calib.py`
```python
def shuffle_factor(factor, seed, mode="fixed") -> pd.DataFrame    # 固定 donor 零因子;"fixed_repack_legacy"/"daily" 仅作对照
def null_distribution(factor, run_fn, n=200, seed=7, keys=None, mode="fixed") -> dict   # 零分布 + 换手/持仓诊断
def calibrated_verdict(real: dict, null: dict, q=0.95) -> dict    # 逐指标 p=(1+#)/(n+1),p ≤ 1−q 为通过
def null_validity(real: dict, null: dict, band=(0.8, 1.25)) -> dict  # 零分布有效性闸门
```
流水线默认 n = 50。n < 19 时 5% 水平不可能通过。

### `engine/gates.py`
分层门槛(平静期/危机期,危机窗口取合同 A 股四段),主指标 = **多头超额 vs 中证500**。
成立须同时满足:零校准核心指标全部通过且零分布有效;平静期超额夏普 ≥ 0.8、PSR ≥ 0.9、DSR ≥ 0.95(n 为全研究域累计试验数加任务卡申报档数);危机期超额夏普 ≥ 0;平均持仓 ≥ 20。
零校准回答"是否优于同换手的随机选股",绝对门槛回答"扣费后是否足够赚钱",二者缺一不可;未提供零校准时一律不成立。

## 其他机制
泄漏测试(语法树静态扫描 + 前缀重算,面板版)、机制签名查重(另加行为指纹)、PSR/DSR —— 输入均为 {字段: 宽表} 面板。
组合层增量准入(多个通过门槛的因子是否携带重复信息)在设计中保留,尚未实现。
