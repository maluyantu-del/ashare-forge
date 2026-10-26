"""任务卡05:盈利收益率(估值/质量)。主口径 E/P = 1/peTTM,对照口径 B/P = 1/pbMRQ。

## 为什么是 1/PE 而不是 PE:负 PE 的正确处理

价值因子最常见的实现错误,是直接拿 peTTM 排序、把"PE 最低"当成"最便宜":
亏损公司的 peTTM 是负数——实测 2019-01-02→2024-08-30(库内 611 只)每日平均 21.0%、
最高 27.8% 的标的 peTTM<0——升序排列时它们全部排在最前,于是「低估值组合」实际上变成了
「亏损股组合」,回测出来的价值因子是被这批股票的特性(壳价值、重组预期、退市风险)驱动的,
与估值无关。

本实现取**倒数**,这不是为了好看,而是因为 E/P 在盈利跨过零点时是连续且单调的:
    E/P = 每股盈利 / 股价。E 从正变负,E/P 从正连续地穿过 0 变负;
    亏得越多 E/P 越负,盈利越多 E/P 越正——全程与"盈利能力"同向。
而 PE = P/E 在 E→0 处有极点:微亏公司 PE = -10000,巨亏公司 PE = -1,
按 PE 排序时"微亏"反而排在"巨亏"后面,顺序是错的。
所以取倒数既修了排序方向,也修了极点——**负值无需特殊剔除,自然落到最差一端**。

配套的三条处理(都写死在代码里,不靠注意事项):
  - peTTM 恰为 0(本库实测 0 行,但接口无保证)与 ±inf 一律置 NaN,不进入排序。
  - 可选 drop_negative=True 得到"仅正盈利"档,用来检验因子的 IC 是不是全靠亏损股
    那一条尾巴撑起来的(这是任务卡 H4)。默认 False:E/P 连续,负值信息是真实的。
  - 默认**不做截面缩尾**(WINSOR_Q=0.0)。理由:下游 to_weights 用 nlargest 选股、
    rank_ic 用秩,本就对单调变换免疫;而缩尾会在两端造出并列值,nlargest 遇到并列
    按列顺序取前 k 个 → 变成按股票代码字母序选股。需要做合成因子时再显式开启。

## peTTM 的 PIT 风险与默认滞后(本因子最大的不可辩护点)

baostock 的 peTTM 是日频快照字段,peTTM 字段本身不带发布日;发布日须另从
query_profit_data(pubDate)取;切换时点的实证见 docs/PIT-VALUATION-AUDIT.md。
若它按**报告期末**(如 09-30)而非**披露日**(如 10-28)切换,那么
9 月 30 日到 10 月 28 日之间的因子值就含有当时还没公开的盈利数字 —— 这是数据层面的
前视,前缀重算测不出来(前缀重算只能证明"没有用到未来行",不能证明"当前行本身没有
提前知道未来信息")。审计结论(预注册判据,早跳比例 ≤ 1% 为通过):peTTM 早跳
4/203 = 1.97%,判定不通过;4 例均为单日毛刺或小幅修订,未见在披露日前出现本期数值,
本期切换日 95.5% 与首次披露日(含业绩快报)相同;pbMRQ 1/203 = 0.49%,通过。
数值是否为首次披露值(追溯重述)审计测不出。

本平台无法完全排除,故采取**保守默认**:VALUATION_LAG_DAYS = 20 个交易日(约一个月)。
季报的报告期末与法定披露截止日至少相隔一个月(Q1: 03-31→04-30;Q3: 09-30→10-31),
滞后 20 个交易日后,无论 baostock 按哪种约定切换,季报口径都不再领先于公开信息。
半年报(06-30→08-31)与年报(12-31→04-30)的法定窗口更长,20 日不足以覆盖,
故任务卡把 lag=60 列为必做稳健性档;价值因子本身移动缓慢,理应扛得住这个滞后。
根治办法是用 baostock 的 query_profit_data(含 pubDate/statDate)自建 PIT 盈利表,
属数据层工作,已写入任务卡 next_steps。

方向申报(DIRECTION):+1,即**盈利收益率越高 → 预期收益越高**(价值溢价),
组合做多 E/P 最大的一端。依据与对立假设见 tasks/task05_earnings_yield.yaml。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VALUATION_LAG_DAYS = 20   # 估值字段统一滞后的交易日数;见模块说明"PIT 风险"
DIRECTION = 1.0           # +1 = 价值溢价方向(高 E/P 预期高收益);拉数据前写死
DROP_NEGATIVE = False     # 默认保留负盈利(E/P 连续);True = 仅正盈利档
WINSOR_Q = 0.0            # 默认不缩尾;>0 时按逐日截面分位缩尾
MAIN_FIELD = "peTTM"
CONTRAST_FIELD = "pbMRQ"


def _require(panel: dict[str, pd.DataFrame], field: str) -> None:
    if field not in panel:
        raise KeyError(f"earnings_yield 需要 panel 含 '{field}'(合同 columns.valuation)")


def _cs_winsor(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """逐日截面缩尾到 [q, 1-q] 分位。只用当日数据,不构成前视。"""
    if not q or q <= 0.0:
        return df
    lo, hi = df.quantile(q, axis=1), df.quantile(1.0 - q, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def inverse_valuation(panel: dict[str, pd.DataFrame], field: str = MAIN_FIELD,
                      lag: int = VALUATION_LAG_DAYS, drop_negative: bool = DROP_NEGATIVE,
                      winsor_q: float = WINSOR_Q) -> pd.DataFrame:
    """估值倒数(field=peTTM → 盈利收益率 E/P;field=pbMRQ → 账面市值比 B/P)。

    顺序:取倒数 → 清理 0/±inf → (可选)剔负 → 滞后 lag 个交易日 → (可选)截面缩尾。
    滞后用 shift(lag),即 t 日只看得到 t-lag 日的估值快照,恒为向过去看,不引入前视。
    """
    _require(panel, field)
    v = panel[field].astype("float64")
    y = (1.0 / v.where(v != 0.0)).replace([np.inf, -np.inf], np.nan)
    if drop_negative:
        y = y.where(y > 0.0)
    if lag:
        y = y.shift(int(lag))
    return _cs_winsor(y, winsor_q)


def negative_share(panel: dict[str, pd.DataFrame], field: str = MAIN_FIELD) -> pd.Series:
    """逐日"该字段为负"的股票占比(披露用:亏损公司在池中的权重有多大)。"""
    _require(panel, field)
    v = panel[field].astype("float64")
    n = v.notna().sum(axis=1)
    return ((v < 0.0).sum(axis=1) / n.where(n > 0)).rename(f"{field}_negative_share")


def build_factor(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """主因子:盈利收益率 E/P = 1/peTTM,滞后 VALUATION_LAG_DAYS,保留负盈利。"""
    return DIRECTION * inverse_valuation(panel, MAIN_FIELD)


def build_factor_bp(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """对照因子:账面市值比 B/P = 1/pbMRQ。同样的负值逻辑(负净资产约占 0.4%)。

    作用是分离"便宜"的两种来源:E/P 看盈利,B/P 看净资产。
    若只有 B/P 有效而 E/P 无效,说明起作用的是账面规模而非盈利质量。
    """
    return DIRECTION * inverse_valuation(panel, CONTRAST_FIELD)


def make_builder(field: str = MAIN_FIELD, lag: int = VALUATION_LAG_DAYS,
                 drop_negative: bool = DROP_NEGATIVE, winsor_q: float = WINSOR_Q,
                 direction: float = DIRECTION):
    """稳健性档构造器(lag=0/60、仅正盈利、B/P 等)。

    **每调用一次都是一次独立试验,必须计入任务卡 trials_declared 并进 DSR 的 n_trials。**
    """
    def _build(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return direction * inverse_valuation(panel, field, lag, drop_negative, winsor_q)
    _build.__name__ = f"{field}_inv_lag{lag}{'_pos' if drop_negative else ''}"
    return _build


# --------------------------------------------------------------------------- 机制签名
# 供 pipeline.discover_specs / engine.registry 查重与 DSR 计数使用。
# 只声明任务卡 task05 的 T1(E/P 主口径)与 T2(B/P 对照);
# T3(lag=0)与 T4(lag=60 且仅正盈利)是按需稳健性档,由 make_builder 显式调用后单独入账。
# 注:registry 把 shift 视作装饰算子,故 lag 不同的档位共享同一机制签名——这是对的,
# 滞后天数是参数而非机制;而 E/P 与 B/P 的 columns 不同,签名不同。
SPECS = [
    {
        "name": "earnings_yield_ep",
        "task_id": "task05_earnings_yield",
        "family": "value",
        "operators": ["inverse"],
        "direction": "long_only",
        "horizon_days": 1,
        "columns": ["peTTM"],
        "build_factor": build_factor,
        "thesis": ("价值溢价:高盈利收益率(E/P)意味着单位价格买到更多当期盈利。取倒数而非直接用 PE,"
                   "是因为 E/P 在盈利跨零点时连续且单调,亏损公司自然落到最差端,"
                   "不会像按 PE 升序那样把亏损股当成最便宜。对立假设是行业押注、区制依赖,"
                   "以及 peTTM 非 PIT 带来的发布时滞前视。"),
        "params": {"VALUATION_LAG_DAYS": VALUATION_LAG_DAYS, "DIRECTION": DIRECTION,
                   "DROP_NEGATIVE": DROP_NEGATIVE, "WINSOR_Q": WINSOR_Q, "field": MAIN_FIELD},
    },
    {
        "name": "earnings_yield_bp",
        "task_id": "task05_earnings_yield",
        "family": "value",
        "operators": ["inverse"],
        "direction": "long_only",
        "horizon_days": 1,
        "columns": ["pbMRQ"],
        "build_factor": build_factor_bp,
        "thesis": ("账面市值比(B/P)对照档:用于分离「便宜」的两种来源——E/P 看盈利,B/P 看净资产。"
                   "若只有 B/P 有效,说明起作用的是账面规模而非盈利质量。"),
        "params": {"VALUATION_LAG_DAYS": VALUATION_LAG_DAYS, "DIRECTION": DIRECTION,
                   "DROP_NEGATIVE": DROP_NEGATIVE, "WINSOR_Q": WINSOR_Q, "field": CONTRAST_FIELD},
    },
]


# 任务卡 task05 已申报但此前从未运行的 T3/T4(2026-09 预注册履约)。batch_default=False:
# 只在按名点跑时运行一次,不随每次全量批跑重复入账(试验数已按申报计入 DSR)。
SPECS += [
    {**SPECS[0], "name": "earnings_yield_ep_lag0", "batch_default": False,
     "build_factor": make_builder(lag=0),
     "params": {**SPECS[0]["params"], "VALUATION_LAG_DAYS": 0},
     "thesis": "task05 T3:不滞后档,给出性能上界与前视敏感度(是否为合法档取决于 PIT 审计结论)。"},
    {**SPECS[0], "name": "earnings_yield_ep_lag60_pos", "batch_default": False,
     "build_factor": make_builder(lag=60, drop_negative=True),
     "params": {**SPECS[0]["params"], "VALUATION_LAG_DAYS": 60, "DROP_NEGATIVE": True},
     "thesis": "task05 T4:长滞后加仅正盈利,检验效应是否来自披露时点或亏损股尾巴。"},
]

