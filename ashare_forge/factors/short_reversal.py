"""任务卡02:短期反转(A股最经典的截面效应)。

定义:过去 WINDOW 个交易日的累计收益取负,再取当日截面百分位秩。
排名越高 = 过去跌得越多 = 预期后续反弹。

它修的坑(每条都是"反转因子看起来能赚、实盘赚不到"的常见来源):

1. **窗口口径必须写死,且不能含未来**。本因子对"当日"的处理是显式参数 `skip`:
   - `skip=1`(默认,本卡主口径):窗口 = t-WINDOW .. t-1,**剔除当日**。
   - `skip=0`(教科书口径):窗口 = t-WINDOW+1 .. t,含当日收盘。
   两者都不构成前视(t 日收盘价在 t 日收盘时已知),但**代价不同**:本引擎的执行链是
   factor[t] → t+1 开盘建仓 → 吃 t+1→t+2 开盘收益,`skip=1` 等于在这条链上再压一天时滞,
   共延迟两个交易日。短期反转的信息半衰期本就以天计,这层时滞的代价必须被测量而不是被假设——
   故任务卡把它写成 H3 的对照运行,由数据裁定,不事后挑好的那版。

2. **停牌导致的分母缺失**。停牌日多数仍有K行(tradestatus=0),但长停牌段没有K行;
   `close / close.shift(N)` 遇到端点 NaN 会整段失效,或在复牌日把"补涨/补跌"错认为反转。
   本模块:用 `ffill(limit=MAX_STALE_DAYS)` 桥接短停牌(前向填充只用过去,不构成前视),
   超过上限不再沿用旧价;并要求窗口内实际成交日占比 ≥ MIN_TRADED_RATIO,否则该单元置 NaN。

3. **复权**。必须用后复权价(adj_close),原始收盘价在除权日会造出假暴跌,
   而"假暴跌"恰好是反转因子最爱的信号——这会系统性地把除权股选进多头组。

4. **取秩而非取原值**。累计收益的尾部极端值(如翻倍或腰斩)会主导等权组合,
   截面秩把量纲与极值影响压掉;秩只用当日截面数据,不构成前视。

所需面板字段:adj_close(必需);tradestatus 或 volume(可选,用于判定"当日是否真成交",
缺失时退化为"当日有无K行")。
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (保持与其他因子模块一致的导入约定)
import pandas as pd

WINDOW = 20                 # 反转观察窗(交易日),约一个月
SKIP_RECENT = 1             # 剔除当日:窗口右端取 t-SKIP_RECENT
MAX_STALE_DAYS = 5          # 停牌桥接上限:连续 >5 个交易日无价则不再沿用旧价
MIN_TRADED_RATIO = 0.6      # 窗口内实际成交日占比下限,低于此视为样本不可用


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """逐日截面百分位秩(0~1)。只用当日数据,不构成前视。"""
    return df.rank(axis=1, pct=True)


def _traded(panel: dict[str, pd.DataFrame], ref: pd.DataFrame) -> pd.DataFrame:
    """当日是否真的成交(bool 宽表)。优先 tradestatus,其次 volume,最后退化为"有无K行"。"""
    if "tradestatus" in panel:
        ts = panel["tradestatus"].reindex(index=ref.index, columns=ref.columns)
        return (ts == 1) & ref.notna()
    if "volume" in panel:
        v = panel["volume"].reindex(index=ref.index, columns=ref.columns)
        return (v > 0) & ref.notna()
    return ref.notna()


def reversal_score(panel: dict[str, pd.DataFrame], window: int = WINDOW,
                   skip: int = SKIP_RECENT) -> pd.DataFrame:
    """未取秩的连续值:过去 window 日累计收益的**相反数**(越大 = 跌得越多)。

    诊断与稳健性对照用(如与换手、市值的截面相关)。build_factor 在其上再取截面秩。
    """
    if window < 2:
        raise ValueError(f"window 必须 ≥2,收到 {window}")
    if skip < 0:
        raise ValueError(f"skip 必须 ≥0,收到 {skip}")

    raw = panel["adj_close"].astype("float64")
    px = raw.ffill(limit=MAX_STALE_DAYS)            # 桥接短停牌;前向填充只用过去

    end = px.shift(skip)                            # 窗口右端(skip=1 时为 t-1)
    beg = px.shift(skip + window)                   # 窗口左端
    cum = end / beg.where(beg > 0) - 1.0            # 窗口累计收益

    traded = _traded(panel, raw).astype("float64")
    n_traded = traded.shift(skip).rolling(window, min_periods=window).sum()

    ok = raw.notna() & cum.notna() & (n_traded >= MIN_TRADED_RATIO * window)
    out = (-cum).where(ok)
    out.index.name, out.columns.name = raw.index.name, raw.columns.name
    return out


def build_factor(panel: dict[str, pd.DataFrame], window: int = WINDOW,
                 skip: int = SKIP_RECENT) -> pd.DataFrame:
    """短期反转因子(日期×股票的截面百分位秩,越大 = 越看多)。"""
    return _cs_rank(reversal_score(panel, window=window, skip=skip))


def build_factor_skip0(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """预注册对照:skip=0(窗口含当日收盘)。任务卡规定主口径为负时须先排除"时滞吃掉信号"。"""
    return build_factor(panel, window=WINDOW, skip=0)


def _hold(factor: pd.DataFrame, every: int) -> pd.DataFrame:
    """每 every 个交易日才更新一次因子,期间沿用最近一次的值(只用过去,不构成前视)。

    调仓日按面板内的行位置取(位置 0, every, 2·every, ...),前缀截断不改变已有行的位置,
    故与前缀重算泄漏检测一致。沿用的值被 ffill(limit=every-1) 限定在一个周期内。
    """
    if every < 2:
        return factor
    keep = pd.Series((np.arange(len(factor)) % every) == 0, index=factor.index)
    return factor.where(keep, axis=0).ffill(limit=every - 1)


def build_factor_hold5(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """事后追加(非预注册):主口径每 5 个交易日调仓一次,检验降换手能否覆盖成本。"""
    return _hold(build_factor(panel), 5)


def build_factor_hold20(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """事后追加(非预注册):主口径每 20 个交易日调仓一次。"""
    return _hold(build_factor(panel), 20)


# 驱动(pipeline.discover_specs)读取的机制声明。operators 里的装饰性算子(cs_rank/gate)
# 会被 registry 规范化时剔除——保留它们是为了如实描述代码,而非为了做出一个新签名。
SPEC = {
    "name": "short_reversal",
    "task_id": "task02_short_reversal",
    "family": "reversal",
    "operators": ["pct_change", "cs_rank", "gate"],
    "direction": "long_only",
    "horizon_days": 1,
    "columns": ["adj_close", "tradestatus"],     # volume 仅在无 tradestatus 时兜底
    "thesis": ("过去一个月跌幅最大的股票因流动性提供补偿与过度反应修正而反弹;"
               "对立假设是它只是换手率因子的换皮,或收益集中在已被合同剔除的微盘段,"
               "再或跌幅最大的一端多为一字跌停、T+1 下买不进而无法兑现。"),
    "params": {"WINDOW": WINDOW, "SKIP_RECENT": SKIP_RECENT,
               "MAX_STALE_DAYS": MAX_STALE_DAYS, "MIN_TRADED_RATIO": MIN_TRADED_RATIO},
}


# 同一任务卡下的对照与变体。skip0 属预注册参数集;hold5/hold20 为 BATCH01 观察到
# "IC 强但年亏"之后事后追加的降换手检验——看过数据才提出,故只能生成假设、不能确认假设,
# 任何正结果须另立任务卡并在保留段上独立复核。全部计入任务卡 trials_declared。
_POST_HOC = ("事后追加(BATCH01 之后提出,非预注册):检验降低调仓频率能否让成本低于反转收益。"
             "对立假设是反转信息半衰期以天计,持有期拉长后收益衰减快于成本下降。")
SPECS = [
    SPEC,
    {**SPEC, "name": "short_reversal_skip0", "build_factor": build_factor_skip0,
     "params": {**SPEC["params"], "SKIP_RECENT": 0},
     "thesis": "预注册对照:窗口含当日收盘(skip=0),排除'一日时滞吃掉信号'的解释。"},
    {**SPEC, "name": "short_reversal_hold5", "build_factor": build_factor_hold5, "batch_default": False,
     "operators": ["pct_change", "cs_rank", "gate", "hold"],
     "params": {**SPEC["params"], "HOLD_EVERY": 5}, "thesis": _POST_HOC},
    {**SPEC, "name": "short_reversal_hold20", "build_factor": build_factor_hold20, "batch_default": False,
     "operators": ["pct_change", "cs_rank", "gate", "hold"],
     "params": {**SPEC["params"], "HOLD_EVERY": 20}, "thesis": _POST_HOC},
]
