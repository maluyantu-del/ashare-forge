"""任务卡04:Amihud 非流动性(收益冲击 / 成交额)。

Amihud(2002)的原始定义:ILLIQ_i,t = mean_{s in 窗口} ( |r_i,s| / DollarVolume_i,s )。
含义是"单位成交额推动价格的幅度"——同样一亿元砸下去,价格动得越多,这只股票越难交易。
非流动性溢价假说认为投资者要求补偿才愿意持有这类股票,故预期收益更高。

本实现相对教科书公式的四处工程判断(每一处都是实测坑,写在这里而不是"注意事项"里):

1. **分母用 amount 不用 volume**。volume 是股数,10 元的股票和 100 元的股票同样 1 万股,
   资金容量差 10 倍;用股数会把高价股系统性判成"更流动"。amount(成交额,元)才是量纲正确的
   资金口径。amount<=0(停牌、无成交)的日子整天剔除,不参与窗口均值,也不按 0 计入分母
   (除以 0 会得到 inf,inf 参与排序会把停牌股顶到最前)。

2. **收益用 pctChg 而不是 adj_close 的日变化**。本库 adj_factor 表的覆盖远落后于 daily 表
   (实测 331 只已同步股票中只有 30 只有复权因子),缺因子的股票 adj_close 退化为原始 close,
   除权日会凭空出现一根 1% 量级的假跌幅,直接污染 |r|。而 baostock 的 preclose 就是除权参考价
   ——实测 sh.600000 于 2018-07-13 除权:preclose=9.47,而前一日 close=9.57,pctChg=+0.21%
   与真实涨跌一致——所以 pctChg/100 本身已是除权调整后的收益,且不依赖复权因子表的完整度。
   需要用后复权口径时传 return_source="adj_close"(复权因子补齐后应改用该档并复核结论)。
   (上述覆盖率为撰写时的状态;复权因子现已全量同步,并有完整性闸门拦截缺因子的股票。
   主口径仍为 pctChg,adj_close 档尚未单独运行。)

3. **|r| 与 amount 必须配对到同一根 K 线,且要求前一交易日也有成交**。停牌数周后复牌当日的
   pctChg 是跨越整个停牌期的累计涨跌,配上复牌当日的正常成交额,会被记成一次极端冲击。
   这里要求 t 与 t-1 两日都有成交才计入该日的冲击值。配对纪律:比值的分子(价格变动)与分母
   (成交额)必须覆盖同一段连续交易区间,否则比值没有经济含义。

4. **窗口包含当日,这不是前视**。t 日的 |r_t| 与 amount_t 在 t 日收盘时已知,而建仓在 t+1 开盘,
   信息与执行的先后关系成立。任务卡01 里"箱顶必须 shift(1)"针对的是另一类结构:被比较的**参照
   水平**若含当日价,当日突破幅度就永远为正,那才是同日前视。两者的区别在于当日值是"被观测量"
   还是"参照系"。本因子的因果性由 engine.leakage.prefix_recompute_test 机械验证。

方向申报(DIRECTION):+1,即**非流动性越高 → 预期收益越高**,组合做多 ILLIQ 最大的一端。
依据见任务卡 task04_illiquidity.yaml 的 direction_declaration;若实测符号翻转,应按"流动性偏好/
拥挤度"的对立假设重新解释,而不是事后把方向改过来——方向在拉数据前已写死。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WIN = 20                 # Amihud 窗口(约一个月交易日)
MIN_OBS = 15             # 窗口内至少这么多个有效交易日才出值,否则 NaN
AMOUNT_UNIT = 1e8        # 成交额折算单位:亿元。冲击值 = 每亿元成交额对应的收益绝对值
DIRECTION = 1.0          # +1 = 非流动性溢价方向(高 ILLIQ 预期高收益);拉数据前写死
RETURN_SOURCE = "pctChg"  # 见模块说明第 2 条;可选 "adj_close"
MIN_NAMES_NEUTRAL = 20   # 市值中性化时单日最少样本数,不足则该日整行 NaN

REQUIRED_FIELDS: tuple[str, ...] = ("amount", "pctChg")
REQUIRED_FIELDS_NEUTRAL: tuple[str, ...] = ("amount", "pctChg", "float_mcap")


def _require(panel: dict[str, pd.DataFrame], fields: tuple[str, ...]) -> None:
    missing = [f for f in fields if f not in panel]
    if missing:
        raise KeyError(f"illiquidity 需要 panel 含 {list(fields)},缺 {missing}")


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """逐日截面百分位秩(0~1)。只用当日数据,不构成前视。"""
    return df.rank(axis=1, pct=True)


def daily_return(panel: dict[str, pd.DataFrame], source: str = RETURN_SOURCE) -> pd.DataFrame:
    """当日收益率(小数)。默认 pctChg/100(已含除权调整,不依赖复权因子表)。"""
    if source == "pctChg":
        return panel["pctChg"].astype("float64") / 100.0
    if source == "adj_close":
        a = panel["adj_close"].astype("float64")
        return a / a.shift(1) - 1.0
    raise ValueError(f"return_source 只支持 'pctChg' 或 'adj_close',收到 {source!r}")


def traded_mask(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """当日是否真的成交过:amount>0,且(若面板含 tradestatus)非停牌。NaN 一律 False。"""
    amt = panel["amount"].astype("float64")
    m = amt > 0.0                                   # NaN > 0 → False
    if "tradestatus" in panel:
        m = m & (panel["tradestatus"] == 1.0)
    return m.fillna(False).astype(bool)


def daily_impact(panel: dict[str, pd.DataFrame], source: str = RETURN_SOURCE) -> pd.DataFrame:
    """逐日收益冲击 = |r_t| / (amount_t / 亿元)。无成交日、以及前一交易日无成交的日子记 NaN。"""
    _require(panel, REQUIRED_FIELDS if source == "pctChg" else ("amount", "adj_close"))
    ret = daily_return(panel, source)
    amt = panel["amount"].astype("float64")
    traded = traded_mask(panel)
    paired = traded & traded.shift(1, fill_value=False)   # t 与 t-1 都要有成交
    denom = (amt / AMOUNT_UNIT).where(amt > 0.0)
    imp = ret.abs() / denom
    return imp.where(paired).replace([np.inf, -np.inf], np.nan)


def illiquidity(panel: dict[str, pd.DataFrame], win: int = WIN, min_obs: int = MIN_OBS,
                source: str = RETURN_SOURCE) -> pd.DataFrame:
    """Amihud ILLIQ:冲击值的 win 日均值(数值越大越不流动)。窗口含当日,见模块说明第 4 条。

    用均值(教科书口径)而非中位数:窗口内一根极端冲击日可主导均值,这是已知的脆弱点,
    须在结论中披露;改中位数是一次独立试验,须计入 DSR 的 n_trials,不得事后挑选。
    """
    imp = daily_impact(panel, source)
    return imp.rolling(win, min_periods=min_obs).mean()


N_SIZE_GROUPS = 10


def _group_neutralize(y: pd.DataFrame, x: pd.DataFrame, n_groups: int, min_names: int) -> pd.DataFrame:
    """逐日按 x 分成 n_groups 组,在组内对 y 取百分位秩并居中(-0.5~+0.5)。

    结果与 x 所在分组无关(每组内均值≈0),因而剔除了 y 对 x 的线性**与非线性**依赖。
    只用当日截面 → 不构成前视;单日有效样本不足 min_names 时该日整行 NaN。
    """
    xv = x.reindex(index=y.index, columns=y.columns)
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    for d in y.index:
        a, b = y.loc[d], xv.loc[d]
        ok = a.notna() & b.notna()
        if int(ok.sum()) < max(min_names, n_groups * 3):
            continue
        g = pd.qcut(b[ok].rank(method="first"), n_groups, labels=False)
        r = a[ok].groupby(g).rank(pct=True) - 0.5
        out.loc[d, r.index] = r.to_numpy()
    return out


def _cs_resid(y: pd.DataFrame, x: pd.DataFrame, min_names: int) -> pd.DataFrame:
    """逐日截面 OLS(y ~ a + b·x)的残差,逐行向量化实现(不引入 scipy)。

    只用当日截面数据 → 不构成前视;单日有效样本不足 min_names 时该日整行 NaN。
    """
    yv = y.to_numpy(dtype="float64")
    xv = x.reindex(index=y.index, columns=y.columns).to_numpy(dtype="float64")
    ok = ~(np.isnan(yv) | np.isnan(xv))
    n = ok.sum(axis=1).astype("float64")
    good = n >= float(min_names)
    nn = np.where(good, n, np.nan)
    my = np.where(ok, yv, 0.0).sum(axis=1) / nn
    mx = np.where(ok, xv, 0.0).sum(axis=1) / nn
    dy = np.where(ok, yv - my[:, None], 0.0)
    dx = np.where(ok, xv - mx[:, None], 0.0)
    vxx = (dx * dx).sum(axis=1)
    beta = np.divide((dx * dy).sum(axis=1), vxx, out=np.zeros_like(vxx), where=vxx > 0.0)
    res = np.where(ok, dy - beta[:, None] * dx, np.nan)
    res[~good] = np.nan
    return pd.DataFrame(res, index=y.index, columns=y.columns)


def build_factor(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """主因子:Amihud 非流动性,按 DIRECTION 定向(+1 = 做多最不流动的一端)。"""
    return DIRECTION * illiquidity(panel)


def build_factor_size_neutral(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """对照因子:剔除流通市值后的非流动性 —— 直接回答"是不是小市值换皮"。

    做法(十分组中性化):当日按流通市值分成 N_SIZE_GROUPS=10 组,在组内对 ILLIQ 取百分位秩
    并居中(见 _group_neutralize)。结果与所在市值组无关,线性与非线性的市值依赖一并剔除。
    只用组内排序、不用原值:ILLIQ 与市值都极度右偏,原值会被少数极端股主导;分组只依赖
    市值的排序,取不取对数结果相同。
    早期版本为"ILLIQ 截面秩对市值截面秩做一元线性回归取残差",只去掉线性部分,
    实测残差与市值的 Spearman 仍为 −0.21;改为十分组后为 −0.017(见下方注释)。
    """
    _require(panel, REQUIRED_FIELDS_NEUTRAL)
    illiq = illiquidity(panel)
    mcap = panel["float_mcap"].astype("float64")
    # 审查发现(P1):原做法是"ILLIQ 秩 对 市值秩"的线性回归取残差,只去掉线性关系;
    # 二者关系是弯的(最大市值档流动性陡降),实测残差与市值的 Spearman 仍 -0.21,
    # 不能读成"剔除市值后"。改用**市值分组内重新排序**(业界标准的分组中性化),
    # 连非线性依赖一并剔除。
    resid = _group_neutralize(illiq, mcap.where(mcap > 0.0), N_SIZE_GROUPS, MIN_NAMES_NEUTRAL)
    return DIRECTION * resid


def make_builder(win: int = WIN, min_obs: int = MIN_OBS, source: str = RETURN_SOURCE,
                 direction: float = DIRECTION, size_neutral: bool = False):
    """稳健性档构造器。**每调用一次都是一次独立试验,必须计入任务卡的 trials_declared。**"""
    def _build(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        illiq = illiquidity(panel, win=win, min_obs=min_obs, source=source)
        if not size_neutral:
            return direction * illiq
        _require(panel, ("float_mcap",))
        mcap = panel["float_mcap"].astype("float64")
        return direction * _cs_resid(_cs_rank(illiq), _cs_rank(mcap.where(mcap > 0.0)),
                                     MIN_NAMES_NEUTRAL)
    _build.__name__ = f"illiq_w{win}_{source}{'_sn' if size_neutral else ''}"
    return _build


# --------------------------------------------------------------------------- 机制签名
# 供 pipeline.discover_specs / engine.registry 查重与 DSR 计数使用。
# 只声明任务卡 task04 的 T1(主口径)与 T2(市值中性化);T3(win=60)属按需稳健性档,
# 由 make_builder 显式调用后单独入账——把它放进 SPECS 会让每次批跑都平白多记一次试验。
# 注:registry 把 cs_rank/clip 等视作装饰算子剔除,而 neutralize 是核心算子
# (中性化改变了因子在赌什么),故 T1 与 T2 的机制签名不同,这正是预期行为。
SPECS = [
    {
        "name": "illiquidity_amihud",
        "task_id": "task04_illiquidity",
        "family": "liquidity",
        "operators": ["pct_change", "abs", "intensity_norm", "rolling_mean"],
        "direction": "long_only",
        "horizon_days": 1,
        "columns": ["amount", "pctChg", "tradestatus"],
        "build_factor": build_factor,
        "thesis": ("Amihud 非流动性溢价:单位成交额推动价格的幅度越大,持有者要求的预期收益越高;"
                   "A股换手率因子长期为负亦指向同一符号。对立假设是它只是流通市值的代理,"
                   "以及冲击成本本身会把毛收益吃光。"),
        "params": {"WIN": WIN, "MIN_OBS": MIN_OBS, "AMOUNT_UNIT": AMOUNT_UNIT,
                   "DIRECTION": DIRECTION, "RETURN_SOURCE": RETURN_SOURCE},
    },
    {
        "name": "illiquidity_amihud_size_neutral",
        "task_id": "task04_illiquidity",
        "family": "liquidity",
        "operators": ["pct_change", "abs", "intensity_norm", "rolling_mean", "neutralize"],
        "direction": "long_only",
        "horizon_days": 1,
        "columns": ["amount", "pctChg", "tradestatus", "float_mcap"],
        "build_factor": build_factor_size_neutral,
        "thesis": ("把 Amihud 非流动性对流通市值做截面正交后的残差,直接回答「非流动性是不是"
                   "小市值换皮」。实测原始 ILLIQ 与流通市值的日均截面秩相关 -0.84,"
                   "故主口径的任何正结果都必须与本档并列汇报。"),
        "params": {"WIN": WIN, "MIN_OBS": MIN_OBS, "DIRECTION": DIRECTION,
                   "MIN_NAMES_NEUTRAL": MIN_NAMES_NEUTRAL, "neutralize_on": "float_mcap(截面秩)"},
    },
]


# 任务卡 task04 已申报但此前从未运行的 T3(2026-09 预注册履约)。batch_default=False:
# 只在按名点跑时运行一次,不随每次全量批跑重复入账。
SPECS += [
    {**SPECS[0], "name": "illiquidity_amihud_w60", "batch_default": False,
     "build_factor": make_builder(win=60, min_obs=40),
     "params": {**SPECS[0]["params"], "WIN": 60, "MIN_OBS": 40},
     "thesis": "task04 T3:窗口稳健性(win=60)。"},
]

