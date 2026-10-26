"""任务卡03:特质波动率(低波异象)。

定义:过去 WINDOW 个交易日的**特质**收益标准差取负,再取当日截面百分位秩。
排名越高 = 波动越低 = 预期收益越高(低波异象的方向)。

"特质"部分的三种口径(`mode`,默认 "capm"):
  - "capm"  :对市场代理做滚动单因子回归,取残差标准差。学术上的 IVOL 口径。
  - "excess":固定 beta=1,取 (个股收益 − 市场收益) 的标准差。少估一个参数、噪声更小。
  - "total" :不扣市场,直接取个股收益标准差。总波动率,作为"特质 vs 总量"的对照。
市场代理优先取面板里的 `bench_close`(指数收盘价);没有则用当日截面等权平均收益。
两者都只用当日及以前数据。

它修的坑:

1. **只有市场一个因子,残差里仍留着规模与风格**。Ang 等人的 IVOL 是对 FF3 取残差;
   本口径缺 SMB/HML,残差中的规模暴露没被扣掉。这不是实现瑕疵而是数据边界,
   直接导致任务卡 known_dead_ends 里那条"低波可能只是市值因子换皮"的风险必须被单独检验,
   不能靠"我们做了特质化"就宣称已排除。

2. **停牌日的"零收益"会把波动率做低**。桥接停牌得到的 0 收益若计入标准差,
   长停牌股会稳定排进"最低波"多头组——而它恰恰是最不该买的。
   本模块把非成交日的收益置 NaN(不计入窗口),并要求窗口内有效观测数 ≥ MIN_OBS_RATIO×窗口。

3. **价格冻结被判为"最优"**。连续一字板或长期停牌会让个股收益恒为 0,取负后排在最前。
   只检查 `vol>0` 并不够:"excess" 口径把 beta 固定为 1,冻结股的超额收益 = −市场收益,
   其标准差恰好等于**市场自身波动**(>0 且不算大),照样能混进低波多头组(本模块自测实测)。
   因此另设一道与口径无关的闸:个股**自身**收益标准差必须 >0,否则该单元置 NaN。

4. **滚动回归的自由度**。残差方差按总体矩算出后乘 n/(n-2) 还原为无偏估计;
   n 是窗口内**有效观测数**(不是窗口长度),因为停牌会让二者不等。

5. **协方差各项的缺失模式必须一致**。个股收益与市场收益逐单元取交集后再算各阶矩,
   否则 E[xy]、E[x]、E[y] 各自跳过不同的 NaN,协方差恒等式不再成立,beta 会被算错。

所需面板字段:adj_close(必需);tradestatus 或 volume(可选,判定当日是否真成交);
bench_close(可选,指数收盘价 Series 或单列 DataFrame,缺失则用截面等权市场代理)。
"""
from __future__ import annotations

import math

import numpy as np  # noqa: F401  (保持与其他因子模块一致的导入约定)
import pandas as pd

WINDOW = 60                 # 波动观察窗(交易日),约一个季度
MODE = "capm"               # capm | excess | total
MAX_STALE_DAYS = 5          # 停牌桥接上限
MIN_OBS_RATIO = 0.7         # 窗口内有效收益观测占比下限
MIN_MARKET_NAMES = 10       # 截面等权市场代理所需的最少股票数
TRADING_DAYS = 243          # 年化用(仅影响可读性,不影响截面排序)
VALID_MODES = ("capm", "excess", "total")


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


def _daily_returns(panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (个股日收益, 当日有K行掩码)。非成交日的收益置 NaN(不计入窗口)。"""
    raw = panel["adj_close"].astype("float64")
    px = raw.ffill(limit=MAX_STALE_DAYS)            # 桥接短停牌;前向填充只用过去
    traded = _traded(panel, raw)
    r = (px / px.shift(1) - 1.0).where(traded)
    return r, raw.notna()


def _market_returns(panel: dict[str, pd.DataFrame], r: pd.DataFrame) -> pd.Series:
    """市场代理日收益。优先指数(bench_close),否则当日截面等权平均。"""
    if "bench_close" in panel:
        b = panel["bench_close"]
        if isinstance(b, pd.DataFrame):
            if b.shape[1] != 1:
                raise ValueError(f"bench_close 为 DataFrame 时必须只有一列,收到 {b.shape[1]} 列")
            b = b.iloc[:, 0]
        b = pd.Series(b).astype("float64").reindex(r.index).ffill()
        rm = b / b.shift(1) - 1.0
    else:
        n = r.notna().sum(axis=1)
        rm = r.mean(axis=1).where(n >= MIN_MARKET_NAMES)   # 截面等权,只用当日数据
    rm.index.name = r.index.name
    return rm.astype("float64")


def idio_volatility(panel: dict[str, pd.DataFrame], window: int = WINDOW,
                    mode: str = MODE, annualize: bool = True) -> pd.DataFrame:
    """未取秩的连续值:特质波动率本身(越大 = 越波动)。诊断与稳健性对照用。"""
    if mode not in VALID_MODES:
        raise ValueError(f"mode 必须是 {VALID_MODES} 之一,收到 {mode!r}")
    if window < 5:
        raise ValueError(f"window 必须 ≥5,收到 {window}")
    min_obs = max(3, int(math.ceil(MIN_OBS_RATIO * window)))

    r, _ = _daily_returns(panel)
    own = r.rolling(window, min_periods=min_obs).std(ddof=1)    # 个股自身波动,兼作"价格冻结"闸

    if mode == "total":
        vol = own
    else:
        rm = _market_returns(panel, r)
        x_full = pd.DataFrame(np.repeat(rm.to_numpy(dtype="float64")[:, None], r.shape[1], axis=1),
                              index=r.index, columns=r.columns)
        valid = r.notna() & x_full.notna()
        x, y = x_full.where(valid), r.where(valid)      # 逐单元取交集:各阶矩的缺失模式一致
        if mode == "excess":
            vol = (y - x).rolling(window, min_periods=min_obs).std(ddof=1)
        else:                                           # capm:滚动单因子回归残差
            cnt = valid.astype("float64").rolling(window, min_periods=1).sum()
            m_y = y.rolling(window, min_periods=min_obs).mean()
            m_x = x.rolling(window, min_periods=min_obs).mean()
            m_xy = (x * y).rolling(window, min_periods=min_obs).mean()
            m_xx = (x * x).rolling(window, min_periods=min_obs).mean()
            m_yy = (y * y).rolling(window, min_periods=min_obs).mean()
            cov = m_xy - m_x * m_y
            var_x = (m_xx - m_x * m_x)
            var_y = (m_yy - m_y * m_y)
            resid_var = (var_y - cov * cov / var_x.where(var_x > 0)).clip(lower=0.0)
            dof = (cnt - 2.0).where(cnt > 2)            # 估了截距与 beta 两个参数
            vol = (resid_var * cnt / dof).pow(0.5)      # 总体矩 → 无偏估计

    # 两道闸都要过:估计量本身为正,且个股自身确实动过(冻结股在 excess 口径下 vol 也会 >0)
    vol = vol.where((vol > 0) & (own > 0))
    if annualize:
        vol = vol * math.sqrt(TRADING_DAYS)
    vol.index.name, vol.columns.name = r.index.name, r.columns.name
    return vol


def build_factor(panel: dict[str, pd.DataFrame], window: int = WINDOW,
                 mode: str = MODE) -> pd.DataFrame:
    """低波因子(日期×股票的截面百分位秩,越大 = 波动越低 = 越看多)。"""
    vol = idio_volatility(panel, window=window, mode=mode)
    has_k = panel["adj_close"].astype("float64").notna()      # 当日无K行者不参与排序
    return _cs_rank((-vol).where(has_k))


# 驱动(pipeline.discover_specs)读取的机制声明,对应**主口径**(window=60, mode=capm)。
# 任务卡03 预注册的 window/mode 对照属于稳健性运行,须各自单独计入 n_trials(DSR 会因此收紧),
# 不在此处声明为独立机制——它们赌的是同一件事。
def build_factor_total(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """对照 A:总波动率(不扣市场)。与主口径并排,检验"特质"是否真的发生。"""
    return build_factor(panel, mode="total")


def build_factor_low_turnover(panel: dict[str, pd.DataFrame], window: int = WINDOW) -> pd.DataFrame:
    """对照 B:纯低换手(过去 window 日平均换手率取负)。

    审查发现(P1):主口径与总波动的截面秩相关 0.966(残差波动/总波动中位 0.90,
    "特质"只剔掉约 10%),与纯低换手 0.527;纯低换手在同一流水线上 IC 0.0440/t 12.33/
    单调性 0.976,与主口径(0.0405/11.84/0.988)基本同构。
    故主口径**不能单独作为"特质波动率异象"的证据**,必须与本对照并排报告;
    差异不显著时,裁定须写明"本口径无法区分 IVOL 与低换手"。
    """
    turn = panel["turn"].astype("float64")
    tv = turn.where(turn > 0)
    avg = tv.shift(1).rolling(window, min_periods=int(window * MIN_OBS_RATIO)).mean()
    return -avg.rank(axis=1, pct=True)


# 驱动(pipeline.discover_specs)读取的机制声明。主口径之外并排两个**对照**:
# 它们不是新想法,而是检验主口径论点(特质波动 ≠ 总波动 ≠ 低换手)所必需的对照组,
# 各自计入 n_trials。
_COMMON = {"task_id": "task03_idio_volatility", "family": "volatility", "horizon_days": 1,
           "params": {"WINDOW": WINDOW, "MODE": MODE, "MAX_STALE_DAYS": MAX_STALE_DAYS,
                      "MIN_OBS_RATIO": MIN_OBS_RATIO, "MIN_MARKET_NAMES": MIN_MARKET_NAMES}}
SPECS = [
    {**_COMMON, "name": "idio_volatility",
     "operators": ["pct_change", "rolling_std", "market_resid", "cs_rank", "gate"],
     "direction": "long_only", "columns": ["adj_close", "tradestatus", "bench_close"],
     "build_factor": build_factor,
     "thesis": ("杠杆约束与彩票偏好使高波动股被系统性高估,做多低特质波动可收取这份补偿;"
                "对立假设是低波组近似大市值组,或超额来自多头组 beta<1 的市场暴露而非选股能力。")},
    {**_COMMON, "name": "idio_volatility_total",
     "operators": ["pct_change", "rolling_std", "cs_rank", "gate"],
     "direction": "long_only", "columns": ["adj_close", "tradestatus"],
     "build_factor": build_factor_total,
     "thesis": "对照 A:总波动率(不扣市场)。若与主口径无显著差异,则'特质'部分未起作用。"},
    {**_COMMON, "name": "low_turnover_control", "family": "liquidity",
     "operators": ["rolling_mean", "cs_rank"],
     "direction": "long_only", "columns": ["turn"],
     "build_factor": build_factor_low_turnover,
     "thesis": "对照 B:纯低换手。审查实测主口径与之秩相关 0.527、业绩同构;用于区分 IVOL 与低换手。"},
]
