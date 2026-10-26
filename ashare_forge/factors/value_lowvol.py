"""任务卡06:便宜又不乱跳(价值 × 低波 合成)。

把两个各自 FAIL 的信息源做等权秩合成:
  - 价值:盈利收益率 E/P(复用 task05 主口径,含 20 日估值滞后与负盈利处理)
  - 低波:特质波动率取负(复用 task03 主口径 capm 残差,含停牌/冻结闸)
合成 = 0.5 × 截面秩(E/P) + 0.5 × 截面秩(低波)。两者缺一则该单元置 NaN(不以单边信息充数)。

为什么值得单独立卡而不是"两个 FAIL 加起来还是 FAIL":
  task05 的 E/P 平静期超额夏普 0.28、task03 的低波 0.71,二者在 BATCH02 诊断里截面秩相关很低
  (价值偏周期/金融,低波偏大盘稳健),若信息正交,等权合成的超额夏普理论上高于两者均值;
  对立假设是二者的超额都来自同一条市值暴露,合成后不升反降。

申报两档(拉数据前写死):
  T1 等权 0.5/0.5(主口径)
  T2 价值偏置 0.7/0.3(对照:检验增量是否来自低波这一边)
只登记这两档,不再调窗口、不再调滞后——那些参数沿用两张母卡的主口径。
"""
from __future__ import annotations

import pandas as pd

from . import earnings_yield as EY
from . import idio_volatility as IV

W_VALUE = 0.5
W_LOWVOL = 0.5
W_VALUE_TILT = 0.7


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def compose(panel: dict[str, pd.DataFrame], w_value: float, w_lowvol: float) -> pd.DataFrame:
    ep = EY.build_factor(panel)        # +E/P,已滞后 20 个交易日
    lv = IV.build_factor(panel)        # 低波秩(0~1),含闸门
    ep, lv = ep.align(lv, join="inner")
    score = w_value * _cs_rank(ep) + w_lowvol * _cs_rank(lv)
    return score.where(ep.notna() & lv.notna())


def build_factor(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return compose(panel, W_VALUE, W_LOWVOL)


def build_factor_value_tilt(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return compose(panel, W_VALUE_TILT, 1.0 - W_VALUE_TILT)


_COMMON = {
    "task_id": "task06_value_lowvol", "family": "value", "direction": "long_only", "horizon_days": 1,
    "columns": ["peTTM", "adj_close", "tradestatus", "bench_close"],
    "params": {"W_VALUE": W_VALUE, "W_LOWVOL": W_LOWVOL, "W_VALUE_TILT": W_VALUE_TILT,
               "EY_LAG": EY.VALUATION_LAG_DAYS, "IV_WINDOW": IV.WINDOW, "IV_MODE": IV.MODE},
}
SPECS = [
    {**_COMMON, "name": "value_lowvol",
     "operators": ["inverse", "shift", "pct_change", "rolling_std", "market_resid", "cs_rank", "gate", "weighted_sum"],
     "build_factor": build_factor,
     "thesis": ("价值(高 E/P)与低特质波动是两条来源不同的信息,等权秩合成后的超额应高于两者均值;"
                "对立假设是二者超额同源于市值暴露,合成后不升反降。")},
    {**_COMMON, "name": "value_lowvol_tilt70",
     "operators": ["inverse", "shift", "pct_change", "rolling_std", "market_resid", "cs_rank", "gate", "weighted_sum"],
     "build_factor": build_factor_value_tilt,
     "thesis": "对照:价值权重 0.7。若优于等权,说明增量主要来自价值一侧;若劣于等权,增量来自低波一侧。"},
]
