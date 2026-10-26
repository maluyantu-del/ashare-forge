"""任务卡01:横盘吸筹 → 放量突破。

把 AB-Screener 的三层筛选改写成**连续打分**的横截面因子(截面排序需要连续值,
而非"入选/不入选"的二值清单)。每一项都只用当日及以前的数据。

与参考实现的实质差异(必须披露):
  - 主力资金净流入不可得(东财源本机被拦),以"量能放大倍数 × 换手"代替。
  - 参考实现自述其股票池有幸存者偏差且"已有研究 FAIL",故不采信其结论,独立重验。

打分构成(均为当日可得):
  1. 横盘紧致度 = 1 / (箱体宽度 / 箱体中枢),越窄分越高
  2. 缩量程度   = 箱体期均量 / 更长期均量,越小分越高(吸筹)
  3. 突破幅度   = (收盘 - 箱顶) / 箱顶,只在为正时计分
  4. 放量倍数   = 当日量 / 箱体期均量,越大分越高(封顶避免异常值主导)
总分 = 各项截面秩的加权平均(用秩而非原值,避免量纲与极值主导)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BOX_WIN = 60        # 横盘观察窗(约3个月)
LONG_WIN = 120      # 缩量对比的长窗
VOL_CAP = 5.0       # 放量倍数封顶
WEIGHTS = {"tight": 0.25, "quiet": 0.25, "breakout": 0.30, "surge": 0.20}


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """逐日截面百分位秩(0~1)。只用当日数据,不构成前视。"""
    return df.rank(axis=1, pct=True)


def build_factor(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # 审查发现(P0):原实现箱体用**原始** high/low、突破价用**后复权** adj_close,
    # 两套基准混算。对有复权因子的股票 adj_close 可达原始价十余倍,breakout 被放大三个量级,
    # 实测 55% 的组合权重被灌给当时仅有的 30 只有复权因子的股票。
    # 修法:把 high/low 乘上同一复权比例,全部统一到后复权基准。
    close = panel["adj_close"].astype("float64")
    ratio = (close / panel["close"].astype("float64").replace(0.0, np.nan)).ffill().fillna(1.0)
    high = panel["high"].astype("float64") * ratio
    low = panel["low"].astype("float64") * ratio
    vol = panel["volume"].astype("float64")

    # 箱体:不含当日(shift(1)),否则"箱顶"会包含当日突破价,构成同日前视
    box_hi = high.shift(1).rolling(BOX_WIN, min_periods=BOX_WIN).max()
    box_lo = low.shift(1).rolling(BOX_WIN, min_periods=BOX_WIN).min()
    box_mid = (box_hi + box_lo) / 2.0

    width = (box_hi - box_lo) / box_mid.replace(0.0, np.nan)
    tight = 1.0 / width.replace(0.0, np.nan)                    # 越窄越高

    v_box = vol.shift(1).rolling(BOX_WIN, min_periods=BOX_WIN).mean()
    v_long = vol.shift(1).rolling(LONG_WIN, min_periods=LONG_WIN).mean()
    quiet = -(v_box / v_long.replace(0.0, np.nan))              # 缩量→取负使"越缩越高"

    breakout = ((close - box_hi) / box_hi.replace(0.0, np.nan)).clip(lower=0.0)
    surge = (vol / v_box.replace(0.0, np.nan)).clip(upper=VOL_CAP)

    parts = {"tight": tight, "quiet": quiet, "breakout": breakout, "surge": surge}
    score = None
    for k, w in WEIGHTS.items():
        r = _cs_rank(parts[k]) * w
        score = r if score is None else score.add(r, fill_value=np.nan)

    # 未发生突破的标的不参与排序(该形态的前提就是"已突破")
    return score.where(breakout > 0)
