"""截面因子指标。

它修的坑:
1. **IC 必须逐日在截面上算**,不是把整张面板拉平算一个相关系数(后者混入时序变动,
   数值好看但与"选股能力"无关)。
2. **前瞻收益要用后复权价**,且 t 期因子对应 t→t+1 的收益;用原始收盘价会在除权日
   凭空产生跳空亏损。
3. 每日截面样本数不足时该日不计 IC(而非按 NaN 混入平均)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_CROSS_SECTION = 20          # 单日可用股票少于此数则该日不计 IC


def _masked(factor: pd.DataFrame, fwd: pd.DataFrame,
            mask: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = factor.astype("float64")
    r = fwd.reindex(index=f.index, columns=f.columns).astype("float64")
    if mask is not None:
        m = mask.reindex(index=f.index, columns=f.columns).fillna(False).astype(bool)
        f, r = f.where(m), r.where(m)
    return f, r


def rank_ic(factor: pd.DataFrame, fwd_ret: pd.DataFrame,
            mask: pd.DataFrame | None = None) -> pd.Series:
    """逐日截面 Spearman 秩相关(因子 vs 前瞻收益)。样本不足的日子返回 NaN。"""
    f, r = _masked(factor, fwd_ret, mask)
    out = {}
    for d in f.index:
        a, b = f.loc[d], r.loc[d]
        ok = a.notna() & b.notna()
        if int(ok.sum()) < MIN_CROSS_SECTION:
            out[d] = np.nan
            continue
        out[d] = a[ok].rank().corr(b[ok].rank())
    s = pd.Series(out, name="rank_ic")
    s.index.name = "date"
    return s


def ic_stats(ic: pd.Series, periods_per_year: int = 243) -> dict:
    """IC 序列统计。ICIR = mean/std;t 值用有效天数。"""
    v = pd.Series(ic).dropna()
    n = int(v.size)
    if n < 2:
        return {"n_days": n, "ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
                "t_stat": np.nan, "positive_ratio": np.nan, "icir_ann": np.nan}
    mean, std = float(v.mean()), float(v.std(ddof=1))
    icir = mean / std if std > 0 else np.nan
    return {
        "n_days": n,
        "ic_mean": mean,
        "ic_std": std,
        "icir": icir,
        "icir_ann": icir * np.sqrt(periods_per_year) if std > 0 else np.nan,
        "t_stat": mean / (std / np.sqrt(n)) if std > 0 else np.nan,
        "positive_ratio": float((v > 0).mean()),
    }


def decile_returns(factor: pd.DataFrame, fwd_ret: pd.DataFrame,
                   mask: pd.DataFrame | None = None, n_groups: int = 10) -> pd.DataFrame:
    """十分组日收益(等权)。列 = 组号 0..n-1(0 = 因子最小)。"""
    f, r = _masked(factor, fwd_ret, mask)
    rows = {}
    for d in f.index:
        a, b = f.loc[d], r.loc[d]
        ok = a.notna() & b.notna()
        if int(ok.sum()) < max(MIN_CROSS_SECTION, n_groups * 2):
            continue
        q = pd.qcut(a[ok].rank(method="first"), n_groups, labels=False, duplicates="drop")
        rows[d] = b[ok].groupby(q, observed=True).mean()
    out = pd.DataFrame(rows).T.sort_index()
    out.index.name = "date"
    out.columns.name = "decile"
    return out


def monotonicity(decile_ret: pd.DataFrame) -> float:
    """组号与组平均收益的 Spearman:+1 = 完全单调递增,-1 = 完全递减,0 = 无序。"""
    if decile_ret.empty:
        return np.nan
    m = decile_ret.mean(axis=0)
    if m.size < 3:
        return np.nan
    # Spearman = 先取秩再算 Pearson,自己实现以免引入 scipy 依赖(公开仓库依赖越少越好)
    g = pd.Series(np.asarray(m.index, dtype="float64")).rank()
    v = pd.Series(m.to_numpy(dtype="float64")).rank()
    return float(g.corr(v))


def turnover(weights: pd.DataFrame) -> pd.Series:
    """逐期单边换手 = 0.5 * Σ|w_t - w_{t-1}|(首期按全额建仓计)。"""
    w = weights.fillna(0.0).astype("float64")
    d = w.diff()
    d.iloc[0] = w.iloc[0]
    s = 0.5 * d.abs().sum(axis=1)
    s.name = "turnover"
    return s
