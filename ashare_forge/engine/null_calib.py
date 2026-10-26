"""零假设校准:成立判据不是"大于某个拍脑袋的数",而是"显著优于同换手的随机因子"。

现行构造与判据(第三版,a15a404 起):
    - 零因子:每只股票全样本固定拿一个"生命周期相近"的供值股票(donor)的因子序列——
      按首/末有效日排序后分组,组内循环错位指派、无自映射(见 shuffle_factor);
    - 判据:逐指标经验单尾 p 值 (1 + #{零样本不劣于真实值}) / (n + 1) ≤ 0.05
      (Phipson & Smyth 2010,有限次置换的 p 值不取 0);
    - 有效性闸门:零样本日均换手中位数 / 真实换手须落在 [0.8, 1.25],否则零分布判无效,
      主判据按不通过处理(见 null_validity)。
    前两代实现均被实测证伪,保留作对照:"逐日截面打乱"(mode="daily",零样本换手远高于
    真实因子,比较的是手续费)与"整列置换 + 逐行重装"(mode="fixed_repack_legacy",
    真实数据中缺失布局几乎每天变化,退化为近似逐日打乱)。

为什么不用固定阈值(A股方向评审反方的实测结论):
    对"剩余天数"去趋势后估 AR(1) 半衰期,**纯随机游走有 82% 概率通过"半衰期≤15日"
    的固定判据**(Nelson-Kang 伪回归)。固定阈值对零假设几乎不设防,写进台账也只是
    把曲线拟合写死。

零因子保留与摧毁的信息:
    - 保留:每条序列的时序路径与持续性(因而换手与真实因子同量级)、可交易股票集合、
      成本与调仓规则;每日的取值分布近似保留
    - 摧毁:"哪只股票拿哪个值"的截面信息
    因此零分布刻画的正是"这条流水线在没有选股能力时能凭运气跑出什么",
    样本长度、股票数、换手成本造成的偏差被自动校准——固定阈值做不到这点。
    donor 生命周期相近,零因子会残留少量真实信息,判据因此略偏保守。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 数值越大越好的指标(比较时取右尾);其余指标按绝对值比较
HIGHER_IS_BETTER = ("ic_mean", "icir", "icir_ann", "t_stat", "monotonicity",
                    "ann_return", "sharpe", "excess_ann", "excess_sharpe", "positive_ratio")


GROUP_SIZE = 20          # 生命周期分组的最小组大小(组内派供值股票)
TURNOVER_BAND = (0.8, 1.25)   # 零样本换手中位数 / 真实换手 须落在此区间,否则零分布判无效


def _lifecycle_donors(valid: np.ndarray, rng: np.random.Generator,
                      group_size: int = GROUP_SIZE) -> np.ndarray:
    """为每只股票指定一个固定的供值股票(donor),全样本不变。

    按"首个有效日、末个有效日"排序后切成 ≥group_size 的连续组,组内做循环错位置换
    (derangement:没有股票拿到自己的值,否则零因子会保留真信号)。生命周期相近的股票
    缺失模式相近,因此绝大多数单元格的 donor 值与本股同时有效,不需要逐行重排。
    无任何有效值的股票 donor 指向自己(全 NaN,无影响)。
    """
    n_rows, n_cols = valid.shape
    has = valid.any(axis=0)
    first = np.where(has, valid.argmax(axis=0), n_rows)
    last = np.where(has, n_rows - 1 - valid[::-1].argmax(axis=0), -1)
    order = np.lexsort((last, first))
    live = [int(j) for j in order if has[j]]
    donors = np.arange(n_cols)
    if len(live) < 2:
        return donors
    n_groups = max(1, len(live) // group_size)
    bounds = np.linspace(0, len(live), n_groups + 1).astype(int)      # 组大小 ≥ group_size(末组吸收余数)
    for a, b in zip(bounds[:-1], bounds[1:]):
        grp = np.array(live[a:b])
        if grp.size < 2:
            continue
        cyc = rng.permutation(grp)
        donors[cyc] = np.roll(cyc, -1)                                 # cyc[k] 取 cyc[k+1] 的值
    return donors


def shuffle_factor(factor: pd.DataFrame, seed: int, mode: str = "fixed") -> pd.DataFrame:
    """构造零假设因子。

    mode="fixed"(默认,**正确的零因子**):每只股票全样本固定拿一个 donor 的因子序列,
        donor 在"生命周期相近"的组内循环错位指派(见 _lifecycle_donors)。
        摧毁:"哪只股票拿哪个值"(选股能力)。保留:每条序列的时序路径与持续性 → 换手与真因子一致。
        本股当日无值则 NaN;donor 当日无值也记 NaN(不重新装填,见下)。
    mode="fixed_repack_legacy"(v1 实现,**有偏,仅作对照**):整列置换后逐行"恢复缺失模式",
        只要某行缺失布局变了,该行就按顺序重新装填 → 股票与 donor 的对应几乎每天滑动。
        真实数据中上市/退市/停牌使几乎每行都触发重装,零因子退化为近似逐日打乱:
        对抗审查实测零分布年化收益中位数约 −30%,反推零样本日换手为真实因子的 5~20 倍,
        即又在"比手续费"(与 BATCH01 撤回同型)。合成测试面板无缺失值,因而未能发现。
    mode="daily"(更早实现,**有偏,仅作对照**):逐日独立打乱,摧毁时序路径。
    """
    rng = np.random.default_rng(seed)
    arr = factor.to_numpy(dtype="float64", copy=True)
    n_cols = arr.shape[1]

    if mode == "fixed":
        valid = ~np.isnan(arr)
        donors = _lifecycle_donors(valid, rng)
        out = arr[:, donors]
        out[~valid] = np.nan                  # 本股当日无值 → 无值(与真因子每日可选集一致或更小)
        return pd.DataFrame(out, index=factor.index, columns=factor.columns)

    if mode == "fixed_repack_legacy":
        perm = rng.permutation(n_cols)
        out = arr[:, perm]
        src_nan, dst_nan = np.isnan(out), np.isnan(arr)
        for i in range(arr.shape[0]):
            need = np.flatnonzero(dst_nan[i] != src_nan[i])
            if need.size:
                vals = out[i][~src_nan[i]]
                row = np.full(n_cols, np.nan)
                slots = np.flatnonzero(~dst_nan[i])
                if vals.size and slots.size:
                    take = vals[: slots.size] if vals.size >= slots.size else \
                        np.resize(vals, slots.size)
                    row[slots] = take
                out[i] = row
        return pd.DataFrame(out, index=factor.index, columns=factor.columns)

    if mode != "daily":
        raise ValueError(f"mode 只能是 'fixed'/'fixed_repack_legacy'/'daily',收到 {mode!r}")
    for i in range(arr.shape[0]):
        row = arr[i]
        idx = np.flatnonzero(~np.isnan(row))
        if idx.size > 1:
            row[idx] = row[rng.permutation(idx)]
    return pd.DataFrame(arr, index=factor.index, columns=factor.columns)


def null_distribution(factor: pd.DataFrame, run_fn, n: int = 200, seed: int = 7,
                      keys: tuple[str, ...] | None = None, mode: str = "fixed") -> dict:
    """把 n 份打乱后的因子穿过同一条流水线,收集指标的零分布。

    run_fn(factor) -> stats dict(通常是 backtest.run_backtest(...)["stats"] 的闭包)。
    """
    rows = []
    for i in range(n):
        s = run_fn(shuffle_factor(factor, seed + i, mode=mode))
        rows.append(s)
    df = pd.DataFrame(rows)
    # 构造有效性诊断(不参与裁定):零样本的换手与持仓数必须与真因子同量级,否则比的是手续费
    diag = df[[k for k in ("avg_turnover", "avg_holdings") if k in df.columns]].copy()
    if keys:
        df = df[[k for k in keys if k in df.columns]]
    out = {"n": int(n), "mode": mode, "samples": df, "diag": diag}
    for k in df.columns:
        v = df[k].dropna().to_numpy(dtype="float64")
        if v.size == 0:
            continue
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else np.nan,
                  "q50": float(np.percentile(v, 50)), "q95": float(np.percentile(v, 95)),
                  "q99": float(np.percentile(v, 99)), "n_valid": int(v.size)}
    return out


def calibrated_verdict(real: dict, null: dict, q: float = 0.95) -> dict:
    """逐指标与零分布比较。p_value = 零样本中不劣于真实值的比例(经验单尾 p)。"""
    samples: pd.DataFrame = null.get("samples", pd.DataFrame())
    rows = []
    for k, rv in real.items():
        if k not in samples.columns or not isinstance(rv, (int, float)) or rv != rv:
            continue
        v = samples[k].dropna().to_numpy(dtype="float64")
        if v.size == 0:
            continue
        higher = k in HIGHER_IS_BETTER
        rr, vv = (rv, v) if higher else (abs(rv), np.abs(v))
        thr = float(np.percentile(vv, q * 100))
        # Phipson & Smyth (2010):有限次置换的 p 值不可能为 0,取 (1+#{零≥真})/(n+1)。
        # 判据改为 p ≤ 1−q;v1 的"真值 > 插值 q95"在 n=50 时单指标假阳性约 6.8% 而非 5%。
        p = float((1 + int((vv >= rr).sum())) / (vv.size + 1))
        rows.append({"metric": k, "real": float(rv), f"null_q{int(q*100)}": thr,
                     "null_median": float(np.percentile(vv, 50)),
                     "passed": bool(p <= 1.0 - q + 1e-12), "p_value": p,
                     "direction": "higher" if higher else "abs"})
    tbl = pd.DataFrame(rows)
    return {"table": tbl, "q": q,
            "n_passed": int(tbl["passed"].sum()) if not tbl.empty else 0,
            "n_metrics": int(len(tbl))}


def null_validity(real: dict, null: dict, band: tuple[float, float] = TURNOVER_BAND) -> dict:
    """零分布构造有效性:零样本日均换手中位数 / 真实换手 须在 band 内。

    不在区间内 = 零因子与真因子的交易成本不可比,收益类指标的"超越零分布"不可信,
    主判据按不通过处理(宁可错杀)。
    """
    diag: pd.DataFrame = null.get("diag", pd.DataFrame())
    rt = real.get("avg_turnover")
    if diag is None or diag.empty or "avg_turnover" not in diag or not rt or rt != rt or rt <= 0:
        return {"valid": None, "turnover_ratio": None, "note": "缺换手诊断,未做有效性检查"}
    med = float(np.nanmedian(diag["avg_turnover"].to_numpy(dtype="float64")))
    ratio = med / float(rt)
    ok = band[0] <= ratio <= band[1]
    return {"valid": bool(ok), "turnover_ratio": ratio, "null_turnover_median": med,
            "real_turnover": float(rt), "band": list(band),
            "note": "" if ok else (f"零样本换手中位数为真实的 {ratio:.2f} 倍(要求 {band[0]}~{band[1]}):"
                                   f"零分布与真因子成本不可比,主判据按不通过处理")}


def to_tsv(verdict: dict) -> str:
    """可粘进 Excel 的裁定表。"""
    t: pd.DataFrame = verdict["table"]
    if t.empty:
        return "metric\treal\tnull_q95\tpassed\tp_value\n(无可比指标)"
    cols = [c for c in ("metric", "real", f"null_q{int(verdict['q']*100)}",
                        "null_median", "passed", "p_value") if c in t.columns]
    return t[cols].to_csv(sep="\t", index=False, float_format="%.4f")
