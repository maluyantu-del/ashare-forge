"""分层门槛:零假设校准为主判据,固定阈值为必要条件,PSR/DSR 计多重检验代价。

设计依据(实测):固定数值阈值对零假设几乎不设防——纯随机游走有 82% 概率通过
"半衰期≤15日"这类判据。因此本模块:
  - **主判据** = 零假设校准(null_calib):固定 donor 零分布——每只股票全样本固定取一只
          生命周期相近股票的因子序列(组内循环错位),保留时序与换手结构、摧毁截面对应;
          核心指标(IC 族必须在场,另加收益族)的经验单尾 p 值 (1+#)/(n+1) 均须 ≤ 0.05;
          零样本换手中位数须为真实的 0.8–1.25 倍,否则零分布判无效,主判据按不通过处理
  - 固定阈值 = 合同 gates_draft 的数值门(平静期超额夏普、PSR)、危机期超额夏普 ≥ 0、
          平均持仓 ≥ 20;均为必要条件,零校准缺席时一律不成立
  - 代价项 = DSR ≥ 0.95:n 为全研究域累计试验数 + 本卡申报档数,门槛随 n 收紧
  - 分层 = 平静期 / 危机期(合同 regimes.crisis_windows 的 A 股四段)分开评估,
          危机样本不足时降级为单事件披露,不硬算
  - 披露 = layer='disclosure' 的行(required=False, passed=None):平静期超额夏普的 90% 区间、
          当前 n 下 DSR ≥ 0.95 所需的观测夏普、给定真实夏普时的通过概率、n_trials 构成、
          三态标签 verdict_label。只披露,不进入 overall_pass;在 overall 算定之后才追加。

PSR/DSR 按 Bailey & López de Prado (2014) 自实现;正态分布函数亦自实现,以免引入 scipy。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..contract import load_contract

EULER_GAMMA = 0.5772156649015329
PERIODS_PER_YEAR = 243
MIN_HOLDINGS = 20          # 平均持仓低于此数,组合层结论不可外推


# ---------- 正态分布工具(无 scipy) ----------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Acklam 近似的标准正态分位数(相对误差 ≤1.15e-9)。"""
    if not 0.0 < p < 1.0:
        raise ValueError("p 必须在 (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------- 夏普族 ----------

def _clean(x) -> np.ndarray:
    return pd.Series(x).astype("float64").dropna().to_numpy()


def ann_sharpe(r, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    v = _clean(r)
    if v.size < 2:
        return float("nan")
    s = v.std(ddof=1)
    return float(v.mean() / s * math.sqrt(periods_per_year)) if s > 0 else float("nan")


def psr(r, sr_benchmark_ann: float = 0.0, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """概率夏普比:考虑偏度与峰度后,真实夏普超过基准的置信度。"""
    v = _clean(r)
    if v.size < 30:
        return float("nan")
    s = v.std(ddof=1)
    if s <= 0:
        return float("nan")
    sr = v.mean() / s
    z = (v - v.mean()) / s
    skew, kurt = float((z**3).mean()), float((z**4).mean())
    sr_b = sr_benchmark_ann / math.sqrt(periods_per_year)
    denom = math.sqrt(max(1e-12, 1.0 - skew*sr + (kurt-1.0)/4.0*sr*sr) / (v.size - 1))
    return norm_cdf((sr - sr_b) / denom)


def expected_max_sharpe_ann(n_trials: int, T: int, periods_per_year: int = PERIODS_PER_YEAR,
                            var_sr_trials: float | None = None) -> float:
    """N 次独立尝试下,真实夏普为 0 时纯靠运气可摸到的期望最大年化夏普。"""
    if n_trials <= 1:
        return 0.0
    v = var_sr_trials if var_sr_trials is not None else 1.0 / max(T, 2)
    e = math.sqrt(v) * ((1 - EULER_GAMMA) * norm_ppf(1 - 1.0/n_trials)
                        + EULER_GAMMA * norm_ppf(1 - 1.0/(n_trials*math.e)))
    return float(e * math.sqrt(periods_per_year))


def dsr(r, n_trials: int, periods_per_year: int = PERIODS_PER_YEAR,
        var_sr_trials: float | None = None) -> float:
    """缩水夏普比:以"N 次尝试的运气上限"为基准的 PSR。试得越多,门槛越高。"""
    v = _clean(r)
    if v.size < 30:
        return float("nan")
    bench = expected_max_sharpe_ann(n_trials, v.size, periods_per_year, var_sr_trials)
    return psr(v, bench, periods_per_year)


def max_drawdown(nav: pd.Series) -> float:
    s = pd.Series(nav).astype("float64").dropna()
    return float((s / s.cummax() - 1.0).min()) if len(s) else float("nan")


# ---------- 检验力与区间(只披露,不参与判定) ----------
#
# 下列函数与 psr() 共用同一标准误公式:逐期夏普估计量在偏度 γ3、峰度 γ4(非超额)下的
# 渐近方差 (1 − γ3·SR + (γ4 − 1)/4·SR²)/(T − 1)。psr() 本身未改动;二者逐位一致由
# tests/test_gates_power.py 钉死。

CI_LEVEL = 0.90                       # 平静期超额夏普区间的置信水平
DISCLOSURE_TRUE_SR = (0.8, 1.0)       # 通过概率所假设的真实年化夏普
DSR_MIN = 0.95                        # 与 evaluate 中 DSR 行的阈值一致(仅供披露计算引用)
LABEL_PASS = "PASS"
LABEL_REFUTED = "FAIL-已否证"          # 未通过,且区间上界 < 参照夏普
LABEL_UNDERPOWERED = "FAIL-功效不足"   # 未通过,且区间未排除参照夏普(或区间不可得)
LABEL_UNCLASSIFIED = "FAIL-未分类"     # 未通过,且披露计算失败、无从判断属于上面哪一类


def sample_moments(r) -> tuple[float, float, float, int]:
    """与 psr() 同口径的 (逐期夏普, 偏度, 峰度[非超额], 样本数);样本 <30 或零方差时前三项为 NaN。"""
    v = _clean(r)
    nan = float("nan")
    if v.size < 30:
        return nan, nan, nan, int(v.size)
    s = v.std(ddof=1)
    if s <= 0:
        return nan, nan, nan, int(v.size)
    sr = v.mean() / s
    z = (v - v.mean()) / s
    return float(sr), float((z**3).mean()), float((z**4).mean()), int(v.size)


def sharpe_se(sr: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """逐期夏普估计量的标准误(偏度、峰度修正),与 psr() 分母同式。"""
    return math.sqrt(max(1e-12, 1.0 - skew*sr + (kurt-1.0)/4.0*sr*sr) / (n_obs - 1))


def psr_from_stats(sr: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0,
                   sr_benchmark_ann: float = 0.0,
                   periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """由矩直接算 PSR;传入 sample_moments 的输出时与 psr(r) 逐位一致。"""
    sr_b = sr_benchmark_ann / math.sqrt(periods_per_year)
    return norm_cdf((sr - sr_b) / sharpe_se(sr, n_obs, skew, kurt))


def required_sharpe_ann(prob: float, sr_benchmark_ann: float, n_obs: int,
                        skew: float = 0.0, kurt: float = 3.0,
                        periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """使 PSR(观测夏普; 基准) = prob 的观测年化夏普(二分求根)。

    PSR 在观测夏普上单调递增,故"观测年化夏普 ≥ 返回值"与"PSR ≥ prob"等价(求根容差内)。
    若该样本长度与高阶矩下 PSR 无法达到 prob,返回 inf。
    """
    if n_obs is None or n_obs < 2 or not (0.0 < prob < 1.0):
        return float("nan")
    if not all(math.isfinite(x) for x in (sr_benchmark_ann, skew, kurt)):
        return float("nan")
    k = math.sqrt(periods_per_year)
    b = sr_benchmark_ann / k
    zt = norm_ppf(prob)

    def f(x: float) -> float:
        return (x - b) / sharpe_se(x, n_obs, skew, kurt) - zt

    step = 1.0 / math.sqrt(n_obs)
    lo, hi = b, b + step
    while f(lo) > 0:                   # prob ≤ 0.5 时根在基准之下
        lo -= step
        step *= 2.0
        if lo < b - 50.0:
            return float("nan")
    step = 1.0 / math.sqrt(n_obs)
    while f(hi) < 0:
        hi += step
        step *= 2.0
        if hi > b + 50.0:              # 逐期夏普 50 远超任何现实值:视为不可达
            return float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(1.0, abs(hi)):
            break
    return float(hi * k)


def required_sharpe_dsr_ann(n_trials: int, n_obs: int, prob: float = DSR_MIN,
                            skew: float = 0.0, kurt: float = 3.0,
                            periods_per_year: int = PERIODS_PER_YEAR,
                            var_sr_trials: float | None = None) -> float:
    """当前试验数 n 下 DSR ≥ prob 所需的观测年化夏普;基准取 expected_max_sharpe_ann。"""
    bench = expected_max_sharpe_ann(n_trials, n_obs, periods_per_year, var_sr_trials)
    return required_sharpe_ann(prob, bench, n_obs, skew, kurt, periods_per_year)


def pass_probability(sr_true_ann: float, sr_required_ann: float, n_obs: int,
                     skew: float = 0.0, kurt: float = 3.0,
                     periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """真实年化夏普为 sr_true_ann 时,观测年化夏普 ≥ sr_required_ann 的概率。

    正态近似:观测逐期夏普 ~ N(真实值, sharpe_se(真实值)²);高阶矩按传入值固定。
    """
    if sr_required_ann is None or n_obs is None or n_obs < 2:
        return float("nan")
    if sr_required_ann != sr_required_ann:
        return float("nan")
    if sr_required_ann == float("inf"):
        return 0.0
    k = math.sqrt(periods_per_year)
    t, q = sr_true_ann / k, sr_required_ann / k
    return 1.0 - norm_cdf((q - t) / sharpe_se(t, n_obs, skew, kurt))


def sharpe_interval_ann(sr: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0,
                        level: float = CI_LEVEL,
                        periods_per_year: int = PERIODS_PER_YEAR) -> tuple[float, float, float]:
    """观测夏普的双侧区间 (下界, 上界, 标准误),均为年化;标准误在观测夏普处取值。"""
    nan = float("nan")
    if sr is None or sr != sr or n_obs is None or n_obs < 2:
        return nan, nan, nan
    k = math.sqrt(periods_per_year)
    se = sharpe_se(sr, n_obs, skew, kurt)
    z = norm_ppf(0.5 + level / 2.0)
    return float((sr - z*se) * k), float((sr + z*se) * k), float(se * k)


def verdict_label(overall_pass: bool, ci_hi: float | None, sr_reference: float = 0.8) -> str:
    """三态标签:PASS / FAIL-已否证(区间上界 < 参照夏普)/ FAIL-功效不足(其余未通过情形)。

    "功效不足"仍是未通过;它只说明样本不足以排除参照夏普,不构成任何放行依据。
    """
    if overall_pass:
        return LABEL_PASS
    if ci_hi is not None and ci_hi == ci_hi and ci_hi < sr_reference:
        return LABEL_REFUTED
    return LABEL_UNDERPOWERED


def _format_breakdown(n_trials: int, breakdown: dict | None) -> str:
    if not breakdown:
        return f"未提供构成(total={n_trials})"
    known = [k for k in ("history", "declared") if k in breakdown]
    extra = [k for k in breakdown if k not in ("history", "declared", "total")]
    parts = [f"{k}={breakdown[k]}" for k in known + extra]
    total = breakdown.get("total", n_trials)
    return " + ".join(parts) + f" = total {total}" if parts else f"total={total}"


# ---------- 分层 ----------

def crisis_mask(index: pd.DatetimeIndex, windows: list | None = None) -> pd.Series:
    """按合同 regimes.crisis_windows 给日期打危机标记(A股四段:2015/2018/2020/2024微盘)。"""
    if windows is None:
        windows = load_contract().get("regimes", {}).get("crisis_windows", []) or []
    m = pd.Series(False, index=index)
    for w in windows:
        a, b = (w[0], w[1]) if isinstance(w, (list, tuple)) else (w["start"], w["end"])
        m |= (index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))
    return m


def evaluate(bt: dict, null_verdict: dict | None = None, n_trials: int = 1,
             crisis_min_days: int = 60, n_trials_breakdown: dict | None = None) -> dict:
    """综合裁定。

    bt                 run_backtest 的返回
    null_verdict       null_calib.calibrated_verdict 的返回(主判据;缺省则 overall 一律为 False)
    n_trials           全研究域累计试验数 + 本卡申报档数,计入 DSR
    n_trials_breakdown n_trials 的构成(如 {history, declared, total}),只进披露行

    返回 {overall_pass, verdict_label, rows, notes, n_trials, n_trials_breakdown, null_passed}。
    overall_pass 只由 required=True 且 passed 非 None 的行与零校准决定;layer='disclosure'
    的行 required=False、passed=None,且在 overall 算定之后才追加,不参与判定。
    """
    ret = pd.Series(bt["daily_ret"]).astype("float64")
    excess = pd.Series(bt.get("excess_ret", pd.Series(dtype="float64"))).astype("float64")
    primary = excess if len(excess.dropna()) > 0 else ret      # 主指标:多头超额(无基准则用绝对)
    cm = crisis_mask(pd.DatetimeIndex(primary.index)).reindex(primary.index).fillna(False)

    calm, crisis = primary[~cm], primary[cm]
    rows, notes = [], []

    def add(layer, name, value, threshold=None, op=">=", required=True):
        passed = None
        if value is not None and value == value and threshold is not None:
            passed = bool(value >= threshold) if op == ">=" else bool(value <= threshold)
        rows.append({"layer": layer, "gate": name, "value": value,
                     "threshold": threshold, "passed": passed, "required": required})

    g = load_contract().get("gates_draft", {})
    # 平静层(主战场)
    add("calm", "n_days", int(calm.dropna().size), 120)
    if calm.dropna().size >= 120:
        add("calm", "excess_sharpe", ann_sharpe(calm), g.get("long_excess_sharpe_min"))
        add("calm", "psr", psr(calm), g.get("psr_min"))
        if g.get("use_dsr", True):
            add("calm", f"dsr(n={n_trials})", dsr(calm, n_trials), 0.95)
    # 危机层:样本不足则降级披露,不硬算
    if crisis.dropna().size < crisis_min_days:
        notes.append(f"危机层样本 {crisis.dropna().size} 天 < {crisis_min_days},降级为单事件披露(不计入判定)")
    else:
        add("crisis", "excess_sharpe", ann_sharpe(crisis), 0.0)
        add("crisis", "max_drawdown", max_drawdown((1 + crisis).cumprod()), None, required=False)

    # 主判据:零假设校准
    null_pass = None
    if null_verdict is not None and null_verdict.get("n_metrics", 0) > 0:
        t = null_verdict["table"]
        # 审查发现(P0):IC 类指标为 NaN 时 calibrated_verdict 会静默跳过,core 行数从 4 掉到 2,
        # 于是"只凭两个收益指标"就能拿到 null_passed=True——零技能因子据此攻破过主判据。
        # 现要求 IC 族(选股能力的直接证据)必须在场,缺失即判不通过并留痕。
        ic_core = ["ic_mean", "icir"]
        ret_core = ["excess_sharpe", "sharpe"]
        present = set(t["metric"])
        missing_ic = [m for m in ic_core if m not in present]
        core = t[t["metric"].isin(ic_core + ret_core)]
        if missing_ic:
            null_pass = False
            notes.append(
                f"零校准缺少 IC 族指标 {missing_ic}(截面样本可能不足,rank_ic 返回 NaN):"
                f"无法据收益指标单独判定选股能力,主判据按不通过处理")
        else:
            null_pass = bool(len(core) > 0 and core["passed"].all())
        vld = null_verdict.get("validity") or {}
        if vld.get("valid") is False:
            null_pass = False
            notes.append(vld.get("note") or "零分布构造无效,主判据按不通过处理")
        rows.append({"layer": "null", "gate": f"超越零分布q{int(null_verdict['q']*100)}",
                     "value": f"{int(core['passed'].sum())}/{len(core)}",
                     "threshold": "全部", "passed": null_pass, "required": True})
    else:
        notes.append("未提供零假设校准结果:固定阈值仅作参考,不足以支撑成立结论")

    # 审查发现(P1):稀疏因子持仓仅约 5 只时,夏普与回撤主要反映集中度而非因子本身,不可外推。
    avg_h = (bt.get("stats") or {}).get("avg_holdings")
    if isinstance(avg_h, (int, float)) and avg_h == avg_h and avg_h < MIN_HOLDINGS:
        notes.append(f"平均持仓仅 {avg_h:.1f} 只(<{MIN_HOLDINGS}):组合层结论受集中度主导,不可外推")
        add("portfolio", "avg_holdings", float(avg_h), float(MIN_HOLDINGS))

    evaluated = [r for r in rows if r["passed"] is not None and r["required"]]
    # 审查发现(P1):跳过零校准(--n-null 0)时,仅凭"仅作参考"的固定阈值仍能产出 PASS,
    # 与合同"成立判据=超越零分布"自相矛盾。现要求零校准必须在场且通过,否则一律不成立。
    overall = bool(evaluated) and all(r["passed"] for r in evaluated) and (null_pass is True)
    if null_pass is None:
        notes.append("未提供零假设校准:按合同,固定阈值不足以支撑成立结论,overall 一律为 False")

    # ---- 披露(overall 已定,以下只追加 required=False 的行,不回写 overall) ----
    try:
        disclosure, label = _disclosure(calm, overall, n_trials, n_trials_breakdown, g, notes)
    except Exception as e:  # noqa: BLE001  披露计算失败不得改变判定,也不得让流水线转为 ERROR
        label = LABEL_PASS if overall else LABEL_UNCLASSIFIED
        disclosure = [{"layer": "disclosure", "gate": "verdict_label", "value": label,
                       "threshold": None, "passed": None, "required": False}]
        notes.append(f"披露行计算失败({type(e).__name__}: {e}):仅保留标签,判定不受影响")
    rows.extend(disclosure)
    return {"overall_pass": overall, "verdict_label": label, "rows": rows, "notes": notes,
            "n_trials": n_trials, "n_trials_breakdown": n_trials_breakdown,
            "null_passed": null_pass}


def _disclosure(calm: pd.Series, overall: bool, n_trials: int, breakdown: dict | None,
                g: dict, notes: list) -> tuple[list[dict], str]:
    """披露行与三态标签。高阶矩取平静期样本的观测值(与该样本 PSR/DSR 行同口径)。

    所需夏普:sr_required_dsr95 为当前 n 下 DSR ≥ 0.95 的观测年化夏普;
    sr_required_calm_binding 为平静期三道夏普族门(超额夏普、PSR、DSR)所需值的最大者,
    通过概率即"观测夏普越过该值"的概率——不含零校准、危机层与持仓门,是 overall 通过概率的上界。
    """
    sr, skew, kurt, t_obs = sample_moments(calm)
    ref = g.get("long_excess_sharpe_min")
    ref = float(ref) if ref is not None else 0.8
    lo, hi, se = sharpe_interval_ann(sr, t_obs, skew, kurt, CI_LEVEL)
    nan = float("nan")
    ok = sr == sr
    req_dsr = required_sharpe_dsr_ann(n_trials, t_obs, DSR_MIN, skew, kurt) if ok else nan
    reqs = []
    if g.get("long_excess_sharpe_min") is not None:
        reqs.append(float(g["long_excess_sharpe_min"]))
    if ok and g.get("psr_min") is not None:
        reqs.append(required_sharpe_ann(float(g["psr_min"]), 0.0, t_obs, skew, kurt))
    if ok and g.get("use_dsr", True):
        reqs.append(req_dsr)
    binding = max(reqs) if ok and reqs else nan

    def row(name, value):
        return {"layer": "disclosure", "gate": name, "value": value,
                "threshold": None, "passed": None, "required": False}

    pct = int(round(CI_LEVEL * 100))
    out = [row("excess_sharpe_se_ann", se),
           row(f"excess_sharpe_ci{pct}_lo", lo),
           row(f"excess_sharpe_ci{pct}_hi", hi),
           row(f"sr_required_dsr95(n={n_trials})", req_dsr),
           row("sr_required_calm_binding", binding)]
    for s_true in DISCLOSURE_TRUE_SR:
        p = pass_probability(s_true, binding, t_obs, skew, kurt) if ok else nan
        out.append(row(f"p_pass_calm|true_sr={s_true:.1f}", p))
    out.append(row("n_trials_breakdown", _format_breakdown(n_trials, breakdown)))
    if breakdown and breakdown.get("total") is not None and breakdown.get("total") != n_trials:
        notes.append(f"n_trials 构成合计 {breakdown.get('total')} 与计入 DSR 的 n_trials={n_trials} 不一致")

    label = verdict_label(overall, hi, ref)
    out.append(row("verdict_label", label))
    if not ok:
        notes.append(f"平静期样本 {t_obs} 天(<30 或零方差):夏普区间不可得,未通过时标为{LABEL_UNDERPOWERED}")
    elif label == LABEL_UNDERPOWERED and lo > ref:
        notes.append(f"平静期超额夏普 {pct}% 区间整体高于 {ref}:未通过源于其他门"
                     f"(DSR/零校准/危机层/持仓),而非夏普本身被否证")
    return out, label


def to_tsv(verdict: dict) -> str:
    lines = ["layer\tgate\tvalue\tthreshold\tpassed"]
    for r in verdict["rows"]:
        v = r["value"]
        v = f"{v:.4f}" if isinstance(v, float) else str(v)
        if r.get("layer") == "disclosure":
            lines.append(f'{r["layer"]}\t{r["gate"]}\t{v}\t\t(披露,不参与判定)')
            continue
        lines.append(f'{r["layer"]}\t{r["gate"]}\t{v}\t{r["threshold"]}\t{r["passed"]}')
    for n in verdict["notes"]:
        lines.append(f"note\t\t{n}\t\t")
    lines.append(f'overall\t\t\t\t{verdict["overall_pass"]}')
    if verdict.get("verdict_label"):
        lines.append(f'verdict_label\t\t\t\t{verdict["verdict_label"]}')
    return "\n".join(lines)
