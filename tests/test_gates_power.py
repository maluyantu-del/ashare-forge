"""门槛检验力披露的已知答案测试。

必须钉死的四条:
  ① 披露行(layer='disclosure')存在,required=False、passed=None,且不进入 overall 计算;
  ② 在一组合成回测上,overall_pass / null_passed / 非披露行与改动前的旧实现逐项一致
     (旧实现按提交 2ad836d 原样冻结在本文件,不依赖 git 历史);
  ③ DSR ≥ 0.95 所需的观测夏普随试验数 n 单调上升,且与闭式解、与 gates.dsr 回代一致;
  ④ 三态标签 PASS / FAIL-已否证 / FAIL-功效不足 在构造的边界情形上正确。
"""
from __future__ import annotations

import itertools
import math
import os

import numpy as np
import pandas as pd
import pytest

from ashare_forge.contract import load_contract
from ashare_forge.engine import gates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PPY = gates.PERIODS_PER_YEAR
Z90 = gates.norm_ppf(0.95)


# ======================================================================
# 旧实现冻结副本(提交 2ad836d 的 ann_sharpe / psr / expected_max_sharpe_ann / dsr /
# max_drawdown / evaluate,逐字照录;正态函数与 crisis_mask 未改动,直接引用)
# ======================================================================

def _legacy_clean(x) -> np.ndarray:
    return pd.Series(x).astype("float64").dropna().to_numpy()


def _legacy_ann_sharpe(r, periods_per_year: int = 243) -> float:
    v = _legacy_clean(r)
    if v.size < 2:
        return float("nan")
    s = v.std(ddof=1)
    return float(v.mean() / s * math.sqrt(periods_per_year)) if s > 0 else float("nan")


def _legacy_psr(r, sr_benchmark_ann: float = 0.0, periods_per_year: int = 243) -> float:
    v = _legacy_clean(r)
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
    return gates.norm_cdf((sr - sr_b) / denom)


def _legacy_expected_max_sharpe_ann(n_trials: int, T: int, periods_per_year: int = 243,
                                    var_sr_trials: float | None = None) -> float:
    if n_trials <= 1:
        return 0.0
    v = var_sr_trials if var_sr_trials is not None else 1.0 / max(T, 2)
    e = math.sqrt(v) * ((1 - gates.EULER_GAMMA) * gates.norm_ppf(1 - 1.0/n_trials)
                        + gates.EULER_GAMMA * gates.norm_ppf(1 - 1.0/(n_trials*math.e)))
    return float(e * math.sqrt(periods_per_year))


def _legacy_dsr(r, n_trials: int, periods_per_year: int = 243,
                var_sr_trials: float | None = None) -> float:
    v = _legacy_clean(r)
    if v.size < 30:
        return float("nan")
    bench = _legacy_expected_max_sharpe_ann(n_trials, v.size, periods_per_year, var_sr_trials)
    return _legacy_psr(v, bench, periods_per_year)


def _legacy_max_drawdown(nav: pd.Series) -> float:
    s = pd.Series(nav).astype("float64").dropna()
    return float((s / s.cummax() - 1.0).min()) if len(s) else float("nan")


def _legacy_evaluate(bt: dict, null_verdict: dict | None = None, n_trials: int = 1,
                     crisis_min_days: int = 60) -> dict:
    ret = pd.Series(bt["daily_ret"]).astype("float64")
    excess = pd.Series(bt.get("excess_ret", pd.Series(dtype="float64"))).astype("float64")
    primary = excess if len(excess.dropna()) > 0 else ret
    cm = gates.crisis_mask(pd.DatetimeIndex(primary.index)).reindex(primary.index).fillna(False)

    calm, crisis = primary[~cm], primary[cm]
    rows, notes = [], []

    def add(layer, name, value, threshold=None, op=">=", required=True):
        passed = None
        if value is not None and value == value and threshold is not None:
            passed = bool(value >= threshold) if op == ">=" else bool(value <= threshold)
        rows.append({"layer": layer, "gate": name, "value": value,
                     "threshold": threshold, "passed": passed, "required": required})

    g = load_contract().get("gates_draft", {})
    add("calm", "n_days", int(calm.dropna().size), 120)
    if calm.dropna().size >= 120:
        add("calm", "excess_sharpe", _legacy_ann_sharpe(calm), g.get("long_excess_sharpe_min"))
        add("calm", "psr", _legacy_psr(calm), g.get("psr_min"))
        if g.get("use_dsr", True):
            add("calm", f"dsr(n={n_trials})", _legacy_dsr(calm, n_trials), 0.95)
    if crisis.dropna().size < crisis_min_days:
        notes.append(f"危机层样本 {crisis.dropna().size} 天 < {crisis_min_days},降级为单事件披露(不计入判定)")
    else:
        add("crisis", "excess_sharpe", _legacy_ann_sharpe(crisis), 0.0)
        add("crisis", "max_drawdown", _legacy_max_drawdown((1 + crisis).cumprod()), None, required=False)

    null_pass = None
    if null_verdict is not None and null_verdict.get("n_metrics", 0) > 0:
        t = null_verdict["table"]
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

    avg_h = (bt.get("stats") or {}).get("avg_holdings")
    if isinstance(avg_h, (int, float)) and avg_h == avg_h and avg_h < 20:
        notes.append(f"平均持仓仅 {avg_h:.1f} 只(<20):组合层结论受集中度主导,不可外推")
        add("portfolio", "avg_holdings", float(avg_h), float(20))

    evaluated = [r for r in rows if r["passed"] is not None and r["required"]]
    overall = bool(evaluated) and all(r["passed"] for r in evaluated) and (null_pass is True)
    if null_pass is None:
        notes.append("未提供零假设校准:按合同,固定阈值不足以支撑成立结论,overall 一律为 False")
    return {"overall_pass": overall, "rows": rows, "notes": notes,
            "n_trials": n_trials, "null_passed": null_pass}


def _legacy_overall_from_rows(rows: list[dict], null_pass) -> bool:
    """旧 overall 规则,作用在给定行上。"""
    evaluated = [r for r in rows if r["passed"] is not None and r["required"]]
    return bool(evaluated) and all(r["passed"] for r in evaluated) and (null_pass is True)


# ======================================================================
# 合成回测与零校准结果
# ======================================================================

def _series(mu: float, sd: float, n: int, start: str, seed: int, t_df: int | None = None) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = rng.standard_t(t_df, n) / math.sqrt(t_df / (t_df - 2)) if t_df else rng.standard_normal(n)
    return pd.Series(mu + sd * x, index=pd.bdate_range(start, periods=n))


def _bt(r: pd.Series, *, excess: pd.Series | None = None, avg_holdings=50.0) -> dict:
    bt = {"daily_ret": r, "nav": (1 + r).cumprod()}
    if excess is not None:
        bt["excess_ret"] = excess
    if avg_holdings is not None:
        bt["stats"] = {"avg_holdings": avg_holdings}
    return bt


def _null(kind: str) -> dict | None:
    all_m = ("ic_mean", "icir", "sharpe", "excess_sharpe")
    if kind == "none":
        return None
    if kind == "empty":
        return {"table": pd.DataFrame(columns=["metric", "passed"]), "q": 0.95,
                "n_metrics": 0, "n_passed": 0}
    if kind == "ic_missing":
        tbl = pd.DataFrame([{"metric": m, "passed": True} for m in ("sharpe", "excess_sharpe")])
        return {"table": tbl, "q": 0.95, "n_metrics": 2, "n_passed": 2}
    passed = {m: True for m in all_m}
    validity = {"valid": True}
    if kind == "one_fail":
        passed["icir"] = False
    if kind == "invalid":
        validity = {"valid": False, "note": "零样本换手与真实换手不同量级"}
    tbl = pd.DataFrame([{"metric": m, "passed": passed[m]} for m in all_m])
    return {"table": tbl, "q": 0.95, "n_metrics": 4, "n_passed": int(sum(passed.values())),
            "validity": validity}


def _bt_cases() -> dict[str, dict]:
    strong = _series(0.0020, 0.01, 1500, "2016-01-04", 1)                 # 年化夏普约 3,跨 2018 危机
    near = _series(0.00055, 0.01, 1500, "2016-01-04", 18)                 # 真实约 0.86,平静期观测 0.82
    weak = _series(-0.0003, 0.01, 2351, "2015-01-05", 3)                  # 全样本长度,负技能
    fat = _series(0.0012, 0.01, 1500, "2016-01-04", 4, t_df=4)            # 厚尾
    short = _series(0.002, 0.01, 100, "2021-01-04", 5)                    # 平静期 < 120
    no_crisis = _series(0.0015, 0.01, 600, "2021-01-04", 6)               # 无危机日,危机层降级
    bench = _series(0.0003, 0.012, 1500, "2016-01-04", 7)
    return {
        "strong": _bt(strong),
        "strong_excess": _bt(strong, excess=strong - bench),
        "near": _bt(near),
        "weak": _bt(weak),
        "fat": _bt(fat),
        "short": _bt(short),
        "no_crisis": _bt(no_crisis),
        "thin_holdings": _bt(strong, avg_holdings=7.7),
        "no_stats": _bt(strong, avg_holdings=None),
    }


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a == b


def _core_rows(v: dict) -> list[dict]:
    return [r for r in v["rows"] if r["layer"] != "disclosure"]


def _disc(v: dict) -> dict[str, dict]:
    return {r["gate"]: r for r in v["rows"] if r["layer"] == "disclosure"}


# ======================================================================
# ① ② 披露行存在、不参与判定;与旧实现逐项一致
# ======================================================================

def test_overall_pass_identical_to_legacy_on_synthetic_grid():
    cases = _bt_cases()
    nulls = ("none", "empty", "ic_missing", "all_pass", "one_fail", "invalid")
    outcomes = set()
    for (cname, bt), nk, n in itertools.product(cases.items(), nulls, (1, 2, 24, 38, 200)):
        old = _legacy_evaluate(bt, _null(nk), n_trials=n)
        new = gates.evaluate(bt, _null(nk), n_trials=n,
                             n_trials_breakdown={"history": n - 1, "declared": 1, "total": n})
        tag = f"{cname}/{nk}/n={n}"
        assert new["overall_pass"] == old["overall_pass"], tag
        assert new["null_passed"] == old["null_passed"], tag
        core = _core_rows(new)
        assert len(core) == len(old["rows"]), tag
        for a, b in zip(core, old["rows"]):
            assert a.keys() == b.keys(), tag
            assert all(_same(a[k], b[k]) for k in a), f"{tag}: {a} != {b}"
        assert new["notes"][:len(old["notes"])] == old["notes"], tag
        # 旧 overall 规则作用在"去掉披露行"与"含披露行"两份行上,结论相同
        assert _legacy_overall_from_rows(core, new["null_passed"]) == new["overall_pass"], tag
        assert _legacy_overall_from_rows(new["rows"], new["null_passed"]) == new["overall_pass"], tag
        outcomes.add(new["overall_pass"])
    assert outcomes == {True, False}, "网格未同时覆盖通过与未通过,比对失去意义"


def test_disclosure_rows_present_not_required_and_after_core_rows():
    for cname, bt in _bt_cases().items():
        v = gates.evaluate(bt, _null("all_pass"), n_trials=30)
        d = _disc(v)
        expected = {"excess_sharpe_se_ann", "excess_sharpe_ci90_lo", "excess_sharpe_ci90_hi",
                    "sr_required_dsr95(n=30)", "sr_required_calm_binding",
                    "p_pass_calm|true_sr=0.8", "p_pass_calm|true_sr=1.0",
                    "n_trials_breakdown", "verdict_label"}
        assert set(d) == expected, cname
        for r in d.values():
            assert r["required"] is False and r["passed"] is None and r["threshold"] is None, (cname, r)
            # pipeline 按 gate 前缀 "dsr" 取 DSR 值写台账:披露行不得冒名
            assert not str(r["gate"]).startswith("dsr"), r["gate"]
        layers = [r["layer"] for r in v["rows"]]
        first = layers.index("disclosure")
        assert all(x == "disclosure" for x in layers[first:]), f"{cname}: 披露行须位于判定行之后"
        assert d["verdict_label"]["value"] == v["verdict_label"]


def test_dsr_ledger_value_not_hijacked_when_calm_layer_absent():
    """平静期不足 120 天时没有 DSR 行;pipeline 的前缀取值必须得到 None,而非披露行的数。"""
    v = gates.evaluate(_bt_cases()["short"], _null("all_pass"), n_trials=30)
    dsr_value = next((r["value"] for r in v["rows"] if str(r["gate"]).startswith("dsr")), None)
    assert dsr_value is None
    assert v["overall_pass"] is False


def test_disclosure_values_follow_psr_standard_error():
    bt = _bt_cases()["near"]
    v = gates.evaluate(bt, _null("all_pass"), n_trials=24)
    r = pd.Series(bt["daily_ret"])
    cm = gates.crisis_mask(pd.DatetimeIndex(r.index))
    calm = r[~cm.to_numpy()]
    sr, skew, kurt, t_obs = gates.sample_moments(calm)
    d = _disc(v)
    se_ann = gates.sharpe_se(sr, t_obs, skew, kurt) * math.sqrt(PPY)
    center = next(x["value"] for x in v["rows"] if x["layer"] == "calm" and x["gate"] == "excess_sharpe")
    assert d["excess_sharpe_se_ann"]["value"] == pytest.approx(se_ann, rel=1e-12)
    assert d["excess_sharpe_ci90_lo"]["value"] == pytest.approx(center - Z90 * se_ann, rel=1e-12)
    assert d["excess_sharpe_ci90_hi"]["value"] == pytest.approx(center + Z90 * se_ann, rel=1e-12)
    # 当前 n 下所需夏普:观测夏普越过它 ⟺ DSR 行通过
    req = d["sr_required_dsr95(n=24)"]["value"]
    dsr_row = next(x for x in v["rows"] if x["gate"] == "dsr(n=24)")
    assert (center >= req) == dsr_row["passed"]
    # 绑定值 = 三道夏普族门所需值的最大者
    g = load_contract()["gates_draft"]
    req_psr = gates.required_sharpe_ann(g["psr_min"], 0.0, t_obs, skew, kurt)
    assert d["sr_required_calm_binding"]["value"] == pytest.approx(
        max(g["long_excess_sharpe_min"], req_psr, req), rel=1e-12)
    p8, p10 = d["p_pass_calm|true_sr=0.8"]["value"], d["p_pass_calm|true_sr=1.0"]["value"]
    assert 0.0 < p8 < p10 < 1.0


def test_psr_from_stats_bitwise_equal_to_psr():
    rng = np.random.default_rng(11)
    for i in range(20):
        x = rng.standard_t(3 + i % 5, 200 + 97 * i) * 0.01 + rng.normal(0, 0.002)
        sr, skew, kurt, t_obs = gates.sample_moments(x)
        for b in (0.0, 0.37, 1.2):
            assert gates.psr_from_stats(sr, t_obs, skew, kurt, b) == gates.psr(x, b)
        assert gates.psr(x) == _legacy_psr(x) and gates.dsr(x, 17) == _legacy_dsr(x, 17)


def test_n_trials_breakdown_row_and_mismatch_note():
    bt = _bt_cases()["near"]
    v = gates.evaluate(bt, _null("all_pass"), n_trials=24,
                       n_trials_breakdown={"history": 20, "declared": 4, "total": 24})
    assert _disc(v)["n_trials_breakdown"]["value"] == "history=20 + declared=4 = total 24"
    assert v["n_trials_breakdown"] == {"history": 20, "declared": 4, "total": 24}
    assert not any("不一致" in n for n in v["notes"])
    v2 = gates.evaluate(bt, _null("all_pass"), n_trials=24)
    assert _disc(v2)["n_trials_breakdown"]["value"] == "未提供构成(total=24)"
    v3 = gates.evaluate(bt, _null("all_pass"), n_trials=24,
                        n_trials_breakdown={"history": 20, "declared": 5, "total": 25})
    assert any("不一致" in n for n in v3["notes"])
    assert v3["overall_pass"] == v["overall_pass"]


# ======================================================================
# ③ 所需夏普:单调、闭式解、回代
# ======================================================================

def _closed_form_req_ann(n: int, T: int, prob: float = 0.95) -> float:
    """偏度 0、峰度 3 时 (sr−b)² = z²(1+sr²/2)/(T−1) 的正根。"""
    b = gates.expected_max_sharpe_ann(n, T) / math.sqrt(PPY)
    k = gates.norm_ppf(prob) ** 2 / (T - 1)
    a = 1.0 - k / 2.0
    return (b + math.sqrt(b * b - a * (b * b - k))) / a * math.sqrt(PPY)


@pytest.mark.parametrize("T", [361, 1990])
def test_required_sharpe_increases_with_n(T):
    reqs = [gates.required_sharpe_dsr_ann(n, T) for n in range(1, 201)]
    assert reqs[0] == pytest.approx(_closed_form_req_ann(1, T), rel=1e-12)
    diffs = np.diff(reqs)
    assert (diffs > 0).all(), "所需夏普应随 n 严格上升"
    for n in (1, 2, 20, 30, 38, 50, 200):
        assert gates.required_sharpe_dsr_ann(n, T) == pytest.approx(_closed_form_req_ann(n, T), rel=1e-12)


def test_required_sharpe_monotone_in_evaluate_rows():
    bt = _bt_cases()["near"]
    vals = []
    for n in (1, 20, 30, 38, 50):
        v = gates.evaluate(bt, _null("all_pass"), n_trials=n)
        vals.append(_disc(v)[f"sr_required_dsr95(n={n})"]["value"])
    assert all(b > a for a, b in zip(vals, vals[1:])), vals


def test_required_sharpe_is_exact_dsr_boundary():
    """在所需夏普上下各 1e-7 相对扰动,gates.dsr 恰好跨过 0.95(高阶矩取样本观测值)。"""
    rng = np.random.default_rng(5)
    x = rng.standard_t(5, 1990)
    x = (x - x.mean()) / x.std(ddof=1)
    sr0, skew, kurt, t_obs = gates.sample_moments(x + 0.05)
    for n in (20, 38):
        req_pp = gates.required_sharpe_dsr_ann(n, t_obs, 0.95, skew, kurt) / math.sqrt(PPY)
        above = 0.01 * (x + req_pp * (1 + 1e-7))
        below = 0.01 * (x + req_pp * (1 - 1e-7))
        assert gates.dsr(above, n) >= 0.95 > gates.dsr(below, n)


def test_pass_probability_known_answers():
    T = 1990
    req = gates.required_sharpe_dsr_ann(30, T)
    assert gates.pass_probability(req, req, T) == pytest.approx(0.5, abs=1e-12)
    p = [gates.pass_probability(s, req, T) for s in (0.0, 0.5, 0.8, 1.0, 1.3, 2.0)]
    assert all(b > a for a, b in zip(p, p[1:]))
    assert gates.pass_probability(1.0, float("inf"), T) == 0.0
    # 手算:年化标准误 = √((1 + t²/2)/(T−1))·√243,t 为真实逐期夏普;P = 1 − Φ((所需 − 真实)/标准误)
    t = 1.0 / math.sqrt(PPY)
    se = math.sqrt((1 + t * t / 2) / (T - 1)) * math.sqrt(PPY)
    assert gates.pass_probability(1.0, req, T) == pytest.approx(1 - gates.norm_cdf((req - 1.0) / se), rel=1e-12)


# ======================================================================
# ④ 三态标签
# ======================================================================

def test_verdict_label_pure_boundaries():
    L = gates
    assert L.verdict_label(True, 0.1) == L.LABEL_PASS
    assert L.verdict_label(False, 0.8 - 1e-9) == L.LABEL_REFUTED
    assert L.verdict_label(False, 0.8) == L.LABEL_UNDERPOWERED          # 上界恰为 0.8:区间覆盖 0.8
    assert L.verdict_label(False, 0.8 + 1e-9) == L.LABEL_UNDERPOWERED
    assert L.verdict_label(False, float("nan")) == L.LABEL_UNDERPOWERED
    assert L.verdict_label(False, None) == L.LABEL_UNDERPOWERED
    assert L.verdict_label(False, 0.5, sr_reference=0.4) == L.LABEL_UNDERPOWERED


def _series_with_ci_hi(target_hi: float, rel: float, n: int = 600) -> pd.Series:
    """构造一条无危机日的序列,使其平静期 90% 区间上界 = target_hi·(1+rel)(高阶矩固定)。"""
    rng = np.random.default_rng(9)
    x = rng.standard_normal(n)
    x = (x - x.mean()) / x.std(ddof=1)
    _, skew, kurt, t_obs = gates.sample_moments(x + 0.05)
    k = math.sqrt(PPY)

    def hi_of(sr_pp: float) -> float:
        return gates.sharpe_interval_ann(sr_pp, t_obs, skew, kurt)[1]

    lo, hi = -1.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if hi_of(mid) < target_hi else (lo, mid)
    sr_pp = hi
    # 在上界层面施加相对扰动:上界对 sr 的导数约为 1(年化),故直接平移年化夏普
    sr_pp = sr_pp + rel * target_hi / k
    return pd.Series(0.01 * (x + sr_pp), index=pd.bdate_range("2021-01-04", periods=n))


def test_verdict_label_in_evaluate_at_constructed_boundary():
    ref = load_contract()["gates_draft"]["long_excess_sharpe_min"]
    below = _series_with_ci_hi(ref, -1e-6)
    above = _series_with_ci_hi(ref, +1e-6)
    vb = gates.evaluate(_bt(below), _null("all_pass"), n_trials=30)
    va = gates.evaluate(_bt(above), _null("all_pass"), n_trials=30)
    assert _disc(vb)["excess_sharpe_ci90_hi"]["value"] < ref <= _disc(va)["excess_sharpe_ci90_hi"]["value"]
    assert vb["overall_pass"] is False and va["overall_pass"] is False
    assert vb["verdict_label"] == gates.LABEL_REFUTED
    assert va["verdict_label"] == gates.LABEL_UNDERPOWERED


def test_verdict_label_pass_and_other_gate_failures():
    cases = _bt_cases()
    v = gates.evaluate(cases["strong"], _null("all_pass"), n_trials=30)
    assert v["overall_pass"] is True and v["verdict_label"] == gates.LABEL_PASS
    # 区间整体高于 0.8,但零校准未过:未通过、未被否证,并留注说明原因不在夏普
    v = gates.evaluate(cases["strong"], _null("one_fail"), n_trials=30)
    assert v["overall_pass"] is False and v["verdict_label"] == gates.LABEL_UNDERPOWERED
    assert _disc(v)["excess_sharpe_ci90_lo"]["value"] > 0.8
    assert any("区间整体高于" in n for n in v["notes"])
    # 平静期不足 30 天:区间不可得,标功效不足并留注
    tiny = _series(0.002, 0.01, 20, "2021-01-04", 1)
    v = gates.evaluate(_bt(tiny), _null("all_pass"), n_trials=30)
    assert v["verdict_label"] == gates.LABEL_UNDERPOWERED
    assert any("区间不可得" in n for n in v["notes"])
    # 长样本负技能:区间上界 < 0.8 → 已否证
    v = gates.evaluate(cases["weak"], _null("all_pass"), n_trials=30)
    assert _disc(v)["excess_sharpe_ci90_hi"]["value"] < 0.8
    assert v["verdict_label"] == gates.LABEL_REFUTED


def test_disclosure_failure_never_changes_verdict(monkeypatch):
    """披露计算抛错时,overall 与非披露行照旧,只留注与标签(不让流水线转为 ERROR)。"""
    bt = _bt_cases()["strong"]
    ref = gates.evaluate(bt, _null("all_pass"), n_trials=30)

    def boom(*a, **k):
        raise ValueError("synthetic")

    monkeypatch.setattr(gates, "sample_moments", boom)
    v = gates.evaluate(bt, _null("all_pass"), n_trials=30)
    assert v["overall_pass"] is ref["overall_pass"] is True
    assert v["verdict_label"] == gates.LABEL_PASS
    assert all(_same(a[k], b[k]) for a, b in zip(_core_rows(v), _core_rows(ref)) for k in a)
    assert list(_disc(v)) == ["verdict_label"]
    assert any("披露行计算失败" in n for n in v["notes"])
    weak = gates.evaluate(_bt_cases()["near"], _null("all_pass"), n_trials=30)
    assert weak["overall_pass"] is False
    assert weak["verdict_label"] == gates.LABEL_UNCLASSIFIED, "披露失败时不得臆断为'功效不足'"


def test_to_tsv_marks_disclosure_rows_and_label():
    v = gates.evaluate(_bt_cases()["near"], _null("all_pass"), n_trials=30)
    tsv = gates.to_tsv(v)
    assert "(披露,不参与判定)" in tsv
    assert tsv.splitlines()[-1] == f"verdict_label\t\t\t\t{v['verdict_label']}"


# ======================================================================
# 流水线调用处:n_trials_breakdown 传入门槛
# ======================================================================

def test_pipeline_passes_n_trials_breakdown_into_gates():
    from ashare_forge import pipeline
    from ashare_forge.data import db
    from ashare_forge.engine import registry

    n_days, n_stocks = 200, 40
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    codes = [f"sz.{i:06d}" for i in range(n_stocks)]
    rng = np.random.default_rng(3)
    logp = np.cumsum(rng.normal(0, 0.01, (n_days, n_stocks)), axis=0)

    def frame(v):
        return pd.DataFrame(v, index=dates, columns=codes)

    price = frame(100.0 * np.exp(logp))
    panel = {"adj_close": price, "close": price, "open": price, "high": price * 1.01,
             "low": price * 0.99, "tradestatus": frame(1.0), "isST": frame(0.0),
             "preclose": price.shift(1).bfill(), "volume": frame(1e7), "turn": frame(1.0)}
    masks = {"in_universe": frame(True), "can_buy": frame(True), "can_sell": frame(True)}
    bench = pd.Series(5000.0 * np.exp(np.cumsum(np.full(n_days, 0.0002))), index=dates)
    ctx = {"panel": panel, "masks": masks, "bench": bench, "bench_code": "synthetic",
           "start": str(dates[0].date()), "end": str(dates[-1].date()),
           "n_dates": n_days, "n_codes": n_stocks,
           "first_date": str(dates[0].date()), "last_date": str(dates[-1].date()),
           "avg_universe": float(n_stocks), "total_stocks": n_stocks, "coverage": 1.0,
           "notes": ["合成面板:仅用于机械验证"]}

    def momentum(p):
        a = p["adj_close"].astype("float64")
        return a / a.shift(10) - 1.0

    spec = {"name": "power_probe", "task_id": "test_power_probe", "family": "momentum",
            "operators": ["pct_change", "cs_rank"], "direction": "long_only", "horizon_days": 1,
            "columns": ["adj_close"], "build_factor": momentum, "thesis": "测试", "params": {"win": 10}}
    conn = db.connect(":memory:")
    try:
        registry.ensure_tables(conn)
        r = pipeline.run_factor(conn, spec, ctx=ctx, n_null=0, leak_points=10, strict=True)
    finally:
        conn.close()
    gv = r["gate_verdict"]
    assert gv["n_trials_breakdown"] == r["n_trials_breakdown"]
    bd = r["n_trials_breakdown"]
    assert _disc(gv)["n_trials_breakdown"]["value"] == (
        f"history={bd['history']} + declared={bd['declared']} = total {bd['total']}")
    assert r["verdict"] == "FAIL" and gv["verdict_label"] in (gates.LABEL_REFUTED, gates.LABEL_UNDERPOWERED)


# ======================================================================
# scripts/gate_power.py
# ======================================================================

def _load_script():
    # 不用 importlib:本模块内定义了流水线测试用的因子,泄漏静态扫描会检查本文件全部导入语句
    import sys
    sd = os.path.join(ROOT, "scripts")
    if sd not in sys.path:
        sys.path.insert(0, sd)
    import gate_power
    return gate_power


def test_gate_power_script_analytic_rows():
    gp = _load_script()
    thr = gp.thresholds()
    assert thr == {"sharpe": 0.8, "psr": 0.9, "dsr": 0.95}
    for T in (1990, 361):
        rows = [gp.analytic_row(n, T, thr) for n in (20, 30, 38, 50)]
        reqs = [r["sr_req_dsr95_ann"] for r in rows]
        assert all(b > a for a, b in zip(reqs, reqs[1:]))
        for r in rows:
            assert r["sr_req_dsr95_ann"] == pytest.approx(_closed_form_req_ann(r["n"], T), rel=1e-12)
            assert r["sr_luck_ann"] == gates.expected_max_sharpe_ann(r["n"], T)
            assert r["binding_gate"] == "dsr" and r["sr_req_binding_ann"] == r["sr_req_dsr95_ann"]
            assert r["dsr_at_req"] == pytest.approx(0.95, abs=1e-3)      # gates.dsr 回代
            assert 0.0 < r["p_analytic@0.8"] < r["p_analytic@1.0"] < 1.0
            assert r["sr_true_power80_ann"] > r["sr_req_binding_ann"]
            assert gates.pass_probability(r["sr_true_power80_ann"], r["sr_req_binding_ann"], T) == \
                pytest.approx(0.80, abs=1e-9)
    # BATCH02 记录的"n≈30、平静期约 1990 天时所需夏普约 1.3"
    assert gp.analytic_row(30, 1990, thr)["sr_req_dsr95_ann"] == pytest.approx(1.30, abs=0.01)


def test_gate_power_script_mc_is_seeded_and_monotone_in_n():
    gp = _load_script()
    a, ka = gp.simulate(361, 1.0, "t", reps=60, seed=7)
    b, kb = gp.simulate(361, 1.0, "t", reps=60, seed=7)
    assert a == b and ka == kb
    rates = [a[n] for n in (20, 30, 38, 50)]
    # 公共随机数:同一路径在更大的 n 下 DSR 只会更低,通过比例不增
    assert all(0.0 <= x <= 1.0 for x in rates) and all(y <= x for x, y in zip(rates, rates[1:]))
    rows = [gp.analytic_row(n, 361, gp.thresholds()) for n in (20, 30)]
    for r in rows:
        for k in gp.COLUMNS:
            if k.startswith("p_mc"):
                r[k] = 0.0
    tsv = gp.to_tsv(rows).splitlines()
    assert tsv[0].split("\t") == gp.COLUMNS and len(tsv) == 3


def test_gate_power_ledger_mode_reproduces_evaluate_interval(tmp_path):
    """台账模式由 PSR 行反推的区间与标签,须与 evaluate 当场算出的披露行一致;只取每名最新终局。"""
    import json

    from ashare_forge.data import db
    gp = _load_script()
    cases = _bt_cases()
    path = str(tmp_path / "ledger.db")
    conn = db.connect(path)
    conn.execute("CREATE TABLE forge_trials (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, "
                 "created_at TEXT, verdict TEXT NOT NULL, provenance_json TEXT)")
    cases["crash"] = _bt(_series(-0.003, 0.01, 1500, "2016-01-04", 8))   # 年化约 −4.7,PSR 下溢为 0
    expect = {}
    for i, (cname, nk) in enumerate([("near", "all_pass"), ("weak", "all_pass"),
                                     ("strong", "all_pass"), ("crash", "all_pass"),
                                     ("near", "one_fail")]):
        v = gates.evaluate(cases[cname], _null(nk), n_trials=20 + i)
        name = f"{cname}_probe"
        pv = {"gates": {"overall_pass": v["overall_pass"], "rows": v["rows"]}}
        conn.execute("INSERT INTO forge_trials (name, created_at, verdict, provenance_json) VALUES (?,?,?,?)",
                     (name, f"2026-01-0{i + 1}", "pass" if v["overall_pass"] else "fail",
                      json.dumps(pv, default=str)))
        conn.execute("INSERT INTO forge_trials (name, created_at, verdict) VALUES (?,?,?)",
                     (name, "2026-02-01", "draft"))          # 草稿行不得被当作终局
        expect[name] = v
    conn.commit()
    conn.close()
    got = {r["name"]: r for r in gp.ledger_labels(path)}
    assert set(got) == {"near_probe", "weak_probe", "strong_probe", "crash_probe"}
    for name, r in got.items():
        v = expect[name]
        d = _disc(v)
        assert r["verdict"] == ("pass" if v["overall_pass"] else "fail")
        # 反推口径与 evaluate 同式,须高精度一致;PSR 贴近 1 时退回偏度0峰度3近似,只要求近似一致
        tol = 1e-6 if r["se_source"] == "psr_backout" else 2e-2
        assert r["ci90_hi"] == pytest.approx(d["excess_sharpe_ci90_hi"]["value"], rel=tol)
        assert r["ci90_lo"] == pytest.approx(d["excess_sharpe_ci90_lo"]["value"], rel=tol)
        assert r["label"] == v["verdict_label"]
    assert got["near_probe"]["se_source"] == "psr_backout"
    assert got["crash_probe"]["se_source"] == "approx_skew0_kurt3"     # PSR 下溢,反推不可用
    assert got["crash_probe"]["label"] == gates.LABEL_REFUTED
    assert got["near_probe"]["n"] == 24          # 同名取最新一条终局(第五条)
    assert gp.ledger_tsv(list(got.values())).splitlines()[0].split("\t") == gp.LEDGER_COLUMNS
