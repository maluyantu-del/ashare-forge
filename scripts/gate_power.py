#!/usr/bin/env python
"""门槛检验力:DSR ≥ 0.95 所需的观测夏普,及给定真实夏普时的通过概率(只披露,不改门槛)。

解析部分直接调用 gates.expected_max_sharpe_ann、gates.required_sharpe_dsr_ann、
gates.required_sharpe_ann 与 gates.pass_probability;它们与 gates.psr 共用同一标准误公式
(逐期夏普估计量的偏度峰度修正方差),此处取偏度 0、峰度 3。"通过"指平静期三道夏普族门同时
通过:超额夏普 ≥ 0.8、PSR ≥ 0.9、DSR ≥ 0.95(前两项阈值读合同 gates_draft)。零校准、危机层
与持仓门不在其内,故所列概率是整体通过概率的上界。

蒙特卡洛部分对模拟日收益直接调用 gates.ann_sharpe / gates.psr / gates.dsr 判三道门:
正态与 t 分布(自由度 4,缩放至单位方差)各一组,固定种子;同一条模拟路径在各 n 之间复用
(公共随机数)。正态组校验解析近似本身,t(4) 组检验厚尾下的偏差。t(4) 的四阶矩不存在,
样本峰度不收敛,PSR 公式中的峰度项因此随样本波动。

T=1990 为 BATCH02 样本的平静期交易日数,T=361 为危机期交易日数。危机层的实际门槛是
超额夏普 ≥ 0,不含 DSR;T=361 行仅说明该样本长度下夏普的分辨力。

不读数据库、不读任何行情数据。

用法:
    .venv/bin/python scripts/gate_power.py                  # 打印 TSV 并写 reports/gate_power.tsv
    .venv/bin/python scripts/gate_power.py --reps 2000 --no-write
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ashare_forge.contract import load_contract  # noqa: E402
from ashare_forge.engine import gates  # noqa: E402

PPY = gates.PERIODS_PER_YEAR
N_LIST = (20, 30, 38, 50)
T_LIST = (1990, 361)
TRUE_SR = (0.8, 1.0)
SIGMA = 0.01                 # 日收益标准差;夏普与尺度无关,任取
T_DF = 4
DEFAULT_OUT = os.path.join(ROOT, "reports", "gate_power.tsv")


def _log(msg: str) -> None:
    print(f"# [{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def thresholds() -> dict:
    """平静期三道夏普族门的阈值:与 gates.evaluate 同源(合同 gates_draft + DSR 0.95)。"""
    g = load_contract().get("gates_draft", {}) or {}
    return {"sharpe": float(g.get("long_excess_sharpe_min", 0.8)),
            "psr": float(g.get("psr_min", 0.9)),
            "dsr": gates.DSR_MIN}


def normal_scores(T: int) -> np.ndarray:
    """确定性的近正态序列(Blom 分位点),标准化为均值 0、样本标准差 1;偏度为 0,峰度略低于 3。"""
    p = (np.arange(1, T + 1) - 0.375) / (T + 0.25)
    x = np.array([gates.norm_ppf(float(q)) for q in p])
    return (x - x.mean()) / x.std(ddof=1)


def true_sr_for_power(power: float, sr_required_ann: float, T: int) -> float:
    """使解析通过概率恰为 power 的真实年化夏普(二分;偏度 0、峰度 3)。"""
    lo, hi = sr_required_ann - 10.0, sr_required_ann + 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gates.pass_probability(mid, sr_required_ann, T) < power:
            lo = mid
        else:
            hi = mid
    return hi


def analytic_row(n: int, T: int, thr: dict) -> dict:
    """偏度 0、峰度 3 下的解析结果。"""
    luck = gates.expected_max_sharpe_ann(n, T)
    req_dsr = gates.required_sharpe_dsr_ann(n, T, thr["dsr"])
    req_psr = gates.required_sharpe_ann(thr["psr"], 0.0, T)
    cands = {"excess_sharpe": thr["sharpe"], "psr": req_psr, "dsr": req_dsr}
    binding_gate = max(cands, key=cands.get)
    binding = cands[binding_gate]
    # 回代:在所需夏普处构造一条近正态序列,直接用 gates.dsr 计算,应约等于 0.95
    x = normal_scores(T)
    dsr_check = gates.dsr(SIGMA * (x + req_dsr / math.sqrt(PPY)), n)
    row = {"n": n, "T": T, "sr_luck_ann": luck, "sr_req_dsr95_ann": req_dsr,
           "dsr_at_req": dsr_check, "binding_gate": binding_gate, "sr_req_binding_ann": binding}
    for s in TRUE_SR:
        row[f"p_analytic@{s:.1f}"] = gates.pass_probability(s, binding, T)
    row["sr_true_power80_ann"] = true_sr_for_power(0.80, binding, T)
    return row


def simulate(T: int, sr_true_ann: float, dist: str, reps: int, seed: int,
             n_list=N_LIST, thr: dict | None = None) -> tuple[dict[int, float], float]:
    """返回 ({n: 三道门同时通过的比例}, 样本峰度中位数)。"""
    thr = thr or thresholds()
    ss = np.random.SeedSequence([seed, T, int(round(sr_true_ann * 1000)), 0 if dist == "normal" else T_DF])
    rng = np.random.default_rng(ss)
    mu = sr_true_ann / math.sqrt(PPY)
    scale = math.sqrt(T_DF / (T_DF - 2.0))
    hits = {n: 0 for n in n_list}
    kurts = np.empty(reps)
    for i in range(reps):
        x = rng.standard_normal(T) if dist == "normal" else rng.standard_t(T_DF, T) / scale
        r = SIGMA * (x + mu)
        kurts[i] = gates.sample_moments(r)[2]
        if not (gates.ann_sharpe(r) >= thr["sharpe"] and gates.psr(r) >= thr["psr"]):
            continue
        for n in n_list:
            if gates.dsr(r, n) >= thr["dsr"]:
                hits[n] += 1
    return {n: hits[n] / reps for n in n_list}, float(np.median(kurts))


COLUMNS = ["n", "T", "sr_luck_ann", "sr_req_dsr95_ann", "dsr_at_req", "binding_gate",
           "sr_req_binding_ann",
           "p_analytic@0.8", "p_mc_normal@0.8", "p_mc_t4@0.8",
           "p_analytic@1.0", "p_mc_normal@1.0", "p_mc_t4@1.0", "sr_true_power80_ann"]


def build_table(reps: int, seed: int, n_list=N_LIST, t_list=T_LIST) -> tuple[list[dict], dict]:
    thr = thresholds()
    rows = {(n, T): analytic_row(n, T, thr) for T in t_list for n in n_list}
    diag = {"reps": reps, "seed": seed, "thresholds": thr, "median_kurtosis": {}}
    for T in t_list:
        for s in TRUE_SR:
            for dist, tag in (("normal", "normal"), ("t", "t4")):
                t0 = time.time()
                rates, med_k = simulate(T, s, dist, reps, seed, n_list, thr)
                diag["median_kurtosis"][(T, s, tag)] = med_k
                for n in n_list:
                    rows[(n, T)][f"p_mc_{tag}@{s:.1f}"] = rates[n]
                _log(f"T={T} 真实夏普={s:.1f} {tag}: {reps} 条路径,{time.time() - t0:.1f}s,"
                     f"样本峰度中位数 {med_k:.2f}")
    return [rows[(n, T)] for T in t_list for n in n_list], diag


def to_tsv(rows: list[dict]) -> str:
    def fmt(k, v):
        if isinstance(v, str) or isinstance(v, (int, np.integer)):
            return str(v)
        return f"{v:.3f}" if k.startswith("p_") else f"{v:.4f}"
    lines = ["\t".join(COLUMNS)]
    for r in rows:
        lines.append("\t".join(fmt(k, r[k]) for k in COLUMNS))
    return "\n".join(lines)


LEDGER_COLUMNS = ["id", "name", "created_at", "verdict", "n", "T_calm", "sr_calm",
                  "ci90_lo", "ci90_hi", "se_source", "sr_req_dsr95_ann", "label"]


def ledger_labels(db_path: str) -> list[dict]:
    """只读:对台账中每个因子名的最新一条终局记录,按披露口径给出区间与三态归类。

    数据只取台账 provenance 中平静期的 n_days / excess_sharpe / psr / dsr 行,不读行情。
    标准误由 PSR 行反推:PSR(基准 0) = Φ(SR/SE),故 SE = SR/Φ⁻¹(PSR),与 evaluate 的区间
    同口径(观测偏度峰度)。PSR 落在 (1e-9, 1−1e-9) 之外(双精度下 1−PSR 的量化误差过大)
    或夏普贴近 0 时反推不稳定,改用偏度 0、峰度 3 的近似,并在 se_source 列注明。
    所需夏普一列按偏度 0、峰度 3 计算,仅供对照。
    归类只作披露,不改变任何既往裁定。
    """
    import json
    import sqlite3

    ref = thresholds()["sharpe"]
    k = math.sqrt(PPY)
    z90 = gates.norm_ppf(0.5 + gates.CI_LEVEL / 2.0)
    conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
    try:
        recs = conn.execute(
            "SELECT id, name, created_at, verdict, provenance_json FROM forge_trials "
            "WHERE verdict IN ('pass', 'fail') AND id IN "
            "(SELECT MAX(id) FROM forge_trials WHERE verdict IN ('pass', 'fail') GROUP BY name) "
            "ORDER BY id").fetchall()
    finally:
        conn.close()
    out = []
    for rid, name, created, verdict, pj in recs:
        grows = ((json.loads(pj) if pj else {}).get("gates") or {}).get("rows") or []
        calm = {r["gate"]: r for r in grows if r.get("layer") == "calm"}
        dsr_row = next((r for r in calm.values() if str(r["gate"]).startswith("dsr(")), None)
        if "excess_sharpe" not in calm or "psr" not in calm or dsr_row is None:
            continue
        T = int(calm["n_days"]["value"])
        sr = float(calm["excess_sharpe"]["value"])
        p = float(calm["psr"]["value"])
        n = int(str(dsr_row["gate"])[len("dsr(n="):-1])
        z = gates.norm_ppf(p) if 1e-9 < p < 1.0 - 1e-9 else float("nan")
        if z == z and abs(z) > 0.01:
            se, src = sr / z, "psr_backout"
        else:
            se, src = gates.sharpe_se(sr / k, T) * k, "approx_skew0_kurt3"
        lo, hi = sr - z90 * se, sr + z90 * se
        out.append({"id": rid, "name": name, "created_at": created, "verdict": verdict, "n": n,
                    "T_calm": T, "sr_calm": sr, "ci90_lo": lo, "ci90_hi": hi, "se_source": src,
                    "sr_req_dsr95_ann": gates.required_sharpe_dsr_ann(n, T),
                    "label": gates.verdict_label(verdict == "pass", hi, ref)})
    return out


def ledger_tsv(rows: list[dict]) -> str:
    lines = ["\t".join(LEDGER_COLUMNS)]
    for r in rows:
        lines.append("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                               for c in LEDGER_COLUMNS))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=20000, help="每组蒙特卡洛路径数(默认 20000)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-write", action="store_true", help="只打印,不写文件")
    ap.add_argument("--ledger", action="store_true",
                    help="改为只读台账,输出各因子最新终局记录的区间与三态归类(须同时给 --db)")
    ap.add_argument("--db", default=None, help="台账所在 SQLite 路径(只读打开)")
    a = ap.parse_args(argv)

    if a.ledger:
        if not a.db:
            ap.error("--ledger 须显式给出 --db")
        print(ledger_tsv(ledger_labels(a.db)))
        return 0

    rows, diag = build_table(a.reps, a.seed)
    tsv = to_tsv(rows)
    print(tsv)
    mc_se = 0.5 / math.sqrt(a.reps)
    _log(f"蒙特卡洛比例的标准误上界 {mc_se:.4f}(p=0.5 时);阈值 {diag['thresholds']}")
    if not a.no_write:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(tsv + "\n")
        _log(f"已写 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
