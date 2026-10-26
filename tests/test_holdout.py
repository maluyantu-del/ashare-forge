"""保留段一次性执行器(scripts/holdout_eval.py)的测试。

全部使用临时 git 仓库与临时 SQLite 库,不触碰 data/ashare.db。钉死的几条:
  ① 拒绝路径:票据未提交 / 仅暂存 / 提交后又改 / 工作区脏 / 评估相关路径在票据提交后被改 /
     样本内未全过 / 票据已用 / 保留段已被消耗 —— 一律拒绝,且不写库。
  ② dry-run 只读:meta 解锁日志、表结构、台账一概不变;只读连接本身拒绝写入。
  ③ 真实执行:登记 started → 解锁 → 同口径回测 → 两值标签 → completed;同一票据再次运行被拒。
  ④ 标签规则的边界(严格大于 tau)、标准误与 gates.psr 同式、零假设概率与蒙特卡洛一致。
  ⑤ 出处漂移时不出标签;预热不足的因子在解锁前被拦下;解锁上下文与 pipeline.load_context 同口径。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sqlite3
import subprocess
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
import yaml

from ashare_forge import pipeline
from ashare_forge.config import CONTRACT_PATH
from ashare_forge.data import db, loader
from ashare_forge.engine import gates, registry

_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "holdout_eval.py")
_spec = importlib.util.spec_from_file_location("holdout_eval", _P)
he = importlib.util.module_from_spec(_spec)
sys.modules["holdout_eval"] = he
_spec.loader.exec_module(he)

CUTOFF = "2024-09-01"
DATES = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", "2024-10-31")]
CODES = [f"sz.{i:06d}" for i in range(1, 31)]
N_HOLDOUT_ROWS = sum(d >= CUTOFF for d in DATES) - 1          # 末行次日开盘不可得,剔除


# ----------------------------------------------------------------------------- 合成因子
def _rev5(panel):
    px = panel["adj_close"]
    return -(px / px.shift(5) - 1.0)


def _expanding_memory(panel):
    """无限记忆的因子:值依赖全部历史,预热长度不同则末日取值不同。"""
    px = panel["adj_close"]
    return px.expanding().mean() / px


def _make_spec(fn=_rev5, name="t_rev5") -> dict:
    return {"name": name, "family": "reversal", "operators": ["pct_change"], "direction": "long_only",
            "horizon_days": 1, "columns": ["adj_close"], "build_factor": fn, "params": {"WIN": 5},
            "module": "tests.test_holdout"}


# ----------------------------------------------------------------------------- 临时 git 仓库
def _git(root, *args) -> str:
    r = subprocess.run(["git", "-C", str(root), "-c", "user.name=tester", "-c", "user.email=tester@example.invalid",
                        "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _commit_all(root, msg="c") -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)
    return _git(root, "rev-parse", "HEAD")


def _sha256(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ----------------------------------------------------------------------------- 临时库
def _build_db(path) -> sqlite3.Connection:
    c = db.connect(str(path))
    db.ensure_schema(c)
    rng = np.random.default_rng(3)
    rows = []
    for code in CODES:
        p = 10.0
        for d in DATES:
            r = float(np.clip(rng.normal(0.0005, 0.02), -0.05, 0.05))
            pre, p = p, p * (1.0 + r)
            rows.append({"code": code, "date": d, "open": p, "high": p * 1.01, "low": p * 0.99, "close": p,
                         "preclose": pre, "volume": 1e7, "amount": p * 1e7, "turn": 1.0, "tradestatus": 1,
                         "pctChg": r * 100.0, "peTTM": 10.0, "pbMRQ": 1.0, "psTTM": 1.0, "pcfNcfTTM": 1.0,
                         "isST": 0})
    db.upsert_many(c, "daily", rows)
    for col in ("ipo_date", "out_date"):
        c.execute(f"ALTER TABLE stocks ADD COLUMN {col} TEXT")
    db.upsert_many(c, "stocks", [{"code": k, "name": k, "board": "main", "first_seen": DATES[0],
                                  "last_seen": DATES[-1], "delisted": 0, "ipo_date": "2010-01-04",
                                  "out_date": None} for k in CODES])
    db.upsert_many(c, "adj_factor", [{"code": k, "date": "2010-01-04", "foreAdjustFactor": 1.0,
                                      "backAdjustFactor": 1.0, "adjustFactor": 1.0} for k in CODES])
    db.upsert_many(c, "index_daily", [
        {"code": ic, "date": d, "open": 5000.0 * 1.0003 ** i, "high": 5000.0 * 1.0003 ** i,
         "low": 5000.0 * 1.0003 ** i, "close": 5000.0 * 1.0003 ** i, "volume": 1e9, "amount": 1e12,
         "pctChg": 0.03}
        for ic in ("sh.000905", "sh.000001") for i, d in enumerate(DATES)])
    registry.ensure_tables(c)
    return c


def _log_trial(conn, spec, commit, *, sr=1.13, n_calm=1990, n_trials=38, verdict="pass", overall=True) -> int:
    sig, canon = registry.signature_from_spec(spec)
    return registry.log_trial(
        conn, sig, canon, verdict, run_ref="test", name=spec["name"], params=spec["params"], sr_calm=sr,
        provenance={"code_sha": registry.factor_code_sha(spec["build_factor"]), "git_commit": commit[:12],
                    "provenance_drift": None, "leak": {"passed": True}, "null": {"passed": True},
                    "gates": {"overall_pass": overall, "n_trials": n_trials,
                              "rows": [{"layer": "calm", "gate": "n_days", "value": n_calm}]},
                    "data": {"first_date": "2015-01-05", "last_date": "2024-08-30", "bench_code": "sh.000905"},
                    "params": spec["params"], "stats": {}})


def _write_ticket(root, tid, commit, trial_id, spec, *, sr=1.13, n_calm=1990, extra_top=None, sha=None,
                  **ev_over):
    d = {"schema": he.TICKET_SCHEMA, "ticket_id": tid, "created_utc": "2026-09-27T00:00:00Z",
         "code": {"git_commit": commit[:12], "contract_sha256": _sha256(CONTRACT_PATH)},
         "evaluation": {"cutoff": CUTOFF, "warmup_trading_days": 60, "top_pct": 0.1, "bench_code": "sh.000905",
                        "expected_holdout_days": 43, **ev_over},
         "candidates": [{"factor_name": spec["name"], "params": spec["params"],
                         "factor_module_sha": sha or registry.factor_code_sha(spec["build_factor"]),
                         "in_sample": {"trial_id": trial_id, "calm_excess_sharpe": sr, "calm_n_days": n_calm}}]}
    d.update(extra_top or {})
    path = os.path.join(root, "tasks", "holdout_tickets", f"{tid}.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(d, fh, allow_unicode=True, sort_keys=False)
    return path


@pytest.fixture()
def env(tmp_path):
    root = tmp_path / "repo"
    (root / "tasks" / "holdout_tickets").mkdir(parents=True)
    (root / "ashare_forge").mkdir()
    (root / "tasks" / "holdout_tickets" / "README.md").write_text("tickets\n", encoding="utf-8")
    (root / "ashare_forge" / "x.py").write_text("X = 1\n", encoding="utf-8")
    _git(root, "init", "-q")
    code_commit = _commit_all(root, "code")
    conn = _build_db(tmp_path / "t.db")
    spec = _make_spec()
    trial = _log_trial(conn, spec, code_commit)
    yield {"root": str(root), "db": str(tmp_path / "t.db"), "conn": conn, "spec": spec, "trial": trial,
           "code_commit": code_commit, "tmp": tmp_path}
    conn.close()


def _run(env, ticket, *, dry_run=True, confirm=None, spec=None):
    spec = spec or env["spec"]
    out: list[str] = []
    code, rep = he.run_ticket(ticket, env["db"], dry_run=dry_run, confirm=confirm, repo_root=env["root"],
                              resolver=lambda n: [spec] if n == spec["name"] else [], echo=out.append)
    return code, rep, "\n".join(out)


def _db_state(path) -> dict:
    c = sqlite3.connect(path)
    try:
        return {"tables": sorted(r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")),
                "meta": sorted(c.execute("SELECT key, value FROM meta").fetchall()),
                "trials": c.execute("SELECT COUNT(*) FROM forge_trials").fetchone()[0]}
    finally:
        c.close()


# ============================================================================= ④ 纯函数:规则与概率
def test_label_is_strictly_greater_than_tau():
    thr = he.consistency_threshold(1.5, 1990, 497, 0.10)
    tau = thr["tau"]
    assert tau > 0
    assert he.holdout_label(tau, tau) == he.LABEL_BAD                      # 等于门限:不一致
    assert he.holdout_label(tau + 1e-9, tau) == he.LABEL_OK
    assert he.holdout_label(float("nan"), tau) == he.LABEL_BAD             # 不可得:无法确立一致
    assert he.holdout_label(None, tau) == he.LABEL_BAD
    assert he.holdout_label(-0.5, tau) == he.LABEL_BAD


def test_tau_floors_at_zero_so_sign_must_be_positive():
    thr = he.consistency_threshold(0.9, 1990, 497, 0.10)                   # 衰减容许幅度 > S_is
    assert thr["tau"] == 0.0
    assert he.holdout_label(0.0, thr["tau"]) == he.LABEL_BAD
    assert he.holdout_label(1e-6, thr["tau"]) == he.LABEL_OK


def test_threshold_known_answer_and_monotone():
    s_is, n_is, n_h = 1.5, 1990, 497
    thr = he.consistency_threshold(s_is, n_is, n_h, 0.10)
    srd = s_is / math.sqrt(243)
    se_is = math.sqrt((1 + srd ** 2 / 2) / (n_is - 1)) * math.sqrt(243)
    se_h = math.sqrt((1 + srd ** 2 / 2) / (n_h - 1)) * math.sqrt(243)
    z = NormalDist().inv_cdf(0.90)
    assert thr["tau"] == pytest.approx(s_is - z * math.sqrt(se_is ** 2 + se_h ** 2), abs=1e-7)
    # 检验水平越宽松(alpha 越小),容许衰减越大,门限越低;保留段越长,门限越高
    assert he.consistency_threshold(s_is, n_is, n_h, 0.05)["tau"] < thr["tau"]
    assert he.consistency_threshold(s_is, n_is, 2000, 0.10)["tau"] > thr["tau"]


def test_sharpe_se_matches_gates_psr_denominator():
    rng = np.random.default_rng(11)
    r = pd.Series(rng.standard_t(5, 800) * 0.01 + 0.0006)
    s = he.sample_sharpe(r)
    srd, sed = s["sr_ann"] / math.sqrt(243), s["se_ann"] / math.sqrt(243)
    assert gates.norm_cdf(srd / sed) == pytest.approx(gates.psr(r), abs=1e-9)
    assert he.sharpe_se_ann(0.0, 497) == pytest.approx(he.se_null(497))


def test_null_probability_known_answers():
    n_is, n_h = 1990, 497
    # tau=0 时零技能被判一致的概率恰为 1/2(只剩符号);入选线低时绝大多数入选值的 tau 为 0,
    # 只有约 1% 的入选值高于容许幅度(约 1.0)而 tau>0,故积分略低于 1/2
    assert he.p_consistent_given_true(0.0, 0.0, n_h) == pytest.approx(0.5)
    p_low = he.p_consistent_null_selected(0.3, n_is, n_h, 0.10)
    assert 0.49 < p_low < 0.5
    # 条件概率 = 1 − Φ(tau / se_null)
    tau = he.consistency_threshold(2.0, n_is, n_h, 0.10)["tau"]
    assert he.p_consistent_given_true(0.0, tau, n_h) == pytest.approx(
        1 - NormalDist().cdf(tau / he.se_null(n_h)), abs=1e-12)
    # 入选线越高,被挑中的零技能策略越难"一致"
    ps = [he.p_consistent_null_selected(g, n_is, n_h, 0.10) for g in (0.8, 1.3, 2.0)]
    assert ps[0] > ps[1] > ps[2] > 0


def test_null_probability_matches_monte_carlo():
    """解析积分与模拟一致:零技能策略样本内按运气越线 → 保留段按同一规则判定。"""
    n_is, n_h, g, alpha = 1990, 497, 1.2, 0.10
    se_is0, se_h0 = he.se_null(n_is), he.se_null(n_h)
    rng = np.random.default_rng(2024)
    nd = NormalDist()
    lo = nd.cdf(g / se_is0)
    u = rng.uniform(lo, 1.0, 60_000)
    s_is = np.array([nd.inv_cdf(x) for x in u]) * se_is0           # 截断正态抽样(逆变换)
    tau = np.array([he.consistency_threshold(x, n_is, n_h, alpha)["tau"] for x in s_is])
    s_h = rng.normal(0.0, se_h0, s_is.size)
    mc = float((s_h > tau).mean())
    an = he.p_consistent_null_selected(g, n_is, n_h, alpha)
    assert abs(mc - an) < 0.01, (mc, an)


def test_selection_lines_follow_contract_and_dsr():
    lines = he.selection_lines(1990, 38)
    assert lines["contract_excess_sharpe_min"] == pytest.approx(0.8)
    assert lines["dsr_line"] > lines["contract_excess_sharpe_min"]
    assert lines["effective"] == pytest.approx(max(lines["contract_excess_sharpe_min"], lines["psr_line"],
                                                   lines["dsr_line"]))
    # DSR 线的定义自洽:在该线上 DSR 恰为 0.95
    s = lines["dsr_line"]
    b = gates.expected_max_sharpe_ann(38, 1990)
    se = he.sharpe_se_ann(s, 1990)
    assert gates.norm_cdf((s - b) / se) == pytest.approx(0.95, abs=1e-6)


def test_holdout_slice_excludes_in_sample_and_last_row_and_aligns_ic():
    idx = pd.bdate_range("2024-08-26", periods=20)
    h_days = [d for d in idx if d >= pd.Timestamp(CUTOFF)][:-1]
    p0 = list(idx).index(h_days[0])
    ex = pd.Series(0.0, index=idx)
    ex[idx < pd.Timestamp(CUTOFF)] = -0.5                       # 样本内的大亏损不得混入
    ex[h_days] = 0.002 + 0.001 * np.sin(np.arange(len(h_days)))
    ex.iloc[-1] = 9.9                                          # 末行不得计入
    ic = pd.Series(0.1, index=idx)
    ic.iloc[p0 - 1] = 0.9                                      # 衡量首个保留段区间,须计入
    ic.iloc[-2] = 5.0                                          # 区间越出数据末尾,须剔除
    bt = {"daily_ret": ex, "excess_ret": ex, "ic": ic, "turnover": pd.Series(0.1, index=idx),
          "n_holdings": pd.Series(30.0, index=idx), "cost": pd.Series(0.0, index=idx)}
    res = he.evaluate_holdout(bt, CUTOFF, None, [dict(s) for s in he.DEFAULT_SEGMENTS], 1.5, 1990, 38, 0.10, 0.90)
    hs = res["holdout"]
    assert hs["n_days"] == len(h_days)
    assert hs["first_day"] == "2024-09-02" and hs["last_day"] == str(h_days[-1].date())
    assert hs["excess_ann"] == pytest.approx(float(np.mean(ex[h_days])) * 243)
    assert hs["ic_mean"] == pytest.approx((0.9 + 0.1 * (len(h_days) - 1)) / len(h_days))
    assert res["label"] == he.LABEL_OK                         # 持续正超额、波动极小
    assert [s["name"] for s in res["segments"]] == ["pre_policy", "policy_rally", "post_rally"]
    assert sum(s["n_days"] for s in res["segments"]) == hs["n_days"]
    assert res["segments"][0]["excess_sharpe"] is None         # 不足 60 日只报累计超额
    bt["excess_ret"] = -ex.abs()
    assert he.evaluate_holdout(bt, CUTOFF, None, [], 1.5, 1990, 38, 0.10, 0.90)["label"] == he.LABEL_BAD


# ============================================================================= 票据格式
def test_ticket_schema_rejects_typos_and_mismatch(env, tmp_path):
    p = _write_ticket(env["root"], "HT_bad", env["code_commit"], env["trial"], env["spec"],
                      extra_top={"ticket_id": "HT_other"}, decay_alfa=0.2)
    tk, errs = he.load_ticket(p)
    joined = " ".join(errs)
    assert "文件名" in joined and "decay_alfa" in joined


def test_readme_template_uses_only_known_fields(tmp_path):
    """tasks/holdout_tickets/README.md 的模板与执行器认识的字段保持同步(文档与代码不漂移)。"""
    readme = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "tasks", "holdout_tickets", "README.md")
    with open(readme, encoding="utf-8") as fh:
        text = fh.read()
    block = text.split("```yaml", 1)[1].split("```", 1)[0]
    p = tmp_path / "HT001_example.yaml"
    p.write_text(block, encoding="utf-8")
    tk, errs = he.load_ticket(str(p))
    assert tk is not None
    assert not [e for e in errs if "未知" in e], errs
    assert tk["evaluation"]["segments"] == [dict(s) for s in he.DEFAULT_SEGMENTS]


def test_ticket_cutoff_must_match_contract(env):
    p = _write_ticket(env["root"], "HT_cut", env["code_commit"], env["trial"], env["spec"], cutoff="2024-06-01")
    _commit_all(env["root"], "ticket")
    code, rep, _ = _run(env, p)
    assert code == 2 and "contract" in rep["failed"]


# ============================================================================= ① 拒绝路径
def test_refuses_untracked_ticket(env):
    p = _write_ticket(env["root"], "HT_untracked", env["code_commit"], env["trial"], env["spec"])
    before = _db_state(env["db"])
    code, rep, out = _run(env, p)
    assert code == 2
    assert "ticket_committed" in rep["failed"] and "worktree_clean" in rep["failed"]
    assert "未解锁" in out
    assert _db_state(env["db"]) == before


def test_refuses_staged_but_uncommitted_ticket(env):
    p = _write_ticket(env["root"], "HT_staged", env["code_commit"], env["trial"], env["spec"])
    _git(env["root"], "add", "-A")
    code, rep, _ = _run(env, p)
    assert code == 2 and "ticket_committed" in rep["failed"]


def test_refuses_ticket_modified_after_commit(env):
    p = _write_ticket(env["root"], "HT_mod", env["code_commit"], env["trial"], env["spec"])
    _commit_all(env["root"], "ticket")
    with open(p, "a", encoding="utf-8") as fh:
        fh.write("notes: changed after commit\n")
    code, rep, _ = _run(env, p)
    assert code == 2 and "ticket_committed" in rep["failed"]


def test_refuses_dirty_worktree(env):
    p = _write_ticket(env["root"], "HT_dirty", env["code_commit"], env["trial"], env["spec"])
    _commit_all(env["root"], "ticket")
    with open(os.path.join(env["root"], "ashare_forge", "x.py"), "a", encoding="utf-8") as fh:
        fh.write("Y = 2\n")
    code, rep, _ = _run(env, p)
    assert code == 2
    assert rep["failed"] == ["worktree_clean"]


def test_refuses_when_eval_paths_changed_after_code_commit(env):
    with open(os.path.join(env["root"], "ashare_forge", "x.py"), "a", encoding="utf-8") as fh:
        fh.write("Y = 2\n")
    _commit_all(env["root"], "post-hoc change")
    p = _write_ticket(env["root"], "HT_drift", env["code_commit"], env["trial"], env["spec"])
    _commit_all(env["root"], "ticket")
    code, rep, _ = _run(env, p)
    assert code == 2 and "code_commit" in rep["failed"]


def test_refuses_candidate_that_did_not_pass_in_sample(env):
    fail_id = _log_trial(env["conn"], env["spec"], env["code_commit"], verdict="fail", overall=False)
    p = _write_ticket(env["root"], "HT_fail", env["code_commit"], fail_id, env["spec"])
    _commit_all(env["root"], "ticket")
    code, rep, _ = _run(env, p)
    assert code == 2 and rep["failed"] == ["in_sample"]


def test_refuses_when_ticket_numbers_disagree_with_ledger(env):
    p = _write_ticket(env["root"], "HT_num", env["code_commit"], env["trial"], env["spec"], sr=1.30)
    _commit_all(env["root"], "ticket")
    code, rep, _ = _run(env, p)
    assert code == 2 and rep["failed"] == ["in_sample"]


def test_refuses_when_factor_code_differs_from_ticket(env):
    """票据登记的模块源码 sha 与当前加载的因子模块不符(模块在票据之后被改)→ 拒绝。"""
    p = _write_ticket(env["root"], "HT_sha", env["code_commit"], env["trial"], env["spec"], sha="0" * 16)
    _commit_all(env["root"], "ticket")
    code, rep, _ = _run(env, p)
    assert code == 2 and "candidates" in rep["failed"] and "in_sample" in rep["failed"]


def test_warmup_check_catches_long_memory_factor(env):
    spec = _make_spec(fn=_expanding_memory, name="t_expanding")
    trial = _log_trial(env["conn"], spec, env["code_commit"])
    p = _write_ticket(env["root"], "HT_warm", env["code_commit"], trial, spec)
    _commit_all(env["root"], "ticket")
    code, rep, _ = _run(env, p, spec=spec)
    assert code == 2 and rep["failed"] == ["warmup"]


# ============================================================================= ② dry-run 只读
def test_readonly_connection_rejects_writes(env):
    c = he.connect_readonly(env["db"])
    try:
        with pytest.raises(sqlite3.OperationalError):
            c.execute("CREATE TABLE should_fail (a INTEGER)")
    finally:
        c.close()


def test_dry_run_passes_and_leaves_db_untouched(env):
    p = _write_ticket(env["root"], "HT_dry", env["code_commit"], env["trial"], env["spec"])
    _commit_all(env["root"], "ticket")
    meta_before = db.get_meta(env["conn"], loader.HOLDOUT_LOG_KEY)
    before = _db_state(env["db"])
    code, rep, out = _run(env, p)
    assert code == 0, out
    assert rep["failed"] == []
    assert _db_state(env["db"]) == before
    assert "holdout_uses" not in before["tables"]
    assert db.get_meta(env["conn"], loader.HOLDOUT_LOG_KEY) == meta_before
    assert not os.path.exists(p.replace(".yaml", ".result.json"))
    assert rep["rule_preview"][0]["n_h"] == 43


def test_real_run_requires_explicit_confirm(env):
    p = _write_ticket(env["root"], "HT_conf", env["code_commit"], env["trial"], env["spec"])
    _commit_all(env["root"], "ticket")
    before = _db_state(env["db"])
    code, _, _ = _run(env, p, dry_run=False, confirm=None)
    assert code == 2
    code, rep, _ = _run(env, p, dry_run=False, confirm="HT_other")
    assert code == 2 and rep.get("refused") == "confirm_mismatch"
    assert _db_state(env["db"]) == before


# ============================================================================= ③ 真实执行
def test_real_run_end_to_end_then_refuses_reuse(env):
    p = _write_ticket(env["root"], "HT_e2e", env["code_commit"], env["trial"], env["spec"])
    _commit_all(env["root"], "ticket")
    log_before = db.get_meta(env["conn"], loader.HOLDOUT_LOG_KEY)
    code, rep, out = _run(env, p, dry_run=False, confirm="HT_e2e")
    assert code == 0, out
    res = rep["result"]
    assert res["valid"] is True and res["status"] == "completed"
    cand = res["candidates"][0]
    assert cand["label"] in (he.LABEL_OK, he.LABEL_BAD)
    assert cand["holdout"]["n_days"] == N_HOLDOUT_ROWS
    assert cand["holdout"]["first_day"] == "2024-09-02"
    assert 0.0 < cand["p_consistent_null"]["selected_at_effective_line"] < 1.0
    with open(p.replace(".yaml", ".result.json"), encoding="utf-8") as fh:
        assert json.load(fh)["ticket_id"] == "HT_e2e"

    c = sqlite3.connect(env["db"])
    try:
        status, result_json = c.execute(
            "SELECT status, result_json FROM holdout_uses WHERE ticket_id='HT_e2e'").fetchone()
        log_after = c.execute("SELECT value FROM meta WHERE key=?", (loader.HOLDOUT_LOG_KEY,)).fetchone()[0]
    finally:
        c.close()
    assert status == "completed" and json.loads(result_json)["candidates"][0]["label"] == cand["label"]
    new = [json.loads(x) for x in log_after.splitlines()]
    assert log_before is None or log_after.startswith(log_before)         # 旧记录原样保留
    assert new[0] == {**new[0], "func": "holdout_eval", "phase": "begin", "ticket_id": "HT_e2e"}
    assert new[-1]["phase"] == "end" and new[-1]["status"] == "completed"
    assert {r["func"] for r in new[1:-1]} == {"load_panel", "index_series"}

    code, rep, _ = _run(env, p, dry_run=False, confirm="HT_e2e")            # 同一票据再跑
    assert code == 2 and "single_use" in rep["failed"]
    p2 = _write_ticket(env["root"], "HT_second", env["code_commit"], env["trial"], env["spec"])
    _commit_all(env["root"], "second ticket")
    code, rep, out = _run(env, p2)                                          # 保留段已被消耗
    assert code == 2 and "single_use" in rep["failed"] and "消耗" in out


def test_aborted_use_blocks_until_superseded(env):
    c = env["conn"]
    c.executescript(he.USES_DDL)
    c.execute("INSERT INTO holdout_uses (ticket_id, cutoff, status, ticket_sha256, head_commit, started_at)"
              " VALUES ('HT_old', ?, 'aborted', 'x', 'y', '2026-09-27T00:00:00+00:00')", (CUTOFF,))
    c.commit()
    p = _write_ticket(env["root"], "HT_new", env["code_commit"], env["trial"], env["spec"])
    _commit_all(env["root"], "ticket")
    code, rep, _ = _run(env, p)
    assert code == 2 and rep["failed"] == ["single_use"]
    p = _write_ticket(env["root"], "HT_new2", env["code_commit"], env["trial"], env["spec"],
                      extra_top={"supersedes": "HT_old"})
    _commit_all(env["root"], "ticket superseding")
    code, rep, out = _run(env, p)
    assert code == 0, out


def test_concurrent_registration_is_refused_before_unlock(env, monkeypatch):
    """两进程同时通过检查:后登记者在写事务内复核时被拒,不解锁、不写解锁日志。"""
    p = _write_ticket(env["root"], "HT_race", env["code_commit"], env["trial"], env["spec"])
    _commit_all(env["root"], "ticket")
    real = he.run_checks

    def racing(*a, **k):
        out = real(*a, **k)
        c = sqlite3.connect(env["db"])
        c.executescript(he.USES_DDL)
        c.execute("INSERT INTO holdout_uses (ticket_id, cutoff, status, ticket_sha256, head_commit, started_at)"
                  " VALUES ('HT_other_proc', ?, 'started', 'x', 'y', '2026-09-27T00:00:00+00:00')", (CUTOFF,))
        c.commit()
        c.close()
        return out

    monkeypatch.setattr(he, "run_checks", racing)
    log_before = db.get_meta(env["conn"], loader.HOLDOUT_LOG_KEY)
    code, rep, out = _run(env, p, dry_run=False, confirm="HT_race")
    assert code == 2 and rep.get("refused") == "registration", out
    c = sqlite3.connect(env["db"])
    try:
        assert c.execute("SELECT COUNT(*) FROM holdout_uses WHERE ticket_id='HT_race'").fetchone()[0] == 0
        log_after = c.execute("SELECT value FROM meta WHERE key=?", (loader.HOLDOUT_LOG_KEY,)).fetchone()
    finally:
        c.close()
    assert (log_after[0] if log_after else None) == log_before


def test_segments_must_lie_inside_holdout(env):
    p = _write_ticket(env["root"], "HT_seg", env["code_commit"], env["trial"], env["spec"],
                      segments=[{"name": "leaky", "start": "2024-08-01", "end": None}])
    _, errs = he.load_ticket(p)
    assert any("早于 cutoff" in e for e in errs)


def test_provenance_drift_withholds_label(env, monkeypatch):
    p = _write_ticket(env["root"], "HT_prov", env["code_commit"], env["trial"], env["spec"])
    _commit_all(env["root"], "ticket")
    real = he.provenance_snapshot
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        s = real(*a, **k)
        if calls["n"] >= 2:                                   # 结束时的快照:模拟合同在运行中被改
            s = {**s, "contract_sha256": "0" * 64}
        return s

    monkeypatch.setattr(he, "provenance_snapshot", fake)
    code, rep, out = _run(env, p, dry_run=False, confirm="HT_prov")
    assert code == 0, out
    res = rep["result"]
    assert res["valid"] is False and res["provenance"]["drift"] == ["contract_sha256"]
    assert res["candidates"][0]["label"] is None
    c = sqlite3.connect(env["db"])
    try:
        assert c.execute("SELECT status FROM holdout_uses WHERE ticket_id='HT_prov'").fetchone()[0] == "completed"
    finally:
        c.close()


# ============================================================================= ⑤ 同口径
def test_context_parity_with_pipeline(env):
    """解锁版上下文在 cutoff 之前的窗口上与 pipeline.load_context 逐项相同(防复刻漂移)。"""
    start, end = "2024-01-02", "2024-08-30"
    a = he.build_context(env["conn"], start, end, "sh.000905", unlock_holdout=True)
    b = pipeline.load_context(env["conn"], start=start, end=end)
    assert set(a["panel"]) == set(b["panel"])
    for k in a["panel"]:
        if isinstance(a["panel"][k], pd.DataFrame):
            pd.testing.assert_frame_equal(a["panel"][k], b["panel"][k])
        else:
            pd.testing.assert_series_equal(a["panel"][k], b["panel"][k])
    for k in ("in_universe", "can_buy", "can_sell", "delisted"):
        pd.testing.assert_frame_equal(a["masks"][k], b["masks"][k])
    pd.testing.assert_series_equal(a["bench"], b["bench"])
    pd.testing.assert_series_equal(a["bench_open"], b["bench_open"])
