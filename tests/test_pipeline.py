"""收口驱动的已知答案测试:全部用合成面板,不连真实库(库里正在跑全量同步)。

必须钉死的三条:
  ① **泄漏因子被拦在回测之前** —— 不是"先回测再报泄漏"。用探针数 run_backtest 的调用次数,
     泄漏因子必须是 0 次;先看见漂亮数字再判作弊,人就舍不得砍了。
  ② 正常因子走完全流程并出裁定(泄漏→回测→零校准→门槛→记账,键齐全、口径一致)。
  ③ **台账确实记了账** —— 草稿一行 + 终局一行;泄漏那次记 abandoned;同族试验次数
     真的会累加(它直接决定 DSR 的 n_trials,记不准 = 多重检验白算)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_forge import pipeline
from ashare_forge.data import db
from ashare_forge.engine import backtest, registry

N_DAYS, N_STOCKS = 260, 60
DATES = pd.bdate_range("2023-01-02", periods=N_DAYS)       # 全段落在合同危机窗口之外
CODES = [f"sz.{i:06d}" for i in range(N_STOCKS)]
REV_WIN = 20


def _frame(v) -> pd.DataFrame:
    return pd.DataFrame(v, index=DATES, columns=CODES)


def _world() -> tuple[dict, dict, pd.Series]:
    """构造一个"20日反转真的能预测**可交易那段**收益"的世界。

    T+1 下 t 日因子要到 t+1 开盘才建得上仓、吃 t+1→t+2 那段,所以世界必须写成
    ret[t] 由 t-2 日可得的动量决定(滞后两期),否则真信号也赚不到钱。
    """
    rng = np.random.default_rng(42)
    logp = np.zeros((N_DAYS, N_STOCKS))
    ret = np.zeros((N_DAYS, N_STOCKS))
    for t in range(N_DAYS):
        if t >= REV_WIN + 2:
            mom = logp[t - 2] - logp[t - 2 - REV_WIN]      # t-2 日就能算出来的动量
            ret[t] = -0.35 * mom + rng.normal(scale=0.008, size=N_STOCKS)
        else:
            ret[t] = rng.normal(scale=0.008, size=N_STOCKS)
        logp[t] = (logp[t - 1] if t else 0.0) + ret[t]

    price = _frame(100.0 * np.exp(logp))
    panel = {
        "adj_close": price, "close": price, "open": price,
        "high": price * 1.01, "low": price * 0.99,
        "tradestatus": _frame(1.0), "isST": _frame(0.0),
        "preclose": price.shift(1).bfill(), "volume": _frame(1e7), "turn": _frame(1.0),
    }
    masks = {"in_universe": _frame(True), "can_buy": _frame(True), "can_sell": _frame(True)}
    bench = pd.Series(5000.0 * np.exp(np.cumsum(np.full(N_DAYS, 0.0002))), index=DATES,
                      name="sh.000905")
    return panel, masks, bench


@pytest.fixture(scope="module")
def ctx() -> dict:
    panel, masks, bench = _world()
    return {
        "panel": panel, "masks": masks, "bench": bench, "bench_code": "synthetic",
        "start": str(DATES[0].date()), "end": str(DATES[-1].date()),
        "n_dates": N_DAYS, "n_codes": N_STOCKS,
        "first_date": str(DATES[0].date()), "last_date": str(DATES[-1].date()),
        "avg_universe": float(N_STOCKS), "total_stocks": N_STOCKS, "coverage": 1.0,
        "notes": ["合成面板:仅用于机械验证,不含任何真实市场结论"],
    }


@pytest.fixture()
def ledger():
    """只装台账的内存库(不碰 data/ashare.db,全量同步在跑)。"""
    c = db.connect(":memory:")
    registry.ensure_tables(c)
    yield c
    c.close()


# --------------------------------------------------------------------- 因子与 spec
def causal_reversal(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """20 日反转:只用当日及以前的数据。"""
    a = panel["adj_close"].astype("float64")
    return -(a / a.shift(REV_WIN) - 1.0)


def peeking_factor(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """明目张胆读未来一日(泄漏)。"""
    a = panel["adj_close"].astype("float64")
    return a.shift(-1) / a - 1.0


def sneaky_factor(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """隐蔽泄漏:截面秩没问题,但除以**全样本时序 std** 用到了未来。"""
    a = panel["adj_close"].astype("float64")
    mom = a / a.shift(REV_WIN) - 1.0
    return mom / mom.stack().std()


def _spec(name: str, fn, family: str, **kw) -> dict:
    s = {
        "name": name, "task_id": f"test_{name}", "family": family,
        "operators": ["pct_change", "cs_rank"], "direction": "long_only",
        "horizon_days": 1, "columns": ["adj_close"], "build_factor": fn,
        "thesis": f"测试用因子 {name}", "params": {"win": REV_WIN},
    }
    s.update(kw)
    return s


# --------------------------------------------------------------------- ① 泄漏拦在回测前
@pytest.mark.parametrize("fn,label", [(peeking_factor, "读未来"), (sneaky_factor, "全样本归一化")])
def test_leaky_factor_blocked_before_backtest(ledger, ctx, monkeypatch, fn, label):
    """泄漏因子必须在回测之前被判死:run_backtest 一次都不许被调用。"""
    calls = {"n": 0}
    real = backtest.run_backtest

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(backtest, "run_backtest", spy)

    spec = _spec(f"leaky_{label}", fn, "pattern")
    r = pipeline.run_factor(ledger, spec, ctx=ctx, n_null=5, top_pct=0.1, leak_points=10)

    assert r["verdict"] == "LEAK_FAIL", r
    assert calls["n"] == 0, f"泄漏因子却跑了 {calls['n']} 次回测(泄漏测试没挡在回测之前)"
    assert r["leak"]["passed"] is False and r["leak"]["failures"]
    # 业绩数字一律不得产生
    assert r["stats"] is None and r["null_verdict"] is None and r["gate_verdict"] is None


def test_leak_fail_is_logged_as_abandoned(ledger, ctx):
    """泄漏被毙也要入账(失败是路标),且消耗一次试验计数。"""
    spec = _spec("leaky_logged", peeking_factor, "pattern")
    r = pipeline.run_factor(ledger, spec, ctx=ctx, n_null=0, leak_points=10)
    assert r["verdict"] == "LEAK_FAIL"

    rows = ledger.execute("SELECT verdict FROM forge_trials WHERE signature = ? ORDER BY id",
                          (r["signature"],)).fetchall()
    assert [x[0] for x in rows] == ["draft", "abandoned"]
    assert registry.trial_count(ledger, "ashare_cs", "pattern") == 1


# --------------------------------------------------------------------- ② 正常因子走完全流程
def test_causal_factor_runs_full_pipeline(ledger, ctx):
    """真因子走完 泄漏→回测→零校准→门槛→记账,键齐全、口径一致、真信号能超越零分布。"""
    spec = _spec("causal_reversal", causal_reversal, "reversal")
    r = pipeline.run_factor(ledger, spec, ctx=ctx, n_null=19, top_pct=0.1, seed=3)   # p 下界 1/20=0.05

    assert r["verdict"] in ("PASS", "FAIL"), r.get("error") or r
    for k in ("name", "signature", "leak", "stats", "null_verdict", "gate_verdict", "verdict"):
        assert k in r, f"返回缺键 {k}"
    assert r["leak"]["passed"] is True
    assert r["stage"] == "done"

    st = r["stats"]
    assert st["n_days"] > 200 and st["ic_mean"] > 0.05, f"构造的真信号 IC 没起来: {st}"
    # 口径自检:IC 由回测器对齐可交易收益产出,不该出现"IC 高而夏普极负"的荒谬组合
    assert st["sharpe"] > 0, f"IC {st['ic_mean']:.3f} 却夏普 {st['sharpe']:.2f}:口径对不上"

    tbl = r["null_verdict"]["table"]
    assert not tbl.empty and r["null_verdict"]["n_metrics"] >= 4
    assert bool(tbl.loc[tbl["metric"] == "ic_mean", "passed"].iloc[0]), \
        f"真信号的 IC 未能超越零分布 q95:\n{tbl}"

    assert r["gate_verdict"]["rows"], "门槛没有产出任何一行"
    assert r["n_trials"] == 1                      # 台账里本族第一次
    assert isinstance(r["seconds"], float)


def test_pure_noise_factor_does_not_pass(ledger, ctx):
    """纯噪声(用价格算但与未来无关)不得通过门槛——零校准挡不住时固定阈值也要挡。"""
    def noise(panel):
        a = panel["adj_close"].astype("float64")
        # 只用当日及以前:昨日收益的确定性打散,与未来收益无关
        return (a / a.shift(1) - 1.0).rolling(3, min_periods=3).std()

    r = pipeline.run_factor(ledger, _spec("noise", noise, "volatility"),
                            ctx=ctx, n_null=19, top_pct=0.1, seed=5)
    assert r["verdict"] == "FAIL", f"噪声因子却通过了门槛:\n{pipeline.scorecard(r)}"


def test_bad_spec_is_error_not_crash(ledger, ctx):
    """spec 缺字段:批量跑时记成 ERROR 一行而不是炸掉整批;strict=True 时照抛。"""
    bad = _spec("bad", causal_reversal, "reversal")
    bad.pop("direction")
    r = pipeline.run_factor(ledger, bad, ctx=ctx, n_null=0)
    assert r["verdict"] == "ERROR" and "direction" in r["error"]
    with pytest.raises(ValueError):
        pipeline.run_factor(ledger, bad, ctx=ctx, n_null=0, strict=True)


# --------------------------------------------------------------------- ③ 台账确实记了账
def test_ledger_records_every_run(ledger, ctx):
    """同一机制跑两次:机制只注册一次,每次两行(draft+终局),n_trials 真的累加。"""
    spec = _spec("ledger_check", causal_reversal, "momentum")
    r1 = pipeline.run_factor(ledger, spec, ctx=ctx, n_null=0, top_pct=0.1)
    r2 = pipeline.run_factor(ledger, spec, ctx=ctx, n_null=0, top_pct=0.1)
    assert r1["verdict"] in ("PASS", "FAIL") and r2["verdict"] in ("PASS", "FAIL")

    sig = r1["signature"]
    assert r2["signature"] == sig                       # 同机制 → 同签名
    n_mech = ledger.execute("SELECT COUNT(*) FROM forge_mechanisms WHERE signature = ?",
                            (sig,)).fetchone()[0]
    assert n_mech == 1, "同一机制被重复注册"

    verdicts = [x[0] for x in ledger.execute(
        "SELECT verdict FROM forge_trials WHERE signature = ? ORDER BY id", (sig,)).fetchall()]
    assert len(verdicts) == 4 and verdicts.count("draft") == 2
    assert all(v in ("draft", "pass", "fail") for v in verdicts)

    # n_trials 是 DSR 的输入:第一次 1、第二次 2(draft 不计数)
    assert (r1["n_trials"], r2["n_trials"]) == (1, 2)
    assert registry.trial_count(ledger, "ashare_cs", "momentum") == 2
    assert r2["dedup_known"] is True, "第二次跑同机制却没给出撞车预警"

    # 业绩数字真的落了库,而不是只落了一个 verdict
    row = ledger.execute(
        "SELECT ic_mean, n_days, n_codes, task_id FROM forge_trials"
        " WHERE signature = ? AND verdict != 'draft' ORDER BY id DESC LIMIT 1", (sig,)).fetchone()
    assert row[0] is not None and row[1] > 200 and row[2] == N_STOCKS
    assert row[3] == spec["task_id"]


def test_signature_collides_on_reskinned_variant():
    """换皮防线:加 clip/smooth/截面秩、换名字、同义算子 → 必须撞到同一签名。"""
    a, _ = registry.mechanism_signature("reversal", ["pct_change"], "long_only", 1, ["adj_close"])
    b, _ = registry.mechanism_signature(
        "reversal", ["ret", "clip", "smooth", "cs_rank"], "long_only", 2, ["adj_close"])
    assert a == b, "换皮变体没有撞到同一签名(查重形同虚设)"
    c, _ = registry.mechanism_signature("momentum", ["pct_change"], "long_only", 1, ["adj_close"])
    assert c != a, "不同信号族却撞签名"


# --------------------------------------------------------------------- 批量与汇总
def test_run_all_summary_and_pool(ledger, ctx):
    """批量跑:汇总表可粘进 Excel;泄漏的那个不进最终池。"""
    specs = [_spec("batch_causal", causal_reversal, "reversal"),
             _spec("batch_leaky", peeking_factor, "pattern")]
    df = pipeline.run_all(ledger, specs, ctx=ctx, n_null=0, top_pct=0.1)

    assert list(df["name"]) == ["batch_causal", "batch_leaky"]
    assert list(df.columns) == list(pipeline.SUMMARY_COLS)
    assert df.loc[df["name"] == "batch_leaky", "verdict"].iloc[0] == "LEAK_FAIL"
    assert "batch_leaky" not in pipeline.final_pool(df)
    assert len(df.attrs["results"]) == 2

    tsv = pipeline.to_tsv(df)
    head = tsv.splitlines()[0].split("\t")
    assert head == list(pipeline.SUMMARY_COLS) and "\t" in tsv.splitlines()[1]
    assert "=== " in pipeline.scorecard(df.attrs["results"][0])


def test_discover_specs_yields_runnable_specs():
    """CLI 的因子发现:拿到的 spec 必须是能直接跑的(签名字段齐全、build_factor 可调用)。"""
    specs = pipeline.discover_specs()
    assert specs, "ashare_forge/factors 下没有发现任何可跑的因子"
    for s in specs:
        pipeline._check_spec(s)                     # 缺字段会抛
        sig, canon = registry.signature_from_spec(s)
        assert len(sig) == 16 and canon["universe"] == "ashare_cs"
    names = {s["name"] for s in specs}
    assert "accumulation_breakout" in names
    assert pipeline.discover_specs(["accumulation_breakout"])[0]["name"] == "accumulation_breakout"
    assert pipeline.discover_specs(["不存在的因子"]) == []


def test_load_context_refuses_when_adj_factors_missing(conn):
    """整只股票缺复权因子时,研究运行必须拒绝开跑(缺省),或在 note 模式下显式留痕。

    回归 2026-09-21 事故:复权补拉登录失败,154 只退市股无因子,批跑照常开跑。
    夹具库只给 NORMAL 一只配了因子,其余有行情的股票均缺。
    """
    with pytest.raises(pipeline.DataIncompleteError):
        pipeline.load_context(conn, start="2024-01-01", end="2024-08-30")
    ctx = pipeline.load_context(conn, start="2024-01-01", end="2024-08-30", adj_policy="note")
    assert ctx["n_missing_adj"] >= 1
    assert any("复权因子缺失" in n for n in ctx["notes"])


def test_load_context_passes_gate_when_adj_complete(conn):
    """补齐因子后闸门放行,n_missing_adj == 0。"""
    codes = [r[0] for r in conn.execute("SELECT DISTINCT code FROM daily")]
    db.upsert_many(conn, "adj_factor", [
        {"code": c, "date": "2024-01-01", "foreAdjustFactor": 1.0, "backAdjustFactor": 1.0,
         "adjustFactor": 1.0} for c in codes])
    ctx = pipeline.load_context(conn, start="2024-01-01", end="2024-08-30")
    assert ctx["n_missing_adj"] == 0


def test_provenance_snapshot_taken_at_start_not_at_log_time(ledger, ctx, monkeypatch):
    """批次运行中途发生新提交:台账须记启动时的提交,并标记出处漂移(回归 id36 出处错记)。"""
    import json
    calls = {"n": 0}

    def fake_commit(*a, **k):
        calls["n"] += 1
        return "aaaaaaaaaaaa" if calls["n"] == 1 else "bbbbbbbbbbbb"

    monkeypatch.setattr(registry, "git_commit", fake_commit)
    spec = _spec("snap_rev", causal_reversal, "reversal")
    df = pipeline.run_all(ledger, [spec], ctx=ctx, n_null=0, top_pct=0.1, leak_points=10)
    assert df.iloc[0]["verdict"] in ("PASS", "FAIL")
    row = ledger.execute("SELECT git_commit, provenance_json FROM forge_trials "
                         "WHERE name='snap_rev' AND verdict!='draft'").fetchone()
    assert row[0] == "aaaaaaaaaaaa"
    pv = json.loads(row[1])
    assert pv["provenance_drift"]["git_commit_at_log"] == "bbbbbbbbbbbb"


def test_optional_specs_run_only_when_named():
    """申报档/事后变体(batch_default=False)不随全量批跑重复入账;按名点跑或 include_optional 才包含。"""
    batch = {s["name"] for s in pipeline.discover_specs()}
    every = {s["name"] for s in pipeline.discover_specs(include_optional=True)}
    optional = every - batch
    assert {"earnings_yield_ep_lag0", "earnings_yield_ep_lag60_pos", "illiquidity_amihud_w60",
            "short_reversal_hold5", "short_reversal_hold20"} <= optional
    by_task = {s["name"] for s in pipeline.discover_specs(["task05_earnings_yield"])}
    assert "earnings_yield_ep_lag0" not in by_task and "earnings_yield_ep" in by_task
    named = pipeline.discover_specs(["earnings_yield_ep_lag0"])
    assert [s["name"] for s in named] == ["earnings_yield_ep_lag0"]
