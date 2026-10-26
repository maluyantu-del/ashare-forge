"""全收益基准(中证500全收益 H00905)披露列的已知答案测试。全部离线,不连真实库与网络。

必须钉死的几条:
  ① 分红补偿落在"包含除息日开盘缺口"的那一行:t+1 为除息日时补在第 t 行,数值按
     (全收益 收→收) − (价格指数 收→收) 手算可得。
  ② 全收益开→开 = 价格指数开→开 + 同一行的补偿;全收益端缺值的步长补偿按 0 处理并计数。
  ③ 不传该序列时回测行为与引入前完全一致;传入时主口径指标、门槛一律不变,只多出披露列。
  ④ 同步按 sh.000001 交易日历剔除休市日行,只写 csi.H00905 这一个代码,开盘等列留空,可重复运行。
  ⑤ load_context 拿不到序列时为 None 并在 notes 说明;主回测透传,零假设校准不传。
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

from ashare_forge import pipeline
from ashare_forge.contract import load_contract
from ashare_forge.data import db, loader, sync
from ashare_forge.engine import backtest, gates, registry

TR_CODE = "csi.H00905"
TR_KEYS = ("bench_tr_ann", "excess_ann_tr", "excess_sharpe_tr")
# 以字符串路径打桩:本文件不写外部行情库的 import,否则下方 run_factor 用例里本模块定义的
# 因子会被泄漏检测的静态扫描(按模块扫描外部取数)正确拦下
AK_FN = "akshare.stock_zh_index_hist_csindex"


# ============================================================ ① ② 派生函数
def _idx(n: int = 10) -> pd.DatetimeIndex:
    return pd.bdate_range("2023-05-08", periods=n)


def test_dividend_compensation_lands_on_row_before_ex_date():
    """除息日为 idx[5] 且价格指数当日收平:补偿恰为股息率 2%,落在第 4 行,其余行为 0。"""
    idx = _idx()
    p = pd.Series([100, 101, 102, 101, 100, 100, 101, 103, 102, 104], index=idx, dtype=float)
    tr = p * np.where(np.arange(len(idx)) >= 5, 1.02, 1.0)        # 全收益在除息日多出 2%
    comp = loader.dividend_compensation(p, tr)
    assert comp.index.equals(idx)
    assert comp.loc[idx[4]] == pytest.approx(0.02, abs=1e-12), "t+1 为除息日时补偿应在第 t 行"
    others = comp.drop(idx[4]).iloc[:-1]
    assert np.allclose(others.to_numpy(), 0.0, atol=1e-12), f"非除息步长补偿应为 0:\n{others}"
    assert np.isnan(comp.iloc[-1]), "最后一行没有 t+1,补偿应为 NaN"
    assert not np.isclose(comp.loc[idx[5]], 0.02), "补偿不得落在除息日本行"


def test_dividend_compensation_is_difference_of_close_returns():
    """除息日价格指数同时下跌:补偿 = 两条收→收收益之差(手算 0.98×1.02−1 − (0.98−1) = 0.0196)。"""
    idx = _idx(6)
    p = pd.Series([100, 100, 100, 98, 98, 99], index=idx, dtype=float)
    tr = p * np.where(np.arange(len(idx)) >= 3, 1.02, 1.0)
    comp = loader.dividend_compensation(p, tr)
    assert comp.loc[idx[2]] == pytest.approx(0.98 * 1.02 - 1.0 - (0.98 - 1.0), abs=1e-12)
    assert comp.loc[idx[2]] == pytest.approx(0.0196, abs=1e-12)
    assert np.allclose(comp.drop(idx[2]).iloc[:-1].to_numpy(), 0.0, atol=1e-12)


def test_total_return_open_to_open_adds_compensation_on_same_row():
    idx = _idx()
    p_close = pd.Series([100, 101, 102, 101, 100, 100, 101, 103, 102, 104], index=idx, dtype=float)
    p_open = pd.Series([99, 100.5, 101, 102, 100.5, 99.5, 100, 102, 103, 103], index=idx, dtype=float)
    tr = p_close * np.where(np.arange(len(idx)) >= 5, 1.02, 1.0)
    out = loader.total_return_open_to_open(p_open, p_close, tr)
    p_oo = p_open.shift(-1) / p_open - 1.0
    diff = out - p_oo
    assert diff.loc[idx[4]] == pytest.approx(0.02, abs=1e-12)
    assert np.allclose(diff.drop(idx[4]).iloc[:-1].to_numpy(), 0.0, atol=1e-12)
    assert np.isnan(out.iloc[-1]), "最后一行无 t+1 开盘,与主基准同为 NaN"
    assert out.attrs["n_comp_missing"] == 0


def test_total_return_missing_tr_step_counts_and_zero_compensation():
    """全收益端缺一天:涉及该日的两个步长补偿按 0 处理(不前向填充),缺失计 2。"""
    idx = _idx()
    p_close = pd.Series(np.linspace(100, 109, len(idx)), index=idx)
    p_open = p_close * 0.999
    tr = (p_close * 1.0).drop(idx[7])                               # 第 7 天缺
    out = loader.total_return_open_to_open(p_open, p_close, tr, index=idx)
    p_oo = p_open.shift(-1) / p_open - 1.0
    assert out.attrs["n_comp_missing"] == 2
    assert np.allclose((out - p_oo).iloc[:-1].to_numpy(), 0.0, atol=1e-12)


# ============================================================ ③ 回测:披露列不影响其余一切
def _synced_world(seed: int = 11, n: int = 120, k: int = 10):
    """所有股票的开盘/收盘与价格指数同步:主口径超额(建仓后)恒为 0。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-03-01", periods=n)
    cols = [f"sz.{i:06d}" for i in range(k)]
    b_open = pd.Series(3000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    b_close = b_open * np.exp(rng.normal(0, 0.008, n))
    px_open = pd.DataFrame(np.tile(b_open.to_numpy()[:, None], (1, k)), index=idx, columns=cols)
    px_close = pd.DataFrame(np.tile(b_close.to_numpy()[:, None], (1, k)), index=idx, columns=cols)
    panel = {"adj_close": px_close, "close": px_close, "open": px_open}
    ok = pd.DataFrame(True, index=idx, columns=cols)
    masks = {"in_universe": ok, "can_buy": ok, "can_sell": ok}
    factor = pd.DataFrame(1.0, index=idx, columns=cols)
    # 全收益:第 40、90 天除息(1.5%、0.8%)
    step = np.ones(n)
    step[40:] *= 1.015
    step[90:] *= 1.008
    tr_close = b_close * step
    return idx, panel, masks, factor, b_open, b_close, tr_close


def _stats_equal(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
            continue
        if x != y:
            return False
    return True


def test_backtest_without_tr_series_is_identical():
    idx, panel, masks, factor, b_open, b_close, tr_close = _synced_world()
    r0 = backtest.run_backtest(factor, panel, masks, bench=b_close, bench_open=b_open, top_pct=1.0)
    r1 = backtest.run_backtest(factor, panel, masks, bench=b_close, bench_open=b_open, top_pct=1.0,
                               bench_tr_ret=None)
    assert _stats_equal(r0["stats"], r1["stats"])
    assert not any(k in r0["stats"] for k in TR_KEYS) and "excess_ret_tr" not in r0
    assert set(r0) == set(r1)


def test_backtest_tr_disclosure_known_answer_and_main_stats_untouched():
    idx, panel, masks, factor, b_open, b_close, tr_close = _synced_world()
    btr = loader.total_return_open_to_open(b_open, b_close, tr_close)
    comp = loader.dividend_compensation(b_close, tr_close).fillna(0.0)
    # 除息日 idx[40]:补偿 = 0.015 × P[40]/P[39],落在第 39 行
    assert comp.loc[idx[39]] == pytest.approx(0.015 * b_close.iloc[40] / b_close.iloc[39], abs=1e-12)
    assert comp.loc[idx[89]] == pytest.approx(0.008 * b_close.iloc[90] / b_close.iloc[89], abs=1e-12)
    assert np.allclose(comp.drop([idx[39], idx[89]]).to_numpy(), 0.0, atol=1e-12)

    r0 = backtest.run_backtest(factor, panel, masks, bench=b_close, bench_open=b_open, top_pct=1.0)
    r = backtest.run_backtest(factor, panel, masks, bench=b_close, bench_open=b_open, top_pct=1.0,
                              bench_tr_ret=btr)
    # 主口径指标一项不变
    main = {k: v for k, v in r["stats"].items() if k not in TR_KEYS}
    assert _stats_equal(main, r0["stats"])
    assert r["excess_ret"].equals(r0["excess_ret"])
    # 已知答案:全收益超额 = 主口径超额 − 同一行的分红补偿
    assert np.allclose(r["excess_ret_tr"].to_numpy(), (r["excess_ret"] - comp).to_numpy(), atol=1e-12)
    assert r["excess_ret_tr"].loc[idx[39]] == pytest.approx(r["excess_ret"].loc[idx[39]] - comp.loc[idx[39]])
    ppy = 243
    assert r["stats"]["excess_ann_tr"] == pytest.approx(r["stats"]["excess_ann"] - comp.mean() * ppy, abs=1e-12)
    assert r["stats"]["bench_tr_ann"] == pytest.approx(r["stats"]["bench_ann"] + comp.mean() * ppy, abs=1e-12)
    assert np.isfinite(r["stats"]["excess_sharpe_tr"])
    # 门槛只看主口径:有无披露列,门槛各行完全一致
    g0, g1 = gates.evaluate(r0, None, n_trials=3), gates.evaluate(r, None, n_trials=3)
    assert [(x["gate"], x["value"], x["passed"]) for x in g0["rows"]] == \
           [(x["gate"], x["value"], x["passed"]) for x in g1["rows"]]
    assert g0["overall_pass"] == g1["overall_pass"]


# ============================================================ ④ 同步
def _fake_fetch_factory(closes: dict[str, float], calls: list | None = None):
    def fetch(symbol, start, end):
        if calls is not None:
            calls.append((symbol, start, end))
        return pd.DataFrame({"date": list(closes), "close": list(closes.values())})
    return fetch


def _count(conn, code):
    return conn.execute("SELECT COUNT(*) FROM index_daily WHERE code = ?", (code,)).fetchone()[0]


def test_sync_filters_calendar_and_writes_only_its_rows(conn):
    cal = [r[0] for r in conn.execute("SELECT date FROM index_daily WHERE code='sh.000001' ORDER BY date")]
    closes = {d: 6000.0 + i for i, d in enumerate(cal)}
    closes["2024-01-01"] = 5999.0                   # 休市日(元旦,早于日历首日)
    closes["2024-01-06"] = 6003.5                   # 周六
    closes["2024-11-01"] = 7000.0                   # 晚于日历末日:无法核验,本次不写
    # 以前误写的休市日行必须被整段替换清掉
    db.upsert_many(conn, "index_daily", [{"code": TR_CODE, "date": "2024-01-13", "open": None, "high": None,
                                          "low": None, "close": 1.0, "volume": None, "amount": None,
                                          "pctChg": None}])
    before = {c: _count(conn, c) for c in ("sh.000905", "sh.000001")}
    calls: list = []
    res = sync.sync_total_return_index(conn, fetch=_fake_fetch_factory(closes, calls),
                                       since="2024-01-01", end="2024-11-30")
    assert calls == [("H00905", "2024-01-01", "2024-11-30")]
    assert res["rows"] == len(cal) == _count(conn, TR_CODE)
    assert res["dropped_non_trading"] == ["2024-01-01", "2024-01-06"]
    assert res["dropped_beyond_calendar"] == 1 and res["missing_vs_calendar"] == []
    assert (res["first"], res["last"]) == (cal[0], cal[-1])
    assert {c: _count(conn, c) for c in ("sh.000905", "sh.000001")} == before, "不得改动其他指数的行"
    got = conn.execute(f"SELECT date, open, high, low, close, volume, amount, pctChg FROM index_daily "
                       f"WHERE code='{TR_CODE}' ORDER BY date").fetchall()
    assert [g[0] for g in got] == cal
    assert all(g[1] is None and g[2] is None and g[3] is None and g[5] is None and g[6] is None
               and g[7] is None for g in got), "开盘等列必须留空"
    assert all(g[4] == closes[g[0]] for g in got)
    assert conn.execute(f"SELECT COUNT(*) FROM index_daily WHERE code='{TR_CODE}' "
                        f"AND date IN ('2024-01-01','2024-01-06','2024-01-13','2024-11-01')").fetchone()[0] == 0
    # 可重复运行:结果一致
    res2 = sync.sync_total_return_index(conn, fetch=_fake_fetch_factory(closes),
                                        since="2024-01-01", end="2024-11-30")
    assert res2["rows"] == res["rows"] == _count(conn, TR_CODE)
    log = conn.execute("SELECT status, rows FROM sync_log WHERE task='index_tr' ORDER BY id").fetchall()
    assert [tuple(x) for x in log] == [("ok", len(cal)), ("ok", len(cal))]


def test_sync_reports_missing_calendar_days_and_fetch_failure(conn):
    cal = [r[0] for r in conn.execute("SELECT date FROM index_daily WHERE code='sh.000001' ORDER BY date")]
    closes = {d: 6000.0 for d in cal if d != cal[10]}
    res = sync.sync_total_return_index(conn, fetch=_fake_fetch_factory(closes),
                                       since="2024-01-01", end="2024-11-30")
    assert res["missing_vs_calendar"] == [cal[10]]

    def broken(symbol, start, end):
        raise ConnectionError("unreachable")

    n_before = _count(conn, TR_CODE)
    with pytest.raises(ConnectionError):
        sync.sync_total_return_index(conn, fetch=broken, since="2024-01-01", end="2024-11-30")
    assert _count(conn, TR_CODE) == n_before, "拉取失败不得删改已有行"
    last = conn.execute("SELECT status FROM sync_log WHERE task='index_tr' ORDER BY id DESC LIMIT 1").fetchone()
    assert last[0] == "fail"


def test_sync_refuses_without_calendar(tmp_path):
    c = db.connect(str(tmp_path / "empty.db"))
    db.ensure_schema(c)
    with pytest.raises(RuntimeError, match="交易日历为空"):
        sync.sync_total_return_index(c, fetch=_fake_fetch_factory({"2024-01-02": 1.0}),
                                     since="2024-01-01", end="2024-01-31")
    assert _count(c, TR_CODE) == 0
    c.close()


def test_fetch_csindex_close_parses_and_restores_proxy_env(monkeypatch):
    seen = {}

    def fake(symbol, start_date, end_date):
        seen["no_proxy"] = os.environ.get("NO_PROXY", "")
        seen["args"] = (symbol, start_date, end_date)
        return pd.DataFrame({"日期": [pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date()],
                             "指数代码": [symbol, symbol], "开盘": [np.nan, np.nan],
                             "收盘": [6000.5, 6010.25]})

    monkeypatch.setattr(AK_FN, fake)
    monkeypatch.setenv("NO_PROXY", "localhost")
    out = sync.fetch_csindex_close("H00905", "2024-01-01", "2024-01-31")
    assert seen["args"] == ("H00905", "20240101", "20240131")
    assert "csindex.com.cn" in seen["no_proxy"], "缺省应对中证指数官网直连"
    assert os.environ["NO_PROXY"] == "localhost", "调用结束后须恢复代理环境变量"
    assert out["date"].tolist() == ["2024-01-02", "2024-01-03"]
    assert out["close"].tolist() == [6000.5, 6010.25]

    def wrong(symbol, start_date, end_date):
        return pd.DataFrame({"日期": ["2024-01-02"], "指数代码": ["000905"], "收盘": [1.0]})

    monkeypatch.setattr(AK_FN, wrong)
    with pytest.raises(RuntimeError, match="返回指数代码"):
        sync.fetch_csindex_close("H00905", "2024-01-01", "2024-01-31")


def test_fetch_csindex_close_times_out_instead_of_hanging(monkeypatch):
    def slow(symbol, start_date, end_date):
        time.sleep(1.0)
        return pd.DataFrame()

    monkeypatch.setattr(AK_FN, slow)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="TimeoutError"):
        sync.fetch_csindex_close("H00905", "2024-01-01", "2024-01-31", timeout=0.1, retries=1)
    assert time.monotonic() - t0 < 0.9


def test_cli_exposes_index_tr_switch():
    a = sync._parse_args(["--index-tr", "--db", "x.db"])
    assert a.index_tr is True and a.tr_use_env_proxy is False


# ============================================================ ⑤ load_context 与透传
def _complete_adj(conn):
    codes = [r[0] for r in conn.execute("SELECT DISTINCT code FROM daily")]
    db.upsert_many(conn, "adj_factor", [
        {"code": c, "date": "2024-01-01", "foreAdjustFactor": 1.0, "backAdjustFactor": 1.0,
         "adjustFactor": 1.0} for c in codes])


def test_load_context_tr_none_with_note_when_absent(conn):
    _complete_adj(conn)
    ctx = pipeline.load_context(conn, start="2024-01-01", end="2024-08-30")
    assert ctx["bench_tr_ret"] is None and ctx["bench_tr_code"] == TR_CODE
    assert any(TR_CODE in n and "披露列缺失" in n for n in ctx["notes"])
    assert ctx["bench_open"] is not None, "主基准不受影响"


def test_load_context_builds_tr_series_from_synced_rows(conn):
    """夹具库:sh.000905 开盘恒 5000(开→开收益 0)、收盘 5000+i;全收益于 EX 日多出 1%。
    序列第 t 行即补偿,须落在 EX 的前一个交易日,数值 = 0.01 × P[EX]/P[EX-1];保留段不返回。"""
    _complete_adj(conn)
    ex = "2024-06-03"
    px = dict(conn.execute("SELECT date, close FROM index_daily WHERE code='sh.000905'").fetchall())
    tr = {d: c * (1.01 if d >= ex else 1.0) for d, c in px.items()}
    sync.sync_total_return_index(conn, fetch=_fake_fetch_factory(tr), since="2024-01-01", end="2024-10-31")
    ctx = pipeline.load_context(conn, start="2024-01-01")
    s = ctx["bench_tr_ret"]
    ref = ctx["panel"]["adj_close"].index
    assert s is not None and s.index.equals(ref)
    assert ref.max() < pd.Timestamp("2024-09-01"), "保留段不得进入"
    days = [d.strftime("%Y-%m-%d") for d in ref]
    prev = days[days.index(ex) - 1]
    expect = 0.01 * px[ex] / px[prev]
    assert s.loc[prev] == pytest.approx(expect, abs=1e-12)
    rest = s.drop(pd.Timestamp(prev)).iloc[:-1]
    assert np.allclose(rest.to_numpy(), 0.0, atol=1e-12)
    assert not any("披露列缺失" in n for n in ctx["notes"])


def test_run_factor_passes_tr_to_main_backtest_only(monkeypatch):
    """主回测收到 bench_tr_ret,零假设校准的每次回测都不收到;门槛输入不变。"""
    idx, panel, masks, factor, b_open, b_close, tr_close = _synced_world(n=160, k=30)
    rng = np.random.default_rng(5)
    noise = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), 30)), axis=0)),
                         index=idx, columns=panel["open"].columns)
    panel = {k: v * noise for k, v in panel.items()}
    panel.update({"tradestatus": noise * 0 + 1.0, "isST": noise * 0.0})
    btr = loader.total_return_open_to_open(b_open, b_close, tr_close)
    ctx = {"panel": panel, "masks": masks, "bench": b_close, "bench_open": b_open, "bench_tr_ret": btr,
           "bench_code": "sh.000905", "notes": [], "n_dates": len(idx), "n_codes": 30}
    seen = []
    real = backtest.run_backtest

    def spy(*a, **k):
        seen.append(k.get("bench_tr_ret") is not None)
        return real(*a, **k)

    monkeypatch.setattr(backtest, "run_backtest", spy)

    def mom(p):
        a = p["adj_close"].astype("float64")
        return a / a.shift(5) - 1.0

    spec = {"name": "tr_probe", "family": "momentum", "operators": ["pct_change"], "direction": "long_only",
            "horizon_days": 1, "columns": ["adj_close"], "build_factor": mom, "thesis": "测试用"}
    c = db.connect(":memory:")
    registry.ensure_tables(c)
    r = pipeline.run_factor(c, spec, ctx=ctx, n_null=3, top_pct=0.2, leak_points=10, strict=True)
    c.close()
    assert seen, (r["verdict"], r.get("leak"))
    assert seen[0] is True, "主回测应收到全收益序列"
    assert seen[1:] == [False] * 3, "零假设校准不得传入全收益序列"
    assert all(k in r["stats"] for k in TR_KEYS)
    assert not any(m.endswith("_tr") for m in r["null_verdict"]["table"]["metric"])


# ============================================================ 合同
def test_contract_marks_tr_as_disclosure_only():
    c = load_contract()
    prim = [b for b in c["benchmarks"] if b.get("primary")]
    assert [b["code"] for b in prim] == ["sh.000905"], "主基准不变"
    tr = prim[0]["total_return"]
    assert tr["code"] == TR_CODE and tr["role"] == "disclosure_only"
    assert all(b["code"] != TR_CODE for b in c["benchmarks"]), "全收益序列不得作为独立基准(sync_index 会按 baostock 拉取)"
    ds = c["data_sources"]
    assert "中证指数官网" in ds["csindex"] and "akshare" in ds["csindex"]
    assert "代理" not in ds["unavailable"]
    assert pipeline.BENCH_CODE == "sh.000905"
