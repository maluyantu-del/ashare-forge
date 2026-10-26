"""截面引擎已知答案测试。

最关键的一条:零假设校准必须能挡住纯噪声因子,同时放行真有预测力的因子。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_forge.engine import backtest, metrics, null_calib

N_DAYS, N_STOCKS = 240, 60
DATES = pd.bdate_range("2023-01-02", periods=N_DAYS)
CODES = [f"sz.{i:06d}" for i in range(N_STOCKS)]


def _frame(v) -> pd.DataFrame:
    return pd.DataFrame(v, index=DATES, columns=CODES)


@pytest.fixture()
def world():
    """构造一个"因子能预测**可交易那段**收益"的世界。

    T+1 下 t 日因子要到 t+1 开盘才建得上仓,吃的是 t+1→t+2 那段,
    所以世界必须写成 ret[t] = 0.02*factor[t-2](滞后两期),而不是滞后一期。
    滞后一期的世界在 T+1 执行下是**不可交易**的(IC 再高也赚不到)。
    """
    rng = np.random.default_rng(42)
    factor = _frame(rng.normal(size=(N_DAYS, N_STOCKS)))
    noise = rng.normal(scale=0.01, size=(N_DAYS, N_STOCKS))
    ret = 0.02 * np.vstack([np.zeros((2, N_STOCKS)), factor.to_numpy()[:-2]]) + noise
    price = _frame(100.0 * np.cumprod(1.0 + ret, axis=0))
    panel = {"adj_close": price, "close": price, "open": price,
             "tradestatus": _frame(1.0), "isST": _frame(0.0),
             "preclose": price.shift(1).bfill(), "volume": _frame(1e7), "turn": _frame(1.0)}
    masks = {"in_universe": _frame(True), "can_buy": _frame(True), "can_sell": _frame(True)}
    return factor, panel, masks


# ---------- 指标 ----------

def test_rank_ic_perfect_and_zero():
    """完美因子 IC≈1、取反≈-1;若 IC 被算成整张面板拉平的相关系数,这里会偏离。"""
    rng = np.random.default_rng(1)
    fwd = _frame(rng.normal(size=(N_DAYS, N_STOCKS)))
    assert metrics.rank_ic(fwd, fwd).mean() == pytest.approx(1.0, abs=1e-9)
    assert metrics.rank_ic(-fwd, fwd).mean() == pytest.approx(-1.0, abs=1e-9)


def test_rank_ic_skips_thin_cross_section():
    """单日可用股票不足 MIN_CROSS_SECTION 时该日不计 IC(而非混入平均)。"""
    f = _frame(np.random.default_rng(2).normal(size=(N_DAYS, N_STOCKS)))
    thin = _frame(False)
    thin.iloc[:, :5] = True                      # 每日只有 5 只可用
    assert metrics.rank_ic(f, f, thin).notna().sum() == 0


def test_monotonicity_sign():
    f = _frame(np.tile(np.arange(N_STOCKS, dtype=float), (N_DAYS, 1)))
    fwd = f * 0.001
    assert metrics.monotonicity(metrics.decile_returns(f, fwd)) == pytest.approx(1.0)
    assert metrics.monotonicity(metrics.decile_returns(-f, fwd)) == pytest.approx(-1.0)


# ---------- T+1 与可交易性(A股规则) ----------

def test_t_plus_1_no_same_day_execution(world):
    """t 日因子不能吃到 t 日的收益。若漏掉 shift(1),完美因子会跑出离谱的高夏普。"""
    factor, panel, masks = world
    r = backtest.run_backtest(factor, panel, masks)
    w = r["weights"]
    assert (w.iloc[0] == 0).all(), "首日就建仓 = 没有 T+1 滞后"


def test_limit_up_blocks_increase(world):
    """一字涨停日不可加仓:目标想买、can_buy=False → 权重不得上升。"""
    factor, panel, masks = world
    m = {k: v.copy() for k, v in masks.items()}
    m["can_buy"].iloc[:] = False                 # 全程买不进
    r = backtest.run_backtest(factor, panel, m)
    assert r["weights"].to_numpy().max() == 0.0, "can_buy 全 False 却建起了仓位"


def test_limit_down_blocks_decrease(world):
    """一字跌停日不可减仓:已持仓且 can_sell=False → 权重不得下降。"""
    factor, panel, masks = world
    m = {k: v.copy() for k, v in masks.items()}
    m["can_sell"].iloc[100:] = False             # 第100日起卖不掉
    r = backtest.run_backtest(factor, panel, m)
    w, pre = r["weights"], r["pre_trade_weights"]
    assert (w.iloc[100] > 0).any(), "前置条件:第100日须有持仓,否则本测试空转"
    # 权重会随价格漂移(那不是卖出);"卖"= 交易后权重低于交易前(已漂移)权重
    for i in range(101, min(120, len(w))):
        assert (w.iloc[i] >= pre.iloc[i] - 1e-12).all(), f"第{i}日在 can_sell=False 下减了仓"


def test_costs_reduce_return(world):
    """成本必须真的扣减:净收益 < 毛收益,且换手越高差距越大。"""
    factor, panel, masks = world
    r = backtest.run_backtest(factor, panel, masks)
    assert r["cost"].sum() > 0
    assert r["daily_ret"].sum() < r["gross_ret"].sum()


# ---------- 零假设校准(本期核心设计) ----------

def _runner(panel, masks):
    def run(f):
        return backtest.run_backtest(f, panel, masks)["stats"]
    return run


def test_null_calibration_rejects_pure_noise(world):
    """纯噪声因子必须被零分布挡住(这正是固定阈值挡不住的情形)。"""
    _, panel, masks = world
    noise = _frame(np.random.default_rng(99).normal(size=(N_DAYS, N_STOCKS)))
    run = _runner(panel, masks)
    real = run(noise)
    null = null_calib.null_distribution(noise, run, n=30, seed=5,
                                        keys=("ic_mean", "icir", "sharpe"))
    v = null_calib.calibrated_verdict(real, null)
    assert v["n_metrics"] > 0
    assert v["n_passed"] == 0, f"纯噪声因子通过了零假设校准:\n{null_calib.to_tsv(v)}"


def test_null_calibration_passes_real_signal(world):
    """真有预测力的因子必须能超越零分布(否则校准过严、把真信号也毙了)。"""
    factor, panel, masks = world
    run = _runner(panel, masks)
    real = run(factor)
    null = null_calib.null_distribution(factor, run, n=30, seed=5,
                                        keys=("ic_mean", "icir", "sharpe"))
    v = null_calib.calibrated_verdict(real, null)
    assert v["n_passed"] == v["n_metrics"], f"真信号未能超越零分布:\n{null_calib.to_tsv(v)}"
    assert (v["table"]["p_value"] < 0.05).all()


def test_shuffle_keeps_original_gaps_and_uses_whole_donor_series():
    """零因子性质(v3):原本缺值处必为缺值;每只股票全样本只取一个 donor 的整条序列,且不取自己。"""
    f = _frame(np.random.default_rng(3).normal(size=(N_DAYS, N_STOCKS)))
    f.iloc[5, :10] = np.nan
    s = null_calib.shuffle_factor(f, 1)
    assert s[f.isna()].isna().all().all(), "原本缺值的格子被填了值"
    for j in range(N_STOCKS):
        col = s.iloc[:, j].dropna()
        hits = [k for k in range(N_STOCKS)
                if np.allclose(col.to_numpy(), f.iloc[:, k].reindex(col.index).to_numpy())]
        assert len(hits) == 1 and hits[0] != j, f"第{j}列不是某一只其他股票的整条序列"


def test_ic_measured_on_tradable_horizon_not_close_to_close():
    """回归:只预测"今收→明收"的因子,在 T+1 下不可交易,IC 必须诚实地接近 0。

    若 IC 用 close-to-close 衡量,这种因子会显示 IC≈1 却在回测里亏钱
    (实测曾出现 IC 0.876 / 夏普 -3.57 的荒谬组合)。本测试钉死"IC 与回测同口径"。
    """
    rng = np.random.default_rng(7)
    factor = _frame(rng.normal(size=(N_DAYS, N_STOCKS)))
    # 世界:ret[t] = 0.02*factor[t-1] —— 只预测下一日收盘,T+1 下吃不到
    ret = 0.02 * np.vstack([np.zeros((1, N_STOCKS)), factor.to_numpy()[:-1]]) \
        + rng.normal(scale=0.005, size=(N_DAYS, N_STOCKS))
    price = _frame(100.0 * np.cumprod(1.0 + ret, axis=0))
    panel = {"adj_close": price, "close": price, "open": price,
             "tradestatus": _frame(1.0), "isST": _frame(0.0),
             "preclose": price.shift(1).bfill(), "volume": _frame(1e7), "turn": _frame(1.0)}
    masks = {"in_universe": _frame(True), "can_buy": _frame(True), "can_sell": _frame(True)}
    st = backtest.run_backtest(factor, panel, masks)["stats"]

    assert abs(st["ic_mean"]) < 0.15, (
        f"不可交易的因子却报出高 IC({st['ic_mean']:.3f})——IC 口径没对齐可交易收益")
    # 对照:同一份数据用 close-to-close 衡量会显示极高 IC(这正是要避免的假象)
    naive = metrics.rank_ic(factor, backtest.forward_returns(price, 1), masks["in_universe"])
    assert naive.mean() > 0.8, "对照组构造失效"


# ---------- 面板泄漏测试 ----------

def test_prefix_recompute_passes_causal_factor():
    """只用过去的因子(20日反转)必须通过。"""
    from ashare_forge.engine import leakage
    rng = np.random.default_rng(11)
    px = _frame(100 * np.cumprod(1 + rng.normal(0, 0.01, (N_DAYS, N_STOCKS)), axis=0))
    panel = {"adj_close": px, "close": px}

    def causal(p):
        a = p["adj_close"]
        return -(a / a.shift(20) - 1.0)          # 只用过去

    r = leakage.prefix_recompute_test(causal, panel)
    assert r["passed"], f"因果因子被误判为泄漏: {r['failures'][:2]}"


def test_prefix_recompute_catches_future_peek():
    """读未来一日的因子必须被抓。"""
    from ashare_forge.engine import leakage
    rng = np.random.default_rng(12)
    px = _frame(100 * np.cumprod(1 + rng.normal(0, 0.01, (N_DAYS, N_STOCKS)), axis=0))
    panel = {"adj_close": px, "close": px}

    def peeking(p):
        a = p["adj_close"]
        return a.shift(-1) / a - 1.0             # 明目张胆读未来

    assert not leakage.prefix_recompute_test(peeking, panel)["passed"], "未抓到 shift(-1) 前视"


def test_prefix_recompute_catches_full_sample_normalization():
    """用全样本统计量(时序 std)归一化 = 隐蔽前视,必须被抓。

    这是截面因子最常见的隐蔽泄漏:截面标准化没问题,但一旦除以整段样本的时序 std,
    就用到了未来信息。固定正则/黑名单查不出来,前缀重算能。
    """
    from ashare_forge.engine import leakage
    rng = np.random.default_rng(13)
    px = _frame(100 * np.cumprod(1 + rng.normal(0, 0.01, (N_DAYS, N_STOCKS)), axis=0))
    panel = {"adj_close": px, "close": px}

    def sneaky(p):
        a = p["adj_close"]
        mom = a / a.shift(20) - 1.0
        return mom / mom.std()                   # 全样本 std:含未来

    assert not leakage.prefix_recompute_test(sneaky, panel)["passed"], "未抓到全样本归一化前视"


# ---------- 回归:零假设校准的构造(2026-09-21 对抗审查 P0) ----------

def test_null_fixed_permutation_preserves_turnover():
    """零因子必须与真因子换手一致,否则比的是手续费不是选股能力。

    旧实现逐日独立打乱,零分布日换手≈0.85 而真因子常为 0.01 量级,
    导致任何低换手因子在收益类指标上自动"超越零分布"(审查实测)。
    """
    rng = np.random.default_rng(1)
    slow = _frame(rng.normal(size=(1, N_STOCKS)) + np.cumsum(rng.normal(0, 0.02, (N_DAYS, N_STOCKS)), axis=0))

    def turn(x):
        return float(x.rank(axis=1, pct=True).diff().abs().sum(axis=1).mean() / 2)

    t_real = turn(slow)
    t_fixed = turn(null_calib.shuffle_factor(slow, 1, mode="fixed"))
    t_daily = turn(null_calib.shuffle_factor(slow, 1, mode="daily"))
    assert t_fixed == pytest.approx(t_real, rel=0.15), "固定 donor 未保持换手同量级"
    assert t_daily > t_real * 5, "对照:旧法应显著抬高换手(否则本测试失去意义)"


def test_null_fixed_permutation_blocks_zero_skill_factor(world):
    """可证明零选股能力的静态随机排序,必须被零分布挡在收益类指标之外。

    这是审查用来击穿旧设计的构造:旧法下它靠"持仓稳定"在 ann_return/sharpe 上自动通过。
    """
    _, panel, masks = world
    rng = np.random.default_rng(5)
    static = _frame(np.tile(rng.normal(size=(1, N_STOCKS)), (N_DAYS, 1)))   # 每只股票一个定值
    run = _runner(panel, masks)
    real = run(static)
    keys = ("ic_mean", "icir", "sharpe", "ann_return")
    v = null_calib.calibrated_verdict(
        real, null_calib.null_distribution(static, run, n=20, seed=3, keys=keys, mode="fixed"))
    assert v["n_passed"] == 0, f"零技能因子攻破了零分布:\n{null_calib.to_tsv(v)}"


def test_null_turnover_matches_real_with_staggered_listings_and_suspensions():
    """回归(审查 P0):真实数据有错峰上市/退市/停牌,v1 的"整列置换+逐行重装"让 donor 对应
    几乎每天滑动,零因子换手暴涨(实测零分布年化 −30%)。新构造须在此情形下保持换手同量级,
    并用对照证明旧实现在同一面板上失败(否则本测试失去意义)。"""
    rng = np.random.default_rng(7)
    n_days, n_stocks = 300, 200
    base = rng.normal(size=(1, n_stocks)) + np.cumsum(rng.normal(0, 0.03, (n_days, n_stocks)), axis=0)
    f = pd.DataFrame(base, index=pd.bdate_range("2021-01-04", periods=n_days),
                     columns=[f"sz.{i:06d}" for i in range(n_stocks)])
    start = rng.integers(0, 200, n_stocks)           # 错峰上市
    end = np.where(rng.random(n_stocks) < 0.2, rng.integers(150, 300, n_stocks), n_days)   # 20% 中途退市
    for j in range(n_stocks):
        f.iloc[: start[j], j] = np.nan
        f.iloc[end[j]:, j] = np.nan
    sus = rng.random((n_days, n_stocks)) < 0.02        # 2% 随机停牌日
    f = f.mask(sus)

    def top_turnover(x):
        top = x.rank(axis=1, ascending=False, pct=True) <= 0.10
        w = top.div(top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        return float(0.5 * w.diff().abs().sum(axis=1).iloc[1:].mean())

    t_real = top_turnover(f)
    t_new = np.median([top_turnover(null_calib.shuffle_factor(f, k, mode="fixed")) for k in range(5)])
    t_old = np.median([top_turnover(null_calib.shuffle_factor(f, k, mode="fixed_repack_legacy"))
                       for k in range(5)])
    assert 0.8 <= t_new / t_real <= 1.25, f"新零因子换手 {t_new:.4f} vs 真实 {t_real:.4f}"
    assert t_old / t_real > 2.0, f"对照失效:旧实现换手 {t_old:.4f} 未显著高于真实 {t_real:.4f}"


def test_null_validity_flags_turnover_mismatch():
    """零样本换手与真因子不同量级时,零分布判无效,主判据不得通过。"""
    from ashare_forge.engine import gates
    null = {"diag": pd.DataFrame({"avg_turnover": [0.30] * 20})}
    bad = null_calib.null_validity({"avg_turnover": 0.03}, null)
    assert bad["valid"] is False and bad["turnover_ratio"] == pytest.approx(10.0)
    good = null_calib.null_validity({"avg_turnover": 0.29}, null)
    assert good["valid"] is True
    idx = pd.bdate_range("2020-01-01", periods=300)
    bt = {"daily_ret": pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 300), index=idx),
          "nav": pd.Series(np.ones(300), index=idx)}
    tbl = pd.DataFrame([{"metric": m, "passed": True} for m in ("ic_mean", "icir", "sharpe", "excess_sharpe")])
    v = gates.evaluate(bt, null_verdict={"table": tbl, "q": 0.95, "n_metrics": 4, "n_passed": 4,
                                         "validity": bad}, n_trials=1)
    assert v["null_passed"] is False


def test_phipson_smyth_pvalue_never_zero():
    """有限次置换的 p 值下界为 1/(n+1);n<19 时 5% 水平不可能通过。"""
    null = {"samples": pd.DataFrame({"ic_mean": np.linspace(-0.01, 0.01, 50)})}
    v = null_calib.calibrated_verdict({"ic_mean": 0.5}, null, q=0.95)
    row = v["table"].iloc[0]
    assert row["p_value"] == pytest.approx(1 / 51) and bool(row["passed"])
    small = {"samples": pd.DataFrame({"ic_mean": np.linspace(-0.01, 0.01, 6)})}
    assert not bool(null_calib.calibrated_verdict({"ic_mean": 0.5}, small)["table"].iloc[0]["passed"])


def test_gates_require_ic_metrics_for_null_pass():
    """IC 族缺失(截面太薄→NaN)时不得只凭收益指标判通过(审查 P0)。"""
    from ashare_forge.engine import gates
    idx = pd.bdate_range("2020-01-01", periods=300)
    bt = {"daily_ret": pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 300), index=idx),
          "nav": pd.Series(np.ones(300), index=idx)}
    only_returns = {"table": pd.DataFrame([
        {"metric": "sharpe", "passed": True}, {"metric": "excess_sharpe", "passed": True}]),
        "q": 0.95, "n_metrics": 2, "n_passed": 2}
    v = gates.evaluate(bt, null_verdict=only_returns, n_trials=1)
    assert v["null_passed"] is False, "IC 族缺失却判零校准通过"
    assert any("IC 族" in n for n in v["notes"])


def test_gates_never_pass_without_null_calibration():
    """跳过零校准时一律不成立(合同:固定阈值仅作参考)。"""
    from ashare_forge.engine import gates
    idx = pd.bdate_range("2020-01-01", periods=1500)
    good = pd.Series(np.random.default_rng(2).normal(0.002, 0.01, 1500), index=idx)
    v = gates.evaluate({"daily_ret": good, "nav": (1 + good).cumprod()}, null_verdict=None, n_trials=1)
    assert v["overall_pass"] is False, "跳过零校准仍给出 PASS —— 宣称与执行矛盾"


def test_accumulation_breakout_uses_single_price_basis():
    """回归:箱体与突破价必须同一价格基准(审查 P0)。

    混用时,复权因子大的股票 breakout 会被放大几个量级,组合权重被少数股票垄断。
    构造两只股票:A 无复权(factor=1)、B 复权因子 10 倍,行为应同量级。
    """
    from ashare_forge.factors.accumulation_breakout import build_factor
    n = 200
    idx = pd.bdate_range("2022-01-03", periods=n)
    codes = ["sz.000001", "sz.000002"]
    base = np.tile(np.linspace(10.0, 12.0, n).reshape(-1, 1), (1, 2))
    close = pd.DataFrame(base, index=idx, columns=codes)
    adj = close.copy()
    adj["sz.000002"] = adj["sz.000002"] * 10.0            # B 的后复权价是原始价 10 倍
    panel = {"close": close, "adj_close": adj,
             "high": close * 1.01, "low": close * 0.99,
             "volume": pd.DataFrame(np.full((n, 2), 1e7), index=idx, columns=codes)}
    f = build_factor(panel)
    a, b = f["sz.000001"].dropna(), f["sz.000002"].dropna()
    if len(a) and len(b):
        assert abs(a.mean() - b.mean()) < 0.5, (
            f"两只走势相同、仅复权因子不同的股票得分差异过大(A={a.mean():.3f} B={b.mean():.3f})"
            " —— 价格基准仍未统一")


# ---------- 回归:泄漏测试的两个审查 P0 ----------

def _write_factor_module(tmp_path, name, body):
    import importlib.util
    f = tmp_path / f"{name}.py"
    f.write_text(body, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_factor


def test_static_scan_catches_out_of_panel_data_access(tmp_path):
    """因子自己开数据库读未来数据,前缀重算察觉不到(审查实测零告警、IC 0.40)——静态扫描必须抓。"""
    from ashare_forge.engine.leakage import scan_factor_source
    cheats = {
        "cheat_sqlite": "import sqlite3\ndef build_factor(p):\n    c=sqlite3.connect('x.db')\n    return p['close']\n",
        "cheat_csv": "import pandas as pd\ndef build_factor(p):\n    f=pd.read_csv('future.csv')\n    return p['close']\n",
        "cheat_open": "def build_factor(p):\n    open('x').read()\n    return p['close']\n",
        "cheat_baostock": "from baostock import query_history_k_data_plus\ndef build_factor(p):\n    return p['close']\n",
        "cheat_env": "import os\ndef build_factor(p):\n    k=os.environ['X']\n    return p['close']\n",
    }
    for name, body in cheats.items():
        assert scan_factor_source(_write_factor_module(tmp_path, name, body)), f"{name} 未被静态扫描抓到"


def test_static_scan_no_false_positive_on_docstrings(tmp_path):
    """文档里提到 baostock/sqlite3/open 不得误报(首版正则曾把 4 个正常因子全部误拦)。"""
    from ashare_forge.engine.leakage import scan_factor_source
    body = ('"""数据来自 baostock 的 peTTM;库是 sqlite3;open 价格用于开盘。"""\n'
            "def build_factor(p):\n"
            "    # 注释里写 read_csv 也不算\n"
            "    return -p['adj_close'].pct_change(20)\n")
    assert scan_factor_source(_write_factor_module(tmp_path, "clean_doc", body)) == []


def test_dense_sampling_catches_sparse_peeking():
    """只在约 2% 的日期偷看未来的因子,默认采样密度下必须被抓(审查实测旧密度抓获率约 10%)。"""
    from ashare_forge.engine import leakage
    rng = np.random.default_rng(21)
    n, s = 600, 60
    idx = pd.bdate_range("2020-01-02", periods=n)
    cols = [f"sz.{i:06d}" for i in range(s)]
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, s)), axis=0), index=idx, columns=cols)
    panel = {"adj_close": px, "close": px}
    leak_days = set(range(40, n - 1, 50))                  # 每 50 天偷看一次 ≈ 2%

    def sparse_peek(p):
        a = p["adj_close"]
        f = -(a / a.shift(20) - 1.0)
        nxt = a.shift(-1) / a - 1.0                          # 次日收益 = 未来信息
        for t in leak_days:
            if t < len(f):
                f.iloc[t] = nxt.iloc[t].rank()
        return f

    caught = sum(not leakage.prefix_recompute_test(sparse_peek, panel, seed=sd)["passed"]
                 for sd in range(8))
    assert caught >= 7, f"稀疏偷看在 8 个随机种子下只被抓到 {caught} 次"


def test_static_scan_follows_helper_calls_but_ignores_unrelated_code(tmp_path):
    """辅助函数里的 I/O 要抓(经调用闭包);与因子无关的 __main__ 调试块不误伤。"""
    from ashare_forge.engine.leakage import scan_factor_source
    helper_cheat = ("def _load():\n    return open('future.txt').read()\n"
                    "def build_factor(p):\n    _load()\n    return p['close']\n")
    assert scan_factor_source(_write_factor_module(tmp_path, "h_cheat", helper_cheat)), "辅助函数里的 open 漏网"
    unrelated = ("def build_factor(p):\n    return -p['adj_close'].pct_change(20)\n"
                 "def _debug():\n    open('log.txt','w')\n"
                 "if __name__ == '__main__':\n    _debug()\n")
    assert scan_factor_source(_write_factor_module(tmp_path, "unrel", unrelated)) == [], \
        "未被因子调用的调试代码被误伤"


def test_size_neutral_actually_removes_size_dependence():
    """回归(审查 P1):市值中性化后,因子与市值的截面 Spearman 须 |ρ|<0.05。

    旧实现(秩对秩线性回归残差)只去掉线性关系,实测残差 Spearman 仍 -0.21,
    "中性化后翻转"的判读因此不成立。
    """
    from ashare_forge.factors.illiquidity import _group_neutralize
    rng = np.random.default_rng(31)
    mcap = _frame(np.exp(rng.normal(22, 1.2, (N_DAYS, N_STOCKS))))
    # 构造与市值**非线性**相关的因子(线性回归去不干净的那种)
    y = _frame(np.exp(-0.9 * np.log(mcap.to_numpy())) * (1 + 0.3 * rng.normal(size=(N_DAYS, N_STOCKS))))
    neu = _group_neutralize(y, mcap, 10, 20)
    rho = np.nanmean([neu.iloc[i].rank().corr(mcap.iloc[i].rank()) for i in range(0, N_DAYS, 10)])
    assert abs(rho) < 0.05, f"中性化后与市值 Spearman 仍为 {rho:.3f}"


# ---------- 漂移再平衡成本(审查 P2:成本低估) ----------

def test_drift_rebalance_hand_computed():
    """两只股等权,A 涨 10%、B 平 → 次日交易前权重 = 1.1/2.1 与 1/2.1,
    回到等权需卖 A 买 B 各 0.5/2.1−... 手算单边换手 = 0.5·Σ|Δw| = 1.1/2.1 − 0.5。"""
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    cols = ["A", "B"]
    tgt = pd.DataFrame(0.5, index=idx, columns=cols)
    ok = pd.DataFrame(True, index=idx, columns=cols)
    ret = pd.DataFrame([[0.10, 0.0], [0.0, 0.0], [0.0, 0.0]], index=idx, columns=cols)
    w, cash, pre, _ = backtest._apply_tradability(tgt, ok, ok, ret=ret)
    assert pre.iloc[1, 0] == pytest.approx(1.1 / 2.1)
    assert pre.iloc[1, 1] == pytest.approx(1.0 / 2.1)
    assert w.iloc[1].tolist() == pytest.approx([0.5, 0.5])
    turn = 0.5 * (w - pre).abs().sum(axis=1)
    assert turn.iloc[1] == pytest.approx(1.1 / 2.1 - 0.5)
    # 旧口径(不漂移)在同一场景下换手为 0 —— 这正是被漏记的那部分
    w0, _, pre0, _ = backtest._apply_tradability(tgt, ok, ok, ret=None)
    assert (0.5 * (w0 - pre0).abs().sum(axis=1)).iloc[1] == pytest.approx(0.0)


def test_drift_keeps_budget_and_raises_cost(world):
    """漂移口径下仍无杠杆(权重+现金≡1),且总成本不低于旧口径。"""
    factor, panel, masks = world
    r_new = backtest.run_backtest(factor, panel, masks)
    r_old = backtest.run_backtest(factor, panel, masks, drift=False)
    tot = r_new["weights"].sum(axis=1) + r_new["cash"]
    live = r_new["weights"].sum(axis=1) > 0
    assert np.allclose(tot[live], 1.0, atol=1e-9)
    assert r_new["stats"]["max_gross_exposure"] <= 1.0 + 1e-9
    assert r_new["stats"]["total_cost"] >= r_old["stats"]["total_cost"] - 1e-12
    assert r_new["stats"]["cost_model"] == "drift_rebalance"


def test_hold_variant_only_uses_past_and_is_prefix_consistent():
    """降频持有:非调仓日沿用最近调仓日的值;前缀截断后已有行不变(与前缀重算泄漏检测一致)。"""
    from ashare_forge.factors import short_reversal as sr
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    f = pd.DataFrame({"A": np.arange(12, dtype=float), "B": np.arange(12, dtype=float) * -1},
                     index=idx)
    h = sr._hold(f, 5)
    assert h["A"].tolist() == [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 10, 10]
    for t in (3, 7, 11):
        pre = sr._hold(f.iloc[: t + 1], 5)
        assert pre.equals(h.iloc[: t + 1])
    assert sr._hold(f, 1).equals(f)


# ---------- 基准时段对齐(审查 P0:度量缺陷) ----------

def test_benchmark_aligned_open_to_open_known_answer():
    """所有股票的开盘价与指数开盘价同步:组合开盘→开盘收益 ≡ 指数开盘→开盘收益。
    对齐口径下,首日建仓成本之后超额恒为 0;旧收盘→收盘口径下超额非零(隔夜段错位)。"""
    rng = np.random.default_rng(11)
    n = 120
    idx = pd.bdate_range("2023-03-01", periods=n)
    cols = [f"sz.{i:06d}" for i in range(10)]
    b_open = pd.Series(3000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    b_close = b_open * np.exp(rng.normal(0, 0.008, n))            # 收盘与开盘不同(有日内与隔夜)
    px_open = pd.DataFrame(np.tile(b_open.to_numpy()[:, None], (1, 10)), index=idx, columns=cols)
    px_close = pd.DataFrame(np.tile(b_close.to_numpy()[:, None], (1, 10)), index=idx, columns=cols)
    panel = {"adj_close": px_close, "close": px_close, "open": px_open}
    ok = pd.DataFrame(True, index=idx, columns=cols)
    masks = {"in_universe": ok, "can_buy": ok, "can_sell": ok}
    factor = pd.DataFrame(1.0, index=idx, columns=cols)             # 恒定等权持有全部 10 只
    r = backtest.run_backtest(factor, panel, masks, bench=b_close, bench_open=b_open, top_pct=1.0)
    ex = r["excess_ret"]
    assert r["stats"]["bench_alignment"] == "open_to_open"
    assert np.allclose(ex.iloc[2:].to_numpy(), 0.0, atol=1e-12), "对齐口径下同步持仓的超额应恒为 0"
    legacy = backtest.run_backtest(factor, panel, masks, bench=b_close, top_pct=1.0)
    assert legacy["stats"]["bench_alignment"] == "close_to_close_legacy"
    assert legacy["excess_ret"].iloc[2:].std() > 1e-3, "旧口径应暴露时段错位产生的伪跟踪误差"
    assert np.isfinite(r["stats"]["excess_sharpe_cc_legacy"])


def test_adj_warning_only_when_first_factor_not_one():
    """新股首条因子=1.0 且晚于首个交易日:按 1.0 处理正确,不告警;首条≠1.0 才是截断。"""
    import warnings as _w
    from ashare_forge.data import loader
    idx = pd.bdate_range("2023-01-02", periods=6)
    close = pd.DataFrame({"A": np.arange(6.0) + 10, "B": np.arange(6.0) + 10}, index=idx)
    ok_adj = pd.DataFrame({"code": ["A", "A"], "date": [idx[2], idx[4]],
                           "backAdjustFactor": [1.0, 1.1]})
    with _w.catch_warnings():
        _w.simplefilter("error")
        out = loader.apply_adjust(close[["A"]], ok_adj)
    assert out["A"].iloc[0] == close["A"].iloc[0]
    bad_adj = pd.DataFrame({"code": ["B"], "date": [idx[3]], "backAdjustFactor": [12.39]})
    with pytest.warns(UserWarning, match="后复权因子历史不完整"):
        loader.apply_adjust(close[["B"]], bad_adj)


def test_v2_delisted_position_is_liquidated_with_haircut():
    """持仓股摘牌时仍未卖出(停牌后直接退市):摘牌日按 haircut 计损失并释放权重(v1 永久冻结=零损失)。"""
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    cols = ["A", "B"]
    tgt = pd.DataFrame(0.5, index=idx, columns=cols)
    tgt.loc[idx[1]:, "A"] = 0.0                       # 之后不再想持有 A(已不在池)
    ok = pd.DataFrame(True, index=idx, columns=cols)
    cannot_sell_a = ok.copy()
    cannot_sell_a.loc[idx[1]:, "A"] = False           # A 停牌,卖不掉
    dl = pd.DataFrame(False, index=idx, columns=cols)
    dl.loc[idx[2]:, "A"] = True                       # 第 3 天摘牌
    ret = pd.DataFrame(0.0, index=idx, columns=cols)
    w, cash, pre, loss = backtest._apply_tradability(tgt, ok, cannot_sell_a, ret=ret,
                                                     delisted=dl, haircut=1.0)
    assert w.loc[idx[1], "A"] == pytest.approx(0.5), "停牌期间应冻结"
    assert loss.loc[idx[2]] == pytest.approx(0.5), "摘牌日全损应记 0.5 净值"
    assert w.loc[idx[2], "A"] == 0.0 and w.loc[idx[3], "A"] == 0.0
    assert (w.loc[idx[2]].sum() + cash.loc[idx[2]]) == pytest.approx(1.0)
    _, _, _, loss0 = backtest._apply_tradability(tgt, ok, cannot_sell_a, ret=ret, delisted=dl, haircut=0.0)
    assert loss0.sum() == 0.0
