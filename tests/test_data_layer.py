"""数据层已知答案测试:每条断言都对应"某种错误实现会让它变红"。

它修的坑:原版 AB-Screener 一类工具把 T+1/涨跌停/停牌/ST/次新/微盘写在"注意事项"里靠人记,
本测试把合同 tradability_rules 逐条钉成断言,规则被改坏会立刻红。
"""
from __future__ import annotations

import sqlite3
import warnings

import numpy as np
import pandas as pd
import pytest

from ashare_forge.contract import limit_pct, load_contract
from ashare_forge.data import db, loader, sync
from conftest import (ADJ_DAY, AS_OF, CUTOFF, DELISTED, DELISTED_LAST, LIMIT_DOWN_DAY,
                      LIMIT_UP_DAY, MICRO, NEW, NORMAL, NORMAL_OK_DAY, ST, STAR, SUSPEND_DAY,
                      TURN, BIG_VOL, MICRO_VOL, _dates, _new_stock_window)

START = "2024-01-02"


def _panel(conn, **kw):
    return loader.load_panel(conn, START, AS_OF, **kw)


# ---------- 合同规则:板块涨跌停幅度 ----------

def test_limit_pct_by_board():
    """幅度取错板块(如全用10%)会红。"""
    assert limit_pct(NORMAL, False) == pytest.approx(0.10)
    assert limit_pct(NORMAL, True) == pytest.approx(0.05)    # 主板 ST
    assert limit_pct(NEW, False) == pytest.approx(0.20)      # 创业板
    assert limit_pct(STAR, False) == pytest.approx(0.20)     # 科创板


# ---------- holdout 下沉数据层 ----------

def test_load_panel_excludes_holdout_by_default(conn):
    """若 loader 忘记过滤 cutoff(最常见的"先看图"泄漏路径),这里立刻红。"""
    p = _panel(conn)
    dates = p["close"].index
    assert len(dates) > 0
    assert dates.max() < pd.Timestamp(CUTOFF), f"默认模式返回了保留段数据:最大日期 {dates.max()}"
    assert not any(d >= pd.Timestamp(CUTOFF) for d in dates)


def test_unlock_holdout_returns_more_and_writes_log(conn):
    """解锁必须同时满足:①真的多出保留段数据 ②在 meta 留台账(无台账=偷看不留痕)。"""
    locked = _panel(conn)["close"]
    unlocked = _panel(conn, unlock_holdout=True)["close"]
    assert unlocked.index.max() >= pd.Timestamp(CUTOFF)
    assert len(unlocked) > len(locked)
    log = db.get_meta(conn, "holdout_unlocked_log")
    assert log and "load_panel" in log, "解锁未写 meta 台账"


def test_index_series_also_respects_holdout(conn):
    """指数序列是绕过 holdout 最容易被忽略的旁路。"""
    s = loader.index_series(conn, "sh.000905", START, AS_OF)
    assert len(s) > 0 and s.index.max() < pd.Timestamp(CUTOFF)
    s2 = loader.index_series(conn, "sh.000905", START, AS_OF, unlock_holdout=True)
    assert s2.index.max() >= pd.Timestamp(CUTOFF)


# ---------- 复权 ----------

def test_apply_adjust_forward_fill_and_default_one(conn):
    """后复权因子必须从生效日"向后"携带;若误用 bfill,生效日之前也会变成 2.0 → 红。
    无因子记录的股票必须是 1.0(等于原价),若实现成 NaN 会红。"""
    p = _panel(conn)
    close, adj = p["close"], p["adj_close"]
    before, after = pd.Timestamp("2024-05-14"), pd.Timestamp(ADJ_DAY)
    assert adj.loc[before, NORMAL] == pytest.approx(close.loc[before, NORMAL] * 1.0)
    assert adj.loc[after, NORMAL] == pytest.approx(close.loc[after, NORMAL] * 2.0)
    # 无复权因子记录的股票 → 因子 1.0
    assert adj.loc[after, ST] == pytest.approx(close.loc[after, ST])


# ---------- 流通市值 ----------

def test_derive_float_mcap_matches_hand_calc_and_turn_zero(conn):
    """流通市值 = close*volume/(turn/100);把 turn 当小数(不除100)会差100倍 → 红。
    停牌日 turn=0 必须是 NaN,若除零得 inf 会红。"""
    p = _panel(conn)
    fmc = p["float_mcap"]
    d = pd.Timestamp(NORMAL_OK_DAY)
    assert fmc.loc[d, NORMAL] == pytest.approx(10.0 * BIG_VOL / (TURN / 100.0))   # 5e9
    assert fmc.loc[d, MICRO] == pytest.approx(10.0 * MICRO_VOL / (TURN / 100.0))  # 1e8
    assert np.isnan(fmc.loc[pd.Timestamp(SUSPEND_DAY), NORMAL])                   # turn=0
    assert not np.isinf(fmc.to_numpy(dtype="float64")).any()


# ---------- 可交易性三张面具(合同核心) ----------

def test_masks_normal_stock_tradable(conn):
    p = _panel(conn)
    m = loader.tradable_masks(p)
    d = pd.Timestamp(NORMAL_OK_DAY)
    assert m["in_universe"].loc[d, NORMAL]
    assert m["can_buy"].loc[d, NORMAL]
    assert m["can_sell"].loc[d, NORMAL]


def test_masks_exclusions(conn):
    """ST/次新/微盘三条剔除规则各自失效都会红。"""
    p = _panel(conn)
    m = loader.tradable_masks(p)
    d_st = pd.Timestamp(NORMAL_OK_DAY)
    assert not m["in_universe"].loc[d_st, ST], "ST 未被剔除"
    assert not m["in_universe"].loc[d_st, MICRO], "微盘未被剔除"
    d_new = pd.Timestamp(_new_stock_window()[-1])
    assert not m["in_universe"].loc[d_new, NEW], "次新(上市<120日)未被剔除"


def test_masks_st_can_still_sell(conn):
    """合同明写:卖出不受 ST/微盘限制。若把 can_sell 写成 in_universe & ... 会红。"""
    p = _panel(conn)
    m = loader.tradable_masks(p)
    d = pd.Timestamp(NORMAL_OK_DAY)
    assert m["can_sell"].loc[d, ST], "ST 股被错误地禁止卖出"
    assert m["can_sell"].loc[d, MICRO], "微盘股被错误地禁止卖出"


def test_masks_suspension_blocks_everything(conn):
    p = _panel(conn)
    m = loader.tradable_masks(p)
    d = pd.Timestamp(SUSPEND_DAY)
    assert not m["in_universe"].loc[d, NORMAL]
    assert not m["can_buy"].loc[d, NORMAL]
    assert not m["can_sell"].loc[d, NORMAL], "停牌日仍被判为可卖"


def test_masks_limit_up_blocks_buy_not_sell(conn):
    """一字涨停:买不进、但卖得掉。若把涨停也当不可卖会红;若忘记涨停判定 can_buy 会是 True → 红。"""
    p = _panel(conn)
    m = loader.tradable_masks(p)
    d = pd.Timestamp(LIMIT_UP_DAY)
    assert m["in_universe"].loc[d, NORMAL], "前置:该日本应在池内,否则本测试会因错误理由通过"
    assert not m["can_buy"].loc[d, NORMAL], "一字涨停仍被判为可买"
    assert m["can_sell"].loc[d, NORMAL], "一字涨停被错误地禁止卖出"


def test_masks_limit_down_blocks_sell_not_buy(conn):
    p = _panel(conn)
    m = loader.tradable_masks(p)
    d = pd.Timestamp(LIMIT_DOWN_DAY)
    assert m["in_universe"].loc[d, NORMAL], "前置:该日本应在池内,否则本测试会因错误理由通过"
    assert not m["can_sell"].loc[d, NORMAL], "一字跌停仍被判为可卖"
    assert m["can_buy"].loc[d, NORMAL], "一字跌停被错误地禁止买入"


def test_masks_are_boolean_and_aligned(conn):
    p = _panel(conn)
    m = loader.tradable_masks(p)
    for k, v in m.items():
        assert v.dtypes.eq(bool).all(), f"{k} 不是布尔表"
        assert v.index.equals(p["close"].index) and v.columns.equals(p["close"].columns)


# ---------- 退市标记与质量报告 ----------

def test_mark_delisted(conn):
    """退市股(K行止于 2024-03-01)在 as_of=2024-10-31、gap=60 下必须被标记;正常股不能被误标。"""
    n = sync.mark_delisted(conn, AS_OF, gap_days=60)
    assert n >= 1
    rows = dict(conn.execute("SELECT code, delisted FROM stocks").fetchall() and
                [(r[0], r[1]) for r in conn.execute("SELECT code, delisted FROM stocks")])
    assert rows[DELISTED] == 1, "退市股未被标记"
    assert rows[NORMAL] == 0, "正常股被误标为退市"


def test_data_quality_report_keys(conn):
    rep = sync.data_quality_report(conn)
    assert isinstance(rep, dict) and rep
    joined = " ".join(rep.keys()).lower()
    for kw in ("stock", "daily", "date"):
        assert kw in joined, f"质量报告缺少与 {kw} 相关的键:{list(rep.keys())}"


# ---------- 存储层 ----------

def test_upsert_many_is_idempotent(conn):
    before = conn.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
    row = dict(conn.execute("SELECT * FROM daily LIMIT 1").fetchone())
    db.upsert_many(conn, "daily", [row])
    db.upsert_many(conn, "daily", [row])
    after = conn.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
    assert after == before, "重复 upsert 产生了重复行"


def test_upsert_many_empty_returns_zero(conn):
    assert db.upsert_many(conn, "daily", []) == 0


def test_last_date(conn):
    assert db.last_date(conn, "daily", DELISTED) == DELISTED_LAST
    assert db.last_date(conn, "daily", "sz.999999") is None


# ---------- 回归:累计复权因子被截断造成的假跳变(2026-09-20 实测踩坑) ----------

def test_apply_adjust_warns_when_factor_history_truncated():
    """baostock 的 backAdjustFactor 是"自上市累计"值。

    若只同步窗口内的因子,窗口前段无记录按 1.0、到首个因子日突然跳到累计值(实测 sh.600000
    在 2024-07-18 假跳 +1139%),会毁掉所有收益率计算。此处要求 apply_adjust 必须告警。
    """
    dates = pd.bdate_range("2024-01-02", "2024-08-30")
    close = pd.DataFrame({"sh.600000": np.linspace(6.6, 8.8, len(dates))}, index=dates)
    truncated = pd.DataFrame([{"code": "sh.600000", "date": "2024-07-18", "backAdjustFactor": 12.38831}])
    with pytest.warns(UserWarning, match="复权因子历史不完整"):
        adj = loader.apply_adjust(close, truncated)
    ratio = (adj["sh.600000"] / close["sh.600000"])
    assert ratio.iloc[0] == pytest.approx(1.0)        # 截断前被迫按 1.0
    assert ratio.iloc[-1] == pytest.approx(12.38831)  # 跳变确实存在(这正是要告警的原因)


def test_apply_adjust_no_warning_with_full_history():
    """补齐窗口前的因子后:不告警,且复权收益率与原始收益率一致(除权日除外)。"""
    dates = pd.bdate_range("2024-01-02", "2024-08-30")
    close = pd.DataFrame({"sh.600000": np.linspace(6.6, 8.8, len(dates))}, index=dates)
    full = pd.DataFrame([
        {"code": "sh.600000", "date": "1999-11-10", "backAdjustFactor": 1.0},
        {"code": "sh.600000", "date": "2023-07-20", "backAdjustFactor": 11.9498},
        {"code": "sh.600000", "date": "2024-07-18", "backAdjustFactor": 12.38831},
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # 任何告警都会让本测试红
        adj = loader.apply_adjust(close, full)
    r_close = close["sh.600000"].pct_change()
    r_adj = adj["sh.600000"].pct_change()
    assert r_adj.max() < 0.5, "补齐历史后仍出现 >50% 的单日假涨幅"
    non_exdiv = r_adj.index != pd.Timestamp("2024-07-18")
    assert np.allclose(r_adj[non_exdiv].dropna(), r_close[non_exdiv].dropna(), atol=1e-9), \
        "非除权日的复权收益率应与原始收益率相同"


def test_sync_adj_uses_full_history_not_since():
    """sync_adj 必须忽略 since 拉全历史,否则断点续传会让历史因子永远补不回来。"""
    import inspect
    src = inspect.getsource(sync.sync_adj)
    assert "ADJ_FULL_HISTORY_SINCE" in src, "sync_adj 未使用全历史起点"
    assert "full_refresh=True" in src, "sync_adj 未关闭断点续传(会补不回历史因子)"


# ---------- 回归:增量同步必须同时补"头部缺口"(2026-09-21 实测踩坑) ----------

def test_pending_ranges_backfills_head_gap(conn):
    """只补尾部会让"先用晚起点同步过"的股票永远补不回更早历史。

    实测:30 只按 2024-01-01 同步过的股票,在全量 2015 同步时被全部 skip,
    最终得到前后不一致的面板(有的股票只有 2024 起、有的从 2015 起)。
    """
    from ashare_forge.data.sync import _pending_ranges
    code = NORMAL  # 库内已有 2024-01-02 ~ 2024-10-31

    # 起点更早 → 必须产出头部缺口区间
    r = _pending_ranges(conn, "daily", code, "2015-01-01", AS_OF)
    assert r, "改用更早起点时没有产出任何待拉区间(历史永远补不回来)"
    assert r[0][0] == "2015-01-01" and r[0][1] < "2024-01-02", f"头部缺口区间不对: {r}"

    # 区间已完整 → 不应重复拉
    assert _pending_ranges(conn, "daily", code, "2024-01-02", AS_OF) == []

    # 末端需要延伸 → 产出尾部增量
    r2 = _pending_ranges(conn, "daily", code, "2024-01-02", "2026-09-18")
    assert r2 and r2[-1][1] == "2026-09-18" and r2[-1][0] > AS_OF

    # 库内无该股 → 整段
    assert _pending_ranges(conn, "daily", "sz.999999", "2015-01-01", AS_OF) == [("2015-01-01", AS_OF)]

    # full_refresh → 无视库内已有,整段重拉(复权因子用)
    assert _pending_ranges(conn, "daily", code, "2015-01-01", AS_OF, full_refresh=True) \
        == [("2015-01-01", AS_OF)]


# ---------- 回归:上市天数必须以真实上市日期为起点(2026-09-21 审查 P0) ----------

def test_listing_age_uses_real_ipo_date(conn):
    """库从窗口起点开始时,老股票不得被当成次新。

    原实现从"库内第一根K线"起算:库从 2015-01-05 开始 → 平安银行(1991年上市)被当成新股,
    样本首 119 个交易日全市场在池恒为 0,2015 股灾起点丢失。
    """
    from ashare_forge.data.sync import _ensure_stock_date_columns
    _ensure_stock_date_columns(conn)
    # 正常股很早就上市;次新股的上市日取其首根K线
    conn.execute("UPDATE stocks SET ipo_date='1991-04-03' WHERE code=?", (NORMAL,))
    conn.execute("UPDATE stocks SET ipo_date=? WHERE code=?", (_new_stock_window()[0], NEW))
    conn.commit()
    # 交易日历:用合成指数的日期
    p = _panel(conn)
    age = p["listing_age"]
    first = age.index[0]
    assert age.loc[first, NORMAL] >= 120, (
        f"1991年上市的老股在窗口首日被算成上市 {age.loc[first, NORMAL]} 天——仍在按库内首根K线起算")
    m = loader.tradable_masks(p)
    assert m["in_universe"].loc[first, NORMAL], "老股在窗口首日被次新规则误剔除"
    # 真次新股仍须被剔除
    d_new = pd.Timestamp(_new_stock_window()[-1])
    assert age.loc[d_new, NEW] < 120 and not m["in_universe"].loc[d_new, NEW]


# ---------- 合同 v2:交易规则事实修正 ----------

def test_v2_chinext_limit_depends_on_reform_date():
    from ashare_forge.contract import limit_pct
    assert limit_pct("sz.300001", False, "2019-06-03") == pytest.approx(0.10)
    assert limit_pct("sz.300001", False, "2020-08-24") == pytest.approx(0.20)
    assert limit_pct("sz.300001", True, "2019-06-03") == pytest.approx(0.10)
    assert limit_pct("sh.600000", True, "2019-06-03") == pytest.approx(0.05)


def test_v2_stamp_tax_halved_on_2023_08_28():
    from ashare_forge.contract import stamp_tax_series
    s = stamp_tax_series(pd.to_datetime(["2019-01-02", "2023-08-25", "2023-08-28", "2024-01-02"]))
    assert s.tolist() == pytest.approx([0.001, 0.001, 0.0005, 0.0005])


def test_v2_limit_price_rounded_to_cent():
    """preclose 10.03:未取整涨停 11.033,交易所涨停价 11.03。开盘 11.03 即封板,必须不可买(v1 漏判)。
    创业板 2019 年按 10%:preclose 20.00 开盘 22.00 封板;v1 按 20% 判定会漏。"""
    from ashare_forge.data import loader
    d = pd.to_datetime(["2019-06-03", "2019-06-04"])
    cols = ["sh.600000", "sz.300001", "sh.600001"]
    pre = pd.DataFrame([[10.00, 20.00, 10.03], [10.03, 20.00, 10.07]], index=d, columns=cols)
    opn = pd.DataFrame([[10.00, 20.00, 10.03], [11.03, 22.00, 9.06]], index=d, columns=cols)
    one = pd.DataFrame(1.0, index=d, columns=cols)
    panel = {"tradestatus": one, "isST": one * 0, "open": opn, "preclose": pre,
             "float_mcap": one * 1e10, "listing_age": one * 1000}
    m = loader.tradable_masks(panel)
    assert not bool(m["can_buy"].loc["2019-06-04", "sh.600000"]), "开盘=取整涨停价 11.03 未判为不可买"
    assert not bool(m["can_buy"].loc["2019-06-04", "sz.300001"]), "2019 创业板按 10% 封板未判为不可买"
    assert not bool(m["can_sell"].loc["2019-06-04", "sh.600001"]), "10.07×0.9=9.063→跌停价 9.06 未判为不可卖"
    assert bool(m["can_buy"].loc["2019-06-03", "sh.600000"])
