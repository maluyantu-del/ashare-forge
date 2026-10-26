"""合成数据夹具:每个构造点对应合同 tradability_rules 的一条规则,便于测试"能抓错"。

构造点一览(日期跨 holdout cutoff 2024-09-01 前后):
  sz.000001 正常股(主板,10%)    —— 含 1 个停牌日、1 个一字涨停日、1 个一字跌停日、1 条复权因子生效
  sz.000002 ST 股(主板 ST,5%)   —— isST=1 全程
  sz.300001 次新股(创业板,20%)  —— 仅最后 30 个交易日有 K 行(不足 120 日)
  sz.000003 微盘股(主板)        —— 流通市值 1 亿 < 20 亿门槛
  sh.600000 已退市股            —— K 行止于 2024-03-01
  sh.688001 科创板(20%)         —— 用于逐板块涨跌停幅度校验
"""
from __future__ import annotations

import os
import sqlite3
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ashare_forge.data import db  # noqa: E402

CUTOFF = "2024-09-01"          # 必须与 contracts/ashare.yaml holdout.cutoff 一致
NORMAL, ST, NEW, MICRO, DELISTED, STAR = (
    "sz.000001", "sz.000002", "sz.300001", "sz.000003", "sh.600000", "sh.688001")

# 特例日必须落在"上市满 120 个交易日之后、holdout cutoff 之前",
# 否则会被次新规则先剔除,涨跌停断言就会"因为错误的理由"通过(本测试自身踩过这个坑)。
def _bdays() -> list[str]:
    return [x.strftime("%Y-%m-%d") for x in pd.bdate_range("2024-01-02", "2024-10-31")]


_D = _bdays()
NORMAL_OK_DAY = _D[130]        # 满 131 个交易日,远超 120 门槛
SUSPEND_DAY = _D[135]          # 正常股停牌日
LIMIT_UP_DAY = _D[140]         # 正常股一字涨停日(开盘即 +10%)
LIMIT_DOWN_DAY = _D[141]       # 正常股一字跌停日(开盘即 -10%)
ADJ_DAY = "2024-05-15"         # 复权因子生效日:之前 1.0,之后 2.0
DELISTED_LAST = "2024-03-01"
AS_OF = "2024-10-31"

# 流通市值 = close * volume / (turn/100)
#   大盘:10 * 5e6 / 0.01 = 5e9 元(50 亿 ≥ 20 亿门槛)
#   微盘:10 * 1e5 / 0.01 = 1e8 元(1 亿 < 20 亿门槛)
BIG_VOL, MICRO_VOL, TURN = 5_000_000.0, 100_000.0, 1.0


def _dates() -> list[str]:
    d = pd.bdate_range("2024-01-02", AS_OF)
    return [x.strftime("%Y-%m-%d") for x in d]


def _row(code: str, date: str, *, close=10.0, preclose=10.0, open_=None,
         volume=BIG_VOL, turn=TURN, tradestatus=1, is_st=0) -> dict:
    return {
        "code": code, "date": date,
        "open": close if open_ is None else open_, "high": close, "low": close,
        "close": close, "preclose": preclose,
        "volume": volume, "amount": close * volume, "turn": turn,
        "tradestatus": tradestatus, "pctChg": 0.0,
        "peTTM": 12.0, "pbMRQ": 1.5, "psTTM": 2.0, "pcfNcfTTM": 8.0, "isST": is_st,
    }


def _new_stock_window() -> list[str]:
    """次新股:holdout cutoff 之前的最后 40 个交易日(<120 日门槛,且默认模式下取得到)。"""
    dates = [d for d in _dates() if d < CUTOFF]
    return dates[-40:]


def _build_rows() -> list[dict]:
    dates, rows = _dates(), []
    new_window = set(_new_stock_window())
    for d in dates:
        # 正常股:停牌/一字涨停/一字跌停三个特例日
        if d == SUSPEND_DAY:
            rows.append(_row(NORMAL, d, tradestatus=0, volume=0.0, turn=0.0))
        elif d == LIMIT_UP_DAY:
            rows.append(_row(NORMAL, d, close=11.0, preclose=10.0, open_=11.0))   # 开=前收×1.10
        elif d == LIMIT_DOWN_DAY:
            rows.append(_row(NORMAL, d, close=9.0, preclose=10.0, open_=9.0))     # 开=前收×0.90
        else:
            rows.append(_row(NORMAL, d))
        rows.append(_row(ST, d, is_st=1))
        rows.append(_row(MICRO, d, volume=MICRO_VOL))
        rows.append(_row(STAR, d))
        if d in new_window:                       # 次新股:cutoff 前最后 40 个交易日
            rows.append(_row(NEW, d))
        if d <= DELISTED_LAST:                    # 退市股:K 行早停
            rows.append(_row(DELISTED, d))
    return rows


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.db"))
    db.ensure_schema(c)
    db.upsert_many(c, "daily", _build_rows())
    db.upsert_many(c, "stocks", [
        {"code": NORMAL, "name": "正常", "board": "main", "first_seen": "2024-01-02", "last_seen": AS_OF, "delisted": 0},
        {"code": ST, "name": "ST", "board": "main", "first_seen": "2024-01-02", "last_seen": AS_OF, "delisted": 0},
        {"code": NEW, "name": "次新", "board": "chinext", "first_seen": "2024-07-05", "last_seen": AS_OF, "delisted": 0},
        {"code": MICRO, "name": "微盘", "board": "main", "first_seen": "2024-01-02", "last_seen": AS_OF, "delisted": 0},
        {"code": DELISTED, "name": "退市", "board": "main", "first_seen": "2024-01-02", "last_seen": DELISTED_LAST, "delisted": 0},
        {"code": STAR, "name": "科创", "board": "star", "first_seen": "2024-01-02", "last_seen": AS_OF, "delisted": 0},
    ])
    db.upsert_many(c, "adj_factor", [
        {"code": NORMAL, "date": "2024-01-01", "foreAdjustFactor": 1.0, "backAdjustFactor": 1.0, "adjustFactor": 1.0},
        {"code": NORMAL, "date": ADJ_DAY, "foreAdjustFactor": 1.0, "backAdjustFactor": 2.0, "adjustFactor": 2.0},
    ])
    db.upsert_many(c, "index_daily", [
        {"code": code_, "date": d, "open": 5000.0, "high": 5000.0, "low": 5000.0,
         "close": 5000.0 + i, "volume": 1e9, "amount": 1e12, "pctChg": 0.01}
        for code_ in ("sh.000905", "sh.000001")          # sh.000001 同时充当交易日历
        for i, d in enumerate(_dates())
    ])
    yield c
    c.close()
