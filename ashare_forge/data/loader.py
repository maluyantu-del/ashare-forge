"""面板加载器:把 SQLite 里的长表变成研究用的 {field: 宽表},并执行合同里的可交易性规则。

本模块修了常见做法的这些坑:
1. **保留段靠"注意事项"提醒** → 这里在数据层强制:contract.holdout.cutoff 及之后的日期一律不返回,
   只有显式 unlock_holdout=True 才放行,且每次放行都往 meta.holdout_unlocked_log 追加一条含 UTC 时间的台账。
2. **直接用未复权 close 算收益** → 除权日出现假跳空。这里按后复权因子(backAdjustFactor)以"生效日起前向填充"
   的口径得到 adj_close;没有因子记录的日期/股票视为 1.0,而不是 NaN 把整列收益打穿。
3. **次新股只按窗口内数据数天数** → 老股在窗口开头前 120 个交易日被误判成次新。这里的上市天数从该股在库中的
   **首个K行**起算(窗口之前的行数用 SQL 聚合补偿),与加载窗口无关。
4. **涨跌停用当日涨幅判定** → 把"触板回落、其实能成交"的日子也当成不可交易。这里按合同的一字板口径:
   open >= preclose*(1+limit)-1e-6 才算不可买,open <= preclose*(1-limit)+1e-6 才算不可卖;limit 按板块/ST 取合同值。
5. **停牌日 turn=0 直接除** → float_mcap 变成 inf 或 0。这里 turn<=0 或 NaN 一律给 NaN。
6. **逐列 pivot 宽表** → 全市场十几列要重复分组十几次。这里 (date,code) 设索引后一次 unstack。
7. **卖出也套用 ST/微盘剔除** → 已持仓的股票变 ST 后"卖不掉"。这里 can_sell 只看停牌与一字跌停。

口径说明:
- 日期入库为 'YYYY-MM-DD' 字符串,加载后转 DatetimeIndex;end 含当日(<=),holdout cutoff 不含(<)。
- listing_age 在该股首个K行当日记 1(该日本身算一个已上市交易日),第 n 个K行记 n。
- 宽表中没有K行的 (date,code) 单元为 NaN;布尔表中对应位置一律 False。
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import sqlite3
import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from ..contract import board_of, limit_pct, load_contract
from .db import get_meta, set_meta

# daily 表的全部数值列(顺序即 data_schema.sql 中的顺序)
DAILY_NUMERIC_COLS: tuple[str, ...] = (
    "open", "high", "low", "close", "preclose",
    "volume", "amount", "turn",
    "tradestatus", "pctChg",
    "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "isST",
)
# 派生字段及其依赖的原始列
DERIVED_FIELDS: dict[str, tuple[str, ...]] = {
    "adj_close": ("close",),
    "float_mcap": ("close", "volume", "turn"),
    "listing_age": (),
}
DEFAULT_FIELDS: tuple[str, ...] = DAILY_NUMERIC_COLS + ("adj_close", "float_mcap", "listing_age")
HOLDOUT_LOG_KEY = "holdout_unlocked_log"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ----------------------------------------------------------------------------- 内部工具
def _check_date(s: str, name: str) -> str:
    if not isinstance(s, str) or not _DATE_RE.match(s):
        raise ValueError(f"{name} 必须是 'YYYY-MM-DD' 字符串,收到 {s!r}")
    _dt.date.fromisoformat(s)  # 非法日期(如 2024-13-40)在这里抛 ValueError
    return s


def _holdout_cutoff() -> str:
    return str(load_contract()["holdout"]["cutoff"])


def _effective_bounds(start: str, end: str | None, unlock_holdout: bool) -> tuple[str, str | None, str | None]:
    """返回 (start, end_inclusive_or_None, holdout_exclusive_or_None)。"""
    start = _check_date(start, "start")
    if end is not None:
        end = _check_date(end, "end")
        if end < start:
            raise ValueError(f"end({end}) 早于 start({start})")
    cutoff = None if unlock_holdout else _holdout_cutoff()
    return start, end, cutoff


def _log_holdout_unlock(conn: sqlite3.Connection, func: str, **kw) -> None:
    """往 meta.holdout_unlocked_log 追加一条 JSON 行(含 UTC 时间),不覆盖历史。"""
    record = {"ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "func": func}
    record.update({k: v for k, v in kw.items() if v is not None})
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    existing = get_meta(conn, HOLDOUT_LOG_KEY, None)
    value = f"{existing}\n{line}" if existing else line
    set_meta(conn, HOLDOUT_LOG_KEY, value)
    try:
        conn.commit()
    except sqlite3.Error:  # set_meta 已自行提交或 conn 处于自动提交模式时无需处理
        pass


def _query_df(conn: sqlite3.Connection, sql: str, params: Iterable = ()) -> pd.DataFrame:
    """用元组游标读取(绕开 conn.row_factory=sqlite3.Row),按 cursor.description 命名列。"""
    cur = conn.cursor()
    cur.row_factory = None
    cur.execute(sql, tuple(params))
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return pd.DataFrame.from_records(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _date_filter(col: str, start: str, end: str | None, cutoff: str | None) -> tuple[str, list]:
    clauses, params = [f"{col} >= ?"], [start]
    if end is not None:
        clauses.append(f"{col} <= ?")
        params.append(end)
    if cutoff is not None:
        clauses.append(f"{col} < ?")
        params.append(cutoff)
    return " AND ".join(clauses), params


def _empty_wide() -> pd.DataFrame:
    return pd.DataFrame(index=pd.DatetimeIndex([], name="date"), columns=pd.Index([], name="code"), dtype="float64")


def _to_wide_from_long(long: pd.DataFrame, value_cols: list[str]) -> dict[str, pd.DataFrame]:
    """长表 (code,date,cols...) → {col: 宽表},一次 unstack。"""
    if long.empty:
        return {c: _empty_wide() for c in value_cols}
    idx = long.set_index(["date", "code"])[value_cols]
    wide_all = idx.unstack("code")  # columns: MultiIndex (field, code)
    wide_all = wide_all.sort_index(axis=0).sort_index(axis=1)
    out = {}
    for c in value_cols:
        w = wide_all[c]
        w.index.name = "date"
        w.columns.name = "code"
        out[c] = w.astype("float64")
    return out


def _bool_frame(x: pd.DataFrame) -> pd.DataFrame:
    """把可能含 NaN 的比较结果压成纯 bool(NaN→False)。"""
    return x.fillna(False).astype(bool) if x.dtypes.map(lambda d: d != bool).any() else x


# ----------------------------------------------------------------------------- 公开函数
def apply_adjust(close_wide: pd.DataFrame, adj_long: pd.DataFrame) -> pd.DataFrame:
    """后复权:adj_close = close * backAdjustFactor,因子按 (code,date) 从生效日起前向填充到每日,缺失视为 1.0。

    adj_long 至少含列 code, date, backAdjustFactor;date 可为字符串或 datetime。
    生效日早于 close_wide 首日的因子照样向后携带;生效日不是交易日也能正确落到下一个交易日。
    """
    if close_wide.empty:
        return close_wide.copy()
    dates = pd.DatetimeIndex(close_wide.index)
    codes = pd.Index(close_wide.columns)

    if adj_long is None or len(adj_long) == 0:
        return close_wide.astype("float64").copy()
    need = {"code", "date", "backAdjustFactor"}
    if not need.issubset(adj_long.columns):
        raise ValueError(f"adj_long 缺列: {sorted(need - set(adj_long.columns))}")

    adj = adj_long.loc[adj_long["code"].isin(codes), ["code", "date", "backAdjustFactor"]].copy()
    if adj.empty:
        return close_wide.astype("float64").copy()
    adj["date"] = pd.to_datetime(adj["date"])
    adj["backAdjustFactor"] = pd.to_numeric(adj["backAdjustFactor"], errors="coerce")
    adj = adj.dropna(subset=["backAdjustFactor"])
    adj = adj.sort_values(["code", "date"]).drop_duplicates(["code", "date"], keep="last")
    # 只保留 close_wide 末日及之前的因子(之后的因子对本窗口无意义,也避免保留段信息混入)
    adj = adj[adj["date"] <= dates.max()]
    if adj.empty:
        return close_wide.astype("float64").copy()

    # 防守:baostock 的 backAdjustFactor 是"自上市累计"值。若某股在窗口首日之前没有任何因子记录,
    # 却在窗口内出现因子,说明历史因子缺失——ffill 会让窗口前段按 1.0、到首个因子日突然跳到累计值,
    # 造出几百倍的假涨幅。此处显式告警(根治办法是全历史同步复权因子,见 sync.sync_adj)。
    first_close = close_wide.notna().idxmax()
    _suspect = []
    for _c in adj["code"].unique():
        _rows = adj.loc[adj["code"] == _c]
        _i = _rows["date"].idxmin()
        _f, _v = _rows.at[_i, "date"], float(_rows.at[_i, "backAdjustFactor"])
        # 首条因子=1.0 时,之前按 1.0 处理恰好正确(新股上市后数日才出首条记录属常态,实测 54 只),
        # 不构成假跳变;只有首条因子≠1.0 才说明前段历史被截断。
        if (_c in first_close.index and close_wide[_c].notna().any() and _f > first_close[_c]
                and abs(_v - 1.0) > 1e-9):
            _suspect.append(_c)
    if _suspect:
        warnings.warn(
            f"后复权因子历史不完整:{len(_suspect)} 只股票(如 {_suspect[:3]})在窗口首日之前无因子记录,"
            f"adj_close 会在首个因子日出现假跳变。请用 sync_adj 重新同步全历史复权因子。",
            stacklevel=2)

    fac = adj.set_index(["date", "code"])["backAdjustFactor"].unstack("code")
    union = fac.index.union(dates).sort_values()
    fac = fac.reindex(index=union).ffill()              # 生效日起向后携带
    fac = fac.reindex(index=dates, columns=codes)       # 回到 close 的日历与股票集
    fac = fac.fillna(1.0).astype("float64")             # 无因子记录 = 1.0
    out = close_wide.astype("float64") * fac
    out.index.name = close_wide.index.name
    out.columns.name = close_wide.columns.name
    return out


def derive_float_mcap(close: pd.DataFrame, volume: pd.DataFrame, turn: pd.DataFrame) -> pd.DataFrame:
    """流通市值(元)= close * volume / (turn/100);turn<=0 或 NaN → NaN(停牌日 turn=0 不除零)。"""
    turn_num = turn.astype("float64")
    turn_ok = turn_num.where(turn_num > 0)              # <=0 与 NaN 都变 NaN
    volume, turn_ok = volume.align(turn_ok, join="outer")
    close_a, volume = close.align(volume, join="outer")
    turn_ok = turn_ok.reindex(index=close_a.index, columns=close_a.columns)
    out = close_a.astype("float64") * volume.astype("float64") / (turn_ok / 100.0)
    out = out.replace([np.inf, -np.inf], np.nan)
    out.index.name = "date"
    out.columns.name = "code"
    return out


def listing_age(daily_dates_by_code: dict[str, pd.DatetimeIndex] | pd.DataFrame) -> pd.DataFrame:
    """每 (date,code) 的已上市交易日数:从该股首个K行起算,首个K行记 1,第 n 个K行记 n;无K行处为 NaN。

    输入三选一:
    - dict {code: DatetimeIndex(该股有K行的日期)}
    - 长表 DataFrame,含列 code, date
    - 宽表 DataFrame(index=日期, columns=code),非 NaN 处视为有K行
    注意:这里只能看到传入的日期,若传入的是加载窗口内的数据,窗口前已上市的天数不会被计入;
    load_panel 内部用库中该股首个K行起算,不受窗口影响。
    """
    if isinstance(daily_dates_by_code, dict):
        parts = []
        for code, idx in daily_dates_by_code.items():
            idx = pd.DatetimeIndex(pd.to_datetime(idx)).unique().sort_values()
            parts.append(pd.DataFrame({"code": code, "date": idx}))
        long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["code", "date"])
    elif isinstance(daily_dates_by_code, pd.DataFrame):
        df = daily_dates_by_code
        if {"code", "date"}.issubset(df.columns):
            long = df[["code", "date"]].copy()
            long["date"] = pd.to_datetime(long["date"])
        else:  # 宽表:非 NaN 处有K行
            present = df.notna()
            st = present.stack()
            st = st[st]
            long = pd.DataFrame({"date": pd.to_datetime(st.index.get_level_values(0)),
                                 "code": st.index.get_level_values(1).astype(str)})
    else:
        raise TypeError("listing_age 需要 dict[code, DatetimeIndex] 或 DataFrame")

    if long.empty:
        return _empty_wide()
    long = long.drop_duplicates(["code", "date"]).sort_values(["code", "date"]).reset_index(drop=True)
    long["age"] = long.groupby("code", observed=True).cumcount() + 1
    wide = long.set_index(["date", "code"])["age"].unstack("code").sort_index(axis=0).sort_index(axis=1)
    wide.index.name = "date"
    wide.columns.name = "code"
    return wide.astype("float64")


def _listing_age_from_ipo(conn: sqlite3.Connection, dates: pd.DatetimeIndex,
                          codes: pd.Index, long: pd.DataFrame) -> pd.DataFrame:
    """上市交易日数,**以真实上市日期(stocks.ipo_date)为起点**。

    审查发现(P0):原实现从"库内第一根K线"起算。库从 2015-01-05 开始,于是平安银行
    (1991 年上市)在系统里被当成 2015 年新股,**样本首 119 个交易日全市场被判次新、在池恒为 0**,
    合同 2015 股灾危机窗口丢掉了崩盘起点。

    交易日历取自指数日线(sh.000001,自 2005 起);上市早于日历起点的股票视为足够老。
    若 stocks 表尚无 ipo_date(旧库),退回原口径并告警。
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(stocks)")}
    if "ipo_date" not in cols:
        warnings.warn("stocks 表无 ipo_date:上市天数退回'库内首根K线'口径,样本起点附近会误判次新。"
                      "请运行 sync --basic。", stacklevel=3)
        lg = long[["code", "date"]].sort_values(["code", "date"]).reset_index(drop=True)
        lg["age"] = lg.groupby("code", observed=True).cumcount() + 1.0
        age = lg.set_index(["date", "code"])["age"].unstack("code")
        return age.reindex(index=dates, columns=codes).astype("float64")

    cal = pd.DatetimeIndex(pd.to_datetime(
        [r[0] for r in conn.execute("SELECT date FROM index_daily WHERE code='sh.000001' ORDER BY date")]))
    ipo = pd.Series({r[0]: r[1] for r in conn.execute(
        "SELECT code, ipo_date FROM stocks WHERE ipo_date IS NOT NULL")}, dtype="object")
    ipo = pd.to_datetime(ipo.reindex(codes), errors="coerce")

    if len(cal) == 0:
        warnings.warn("交易日历为空(index_daily 无 sh.000001),上市天数以自然日/1.45 近似", stacklevel=3)
        days = (pd.DatetimeIndex(dates).to_numpy()[:, None] - ipo.to_numpy()[None, :]) / np.timedelta64(1, "D")
        return pd.DataFrame(np.floor(days / 1.45) + 1, index=dates, columns=codes).astype("float64")

    pos_date = cal.searchsorted(pd.DatetimeIndex(dates), side="right")          # 截至 d 的交易日数
    pos_ipo = cal.searchsorted(ipo.fillna(pd.Timestamp("2100-01-01")), side="left")
    old = (ipo < cal[0]).to_numpy()                                              # 早于日历起点
    age = pos_date[:, None] - pos_ipo[None, :]
    age = age.astype("float64")
    age[:, old] = 1e5                                                            # 足够老
    age[:, ipo.isna().to_numpy()] = np.nan                                       # 无上市日期:未知
    age[age < 0] = np.nan                                                        # 尚未上市
    return pd.DataFrame(age, index=dates, columns=codes)


def load_panel(conn: sqlite3.Connection, start: str, end: str | None = None, fields: list[str] | None = None,
               unlock_holdout: bool = False) -> dict[str, pd.DataFrame]:
    """返回 {field: 宽表}(index=DatetimeIndex 升序, columns=code 升序)。

    fields 缺省 = daily 全部数值列 + 'adj_close' + 'float_mcap' + 'listing_age'。
    holdout 强制:contract.holdout.cutoff 及之后的日期一律不返回;unlock_holdout=True 才放行,并写 meta 台账。
    """
    start, end, cutoff = _effective_bounds(start, end, unlock_holdout)
    if unlock_holdout:
        _log_holdout_unlock(conn, "load_panel", start=start, end=end, fields=list(fields) if fields else None)

    fields = list(DEFAULT_FIELDS) if fields is None else list(dict.fromkeys(fields))
    allowed = set(DAILY_NUMERIC_COLS) | set(DERIVED_FIELDS)
    bad = [f for f in fields if f not in allowed]
    if bad:
        raise ValueError(f"未知字段 {bad};可用: {sorted(allowed)}")

    raw_needed: list[str] = []
    for f in fields:
        for c in ((f,) if f in DAILY_NUMERIC_COLS else DERIVED_FIELDS[f]):
            if c not in raw_needed:
                raw_needed.append(c)

    where, params = _date_filter("date", start, end, cutoff)
    sel = ", ".join(["code", "date"] + raw_needed)
    long = _query_df(conn, f"SELECT {sel} FROM daily WHERE {where} ORDER BY code, date", params)
    if long.empty:
        warnings.warn(f"load_panel: daily 表在 [{start}, {end or '∞'}) ∩ holdout({cutoff}) 内没有任何行", stacklevel=2)
        return {f: _empty_wide() for f in fields}

    long["date"] = pd.to_datetime(long["date"], format="%Y-%m-%d")
    long["code"] = long["code"].astype(str)
    for c in raw_needed:
        long[c] = pd.to_numeric(long[c], errors="coerce")

    wide = _to_wide_from_long(long, raw_needed)
    dates = wide[raw_needed[0]].index if raw_needed else pd.DatetimeIndex(sorted(long["date"].unique()), name="date")
    codes = wide[raw_needed[0]].columns if raw_needed else pd.Index(sorted(long["code"].unique()), name="code")

    out: dict[str, pd.DataFrame] = {}
    for f in fields:
        if f in DAILY_NUMERIC_COLS:
            out[f] = wide[f]
        elif f == "float_mcap":
            out[f] = derive_float_mcap(wide["close"], wide["volume"], wide["turn"])
        elif f == "adj_close":
            adj = _query_df(conn,
                            "SELECT code, date, backAdjustFactor FROM adj_factor WHERE date <= ? ORDER BY code, date",
                            [dates.max().strftime("%Y-%m-%d")])
            out[f] = apply_adjust(wide["close"], adj)
        elif f == "listing_age":
            out[f] = _listing_age_from_ipo(conn, dates, codes, long)

    # 统一对齐(所有表同一 index/columns)
    for f in fields:
        if not out[f].index.equals(dates) or not out[f].columns.equals(codes):
            out[f] = out[f].reindex(index=dates, columns=codes)
    return out


def tradable_masks(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """返回 {'in_universe','can_buy','can_sell'} 三张布尔宽表(合同 tradability_rules 由代码执行)。

    in_universe = 非停牌 & 非ST & 上市≥new_listing_days & float_mcap≥min_float_mcap_yuan & 当日有K行
    can_buy     = in_universe & 非一字涨停(open >= preclose*(1+limit)-1e-6 为不可买)
    can_sell    = 非停牌 & 非一字跌停(open <= preclose*(1-limit)+1e-6 为不可卖);卖出不受 ST/微盘限制
    limit 按 contract.limit_pct(code, is_st) 逐单元取值(主板 ST 日用 5%)。
    """
    need = ["tradestatus", "isST", "open", "preclose"]
    missing = [k for k in need if k not in panel]
    if missing:
        raise KeyError(f"tradable_masks 需要 panel 含 {need},缺 {missing}")
    tradestatus, is_st, open_, preclose = (panel[k] for k in need)
    dates, codes = tradestatus.index, tradestatus.columns

    if "float_mcap" in panel:
        fmc = panel["float_mcap"]
    elif {"close", "volume", "turn"}.issubset(panel):
        fmc = derive_float_mcap(panel["close"], panel["volume"], panel["turn"])
    else:
        raise KeyError("tradable_masks 需要 panel 含 float_mcap(或 close/volume/turn 以便派生)")

    if "listing_age" in panel:
        age = panel["listing_age"]
    else:
        warnings.warn("panel 无 listing_age,改用窗口内K行数估算(窗口前已上市的天数不计,老股可能被误判为次新)",
                      stacklevel=2)
        age = listing_age(tradestatus)

    fmc = fmc.reindex(index=dates, columns=codes)
    age = age.reindex(index=dates, columns=codes)
    open_ = open_.reindex(index=dates, columns=codes)
    preclose = preclose.reindex(index=dates, columns=codes)
    is_st = is_st.reindex(index=dates, columns=codes)

    ex = load_contract()["tradability_rules"]["exclusions"]
    new_days = float(ex["new_listing_days"])
    min_mcap = float(ex["min_float_mcap_yuan"])

    has_k = tradestatus.notna() | open_.notna() | preclose.notna()
    if "close" in panel:
        has_k = has_k | panel["close"].reindex(index=dates, columns=codes).notna()
    not_suspended = _bool_frame((tradestatus == 1))                      # NaN → False
    st_flag = _bool_frame((is_st == 1))
    not_st = ~st_flag
    age_ok = _bool_frame(age >= new_days)
    mcap_ok = _bool_frame(fmc >= min_mcap)

    # 逐单元涨跌停幅度:按股票板块取基准,主板且当日 ST 取 5%;创业板改革日前取 10%(合同 v2)
    base = np.array([limit_pct(c, False) for c in codes], dtype="float64")
    st_lim = np.array([limit_pct(c, True) for c in codes], dtype="float64")
    base_w = pd.DataFrame(np.tile(base, (len(dates), 1)), index=dates, columns=codes)
    st_w = pd.DataFrame(np.tile(st_lim, (len(dates), 1)), index=dates, columns=codes)
    lp = load_contract()["tradability_rules"]["limit_pct"]
    if "chinext_reform_date" in lp:
        pre = pd.DatetimeIndex(dates) < pd.Timestamp(str(lp["chinext_reform_date"]))
        cx = [j for j, c in enumerate(codes) if board_of(c) == "chinext"]
        if cx and pre.any():
            pre_val = float(lp.get("chinext_pre_reform", lp["chinext"]))
            base_w.iloc[np.flatnonzero(pre), cx] = pre_val
            st_w.iloc[np.flatnonzero(pre), cx] = pre_val
    limit = base_w.where(~st_flag, st_w)

    # 交易所按"分"四舍五入(round half up)出涨跌停价;v1 用未取整价 -1e-6 比较,
    # 取整向下时开盘封板价低于未取整价而漏判(实测 2019 主板一字涨停漏判约 41%)。
    pc = preclose.astype("float64")
    up_price = np.floor(pc * (1.0 + limit) * 100.0 + 0.5 + 1e-7) / 100.0
    down_price = np.floor(pc * (1.0 - limit) * 100.0 + 0.5 + 1e-7) / 100.0
    up_locked = _bool_frame(open_ >= up_price - 0.005)
    down_locked = _bool_frame(open_ <= down_price + 0.005)

    in_universe = has_k & not_suspended & not_st & age_ok & mcap_ok
    can_buy = in_universe & ~up_locked
    can_sell = has_k & not_suspended & ~down_locked

    out = {"in_universe": in_universe, "can_buy": can_buy, "can_sell": can_sell}
    for k, v in out.items():
        v = v.astype(bool)
        v.index.name, v.columns.name = "date", "code"
        out[k] = v
    return out


def delisted_mask(conn: sqlite3.Connection, dates, codes) -> pd.DataFrame:
    """退市标记宽表:date >= stocks.out_date(真实摘牌日,来自 query_stock_basic)为 True。

    回测据此在摘牌日对"仍未卖出"的持仓做强制清算(合同 v2 delisting 段)。stocks 表无 out_date
    列(旧库)时返回全 False 并告警——此时退市持仓会被冻结在最后价格上(偏乐观)。
    """
    dates, codes = pd.DatetimeIndex(dates), pd.Index(codes)
    out = pd.DataFrame(False, index=dates, columns=codes)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(stocks)")}
    if "out_date" not in cols:
        warnings.warn("stocks 表无 out_date:退市持仓无法强制清算(偏乐观)。请运行 sync --basic。", stacklevel=2)
        return out
    rows = conn.execute("SELECT code, out_date FROM stocks WHERE out_date IS NOT NULL AND out_date != ''").fetchall()
    pos = {c: j for j, c in enumerate(codes)}
    arr = out.to_numpy(copy=True)
    for code, od in rows:
        j = pos.get(code)
        if j is None:
            continue
        try:
            t = pd.Timestamp(str(od))
        except (ValueError, TypeError):
            continue
        arr[dates >= t, j] = True
    return pd.DataFrame(arr, index=dates, columns=codes)


def index_series(conn: sqlite3.Connection, code: str = "sh.000905", start: str = "2005-01-01", end: str | None = None,
                 unlock_holdout: bool = False, field: str = "close") -> pd.Series:
    """指数价格序列(index=DatetimeIndex 升序, name=code),同样受 holdout 约束。

    field="close"(缺省)或 "open"。回测的组合收益是"开盘→次日开盘",基准须用同一时段,
    故回测侧取 open(见 backtest.run_backtest 的 bench_open)。
    """
    if field not in ("close", "open"):
        raise ValueError(f"index_series field 只能是 close/open,收到 {field!r}")
    start, end, cutoff = _effective_bounds(start, end, unlock_holdout)
    if unlock_holdout:
        _log_holdout_unlock(conn, "index_series", code=code, start=start, end=end)
    where, params = _date_filter("date", start, end, cutoff)
    df = _query_df(conn, f"SELECT date, {field} AS v FROM index_daily WHERE code = ? AND {where} ORDER BY date",
                   [code] + params)
    if df.empty:
        warnings.warn(f"index_series: index_daily 中 {code} 在 [{start}, {end or '∞'}) ∩ holdout({cutoff}) 内没有行",
                      stacklevel=2)
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="date"), name=code)
    s = pd.Series(pd.to_numeric(df["v"], errors="coerce").to_numpy(),
                  index=pd.DatetimeIndex(pd.to_datetime(df["date"], format="%Y-%m-%d"), name="date"), name=code)
    return s.sort_index()


# ----------------------------------------------------------------------------- 全收益基准(披露口径)
def dividend_compensation(price_close: pd.Series, tr_close: pd.Series,
                          index: pd.DatetimeIndex | None = None) -> pd.Series:
    """分红补偿序列,与组合收益同一行号约定:第 t 行对应 t → t+1 这一步。

    comp[t] = (全收益 收→收, t→t+1) − (价格指数 收→收, t→t+1)

    指数成分股除息时,价格指数在除息日开盘即反映除息缺口,全收益指数把股息再投资补回。
    组合第 t 行的时段是 t 开盘 → t+1 开盘,除息日 t+1 的开盘缺口正落在这一段内,
    因此"t+1 为除息日"的补偿记在第 t 行。收盘→收盘的差值在除息日 t+1 这一步取得,
    即 gap[t+1] 平移到第 t 行。

    两条序列先对齐到 index(缺省为 price_close 的日期),**不做前向填充**:任一端缺值的步长
    补偿为 NaN,由调用方决定按 0 处理并披露(前向填充会把缺失日的价格变动错记为分红)。
    最后一行没有 t+1,恒为 NaN。
    """
    idx = pd.DatetimeIndex(index if index is not None else price_close.index)
    p = pd.Series(price_close).reindex(idx).astype("float64")
    tr = pd.Series(tr_close).reindex(idx).astype("float64")
    gap = (tr / tr.shift(1) - 1.0) - (p / p.shift(1) - 1.0)     # 第 d 行 = d-1 收 → d 收
    comp = gap.shift(-1)                                         # 第 t 行 = t 收 → t+1 收
    comp.index.name = "date"
    return comp.rename("div_comp")


def total_return_open_to_open(price_open: pd.Series, price_close: pd.Series, tr_close: pd.Series,
                              index: pd.DatetimeIndex | None = None) -> pd.Series:
    """全收益"开盘→次日开盘"基准收益(第 t 行 = t 开盘 → t+1 开盘,与 backtest 组合收益同一时段)。

    = 价格指数开→开收益 + dividend_compensation(第 t 行)

    全收益指数(如中证500全收益 H00905)只发布收盘价,不能直接当作 bench_open 使用
    (那样基准会静默退回收盘口径);这里以价格指数的开→开收益为骨架,只把分红部分叠加上去。
    补偿缺失(全收益端缺值)的行按 0 处理,缺失行数记在返回值 attrs["n_comp_missing"]
    (不含天然缺失的最后一行)。价格指数开盘缺失的行保持 NaN,与主基准一致。
    """
    idx = pd.DatetimeIndex(index if index is not None else price_open.index)
    po = pd.Series(price_open).reindex(idx).astype("float64")
    p_oo = po.shift(-1) / po - 1.0
    comp = dividend_compensation(price_close, tr_close, idx)
    n_missing = int(comp.iloc[:-1].isna().sum()) if len(comp) else 0
    out = (p_oo + comp.fillna(0.0)).rename("bench_tr_oo")
    out.index.name = "date"
    out.attrs["n_comp_missing"] = n_missing
    return out


__all__ = [
    "DAILY_NUMERIC_COLS", "DERIVED_FIELDS", "DEFAULT_FIELDS", "HOLDOUT_LOG_KEY",
    "load_panel", "apply_adjust", "derive_float_mcap", "listing_age", "tradable_masks", "index_series",
    "delisted_mask", "dividend_compensation", "total_return_open_to_open",
]
