"""baostock 数据源封装(AShare Forge 数据层,免 token,公开数据)。

对外接口(签名见 docs/ARCHITECTURE.md,不得改):
    with BaostockSource() as src:
        src.fetch_daily(code, start, end)            # 不复权日线 → daily 表列
        src.fetch_adjust_factors(code, start, end)   # 复权因子 → adj_factor 表列
        src.fetch_all_stock(day)                     # 某日全部证券(不过滤)
        src.fetch_industry()                         # 证监会行业分类(当前快照)
        src.fetch_index_daily(code, start, end)      # 指数日线 → index_daily 表列
    额外提供(非规格要求,给同步层用):fetch_trade_dates / last_trading_day。

它修了常见 baostock 用法的哪些坑:
1. 「失败返回空表」:常见写法 `if rs.error_code != '0': return pd.DataFrame()`,把网络故障伪装成
   "这只股票没数据",同步层就会把它当作已同步完成。本模块任何查询 error_code!="0" 或抛异常都
   指数退避重试 3 次(1s/3s/9s),每次重试前先重登(relogin);耗尽后抛 RuntimeError,消息含
   code、查询名、error_msg。
2. 「只检查一次 error_code」:baostock 每页 2000 行,query_all_stock(约 7400 行)/行业分类
   (约 5500 行)都要翻页;翻页中途出错时 `next()` 只是静默返回 False 并把 error_code 改掉,
   常见写法在循环前检查一次就把截断的半页当完整数据入库。本模块翻页结束后再校验一次 error_code。
3. 「socket 无超时」:baostock 的 socket 从不 settimeout,服务器不响应时 recv 永久挂起,
   进程变僵尸。本模块登录成功后给底层 socket 设超时(默认 60s),超时会被 baostock 内部捕获
   变成 error_code=10002007,进入第 1 条的重登重试。
4. 「非交易日全市场列表为空」:query_all_stock 在周末/节假日返回 error_code=0 + 0 行,
   常被当成"当天没有股票"写进快照。全市场列表为空在语义上不可能是成功,本模块直接抛
   RuntimeError,并提供 last_trading_day(day) 回退到最近交易日。
5. 「字符串数值直接 astype(float)」:baostock 所有字段都是字符串,缺失值是空串 "",
   astype(float) 会炸或把 "" 当 0。本模块数值列一律 pd.to_numeric(errors="coerce"),空串→NaN。
6. 「长区间一次查完」:十年日线一次请求偶发超时/截断,而且失败后整只重来。本模块按自然年分块
   查询后拼接,失败重试的粒度是一年。
7. 「日期格式错时 baostock 返回 None」:query_history_k_data_plus 遇到 "2026/09/01" 这种格式
   会 print 一句然后 return None,常见写法 `rs.error_code` 直接 AttributeError。本模块入口先校验
   'YYYY-MM-DD' 格式(ValueError),并把 None 结果当参数错误处理。
8. 「参数错误也重试」:无效代码(如 xx.999999,error_code=10004011)是客户端参数错误,重试 13 秒
   毫无意义。本模块把 10004xxx 参数类错误码列为不可重试,立即抛 RuntimeError(见 deviations)。

已知限制(照实披露):
- baostock 是进程级全局单 socket,同一进程只能有一个活跃的 BaostockSource;非线程安全。
- 格式合法但不存在的代码(如 sz.999999)服务器返回 error_code=0 + 0 行,与"该区间未上市/
  无交易日"无法区分,fetch_daily 按规格返回带列名的空表;代码是否存在应以 fetch_all_stock 为准。
- tradestatus/isST 在极少数行为空时,按保守口径填(tradestatus→0 停牌、isST→1 视为 ST),即
  "不确定就不可交易",并发出 RuntimeWarning;实测 baostock 日线这两列从不为空。
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import io
import time
import warnings
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import baostock as bs
import baostock.common.context as _bs_context

__all__ = ["BaostockSource", "BaostockError", "BaostockParamError"]

_DATE_FMT = "%Y-%m-%d"

# baostock 客户端/参数类错误码:重试不可能成功,立即抛错(见模块 docstring 第 8 条)
_NON_RETRYABLE_CODES = frozenset({
    "10004005",  # 传入参数为空
    "10004006",  # 参数错误
    "10004007",  # 起始日期格式不正确
    "10004008",  # 截止日期格式不正确
    "10004009",  # 起始日期大于终止日期
    "10004010",  # 日期格式不正确
    "10004011",  # 无效的证券代码
    "10004012",  # 无效的指标
    "10004013",  # 超出日期支持范围
    "10004014",  # 不支持的混合证券品种
    "10004015",  # 不支持的证券代码品种
    "10004018",  # 指标重复
})

# 各返回表的列顺序(与 data_schema.sql 一致)
DAILY_COLUMNS = ["code", "date", "open", "high", "low", "close", "preclose",
                 "volume", "amount", "turn", "tradestatus", "pctChg",
                 "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "isST"]
DAILY_FLOAT_COLUMNS = ["open", "high", "low", "close", "preclose", "volume", "amount",
                       "turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
DAILY_INT_COLUMNS = ["tradestatus", "isST"]
# 空值保守填充:不确定就当不可交易 / 当作 ST(仅在极少数异常行触发,并发 RuntimeWarning)
_FLAG_FILL = {"tradestatus": 0, "isST": 1}

INDEX_COLUMNS = ["code", "date", "open", "high", "low", "close", "volume", "amount", "pctChg"]
INDEX_FLOAT_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "pctChg"]

ADJ_COLUMNS = ["code", "date", "foreAdjustFactor", "backAdjustFactor", "adjustFactor"]
ADJ_FLOAT_COLUMNS = ["foreAdjustFactor", "backAdjustFactor", "adjustFactor"]

ALL_STOCK_COLUMNS = ["code", "tradestatus", "name"]
INDUSTRY_COLUMNS = ["updateDate", "code", "code_name", "industry", "industryClassification"]
TRADE_DATES_COLUMNS = ["calendar_date", "is_trading_day"]


class BaostockError(RuntimeError):
    """baostock 查询在重试耗尽后仍失败(网络/服务端),或返回了语义上不可能成功的结果。"""


class BaostockParamError(BaostockError):
    """baostock 判定为参数错误(无效代码/日期越界等),不重试。"""


# --------------------------------------------------------------------------- 工具函数

def _check_date(value: str, name: str) -> str:
    """校验 'YYYY-MM-DD';格式错是调用方 bug,抛 ValueError(不是网络错误)。"""
    if not isinstance(value, str):
        raise ValueError(f"{name} 必须是 'YYYY-MM-DD' 字符串,收到 {value!r}")
    try:
        _dt.datetime.strptime(value, _DATE_FMT)
    except ValueError as e:
        raise ValueError(f"{name} 必须是 'YYYY-MM-DD' 格式,收到 {value!r}") from e
    return value


def _check_code(code: str) -> str:
    """只做最低限度校验(非空字符串);代码是否有效交给 baostock 判定(10004011 → RuntimeError)。"""
    if not isinstance(code, str) or not code:
        raise ValueError(f"code 必须是 'sz.000001' 形式的非空字符串,收到 {code!r}")
    return code


def _year_chunks(start: str, end: str) -> Iterator[tuple[str, str]]:
    """把 [start, end] 按自然年切成 [(s1,e1),(s2,e2),...];start>end 时不产生任何块。"""
    s = _dt.datetime.strptime(start, _DATE_FMT).date()
    e = _dt.datetime.strptime(end, _DATE_FMT).date()
    if s > e:
        return
    for year in range(s.year, e.year + 1):
        cs = max(s, _dt.date(year, 1, 1))
        ce = min(e, _dt.date(year, 12, 31))
        yield cs.strftime(_DATE_FMT), ce.strftime(_DATE_FMT)


def _to_numeric_cols(df: pd.DataFrame, cols: list[str]) -> None:
    """数值列:空串→NaN,其余 to_numeric(errors='coerce')。原地修改(逐列整列赋值,CoW 安全)。"""
    for c in cols:
        s = df[c]
        if s.dtype == object or str(s.dtype).startswith("str"):
            s = s.mask(s == "", np.nan)
        df[c] = pd.to_numeric(s, errors="coerce").astype("float64")


def _to_flag_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
    """0/1 标志列转 int64;极少数空值按保守口径填并告警(见模块 docstring 已知限制)。"""
    for c in cols:
        s = pd.to_numeric(df[c].mask(df[c] == "", np.nan), errors="coerce")
        n_na = int(s.isna().sum())
        if n_na:
            warnings.warn(
                f"{label}: 列 {c} 有 {n_na} 行空值,按保守口径填 {_FLAG_FILL[c]}(不确定即不可交易)",
                RuntimeWarning, stacklevel=3,
            )
            s = s.fillna(_FLAG_FILL[c])
        df[c] = s.astype("int64")


def _to_str_cols(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        df[c] = df[c].astype("str")


def _frame(fields: list[str], rows: list[list[str]]) -> pd.DataFrame:
    """把 baostock 的 fields+rows 变成 DataFrame;0 行时也保留列名。"""
    if rows:
        return pd.DataFrame(rows, columns=fields)
    return pd.DataFrame({f: pd.Series([], dtype="str") for f in fields})


def _require_fields(fields: list[str], expected: list[str], query: str, code: str) -> None:
    """服务端返回的字段集必须覆盖期望字段,否则说明接口变了,宁可报错也不静默补 NaN。"""
    missing = [f for f in expected if f not in fields]
    if missing:
        raise BaostockError(f"baostock {query} 返回字段缺失 {missing}: code={code} fields={fields}")


# --------------------------------------------------------------------------- 主类

class BaostockSource:
    """baostock 会话封装:上下文管理器 login/logout;所有查询带重登重试;失败抛 RuntimeError。

    参数(都有缺省值,规格未约束):
        retry_delays   每次重试前的等待秒数,长度即重试次数;缺省 (1, 3, 9) = 重试 3 次
        socket_timeout 登录后给 baostock 底层 socket 设的超时秒数;None 表示不设(不建议)
        quiet          True 时吞掉 baostock 的 "login success!" 之类打印
    """

    DAILY_FIELDS = ("date,code,open,high,low,close,preclose,volume,amount,adjustflag,"
                    "turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST")
    INDEX_FIELDS = "date,code,open,high,low,close,volume,amount,pctChg"

    def __init__(self, retry_delays: tuple[float, ...] = (1.0, 3.0, 9.0),
                 socket_timeout: float | None = 60.0, quiet: bool = True) -> None:
        self.retry_delays = tuple(float(d) for d in retry_delays)
        self.socket_timeout = socket_timeout
        self.quiet = quiet
        self.logged_in = False
        self.relogin_count = 0      # 统计:发生过多少次重登(排障用)
        self.retry_count = 0        # 统计:发生过多少次查询重试
        self._sleep: Callable[[float], None] = time.sleep   # 测试可替换

    # ---------------------------------------------------------------- 会话管理
    def __enter__(self) -> "BaostockSource":
        self.login()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.logout()
        return False

    def login(self) -> None:
        """登录;失败按同样的退避重试,耗尽抛 RuntimeError。"""
        last = ""
        for attempt in range(1 + len(self.retry_delays)):
            if attempt:
                self._sleep(self.retry_delays[attempt - 1])
            try:
                lg = self._quiet_call(bs.login)
                if lg is not None and lg.error_code == "0":
                    self.logged_in = True
                    self._apply_socket_timeout()
                    return
                last = (f"error_code={getattr(lg, 'error_code', None)} "
                        f"error_msg={getattr(lg, 'error_msg', None)}")
            except Exception as e:  # noqa: BLE001 - 网络层什么都可能抛
                last = f"exception={e!r}"
        raise BaostockError(f"baostock login 失败(已重试 {len(self.retry_delays)} 次): {last}")

    def logout(self) -> None:
        """登出;失败不抛(会话结束时不需要更多噪音)。"""
        if not self.logged_in:
            return
        try:
            self._quiet_call(bs.logout)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self.logged_in = False

    def _relogin(self) -> None:
        """重试前重建会话:关掉旧 socket(不等它超时)→ 重新 login。登录失败不抛,留给下一次查询暴露。"""
        self.relogin_count += 1
        sock = getattr(_bs_context, "default_socket", None)
        if sock is not None:
            try:
                sock.settimeout(5.0)          # 旧连接若已半死,logout 最多等 5 秒
                self._quiet_call(bs.logout)   # bs.logout 内部会 close socket
            except Exception:  # noqa: BLE001
                pass
            try:
                sock.close()
            except Exception:  # noqa: BLE001
                pass
        self.logged_in = False
        try:
            lg = self._quiet_call(bs.login)
            if lg is not None and lg.error_code == "0":
                self.logged_in = True
                self._apply_socket_timeout()
        except Exception:  # noqa: BLE001
            pass

    def _quiet_call(self, fn: Callable, *args, **kwargs):
        if not self.quiet:
            return fn(*args, **kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args, **kwargs)

    def _apply_socket_timeout(self) -> None:
        if self.socket_timeout is None:
            return
        sock = getattr(_bs_context, "default_socket", None)
        if sock is not None:
            try:
                sock.settimeout(float(self.socket_timeout))
            except Exception:  # noqa: BLE001
                pass

    # ---------------------------------------------------------------- 带重试的底层查询
    def _query(self, query_name: str, fn: Callable, *args, code: str = "") -> tuple[list[str], list[list[str]]]:
        """执行一次 baostock 查询并翻完所有页,返回 (fields, rows)。

        error_code!="0" 或抛异常 → 等待 retry_delays[i] → relogin → 重试;耗尽抛 BaostockError(RuntimeError)。
        参数类错误码(_NON_RETRYABLE_CODES)与返回 None 直接抛 BaostockParamError(RuntimeError),不重试。
        """
        n_retry = len(self.retry_delays)
        last = ""
        for attempt in range(1 + n_retry):
            if attempt:
                self.retry_count += 1
                self._sleep(self.retry_delays[attempt - 1])
                self._relogin()
            try:
                rs = fn(*args)
                if rs is None:
                    raise BaostockParamError(
                        f"baostock {query_name} 返回 None(通常是日期格式错误): code={code} args={args}")
                if rs.error_code != "0":
                    if rs.error_code in _NON_RETRYABLE_CODES:
                        raise BaostockParamError(
                            f"baostock {query_name} 参数错误(不重试): code={code} "
                            f"error_code={rs.error_code} error_msg={rs.error_msg} args={args}")
                    last = f"error_code={rs.error_code} error_msg={rs.error_msg}"
                    continue
                rows: list[list[str]] = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rs.error_code != "0":  # 翻页中途失败:next() 只会静默返回 False
                    last = (f"翻页中断 error_code={rs.error_code} error_msg={rs.error_msg}"
                            f"(已收到 {len(rows)} 行,丢弃)")
                    continue
                return list(rs.fields), rows
            except BaostockParamError:
                raise
            except Exception as e:  # noqa: BLE001
                last = f"exception={e!r}"
        raise BaostockError(
            f"baostock {query_name} 失败,已重试 {n_retry} 次: code={code} args={args} last={last}")

    def _fetch_k_by_year(self, code: str, fields: str, start: str, end: str) -> pd.DataFrame:
        """按自然年分块调用 query_history_k_data_plus(不复权 adjustflag='3')并拼接;0 行时保留列名。"""
        parts: list[pd.DataFrame] = []
        field_list = fields.split(",")
        for cs, ce in _year_chunks(start, end):
            got_fields, rows = self._query(
                "query_history_k_data_plus", bs.query_history_k_data_plus,
                code, fields, cs, ce, "d", "3", code=code)
            _require_fields(got_fields, field_list, "query_history_k_data_plus", code)
            if rows:
                parts.append(pd.DataFrame(rows, columns=got_fields))
        if parts:
            return pd.concat(parts, ignore_index=True)
        return _frame(field_list, [])

    # ---------------------------------------------------------------- 规格接口
    def fetch_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        """不复权日线(adjustflag='3'),按自然年分块查询后拼接。

        返回列 = daily 表列(DAILY_COLUMNS,去掉 adjustflag);date 为 'YYYY-MM-DD' 字符串;
        数值列 float(空串→NaN);tradestatus/isST 为 int。start>end 或区间无 K 行 → 带列名的空表。
        """
        _check_code(code)
        _check_date(start, "start")
        _check_date(end, "end")
        raw = self._fetch_k_by_year(code, self.DAILY_FIELDS, start, end)
        df = raw.drop(columns=["adjustflag"]).loc[:, DAILY_COLUMNS].copy()
        _to_str_cols(df, ["code", "date"])
        _to_numeric_cols(df, DAILY_FLOAT_COLUMNS)
        _to_flag_cols(df, DAILY_INT_COLUMNS, f"fetch_daily {code}")
        df = (df.drop_duplicates(subset=["code", "date"], keep="last")
                .sort_values(["code", "date"], kind="stable")
                .reset_index(drop=True))
        return df

    def fetch_adjust_factors(self, code: str, start: str, end: str) -> pd.DataFrame:
        """复权因子:列 code,date(dividOperateDate→date),foreAdjustFactor,backAdjustFactor,adjustFactor。

        区间内无除权事件 → 带列名的空表(这是合法结果,不是失败)。
        """
        _check_code(code)
        _check_date(start, "start")
        _check_date(end, "end")
        fields, rows = self._query("query_adjust_factor", bs.query_adjust_factor, code, start, end, code=code)
        _require_fields(fields, ["code", "dividOperateDate"] + ADJ_FLOAT_COLUMNS, "query_adjust_factor", code)
        df = _frame(fields, rows).rename(columns={"dividOperateDate": "date"}).loc[:, ADJ_COLUMNS].copy()
        _to_str_cols(df, ["code", "date"])
        _to_numeric_cols(df, ADJ_FLOAT_COLUMNS)
        df = (df.drop_duplicates(subset=["code", "date"], keep="last")
                .sort_values(["code", "date"], kind="stable")
                .reset_index(drop=True))
        return df

    def fetch_all_stock(self, day: str) -> pd.DataFrame:
        """某日全部证券列表(含指数/基金,不过滤):列 code,tradestatus(int),name。

        非交易日 baostock 返回 0 行,这在语义上不可能是"成功",抛 RuntimeError;
        调用方可先用 last_trading_day(day) 回退到最近交易日。
        """
        _check_date(day, "day")
        fields, rows = self._query("query_all_stock", bs.query_all_stock, day, code=day)
        _require_fields(fields, ["code", "tradeStatus", "code_name"], "query_all_stock", day)
        if not rows:
            raise BaostockError(
                f"baostock query_all_stock 返回 0 行: day={day} 很可能不是交易日(周末/节假日),"
                f"请改用 last_trading_day('{day}') 得到的最近交易日再查")
        df = (_frame(fields, rows)
              .rename(columns={"tradeStatus": "tradestatus", "code_name": "name"})
              .loc[:, ALL_STOCK_COLUMNS].copy())
        _to_str_cols(df, ["code", "name"])
        _to_flag_cols(df, ["tradestatus"], f"fetch_all_stock {day}")
        return df.drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)

    def fetch_industry(self) -> pd.DataFrame:
        """证监会行业分类(当前快照,非 PIT):列 updateDate,code,code_name,industry,industryClassification。"""
        fields, rows = self._query("query_stock_industry", bs.query_stock_industry, code="*")
        _require_fields(fields, INDUSTRY_COLUMNS, "query_stock_industry", "*")
        if not rows:
            raise BaostockError("baostock query_stock_industry 返回 0 行:行业分类表不可能为空,视为失败")
        df = _frame(fields, rows).loc[:, INDUSTRY_COLUMNS].copy()
        _to_str_cols(df, INDUSTRY_COLUMNS)
        return df.reset_index(drop=True)

    def fetch_index_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        """指数日线:列 code,date,open,high,low,close,volume,amount,pctChg;同样按自然年分块。"""
        _check_code(code)
        _check_date(start, "start")
        _check_date(end, "end")
        raw = self._fetch_k_by_year(code, self.INDEX_FIELDS, start, end)
        df = raw.loc[:, INDEX_COLUMNS].copy()
        _to_str_cols(df, ["code", "date"])
        _to_numeric_cols(df, INDEX_FLOAT_COLUMNS)
        df = (df.drop_duplicates(subset=["code", "date"], keep="last")
                .sort_values(["code", "date"], kind="stable")
                .reset_index(drop=True))
        return df

    # ---------------------------------------------------------------- 额外辅助(非规格,供同步层用)
    def fetch_trade_dates(self, start: str, end: str) -> pd.DataFrame:
        """交易日历:列 calendar_date(str), is_trading_day(int)。"""
        _check_date(start, "start")
        _check_date(end, "end")
        fields, rows = self._query("query_trade_dates", bs.query_trade_dates, start, end, code=f"{start}..{end}")
        _require_fields(fields, TRADE_DATES_COLUMNS, "query_trade_dates", f"{start}..{end}")
        df = _frame(fields, rows).loc[:, TRADE_DATES_COLUMNS].copy()
        _to_str_cols(df, ["calendar_date"])
        _to_flag_cols(df, ["is_trading_day"], f"fetch_trade_dates {start}..{end}")
        return df.reset_index(drop=True)

    def last_trading_day(self, day: str, lookback_days: int = 30) -> str:
        """≤day 的最近交易日('YYYY-MM-DD');lookback_days 内找不到则抛 RuntimeError。"""
        _check_date(day, "day")
        d = _dt.datetime.strptime(day, _DATE_FMT).date()
        start = (d - _dt.timedelta(days=int(lookback_days))).strftime(_DATE_FMT)
        cal = self.fetch_trade_dates(start, day)
        trading = cal.loc[cal["is_trading_day"] == 1, "calendar_date"]
        if trading.empty:
            raise BaostockError(f"baostock query_trade_dates: {start}..{day} 内没有交易日(日历为空?)")
        return str(trading.max())
