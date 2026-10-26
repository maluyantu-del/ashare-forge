"""数据同步:股池发现 → 指数 → 行业 → 日线 → 复权因子 → 退市标记 → 质量报告。

本模块修了常见"一把梭"同步脚本的这些坑:

1. **断点续传而不是每次全量重拉**:每只股票只从 ``max(last_date+1, since)`` 拉到 ``end``;
   每只写完立即 ``commit``,中途 Ctrl-C 或断网后重跑即接着上次的进度,不会重拉几小时。
2. **单只失败不拖垮整批,但也绝不假装成功**:单只股票拉取异常 → 写 ``sync_log(status='fail')``
   并进 ``failed`` 清单,继续下一只;而"拉到了但确实没有行"记 ``status='empty'``,
   与失败严格区分。股池发现若一张快照都没拿到直接抛 ``RuntimeError``,不返回 0 冒充完成。
3. **``INSERT OR REPLACE`` 会抹掉未提供的列**:``db.upsert_many`` 是整行替换,若股池发现只写
   ``code/name/board`` 会把之前的 ``industry``/``delisted`` 清成 NULL。这里先读旧行合并
   (first_seen 取更早、last_seen 取更晚、保留行业与退市标记)再整行写回;行业更新走 ``UPDATE``。
4. **非交易日快照为空**:``query_all_stock`` 在周末/节假日返回空表,常见做法直接跳过导致季度
   快照缺失。这里从计划日向前最多回溯 15 个自然日找最近交易日,并以**实际交易日**作为
   ``universe_snapshot.day``。
5. **股池里混进指数/基金**:``query_all_stock`` 含 sh.000xxx / sz.399xxx / 基金,统一用
   ``contract.is_stock_code`` 过滤后才进 ``stocks``;快照原样留档便于审计。
6. **numpy 标量 / NaN / pd.NA 直接绑定 sqlite 会报错或写成 NaN**:入库前统一转 Python
   原生类型,缺失一律 ``None``(sqlite NULL);日期为 ``'YYYY-MM-DD'`` 字符串。
7. **进度不可见**:每 50 只打印 已完成/总数、只/分钟、行/秒、预计剩余,长任务可估算;
   全市场同步(未给 ``--limit``/``--codes``)开跑前明确提示预计耗时。
8. **退市判定用 K 线而非快照**:``mark_delisted`` 以该股最后一根 K 行是否早于
   ``as_of - gap_days`` 为准(合同 ``universe.pit_daily``),而非季度快照的 ``last_seen``;
   没有 K 行的股票不判(尚未同步 ≠ 退市)。

9. **全收益指数不能按普通指数同步**:中证500全收益(H00905)只在中证指数官网发布,且只有
   收盘价;官网在查询起点等处会返回休市日行(实测 2005-01-01、2018-06-18)。
   ``sync_total_return_index`` 按 sh.000001 交易日历剔除这些行,以 ``csi.H00905`` 单独入库,
   开盘等列留空。该序列仅作披露口径,不是主基准(见合同 benchmarks)。

CLI:``python -m ashare_forge.data.sync --limit 30``(先小同步验证);默认全流程顺序
discover_universe → sync_index → sync_industry → sync_daily → sync_adj → sync_total_return_index
→ mark_delisted → report。
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable

import pandas as pd

from .. import contract
from ..config import DB_PATH
from . import db
from .baostock_source import BaostockSource

# 表列(与 data_schema.sql 一致;入库时只取这些列,多余列丢弃、缺列报错)
DAILY_COLS = [
    "code", "date", "open", "high", "low", "close", "preclose",
    "volume", "amount", "turn", "tradestatus", "pctChg",
    "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "isST",
]
ADJ_COLS = ["code", "date", "foreAdjustFactor", "backAdjustFactor", "adjustFactor"]
INDEX_COLS = ["code", "date", "open", "high", "low", "close", "volume", "amount", "pctChg"]
SNAPSHOT_COLS = ["day", "code", "tradestatus", "name"]

PROGRESS_EVERY = 50          # 每 50 只打印一次进度
SNAPSHOT_LOOKBACK_DAYS = 15  # 非交易日快照向前回溯上限(覆盖春节长假)
DEFAULT_SINCE = "2015-01-01"
INDEX_SINCE = "2005-01-01"


# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------
def _today() -> str:
    return date.today().isoformat()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _prev_day(d: str) -> str:
    return (date.fromisoformat(d) - timedelta(days=1)).isoformat()


def _next_day(d: str) -> str:
    return (date.fromisoformat(d) + timedelta(days=1)).isoformat()


def _add_months(d: date, months: int) -> date:
    """加 N 个月,日期超出当月天数则取月末(1/31 + 1月 → 2/28)。"""
    m0 = d.month - 1 + months
    y, m = d.year + m0 // 12, m0 % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _records(
    df: pd.DataFrame, cols: list[str], code: str | None = None,
    keys: tuple[str, str] = ("code", "date"),
) -> list[dict]:
    """DataFrame → 可直接绑定 sqlite 的 dict 列表:只取表列、按主键去重、缺失→None、numpy→原生。

    keys = (代码列, 日期列);日线/复权/指数为 (code, date),快照为 (code, day)。
    """
    if df is None or len(df) == 0:
        return []
    df = df.copy()
    code_col, date_col = keys
    if code_col not in df.columns and code is not None:
        df[code_col] = code
    missing = [c for c in keys if c not in df.columns]
    if missing:
        raise RuntimeError(f"数据源返回缺少关键列 {missing}(code={code}),列={list(df.columns)}")
    use = [c for c in cols if c in df.columns]
    sub = df[use].drop_duplicates(subset=list(keys), keep="last")
    sub = sub.astype(object).where(sub.notna(), None)
    out = []
    for rec in sub.to_dict("records"):
        rec[date_col] = str(rec[date_col])[:10]
        rec[code_col] = str(rec[code_col])
        out.append(rec)
    return out


class _Progress:
    """每 N 只打印进度/速率/预计剩余。"""

    def __init__(self, task: str, total: int, every: int = PROGRESS_EVERY):
        self.task, self.total, self.every = task, total, every
        self.t0 = time.monotonic()
        self.done = self.rows = self.failed = 0

    def step(self, rows: int, failed: bool = False) -> None:
        self.done += 1
        self.rows += rows
        self.failed += int(failed)
        if self.done % self.every == 0 or self.done == self.total:
            self.report()

    def report(self) -> None:
        el = max(time.monotonic() - self.t0, 1e-9)
        per_min = self.done / el * 60
        rows_per_s = self.rows / el
        remain = self.total - self.done
        eta_min = remain / per_min if per_min > 0 else float("inf")
        _log(
            f"[{self.task}] {self.done}/{self.total} 只 | {self.rows} 行 | 用时 {el/60:.1f} 分 | "
            f"{per_min:.1f} 只/分 | {rows_per_s:.0f} 行/秒 | 预计剩余 {eta_min:.1f} 分 | 失败 {self.failed}"
        )


# ----------------------------------------------------------------------------
# 股池发现
# ----------------------------------------------------------------------------
def _snapshot_days(start: str, end: str, step_months: int) -> list[str]:
    """从 start 每 step_months 一个计划日,并确保含 end 当日。"""
    if step_months <= 0:
        raise ValueError("step_months 必须 > 0")
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    if s > e:
        raise ValueError(f"start {start} 晚于 end {end}")
    days, d = [], s
    while d <= e:
        days.append(d.isoformat())
        d = _add_months(d, step_months)
    if days[-1] != end:
        days.append(end)
    return sorted(set(days))


def _fetch_snapshot_near(src, planned: str) -> tuple[str | None, pd.DataFrame | None]:
    """取 planned 当日快照;为空(非交易日)则向前回溯,返回 (实际交易日, df)。

    某一天查询抛错也向前试(单日瞬时错误不应让整张快照缺失);
    若回溯窗口内全部抛错则把错误抛出去,全部为空则返回 (None, None) 由上层记录。
    """
    d = date.fromisoformat(planned)
    errors: list[str] = []
    for back in range(SNAPSHOT_LOOKBACK_DAYS + 1):
        day = (d - timedelta(days=back)).isoformat()
        try:
            df = src.fetch_all_stock(day)
        except Exception as e:  # noqa: BLE001 - 记录后继续试前一天
            errors.append(f"{day}: {e}")
            continue
        if df is not None and len(df) > 0:
            return day, df
    if errors and len(errors) > SNAPSHOT_LOOKBACK_DAYS:
        raise RuntimeError(f"快照 {planned} 回溯 {SNAPSHOT_LOOKBACK_DAYS} 天全部失败:{errors[-1]}")
    return None, None


def discover_universe(src, conn, start: str, end: str, step_months: int = 3) -> int:
    """按 step_months 取 query_all_stock 快照(含 end 当日)写 universe_snapshot,
    过滤成股票后合并进 stocks(first_seen/last_seen/board/name);返回 stocks 表股票总数。"""
    planned = _snapshot_days(start, end, step_months)
    _log(f"[universe] 计划快照 {len(planned)} 张:{planned[0]} … {planned[-1]}(每 {step_months} 月)")
    # code -> {first, last, name}
    seen: dict[str, dict] = {}
    got = 0
    used_days: set[str] = set()
    for i, p in enumerate(planned, 1):
        day, df = _fetch_snapshot_near(src, p)
        if day is None:
            msg = f"快照 {p} 向前 {SNAPSHOT_LOOKBACK_DAYS} 天均为空表,跳过"
            _log(f"[universe] 警告:{msg}")
            db.log_sync(conn, "universe", None, 0, "fail", msg)
            conn.commit()
            continue
        if day in used_days:  # 两个计划日回溯到同一交易日(如 end 紧邻上一计划日)
            continue
        used_days.add(day)
        rows = _records(df.assign(day=day), SNAPSHOT_COLS, keys=("code", "day"))
        n = db.upsert_many(conn, "universe_snapshot", rows)
        for r in rows:
            code = r["code"]
            if not contract.is_stock_code(code):
                continue
            s = seen.get(code)
            if s is None:
                seen[code] = {"first": day, "last": day, "name": r.get("name")}
            else:
                s["first"] = min(s["first"], day)
                if day >= s["last"]:
                    s["last"], s["name"] = day, r.get("name") or s["name"]
        db.log_sync(conn, "universe", None, n, "ok", f"snapshot planned={p} actual={day}")
        conn.commit()
        got += 1
        _log(f"[universe] {i}/{len(planned)} 快照 {p}→{day}:{n} 行,累计股票 {len(seen)}")
    if got == 0:
        raise RuntimeError(f"股池发现失败:{start}~{end} 没有拿到任何一张快照")

    # 合并旧行再整行写回(INSERT OR REPLACE 不能只写部分列)
    existing = {
        r["code"]: dict(r)
        for r in conn.execute("SELECT * FROM stocks").fetchall()
    }
    out = []
    for code, s in seen.items():
        old = existing.get(code, {})
        first = min(x for x in (old.get("first_seen"), s["first"]) if x)
        last = max(x for x in (old.get("last_seen"), s["last"]) if x)
        out.append({
            "code": code,
            "name": s["name"] or old.get("name"),
            "board": contract.board_of(code),
            "first_seen": first,
            "last_seen": last,
            "delisted": old.get("delisted") or 0,
            "industry": old.get("industry"),
            "industry_src": old.get("industry_src"),
            "industry_asof": old.get("industry_asof"),
        })
    db.upsert_many(conn, "stocks", out)
    db.set_meta(conn, "last_run_universe", _utc_now())
    db.set_meta(conn, "universe_range", f"{start}~{end}")
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
    _log(f"[universe] 完成:{got} 张快照,本次 {len(out)} 只,stocks 表共 {total} 只")
    return int(total)


# ----------------------------------------------------------------------------
# 逐股增量同步(日线 / 复权因子 / 指数共用)
# ----------------------------------------------------------------------------
def _all_stock_codes(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT code FROM stocks ORDER BY code").fetchall()]


def _pending_ranges(conn, table: str, code: str, since: str, end: str,
                    full_refresh: bool = False) -> list[tuple[str, str]]:
    """返回该 code 仍需拉取的日期区间列表(头部缺口 + 尾部增量)。

    full_refresh=True 直接返回整段。否则按库内已有的 [min,max] 计算:
      头部缺口 = [since, min-1](修"改早起点补不回历史"的坑)
      尾部增量 = [max+1, end]
    中间的空洞不补(停牌/非交易日无法与真缺失区分),这是已披露的限制。
    """
    if full_refresh:
        return [(since, end)] if since <= end else []
    row = conn.execute(
        f"SELECT MIN(date), MAX(date) FROM {table} WHERE code = ?", (code,)  # noqa: S608 表名来自内部常量
    ).fetchone()
    lo, hi = (row[0], row[1]) if row else (None, None)
    if not lo:
        return [(since, end)] if since <= end else []
    out: list[tuple[str, str]] = []
    if since < lo:
        out.append((since, _prev_day(lo)))
    if hi < end:
        out.append((_next_day(hi), end))
    return out


def _sync_per_code(
    src, conn, *, task: str, table: str, cols: list[str],
    fetch: Callable[[str, str, str], pd.DataFrame],
    codes: Iterable[str], since: str, end: str, full_refresh: bool = False,
) -> dict:
    # full_refresh=True:跳过断点续传,每次拉全区间。复权因子必须如此——
    # 增量续传会让"已同步到 2026 的累计因子表"永远补不回更早的历史(实测踩坑)。
    codes = list(codes)
    prog = _Progress(task, len(codes))
    failed: list[dict] = []
    skipped = empty = 0
    t0 = time.monotonic()
    _log(f"[{task}] 开始:{len(codes)} 只,since={since} end={end}")
    for code in codes:
        try:
            # 同时处理「头部缺口」与「尾部增量」:
            # 只补尾部会让"先用晚起点同步过"的股票永远补不回更早历史,
            # 全量同步后得到前后不一致的面板(实测:30 只按 2024 起点同步过的股票被全部跳过)。
            ranges = _pending_ranges(conn, table, code, since, end, full_refresh)
            if not ranges:
                skipped += 1
                db.log_sync(conn, task, code, 0, "skip", f"区间已完整 {since}~{end}")
                conn.commit()
                prog.step(0)
                continue
            n = 0
            for r_start, r_end in ranges:
                df = fetch(code, r_start, r_end)
                rows = _records(df, cols, code=code)
                n += db.upsert_many(conn, table, rows) if rows else 0
            start = ranges[0][0]
            if n == 0:
                empty += 1
                db.log_sync(conn, task, code, 0, "empty", f"{start}~{end} 无行")
            else:
                db.log_sync(conn, task, code, n, "ok", f"{start}~{end}")
            conn.commit()
            prog.step(n)
        except KeyboardInterrupt:
            conn.rollback()
            _log(f"[{task}] 用户中断:已完成 {prog.done}/{len(codes)} 只,重跑即可续传")
            raise
        except Exception as e:  # noqa: BLE001 - 单只失败记录后继续
            conn.rollback()
            msg = f"{type(e).__name__}: {e}"[:500]
            failed.append({"code": code, "error": msg})
            try:
                db.log_sync(conn, task, code, 0, "fail", msg)
                conn.commit()
            except Exception as e2:  # noqa: BLE001
                _log(f"[{task}] 写 sync_log 失败 {code}: {e2}")
            _log(f"[{task}] 失败 {code}: {msg}")
            prog.step(0, failed=True)
    secs = time.monotonic() - t0
    db.set_meta(conn, f"last_run_{task}", _utc_now())
    conn.commit()
    _log(
        f"[{task}] 结束:{len(codes)} 只 | 写入 {prog.rows} 行 | 跳过 {skipped} | 空 {empty} | "
        f"失败 {len(failed)} | {secs:.1f} 秒"
    )
    if failed:
        _log(f"[{task}] 失败清单:{[f['code'] for f in failed]}")
    return {
        "codes": len(codes), "rows": int(prog.rows), "failed": failed,
        "seconds": round(secs, 3), "skipped": skipped, "empty": empty,
    }


def _resolve_codes(conn, codes: list[str] | None, limit: int | None, what: str) -> list[str]:
    if codes:
        out = [c.strip() for c in codes if c and c.strip()]
    else:
        out = _all_stock_codes(conn)
        if not out:
            raise RuntimeError(f"{what}:stocks 表为空,请先运行 discover_universe(或用 --codes 指定)")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit 必须 > 0")
        out = out[:limit]
    return out


def sync_daily(src, conn, codes: list[str] | None, since: str, limit: int | None, end: str | None = None) -> dict:
    """日线增量同步(不复权,含估值/ST/停牌字段)。codes 为空取 stocks 全表按 code 排序。"""
    end = end or _today()
    codes = _resolve_codes(conn, codes, limit, "sync_daily")
    if limit is None and len(codes) > 200:
        _log(f"[daily] 提示:{len(codes)} 只全量同步,baostock 逐股串行,首次约 3~7 小时;可 Ctrl-C 后重跑续传")
    return _sync_per_code(
        src, conn, task="daily", table="daily", cols=DAILY_COLS,
        fetch=src.fetch_daily, codes=codes, since=since, end=end,
    )


# baostock 的 backAdjustFactor 是"自上市累计"的因子,不是当次除权比例。
# 若只同步窗口内的因子,窗口前段无记录按 1.0 处理、到窗口内第一个除权日突然跳到累计值,
# 会在后复权价上造出一个几百倍的假涨幅(实测 sh.600000 于 2024-07-18 假跳 +1139%)。
# 因此复权因子一律全历史同步(每只仅数行,代价可忽略),不受 --since 约束。
ADJ_FULL_HISTORY_SINCE = "1990-01-01"


def sync_adj(src, conn, codes, since, limit) -> dict:
    """复权因子同步(adj_factor 表)。

    与日线不同:**忽略 since,始终拉全历史**。原因见上方注释——累计因子被截断会污染收益率。
    """
    end = _today()
    codes = _resolve_codes(conn, codes, limit, "sync_adj")
    if since and since > ADJ_FULL_HISTORY_SINCE:
        _log(f"[adj] 注意:复权因子忽略 since={since},改用全历史 {ADJ_FULL_HISTORY_SINCE}"
             f"(累计因子被截断会造成后复权假跳变)")
    return _sync_per_code(
        src, conn, task="adj", table="adj_factor", cols=ADJ_COLS,
        fetch=src.fetch_adjust_factors, codes=codes, since=ADJ_FULL_HISTORY_SINCE, end=end,
        full_refresh=True,
    )


def sync_index(src, conn, codes: list[str] | None = None, since: str = INDEX_SINCE) -> dict:
    """指数日线增量同步;codes 缺省取合同 benchmarks。"""
    if not codes:
        codes = [b["code"] for b in contract.load_contract()["benchmarks"]]
    return _sync_per_code(
        src, conn, task="index", table="index_daily", cols=INDEX_COLS,
        fetch=src.fetch_index_daily, codes=list(codes), since=since, end=_today(),
    )


# ----------------------------------------------------------------------------
# 全收益基准(仅披露口径):中证500全收益 H00905,来源中证指数官网(经 akshare)
# ----------------------------------------------------------------------------
TR_INDEX_CODE = "csi.H00905"      # 库内代码;csi. 前缀与 baostock 的 sh./sz. 指数区分,避免被当作可 baostock 拉取
TR_SOURCE_SYMBOL = "H00905"       # 中证指数官网代码
CALENDAR_CODE = "sh.000001"       # 交易日历(与 loader._listing_age_from_ipo 同源)
CSINDEX_HOST = "csindex.com.cn"
CSINDEX_TIMEOUT_S = 60.0
CSINDEX_RETRIES = 3


@contextmanager
def _bypass_env_proxy(host: str):
    """临时把 host 加入 NO_PROXY/no_proxy,退出时原样恢复。

    akshare 的 stock_zh_index_hist_csindex 内部直接 requests.get(不设超时,读取代理环境变量)。
    开发环境中经 HTTP(S)_PROXY 转发到中证指数官网时实测频繁读超时,直连约 1 秒返回;
    未设代理的环境下本函数无副作用。
    """
    keys = ("NO_PROXY", "no_proxy")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            cur = saved[k]
            os.environ[k] = f"{cur},{host}" if cur else host
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _call_with_timeout(fn: Callable, timeout: float, *args, **kwargs):
    """在守护线程里调用 fn,超时抛 TimeoutError(第三方接口不设超时,不能让同步无限挂起)。"""
    box: dict = {}

    def _run():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 - 原样转交调用方
            box["error"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError(f"{getattr(fn, '__name__', fn)} 超过 {timeout:.0f} 秒未返回")
    if "error" in box:
        raise box["error"]
    return box["value"]


def fetch_csindex_close(symbol: str, start: str, end: str, *, timeout: float = CSINDEX_TIMEOUT_S,
                        retries: int = CSINDEX_RETRIES, use_env_proxy: bool = False) -> pd.DataFrame:
    """中证指数官网日线收盘价(经 akshare.stock_zh_index_hist_csindex)。返回列 date('YYYY-MM-DD'), close。

    该接口只有收盘价(开盘/最高/最低为空),不能作为回测基准的开盘→开盘口径来源,
    只用于派生分红补偿(见 loader.dividend_compensation)。单次调用超时或报错时重试
    retries 次,仍失败则抛出,不返回空表冒充成功。
    """
    import akshare as ak

    s, e = start.replace("-", ""), end.replace("-", "")
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        try:
            if use_env_proxy:
                raw = _call_with_timeout(ak.stock_zh_index_hist_csindex, timeout,
                                         symbol=symbol, start_date=s, end_date=e)
            else:
                with _bypass_env_proxy(CSINDEX_HOST):
                    raw = _call_with_timeout(ak.stock_zh_index_hist_csindex, timeout,
                                             symbol=symbol, start_date=s, end_date=e)
            break
        except Exception as ex:  # noqa: BLE001 - 记录后重试
            errors.append(f"第{attempt}次 {type(ex).__name__}: {ex}"[:300])
            _log(f"[index_tr] 拉取 {symbol} 失败({errors[-1]})")
            if attempt < retries:
                time.sleep(2.0 * attempt)
    else:
        raise RuntimeError(f"中证指数官网 {symbol} 拉取 {retries} 次均失败:{errors[-1]}")

    if raw is None or len(raw) == 0:
        raise RuntimeError(f"中证指数官网 {symbol} 在 {start}~{end} 返回空表")
    need = {"日期", "收盘"}
    if not need.issubset(raw.columns):
        raise RuntimeError(f"中证指数官网返回缺列 {sorted(need - set(raw.columns))},列={list(raw.columns)}")
    if "指数代码" in raw.columns:
        got = set(raw["指数代码"].dropna().astype(str))
        if got and got != {symbol}:
            raise RuntimeError(f"请求 {symbol} 却返回指数代码 {sorted(got)}")
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["日期"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "close": pd.to_numeric(raw["收盘"], errors="coerce"),
    })
    return out


def sync_total_return_index(conn, *, fetch: Callable[[str, str, str], pd.DataFrame] | None = None,
                            since: str = INDEX_SINCE, end: str | None = None,
                            code: str = TR_INDEX_CODE, symbol: str = TR_SOURCE_SYMBOL,
                            calendar_code: str = CALENDAR_CODE) -> dict:
    """把全收益指数收盘价写入 index_daily(code=csi.H00905;open/high/low/volume/amount/pctChg 留空)。

    - 数据源:fetch(symbol, start, end) → DataFrame[date, close];缺省为 fetch_csindex_close。
    - 按 calendar_code(sh.000001)的交易日历过滤:日历内的休市日行剔除;晚于日历末日的行
      无法核验是否交易日,本次不写(日历同步后重跑即补上)。
    - 本代码在 [since, end] 内的旧行整段替换(同一事务内先删后写),重复运行结果一致,
      也会清掉以前可能写入的休市日行。只动 code 这一个代码的行。
    - 该序列仅作披露口径:不是主基准,不进门槛(合同 benchmarks)。

    返回 {code, rows, first, last, dropped_non_trading, dropped_beyond_calendar,
          missing_vs_calendar, failed}。
    """
    end = end or _today()
    for d in (since, end):
        date.fromisoformat(d)
    fetch = fetch or fetch_csindex_close
    t0 = time.monotonic()
    _log(f"[index_tr] 开始:{symbol} → {code},{since}~{end}")
    try:
        raw = fetch(symbol, since, end)
    except Exception as e:
        db.log_sync(conn, "index_tr", code, 0, "fail", f"{type(e).__name__}: {e}"[:500])
        conn.commit()
        raise

    df = pd.DataFrame({"date": raw["date"].astype(str).str[:10],
                       "close": pd.to_numeric(raw["close"], errors="coerce")})
    df = df[(df["date"] >= since) & (df["date"] <= end)]
    n_bad = int(df["close"].isna().sum() + (df["close"] <= 0).sum())
    df = df[df["close"].notna() & (df["close"] > 0)]
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)

    cal = [r[0] for r in conn.execute(
        "SELECT date FROM index_daily WHERE code = ? AND date >= ? AND date <= ? ORDER BY date",
        (calendar_code, since, end)).fetchall()]
    if not cal:
        raise RuntimeError(f"交易日历为空:index_daily 无 {calendar_code} 在 {since}~{end} 的行,"
                           f"请先同步指数(--index-only)")
    cal_set, cal_max = set(cal), cal[-1]
    beyond = df["date"] > cal_max
    in_cal = df["date"].isin(cal_set)
    dropped_non_trading = df.loc[~in_cal & ~beyond, "date"].tolist()
    kept = df[in_cal]
    if kept.empty:
        raise RuntimeError(f"{symbol} 在 {since}~{end} 与交易日历无交集,未写入")
    first, last = kept["date"].iloc[0], kept["date"].iloc[-1]
    have = set(kept["date"])
    missing = [d for d in cal if first <= d <= last and d not in have]

    rows = [{"code": code, "date": d, "open": None, "high": None, "low": None, "close": float(c),
             "volume": None, "amount": None, "pctChg": None}
            for d, c in zip(kept["date"], kept["close"])]
    # 先删后写在同一事务里(upsert_many 结束时统一 commit,异常则回滚,删除一并撤销)
    conn.execute("DELETE FROM index_daily WHERE code = ? AND date >= ? AND date <= ?", (code, since, end))
    n = db.upsert_many(conn, "index_daily", rows)

    msg = (f"{first}~{last} 写入 {n} 行 | 休市日剔除 {len(dropped_non_trading)} 行 {dropped_non_trading[:5]} | "
           f"晚于日历末日({cal_max})未写 {int(beyond.sum())} 行 | 日历内缺失 {len(missing)} 天 | "
           f"无效收盘 {n_bad} 行")
    db.log_sync(conn, "index_tr", code, n, "ok", msg[:500])
    db.set_meta(conn, "last_run_index_tr", _utc_now())
    conn.commit()
    _log(f"[index_tr] 完成:{msg} | {time.monotonic() - t0:.1f} 秒")
    if missing:
        _log(f"[index_tr] 警告:交易日历内 {len(missing)} 天无全收益收盘(如 {missing[:5]}),"
             f"对应日期的分红补偿按 0 处理")
    return {"code": code, "rows": int(n), "first": first, "last": last,
            "dropped_non_trading": dropped_non_trading, "dropped_beyond_calendar": int(beyond.sum()),
            "missing_vs_calendar": missing, "failed": []}


# ----------------------------------------------------------------------------
# 行业 / 退市 / 报告
# ----------------------------------------------------------------------------
# ==================== 股票基本信息:真实上市/退市日期 ====================
# 审查与复查发现两个关联的 P0:
#  ① 上市天数原先从"库内第一根K线"算起,库从 2015-01-05 开始 → 平安银行(1991年上市)
#     在系统里被当成 2015 年新股,**样本首 119 个交易日全市场被误判为次新、在池恒为 0**,
#     合同的 2015 股灾危机窗口恰好丢掉崩盘起点。
#  ② 股池快照只从 2024 年取起 → **2015–2023 年间退市的 154 只股票完全不在库**
#     (退市银鸽、退市保千、*ST宋都…),回测主体区间存在幸存者偏差。
# baostock.query_stock_basic 一次调用给出全部股票的 ipoDate/outDate,同时修复两者。

STOCK_TYPE = "1"          # baostock type: 1=股票 2=指数 4=可转债 5=ETF


def _ensure_stock_date_columns(conn) -> None:
    """为 stocks 表幂等补列 ipo_date / out_date(不改 data_schema.sql,兼容旧库)。"""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(stocks)")}
    for c in ("ipo_date", "out_date"):
        if c not in cols:
            conn.execute(f"ALTER TABLE stocks ADD COLUMN {c} TEXT")   # 列名为内部常量
    conn.commit()


def sync_stock_basic(src, conn) -> dict:
    """拉取全部股票的真实上市/退市日期,并把股池快照漏掉的历史退市股补进 stocks。

    返回 {n_total, n_added, n_delisted, added_sample}。
    """
    import baostock as bs  # 本函数直接调用 query_stock_basic(BaostockSource 未封装该接口)

    _ensure_stock_date_columns(conn)
    rows = []
    rs = bs.query_stock_basic()
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0" or not rows:
        raise RuntimeError(f"query_stock_basic 失败或为空: {rs.error_code} {rs.error_msg}")
    fields = rs.fields
    df = pd.DataFrame(rows, columns=fields)
    df = df[(df["type"] == STOCK_TYPE) & df["code"].map(contract.is_stock_code)]

    existing = {r["code"]: dict(r) for r in conn.execute("SELECT * FROM stocks").fetchall()}
    out, added = [], []
    for _, r in df.iterrows():
        code = r["code"]
        ipo = r["ipoDate"] or None
        outd = r["outDate"] or None
        old = existing.get(code)
        if old is None:
            added.append((code, r["code_name"], ipo, outd))
        old = old or {}
        out.append({
            "code": code,
            "name": old.get("name") or r["code_name"],
            "board": contract.board_of(code),
            "first_seen": old.get("first_seen") or ipo,
            "last_seen": old.get("last_seen") or outd or ipo,
            "delisted": 1 if outd else int(old.get("delisted") or 0),
            "industry": old.get("industry"),
            "industry_src": old.get("industry_src"),
            "industry_asof": old.get("industry_asof"),
            "ipo_date": ipo,
            "out_date": outd,
        })
    n = db.upsert_many(conn, "stocks", out)
    db.set_meta(conn, "stock_basic_synced_at", _utc_now())
    db.log_sync(conn, "stock_basic", None, n, "ok", f"added={len(added)}")
    conn.commit()
    n_del = sum(1 for r in out if r["out_date"])
    _log(f"[basic] 股票 {len(out)} 只 | 新增(快照漏掉的) {len(added)} 只 | 有退市日期 {n_del} 只")
    return {"n_total": len(out), "n_added": len(added), "n_delisted": n_del,
            "added_sample": added[:10]}


def sync_industry(src, conn) -> int:
    """写 stocks.industry / industry_src / industry_asof(UPDATE,不动其他列);返回更新股数。"""
    df = src.fetch_industry()
    if df is None or len(df) == 0:
        raise RuntimeError("行业分类返回空表(fetch_industry)")
    need = {"code", "industry"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"行业分类缺列 {need - set(df.columns)},列={list(df.columns)}")
    known = set(_all_stock_codes(conn))
    params, asof_max = [], None
    for rec in df.astype(object).where(df.notna(), None).to_dict("records"):
        code = str(rec["code"])
        if code not in known:
            continue
        ind = rec.get("industry")
        ind = str(ind).strip() or None if ind is not None else None
        cls = rec.get("industryClassification")
        src_tag = "baostock:" + (str(cls).strip() if cls else "industry")
        asof = str(rec.get("updateDate") or "")[:10] or None
        if asof and (asof_max is None or asof > asof_max):
            asof_max = asof
        params.append((ind, src_tag, asof, code))
    cur = conn.executemany(
        "UPDATE stocks SET industry=?, industry_src=?, industry_asof=? WHERE code=?", params
    )
    n = int(cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else len(params))
    db.log_sync(conn, "industry", None, n, "ok", f"asof={asof_max} 源行数={len(df)}")
    db.set_meta(conn, "last_run_industry", _utc_now())
    if asof_max:
        db.set_meta(conn, "industry_asof", asof_max)
    conn.commit()
    _log(f"[industry] 更新 {n} 只(源 {len(df)} 行,asof={asof_max})")
    return n


def mark_delisted(conn, as_of: str, gap_days: int = 60) -> int:
    """最后一根 K 行早于 as_of-gap_days 的股票标 delisted=1(有新 K 行的复位为 0);返回标记数。"""
    threshold = (date.fromisoformat(as_of) - timedelta(days=gap_days)).isoformat()
    cur = conn.execute(
        "UPDATE stocks SET delisted=1 WHERE code IN "
        "(SELECT code FROM daily GROUP BY code HAVING MAX(date) < ?)",
        (threshold,),
    )
    n = int(cur.rowcount)
    conn.execute(
        "UPDATE stocks SET delisted=0 WHERE delisted=1 AND code IN "
        "(SELECT code FROM daily GROUP BY code HAVING MAX(date) >= ?)",
        (threshold,),
    )
    db.set_meta(conn, "delisted_threshold", threshold)
    conn.commit()
    _log(f"[delisted] as_of={as_of} gap={gap_days}d 阈值<{threshold}:标记 {n} 只")
    return n


def _n_vanished(conn) -> int:
    """曾在股池快照出现、但在最新快照中已消失的股票数。"""
    row = conn.execute(
        "SELECT COUNT(*) FROM stocks WHERE last_seen < (SELECT MAX(day) FROM universe_snapshot)"
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def data_quality_report(conn) -> dict:
    """质量报告:股票数/已退市数/日线行数/日期范围/停牌比例/ST比例/缺复权因子股数/每年行数/最近同步/失败清单。"""
    q = lambda sql, *p: conn.execute(sql, p).fetchone()  # noqa: E731
    n_stocks = q("SELECT COUNT(*) FROM stocks")[0]
    n_delisted = q("SELECT COUNT(*) FROM stocks WHERE delisted=1")[0]
    n_rows, dmin, dmax = q("SELECT COUNT(*), MIN(date), MAX(date) FROM daily")
    n_daily_codes = q("SELECT COUNT(DISTINCT code) FROM daily")[0]
    susp = q("SELECT AVG(CASE WHEN tradestatus=0 THEN 1.0 ELSE 0.0 END) FROM daily")[0]
    st = q("SELECT AVG(CASE WHEN isST=1 THEN 1.0 ELSE 0.0 END) FROM daily")[0]
    n_missing_adj = q(
        "SELECT COUNT(*) FROM stocks WHERE code NOT IN (SELECT DISTINCT code FROM adj_factor)"
    )[0]
    rows_per_year = {
        r[0]: r[1] for r in conn.execute(
            "SELECT substr(date,1,4) AS y, COUNT(*) FROM daily GROUP BY y ORDER BY y"
        ).fetchall()
    }
    n_index_rows = q("SELECT COUNT(*) FROM index_daily")[0]
    index_codes = [r[0] for r in conn.execute(
        "SELECT DISTINCT code FROM index_daily ORDER BY code").fetchall()]
    n_snapshots = q("SELECT COUNT(DISTINCT day) FROM universe_snapshot")[0]
    last_sync = q("SELECT MAX(ts) FROM sync_log")[0]
    last_by_task = {
        r[0]: r[1] for r in conn.execute(
            "SELECT task, MAX(ts) FROM sync_log GROUP BY task").fetchall()
    }
    # 失败清单:每个 (task, code) 只看最后一条记录,最后一次仍是 fail 才算未解决
    failures = [
        {"task": r[0], "code": r[1], "ts": r[2], "message": r[3]}
        for r in conn.execute(
            "SELECT task, code, ts, message FROM sync_log s WHERE status='fail' AND id = "
            "(SELECT MAX(id) FROM sync_log t WHERE t.task=s.task AND t.code IS s.code) "
            "ORDER BY ts DESC LIMIT 500"
        ).fetchall()
    ]
    return {
        "n_stocks": int(n_stocks),
        "n_delisted": int(n_delisted),
        "n_daily_rows": int(n_rows),
        "n_daily_codes": int(n_daily_codes),
        "date_range": {"min": dmin, "max": dmax},
        "suspended_ratio": None if susp is None else round(float(susp), 6),
        "st_ratio": None if st is None else round(float(st), 6),
        "n_missing_adj": int(n_missing_adj),
        "rows_per_year": rows_per_year,
        "n_index_rows": int(n_index_rows),
        "index_codes": index_codes,
        "n_snapshots": int(n_snapshots),
        # 幸存者偏差可见性:曾在股池快照出现、但最新快照中已消失的股票数(退市/摘牌的直接证据)。
        # 与 n_delisted(按日线K行判定)不同:后者要等该股日线同步后才算得出来,
        # 两者差额 = 已知消失但尚未取证的股票数。
        "n_vanished_from_universe": int(_n_vanished(conn)),
        "last_sync": last_sync,
        "last_sync_by_task": last_by_task,
        "n_failures": len(failures),
        "failures": failures,
        "generated_at": _utc_now(),
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m ashare_forge.data.sync",
        description="AShare Forge 数据同步(baostock,免 token;全收益指数来自中证指数官网,经 akshare)。"
                    "不带步骤开关时跑默认全流程:discover_universe → sync_index → sync_industry → "
                    "sync_daily → sync_adj → sync_total_return_index → mark_delisted → report",
    )
    p.add_argument("--db", default=DB_PATH, help=f"SQLite 路径(缺省 {DB_PATH})")
    p.add_argument("--since", default=DEFAULT_SINCE, help="日线/复权起始日 YYYY-MM-DD(缺省 2015-01-01)")
    p.add_argument("--end", default=None, help="截止日 YYYY-MM-DD(缺省今天)")
    p.add_argument("--limit", type=int, default=None, help="只同步前 N 只(按 code 排序);验证用 --limit 30")
    p.add_argument("--codes", default=None, help="逗号分隔代码,如 sz.000001,sh.600000")
    p.add_argument("--universe-only", action="store_true", help="只做股池发现")
    p.add_argument("--daily-only", action="store_true", help="只同步日线")
    p.add_argument("--adj-only", action="store_true", help="只同步复权因子")
    p.add_argument("--index-only", action="store_true", help="只同步指数")
    p.add_argument("--industry", action="store_true", help="同步行业分类")
    p.add_argument("--basic", action="store_true",
                   help="同步真实上市/退市日期,并补入股池快照漏掉的历史退市股")
    p.add_argument("--index-tr", action="store_true",
                   help=f"只同步中证500全收益指数 {TR_SOURCE_SYMBOL}(中证指数官网,经 akshare;仅收盘价,"
                        f"入库代码 {TR_INDEX_CODE},按 {CALENDAR_CODE} 交易日历剔除休市日行;仅作披露口径)")
    p.add_argument("--tr-use-env-proxy", action="store_true",
                   help="访问中证指数官网时沿用 HTTP(S)_PROXY 环境变量(缺省对该域名直连)")
    p.add_argument("--report", action="store_true", help="输出数据质量报告(不联网)")
    p.add_argument("--snapshots-from", default=None, help="股池快照起始日(缺省 = --since)")
    p.add_argument("--snapshots-to", default=None, help="股池快照截止日(缺省 = --end)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = _parse_args(argv)
    end = a.end or _today()
    codes = [c.strip() for c in a.codes.split(",") if c.strip()] if a.codes else None
    snap_from, snap_to = a.snapshots_from or a.since, a.snapshots_to or end
    for d in (a.since, end, snap_from, snap_to):
        date.fromisoformat(d)  # 早失败:非法日期直接报错

    selected = {
        "universe": a.universe_only, "index": a.index_only, "industry": a.industry,
        "daily": a.daily_only, "adj": a.adj_only, "report": a.report, "basic": a.basic,
        "index_tr": a.index_tr,
    }
    full = not any(selected.values())
    # basic 排在 universe 之后、daily 之前:先补全股池(含历史退市股),日线才会把它们一起拉下来
    # index_tr 不经 baostock,且依赖 sh.000001 交易日历,排在 baostock 各步之后
    steps = ["universe", "basic", "index", "industry", "daily", "adj", "index_tr", "delisted", "report"] if full \
        else [s for s in ("universe", "basic", "index", "industry", "daily", "adj", "index_tr", "report")
              if selected[s]]
    net_steps = [s for s in steps if s not in ("report", "delisted", "index_tr")]

    conn = db.connect(a.db)
    db.ensure_schema(conn)
    _log(f"[sync] db={a.db} steps={steps} since={a.since} end={end} limit={a.limit} codes={codes}")
    if full and a.limit is None and not codes:
        _log("[sync] 警告:未指定 --limit/--codes,将同步全市场(约 3~7 小时);建议先 --limit 30 验证")
    t0 = time.monotonic()
    summary: dict = {}
    try:
        if net_steps:
            with BaostockSource() as src:
                if "universe" in steps:
                    summary["universe"] = discover_universe(src, conn, snap_from, snap_to)
                if "basic" in steps:
                    summary["basic"] = sync_stock_basic(src, conn)
                if "index" in steps:
                    summary["index"] = sync_index(src, conn, since=INDEX_SINCE)
                if "industry" in steps:
                    summary["industry"] = sync_industry(src, conn)
                if "daily" in steps:
                    summary["daily"] = sync_daily(src, conn, codes, a.since, a.limit, end)
                if "adj" in steps:
                    summary["adj"] = sync_adj(src, conn, codes, a.since, a.limit)
        if "index_tr" in steps:
            use_proxy = a.tr_use_env_proxy
            try:
                summary["index_tr"] = sync_total_return_index(
                    conn, end=end,
                    fetch=lambda s, st, en: fetch_csindex_close(s, st, en, use_env_proxy=use_proxy))
            except Exception as e:  # noqa: BLE001 - 披露口径的数据源不可达不应拖垮其余步骤,但照实计入失败
                msg = f"{type(e).__name__}: {e}"[:500]
                _log(f"[index_tr] 失败:{msg}")
                summary["index_tr"] = {"failed": [{"code": TR_INDEX_CODE, "error": msg}]}
        if "delisted" in steps:
            summary["delisted"] = mark_delisted(conn, end)
        if "report" in steps:
            rep = data_quality_report(conn)
            summary["report"] = rep
            _log(json.dumps(rep, ensure_ascii=False, indent=2))
    finally:
        conn.close()
    _log(f"[sync] 全部完成,用时 {(time.monotonic()-t0)/60:.1f} 分")
    failed_total = sum(len(v.get("failed", [])) for v in summary.values() if isinstance(v, dict) and "failed" in v)
    if failed_total:
        _log(f"[sync] 注意:共 {failed_total} 只失败,见 sync_log / 报告 failures")
    return 1 if failed_total else 0


if __name__ == "__main__":
    sys.exit(main())
