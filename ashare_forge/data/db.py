"""SQLite 连接与写入层(AShare Forge 数据层的最底层,其他模块只通过这里碰数据库)。

它修了常见做法的哪几个坑:
1. sqlite3 默认 DELETE 日志 + synchronous=FULL:全市场几百万行日线写起来慢到不可用,
   而且写库期间读库会被锁死。connect() 固定打开 WAL + synchronous=NORMAL:读写互不阻塞,
   写速度快一个量级;断电最多丢最后一个 checkpoint 之后的事务,库文件本身不会损坏。
   若文件库未能切到 WAL(如某些网络盘)直接抛错,不带病运行。
2. 常见写法是逐行 execute + 逐行 commit,或 DataFrame.to_sql(慢、不支持 upsert、重跑会撞主键)。
   upsert_many() 用 INSERT OR REPLACE + 分批 executemany,一次调用一个事务:
   重复同步同一段日期不会写出重复行(由表主键决定),也不靠"先删后插"制造空窗。
3. pandas 出来的值直接喂 sqlite3 会踩坑:numpy.int64/numpy.float64 报 "unsupported type",
   NaN / pd.NA / NaT 要么报错要么入库成怪值。这里统一规范成 None / Python 原生标量 /
   'YYYY-MM-DD' 字符串,缺失一律入库为 NULL(SQL 里 IS NULL 能查到,NaN 查不到)。
4. data/ 目录不存在时 sqlite3 只抛一句 "unable to open database file",connect() 自动建目录。
5. sync_log.ts 若写本地时间又不带时区,换台机器对账就差 8 小时;这里一律写 UTC ISO 8601。
6. 表名 / 列名拼进 SQL 之前做标识符校验并加引号,防止外来列名(例如 DataFrame 列)夹带 SQL。
7. 任何失败直接抛异常,不 try/except 吞掉后返回 0 或空值冒充成功。
"""
from __future__ import annotations

import os
import re
import sqlite3
from datetime import date, datetime, timezone
from decimal import Decimal

from ..config import DB_PATH, SCHEMA_PATH

__all__ = [
    "connect", "ensure_schema", "upsert_many", "last_date",
    "get_meta", "set_meta", "log_sync",
]

BATCH_SIZE = 5000   # executemany 每批行数;单次 upsert_many 仍是一个事务

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(name: str) -> str:
    """校验并引用 SQL 标识符(表名/列名)。非法名字直接抛错,不进 SQL。"""
    if not isinstance(name, str) or not _IDENT_RE.match(name):
        raise ValueError(f"非法 SQL 标识符: {name!r}")
    return f'"{name}"'


def _sql_value(v):
    """把单个值规范成 sqlite3 能绑定的原生类型;各种"缺失"一律 → None。

    快路径:str / int / float / None 直接返回(NaN → None)。
    慢路径:numpy 标量 → .item();pd.NA / pd.NaT / np.datetime64('NaT') → None;
    datetime / pd.Timestamp / date → 'YYYY-MM-DD'(有时分秒则保留 ISO 全串);Decimal → float。
    其余类型原样交给 sqlite3,不支持会由它显式报错。
    """
    if v is None:
        return None
    t = type(v)
    if t is str or t is int or t is bytes:
        return v
    if t is float:
        return None if v != v else v          # NaN → NULL
    if t is bool:
        return int(v)
    name = t.__name__
    if name in ("NAType", "NaTType"):         # pandas 缺失哨兵,不 import pandas
        return None
    if name == "datetime64":                  # numpy 日期标量;.item() 在 ns 精度下会变成 int,单独处理
        return None if v != v else str(v)[:10]
    if hasattr(v, "dtype") and hasattr(v, "item"):   # numpy 标量 → Python 原生
        return _sql_value(v.item())
    if isinstance(v, datetime):               # 含 pd.Timestamp
        if (v.hour, v.minute, v.second, v.microsecond) == (0, 0, 0, 0):
            return v.strftime("%Y-%m-%d")
        return v.isoformat(sep=" ")
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    """打开数据库:自动建目录;PRAGMA journal_mode=WAL, synchronous=NORMAL;row_factory=sqlite3.Row。

    timeout=30s:另一进程(如同步脚本)持有写锁时等它,而不是立刻 "database is locked"。
    """
    is_file = path != ":memory:" and not path.startswith("file:")
    if is_file:
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    if is_file and str(mode).lower() != "wal":
        conn.close()
        raise RuntimeError(f"无法把 {path} 切到 WAL 模式(当前 {mode!r}),拒绝带病运行")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """读 SCHEMA_PATH 并整体执行。schema 全部 CREATE ... IF NOT EXISTS,重复调用无副作用。"""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        sql = f.read()
    conn.executescript(sql)
    conn.commit()


def upsert_many(conn: sqlite3.Connection, table: str, rows: list[dict]) -> int:
    """INSERT OR REPLACE 批量写入,返回写入行数。

    - 列由 rows[0].keys() 决定;后续行缺列直接 KeyError(不静默补 NULL),多出的键忽略。
    - 空列表返回 0,不碰数据库。
    - 按 BATCH_SIZE 分批 executemany,整个调用是一个事务,结束时 commit;中途异常则回滚。
    - 值经 _sql_value 规范化(numpy 标量 / NaN / pd.NA / 日期)。
    """
    if not rows:
        return 0
    cols = list(rows[0].keys())
    if not cols:
        raise ValueError("upsert_many: rows[0] 没有任何列")
    col_sql = ", ".join(_ident(c) for c in cols)
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT OR REPLACE INTO {_ident(table)} ({col_sql}) VALUES ({placeholders})"

    written = 0
    try:
        cur = conn.cursor()
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            params = [[_sql_value(r[c]) for c in cols] for r in batch]
            cur.executemany(sql, params)
            written += cur.rowcount if cur.rowcount >= 0 else len(batch)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return written


def last_date(conn: sqlite3.Connection, table: str, code: str) -> str | None:
    """SELECT MAX(date) FROM table WHERE code=?;无行返回 None。用于增量同步的断点。"""
    row = conn.execute(
        f"SELECT MAX(date) FROM {_ident(table)} WHERE code = ?", (code,)
    ).fetchone()
    return row[0] if row is not None else None


def get_meta(conn: sqlite3.Connection, key: str, default=None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row is not None else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """写 meta(key 唯一,重复写覆盖)。非 None 值一律按 str 入库,与 get_meta 返回类型对齐。"""
    v = None if value is None else str(value)
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, v))
    conn.commit()


def log_sync(conn: sqlite3.Connection, task: str, code: str | None, rows: int,
             status: str, message: str = "") -> None:
    """写一条 sync_log;ts 为 UTC ISO 8601(秒精度,带 +00:00)。"""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO sync_log (ts, task, code, rows, status, message) VALUES (?, ?, ?, ?, ?, ?)",
        (ts, task, code, int(rows), status, message),
    )
    conn.commit()
