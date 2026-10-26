#!/usr/bin/env python
"""全市场日线并行同步(4 路)。

为什么要并行:baostock 单连接拉 10.7 年日线约 15.4 秒/只,5221 只需 22.4 小时;
实测 4 路并行提速 3.0 倍 → 约 7.5 小时。不再加路数:免费数据源,稳定优先。

并发安全:SQLite WAL 允许多读单写,db.connect 已设 busy timeout=30s;
每只股票一次小事务(约 2800 行),冲突概率低且会自动等待。
断点续传:每只股票从 last_date+1 拉起,中断后重跑即可续,不会重复写。

用法:
    python scripts/sync_all.py --workers 4 --since 2015-01-01
    caffeinate -i python scripts/sync_all.py ...   # 防止 Mac 睡眠中断
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ashare_forge.config import DB_PATH  # noqa: E402
from ashare_forge.data import db, sync  # noqa: E402


def _worker(args: tuple) -> dict:
    wid, codes, since, end, db_path = args
    from ashare_forge.data.baostock_source import BaostockSource
    conn = db.connect(db_path)
    t0 = time.time()
    try:
        with BaostockSource() as src:
            res = sync.sync_daily(src, conn, codes=codes, since=since, limit=None, end=end)
        res["worker"] = wid
        res["seconds"] = time.time() - t0
        return res
    except Exception as e:  # noqa: BLE001
        return {"worker": wid, "error": f"{type(e).__name__}: {e}", "codes": 0, "rows": 0,
                "failed": [], "seconds": time.time() - t0}
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="全市场日线并行同步")
    ap.add_argument("--workers", type=int, default=4, help="并行进程数(实测 4 路最优)")
    ap.add_argument("--since", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--limit", type=int, default=None, help="只同步前 N 只(验证用)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="只重跑'库内无任何日线'的股票(补 baostock 临时故障造成的缺口)")
    a = ap.parse_args()

    conn = db.connect(a.db)
    db.ensure_schema(conn)
    if a.retry_failed:
        # baostock 偶发"接收数据异常",重试 3 次仍可能失败(实测一轮 144 只);
        # 这些股票库内无任何日线,重跑即可补回(已验证同样代码稍后可正常拉取)。
        codes = [r[0] for r in conn.execute(
            "SELECT code FROM stocks WHERE code NOT IN (SELECT DISTINCT code FROM daily) ORDER BY code")]
        print(f"[all] 补漏模式:库内无日线的股票 {len(codes)} 只")
    else:
        codes = [r[0] for r in conn.execute("SELECT code FROM stocks ORDER BY code")]
    # 已同步到末日的股票由 sync_daily 内部 skip,这里不预过滤(保证断点续传语义单一)
    if a.limit:
        codes = codes[: a.limit]
    conn.close()
    if not codes:
        print("stocks 表为空,请先跑 discover_universe")
        return 1

    # 轮转分片:相邻代码分到不同进程,避免某片全是大盘股/全是次新
    shards = [codes[i :: a.workers] for i in range(a.workers)]
    print(f"[all] {len(codes)} 只 × {a.workers} 路 | since={a.since} end={a.end or '今天'}")
    print(f"[all] 每片 {[len(s) for s in shards]} 只 | 库 {a.db}")
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        results = pool.map(_worker, [(i, s, a.since, a.end, a.db) for i, s in enumerate(shards)])
    el = time.time() - t0

    rows = sum(r.get("rows", 0) for r in results)
    done = sum(r.get("codes", 0) for r in results)
    failed = [f for r in results for f in r.get("failed", [])]
    errs = [r for r in results if r.get("error")]
    print(f"\n[all] 完成 {done} 只 | 写入 {rows} 行 | 用时 {el/3600:.2f} 小时 "
          f"({el/max(done,1):.2f} 秒/只) | 单只失败 {len(failed)} | 进程级错误 {len(errs)}")
    for r in errs:
        print(f"  worker{r['worker']} 错误: {r['error']}")
    for f in failed[:10]:
        print(f"  失败 {f}")
    if failed:
        print(f"  (共 {len(failed)} 只失败,重跑本脚本即可只补这些)")
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # baostock 的 socket 不能 fork 继承
    raise SystemExit(main())
