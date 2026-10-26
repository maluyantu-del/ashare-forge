#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地研究看板:把数据层状态 / 任务卡 / 因子裁定台账渲染成一个自包含的单文件网页。

用途是"研究员自己看的工作台"——当前哪些数据到位、哪些任务卡在跑、哪些因子已被裁定,
一屏看完。不对外发布,不连任何外部服务。

它修的坑(与常见"看板脚本"的区别):

1. **只读打开数据库**。后台可能正在跑全量同步(多进程写 WAL)。本脚本一律用
   ``file:...?mode=ro`` 连接,不建表、不写 meta、不触发同步;聚合查询一次全表扫描完成,
   不做逐股循环,避免长时间占住读事务影响 writer 的 checkpoint。
2. **缺数据显示"未运行",不拿别处的数字冒充**。因子裁定台账(``factor_trials``)尚未建立时,
   裁定区给出空态 + 本看板期望的字段清单;任务卡的中期记录原文单独标注为"非台账来源",
   不折算成台账条目,也不参与 KPI。
3. **KPI 诚实**。"门槛通过"按 ``通过/总数`` 原样展示(台账未建时就是 0/0),不挑好看的指标;
   样本覆盖率直接摆在顶部,而不是藏进脚注。
4. **负结果用中性色**。FAIL / 不成立用灰蓝(``--neutral``),不用刺眼红——红色会把
   "如实记录的否定结论"视觉上误导成"程序出故障"。
5. **真正自包含**。无 CDN、无外链、无网络请求、无外部字体;净值 sparkline 由 Python 直接
   生成内联 SVG,禁用 JS 也能看到形状(JS 只做主题切换与表格筛选)。

台账(``factor_trials``)的期望结构 —— 裁定流程落地时按此写入,本看板即自动读出:

    列名(方括号内为等价别名,任选其一)
      factor        [factor_name, name]      因子名
      task_id       [task, card]             任务卡 id
      signature     [mechanism_signature]    机制签名(查重用)
      leakage_passed[leakage]                泄漏测试 True/False
      ic_mean       [ic, rank_ic]            RankIC 均值(须对齐可交易收益)
      icir          [ic_ir]                  ICIR
      excess_sharpe [long_excess_sharpe]     多头超额夏普
      null_passed   [n_null_passed]          零校准通过的指标数
      null_metrics  [n_null_metrics]         零校准比较的指标总数
      verdict       [overall_pass, decision] 门槛裁定 PASS/FAIL
      n_trials      [trials]                 同一(域,因子族)累计试验次数
      nav_json      [nav, excess_nav]        净值序列(JSON:数组 / {日期:值} / [[日期,值],...])
      nav_path      [nav_file]               或者给一个 csv/json 文件路径(相对仓库根)
      ts            [run_at, created_at]     运行时间
      note          [notes]                  备注

    存放位置(按序探测,全部读取并按 trial 去重):
      data/ashare.db  表 factor_trials       ← 推荐(与数据同库,同步进程不受影响)
      data/ledger.db  表 factor_trials
      data/factor_trials.jsonl / .json、reports/factor_trials.jsonl / .json

CLI:
    .venv/bin/python scripts/build_dashboard.py [--out reports/dashboard.html] [--db data/ashare.db]
    .venv/bin/python scripts/build_dashboard.py --selftest     # 用合成台账走通渲染路径
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import html
import io
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "ashare.db")
TASKS_DIR = os.path.join(ROOT, "tasks")
DOCS_DIR = os.path.join(ROOT, "docs")
CONTRACT_PATH = os.path.join(ROOT, "contracts", "ashare.yaml")
ENGINE_DIR = os.path.join(ROOT, "ashare_forge", "engine")
FACTORS_DIR = os.path.join(ROOT, "ashare_forge", "factors")
DEFAULT_OUT = os.path.join(ROOT, "reports", "dashboard.html")

# 集成发现:台账模块建的表名是 forge_trials(与 Forge 同源),看板原先写死 factor_trials,
# 导致裁定区永远显示"未运行"。此处按优先级依次尝试,兼容两种命名。
LEDGER_TABLES = ("forge_trials", "factor_trials")
LEDGER_TABLE = LEDGER_TABLES[0]
LEDGER_SQLITE_CANDIDATES = (DB_PATH, os.path.join(DATA_DIR, "ledger.db"))
LEDGER_FILE_CANDIDATES = (
    os.path.join(DATA_DIR, "factor_trials.jsonl"),
    os.path.join(DATA_DIR, "factor_trials.json"),
    os.path.join(ROOT, "reports", "factor_trials.jsonl"),
    os.path.join(ROOT, "reports", "factor_trials.json"),
)

MISSING = "未运行"          # 全站统一的"没有数据"字样,绝不用 0 或占位数字代替
DASH = "—"


# ============================================================ 读取:数据库(只读)
def _ro_conn(path: str) -> sqlite3.Connection | None:
    """只读连接。库不存在或打不开一律返回 None,由调用方降级为"未运行"。"""
    if not os.path.exists(path):
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=8.0)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


def _one(conn, sql, params=()):
    try:
        row = conn.execute(sql, params).fetchone()
    except sqlite3.Error:
        return None
    return tuple(row) if row is not None else None


def _all(conn, sql, params=()):
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    except sqlite3.Error:
        return []


def _has_table(conn, name: str) -> bool:
    return bool(_one(conn, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)))


def collect_data_layer(db_path: str) -> dict:
    """数据层状态。口径与 sync.data_quality_report 保持一致(同样的 SQL 定义)。"""
    d: dict = {"db_path": db_path, "db_exists": os.path.exists(db_path), "available": False}
    if os.path.exists(db_path):
        try:
            d["db_bytes"] = os.path.getsize(db_path)
        except OSError:
            d["db_bytes"] = None
    conn = _ro_conn(db_path)
    if conn is None:
        return d
    try:
        d["available"] = True
        d["n_stocks"] = (_one(conn, "SELECT COUNT(*) FROM stocks") or (None,))[0]
        d["n_delisted"] = (_one(conn, "SELECT COUNT(*) FROM stocks WHERE delisted=1") or (None,))[0]
        d["board_dist"] = _all(conn, "SELECT board, COUNT(*) FROM stocks GROUP BY board ORDER BY 2 DESC")
        # 一次扫描拿全部 daily 聚合(939k 行以上,分多次扫描没必要)
        agg = _one(conn,
                   "SELECT COUNT(*), MIN(date), MAX(date), "
                   "AVG(CASE WHEN tradestatus=0 THEN 1.0 ELSE 0.0 END), "
                   "AVG(CASE WHEN isST=1 THEN 1.0 ELSE 0.0 END) FROM daily")
        if agg:
            d["n_daily_rows"], d["date_min"], d["date_max"], d["suspended_ratio"], d["st_ratio"] = agg
        d["n_daily_codes"] = (_one(conn, "SELECT COUNT(DISTINCT code) FROM daily") or (None,))[0]
        d["rows_per_year"] = _all(conn,
                                  "SELECT substr(date,1,4) AS y, COUNT(*) FROM daily GROUP BY y ORDER BY y")
        d["n_missing_adj"] = (_one(conn,
                                   "SELECT COUNT(*) FROM stocks WHERE code NOT IN "
                                   "(SELECT DISTINCT code FROM adj_factor)") or (None,))[0]
        # 幸存者偏差可见性:曾在股池快照出现、最新快照已消失(定义与 sync._n_vanished 一致)
        d["n_vanished"] = (_one(conn,
                                "SELECT COUNT(*) FROM stocks WHERE last_seen < "
                                "(SELECT MAX(day) FROM universe_snapshot)") or (None,))[0]
        snap = _one(conn, "SELECT COUNT(DISTINCT day), MIN(day), MAX(day) FROM universe_snapshot")
        if snap:
            d["n_snapshots"], d["snap_min"], d["snap_max"] = snap
        d["index_rows"] = _all(conn,
                               "SELECT code, COUNT(*), MIN(date), MAX(date) FROM index_daily "
                               "GROUP BY code ORDER BY code")
        d["last_sync"] = (_one(conn, "SELECT MAX(ts) FROM sync_log") or (None,))[0]
        d["last_sync_by_task"] = _all(conn, "SELECT task, MAX(ts) FROM sync_log GROUP BY task ORDER BY task")
        d["sync_status"] = _all(conn,
                                "SELECT task, status, COUNT(*) FROM sync_log GROUP BY task, status "
                                "ORDER BY task, status")
        # 未解决失败:每个 (task, code) 只看最后一条,仍是 fail 才算
        d["n_failures"] = (_one(conn,
                                "SELECT COUNT(*) FROM sync_log s WHERE status='fail' AND id = "
                                "(SELECT MAX(id) FROM sync_log t WHERE t.task=s.task AND t.code IS s.code)"
                                ) or (0,))[0]
        d["meta"] = {k: v for k, v in _all(conn, "SELECT key, value FROM meta ORDER BY key")}
        log = d["meta"].get("holdout_unlocked_log") or ""
        d["holdout_unlocks"] = [ln for ln in str(log).splitlines() if ln.strip()]
    finally:
        conn.close()
    return d


# ============================================================ 读取:任务卡 / 合同 / 文档
def _load_yaml(path: str):
    try:
        import yaml
    except ImportError:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:                                    # noqa: BLE001 — 坏文件不该炸掉整张看板
        return None


def collect_tasks(tasks_dir: str = TASKS_DIR) -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(tasks_dir, "*.yaml")) + glob.glob(os.path.join(tasks_dir, "*.yml"))):
        y = _load_yaml(p)
        if not isinstance(y, dict):
            out.append({"task_id": os.path.basename(p), "name": DASH, "status": "解析失败",
                        "core_question": [], "path": p, "notes_files": []})
            continue
        cq = y.get("core_question") or []
        if isinstance(cq, str):
            cq = [cq]
        stem = os.path.splitext(os.path.basename(p))[0]
        notes = sorted(glob.glob(os.path.join(tasks_dir, f"{stem}*.md")))
        hyp = y.get("hypotheses") or {}
        if isinstance(hyp, dict):
            hyp = [f"{k}: {v}" for k, v in hyp.items()]
        elif isinstance(hyp, str):
            hyp = [hyp]
        dead = y.get("known_dead_ends") or []
        if isinstance(dead, str):
            dead = [dead]
        out.append({
            "task_id": str(y.get("task_id") or stem),
            "name": str(y.get("name") or DASH),
            "category": str(y.get("category") or ""),
            "status": str(y.get("status") or MISSING),
            "core_question": [str(x) for x in cq],
            "rationale": str(y.get("economic_rationale") or "").strip(),
            "hypotheses": [str(x) for x in hyp],
            "falsification": str(y.get("falsification") or "").strip(),
            "dead_ends": [str(x) for x in dead],
            "factor_module": str(y.get("factor_module") or ""),
            "path": os.path.relpath(p, ROOT),
            "notes_files": [{"path": os.path.relpath(n, ROOT), "text": _read_text(n)} for n in notes],
        })
    return out


def _read_text(path: str, limit: int = 20000) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read(limit)
    except OSError:
        return ""


def collect_limitations(md_path: str | None = None) -> list[str]:
    """抽 docs/PHASE1-DATA-LAYER.md 的"已知限制"小节,逐条返回(原文,仅去 markdown 记号)。"""
    md_path = md_path or os.path.join(DOCS_DIR, "PHASE1-DATA-LAYER.md")
    text = _read_text(md_path)
    if not text:
        return []
    lines, grabbing, buf = text.splitlines(), False, []
    for ln in lines:
        s = ln.rstrip()
        if s.startswith("#"):
            if grabbing:
                break
            grabbing = "已知限制" in s
            continue
        if grabbing:
            buf.append(s)
    items, cur = [], ""
    for s in buf:
        st = s.strip()
        if not st:
            continue
        is_new = st[:1].isdigit() and "." in st[:4]
        if is_new:
            if cur:
                items.append(cur)
            cur = st.split(".", 1)[1].strip()
        elif st.startswith("-"):
            if cur:
                items.append(cur)
            cur = st.lstrip("-").strip()
        else:
            cur = f"{cur} {st}".strip()
    if cur:
        items.append(cur)
    return [_demark(x) for x in items if x]


def _demark(s: str) -> str:
    """去掉 markdown 的 ** ` 记号(看板用纯文本渲染,避免把记号当内容显示)。"""
    return s.replace("**", "").replace("`", "").strip()


def collect_contract() -> dict:
    y = _load_yaml(CONTRACT_PATH) or {}
    disc = y.get("disclosure") or []
    if isinstance(disc, str):
        disc = [disc]
    gates = y.get("gates_draft") or {}
    return {
        "contract_id": str(y.get("contract_id") or MISSING),
        "version": y.get("version"),
        "holdout_cutoff": str(((y.get("holdout") or {}).get("cutoff")) or MISSING),
        "disclosure": [str(x) for x in disc],
        "gates": gates if isinstance(gates, dict) else {},
        "costs": ((y.get("tradability_rules") or {}).get("costs")) or {},
        "crisis_windows": ((y.get("regimes") or {}).get("crisis_windows")) or [],
        "exists": os.path.exists(CONTRACT_PATH),
    }


# ============================================================ 读取:因子裁定台账
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "factor": ("factor", "factor_name", "name", "signal", "signal_name"),
    "task_id": ("task_id", "task", "card", "task_card"),
    "signature": ("signature", "mechanism_signature", "mech_sig", "sig"),
    "leakage": ("leakage_passed", "leakage", "prefix_recompute_passed", "leak_test", "leakage_test"),
    "ic": ("ic_mean", "ic", "rank_ic", "rank_ic_mean"),
    "icir": ("icir", "ic_ir"),
    "excess_sharpe": ("excess_sharpe", "long_excess_sharpe", "excess_sr"),
    "sharpe": ("sharpe", "ann_sharpe"),
    "null_passed": ("null_passed", "n_null_passed", "null_n_passed"),
    "null_total": ("null_metrics", "n_null_metrics", "null_n_metrics", "null_total"),
    "null_str": ("null_pass", "null_calibration", "null_verdict"),
    "verdict": ("verdict", "overall_pass", "gate_verdict", "gates_verdict", "decision"),
    "n_trials": ("n_trials", "trials", "trial_count"),
    "ts": ("ts", "run_at", "timestamp", "created_at", "date"),
    "nav_json": ("nav_json", "nav", "equity", "nav_series", "excess_nav", "excess_nav_json"),
    "nav_path": ("nav_path", "nav_file", "equity_path"),
    "note": ("note", "notes", "comment"),
    "n_codes": ("n_codes", "n_stocks", "universe_size", "sample_codes"),
    "n_days": ("n_days", "days", "sample_days"),
}


def _pick(row: dict, key: str):
    for alias in FIELD_ALIASES[key]:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    return None


def _as_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "pass", "passed", "ok", "通过", "y"):
        return True
    if s in ("0", "false", "no", "fail", "failed", "不通过", "未通过", "n"):
        return False
    return None


def _as_float(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f                      # NaN → None(显示"未运行"而不是 nan)


def _as_int(v):
    f = _as_float(v)
    return None if f is None else int(round(f))


def _parse_nav(raw, nav_path: str | None) -> list[float]:
    """净值序列:接受 JSON 数组 / {日期:值} / [[日期,值],...],或一个 csv/json 文件路径。"""
    data = None
    if raw is not None:
        data = raw
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except (ValueError, TypeError):
                data = [x for x in (_as_float(p) for p in raw.replace(";", ",").split(",")) if x is not None]
    elif nav_path:
        p = nav_path if os.path.isabs(nav_path) else os.path.join(ROOT, nav_path)
        if not os.path.exists(p):
            return []
        txt = _read_text(p, limit=4_000_000)
        if p.lower().endswith(".json"):
            try:
                data = json.loads(txt)
            except ValueError:
                return []
        else:                                          # csv/tsv:取最后一个可解析为数字的列
            vals = []
            sniff = "\t" if ("\t" in txt.splitlines()[0] if txt.splitlines() else False) else ","
            for rec in csv.reader(io.StringIO(txt), delimiter=sniff):
                for cell in reversed(rec):
                    f = _as_float(cell)
                    if f is not None:
                        vals.append(f)
                        break
            data = vals
    if data is None:
        return []
    if isinstance(data, dict):
        data = [data[k] for k in sorted(data.keys())]
    out = []
    for item in (data if isinstance(data, (list, tuple)) else []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            f = _as_float(item[1])
        elif isinstance(item, dict):
            f = _as_float(item.get("nav", item.get("value")))
        else:
            f = _as_float(item)
        if f is not None:
            out.append(f)
    return out


def _normalize_trial(row: dict, source: str) -> dict:
    v_raw = _pick(row, "verdict")
    v_bool = _as_bool(v_raw)
    if v_bool is True:
        verdict = "PASS"
    elif v_bool is False:
        verdict = "FAIL"
    else:
        verdict = str(v_raw).upper() if v_raw is not None else MISSING
    nav = _parse_nav(_pick(row, "nav_json"), _pick(row, "nav_path"))
    np_, nt_ = _as_int(_pick(row, "null_passed")), _as_int(_pick(row, "null_total"))
    null_str = _pick(row, "null_str")
    if np_ is not None and nt_ is not None:
        null_txt = f"{np_}/{nt_}"
    elif null_str is not None:
        null_txt = str(null_str)
    elif np_ is not None:
        null_txt = str(np_)
    else:
        null_txt = MISSING
    return {
        "factor": str(_pick(row, "factor") or MISSING),
        "task_id": str(_pick(row, "task_id") or DASH),
        "signature": str(_pick(row, "signature") or DASH),
        "leakage": _as_bool(_pick(row, "leakage")),
        "ic": _as_float(_pick(row, "ic")),
        "icir": _as_float(_pick(row, "icir")),
        "excess_sharpe": _as_float(_pick(row, "excess_sharpe")),
        "sharpe": _as_float(_pick(row, "sharpe")),
        "null_txt": null_txt,
        "null_passed": np_, "null_total": nt_,
        "verdict": verdict,
        "n_trials": _as_int(_pick(row, "n_trials")),
        "ts": str(_pick(row, "ts") or ""),
        "nav": nav,
        "note": str(_pick(row, "note") or ""),
        "n_codes": _as_int(_pick(row, "n_codes")),
        "n_days": _as_int(_pick(row, "n_days")),
        "source": source,
    }


def collect_trials() -> dict:
    """探测全部候选台账位置,合并去重。返回 {rows, sources, checked}。"""
    checked, sources, raw_rows = [], [], []
    for db in LEDGER_SQLITE_CANDIDATES:
        checked.append(os.path.relpath(db, ROOT))
        conn = _ro_conn(db)
        if conn is None:
            continue
        try:
            table = next((t for t in LEDGER_TABLES if _has_table(conn, t)), None)
            if table:
                try:
                    cur = conn.execute(f"SELECT * FROM {table}")   # 表名取自常量白名单,非外部输入
                    cols = [c[0] for c in cur.description]
                    got = [dict(zip(cols, tuple(r))) for r in cur.fetchall()]
                except sqlite3.Error:
                    got = []
                if got:
                    sources.append(f"{os.path.relpath(db, ROOT)}::{table}({len(got)})")
                    raw_rows += [(g, os.path.relpath(db, ROOT)) for g in got]
        finally:
            conn.close()
    for p in LEDGER_FILE_CANDIDATES:
        checked.append(os.path.relpath(p, ROOT))
        if not os.path.exists(p):
            continue
        txt = _read_text(p, limit=8_000_000)
        got = []
        if p.endswith(".jsonl"):
            for ln in txt.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    got.append(obj)
        else:
            try:
                obj = json.loads(txt)
            except ValueError:
                obj = None
            if isinstance(obj, list):
                got = [o for o in obj if isinstance(o, dict)]
            elif isinstance(obj, dict):
                inner = obj.get(LEDGER_TABLE) or obj.get("trials") or obj.get("rows")
                got = [o for o in inner if isinstance(o, dict)] if isinstance(inner, list) else [obj]
        if got:
            sources.append(f"{os.path.relpath(p, ROOT)}({len(got)})")
            raw_rows += [(g, os.path.relpath(p, ROOT)) for g in got]

    seen, rows = set(), []
    for g, src in raw_rows:
        t = _normalize_trial(g, src)
        key = (t["factor"], t["task_id"], t["ts"], t["signature"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(t)
    return {**_with_current(rows), "sources": sources, "checked": checked}


def _with_current(rows: list[dict]) -> dict:
    """当前裁定 = 每个因子最近一次非草稿条目。

    更早的裁定(含已撤回批次,如 BATCH01)仍在台账里、仍计入 DSR 的试验次数,
    但不进"当前裁定"表与 KPI,避免新旧口径混读。
    """
    rows = sorted(rows, key=lambda r: (r["ts"] or "", r["factor"]), reverse=True)
    judged = [r for r in rows if r["verdict"] not in ("DRAFT", MISSING)]
    current, seen_f = [], set()
    for r in judged:
        if r["factor"] not in seen_f:
            seen_f.add(r["factor"])
            current.append(r)
    n_hist = len(judged) - len(current)
    # 裁定字段缺失的畸形条目照样显示(显示为缺失占位,便于发现写账问题),但不进 KPI 分母
    for r in rows:
        if r["verdict"] == MISSING and r["factor"] not in seen_f:
            seen_f.add(r["factor"])
            current.append(r)
    return {"rows": rows, "current": current, "n_superseded": n_hist}


# ============================================================ 渲染工具
def esc(s) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def fmt_int(v) -> str:
    return MISSING if v is None else f"{int(v):,}"


def fmt_num(v, nd: int = 4, sign: bool = False) -> str:
    if v is None:
        return MISSING
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def fmt_pct(v, nd: int = 1) -> str:
    return MISSING if v is None else f"{v * 100:.{nd}f}%"


def sparkline(values: list[float], width: int = 128, height: int = 30, pad: int = 3, uid: str = "s") -> str:
    """内联 SVG 净值曲线。涨用中性强调色、跌用中性灰蓝(负结果不用刺眼红)。"""
    if not values or len(values) < 2:
        return f'<span class="na" title="台账未保存净值序列">{MISSING}</span>'
    v = values
    if len(v) > 160:                                   # 等间隔抽样,保形状不保每一点
        step = len(v) / 160.0
        v = [v[min(len(values) - 1, int(i * step))] for i in range(160)]
        v[-1] = values[-1]
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1.0
    n = len(v) - 1
    pts = [(pad + i / n * (width - 2 * pad), pad + (1 - (x - lo) / rng) * (height - 2 * pad))
           for i, x in enumerate(v)]
    col = "var(--accent)" if v[-1] >= v[0] else "var(--neutral)"
    line = " ".join(("M" if i == 0 else "L") + f"{x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts))
    area = (f"M{pts[0][0]:.1f} {height - pad:.1f} "
            + " ".join(f"L{x:.1f} {y:.1f}" for x, y in pts)
            + f" L{pts[-1][0]:.1f} {height - pad:.1f} Z")
    base = ""
    if lo <= 1.0 <= hi:                                 # 净值 1.0 参考线(本金线)
        y1 = pad + (1 - (1.0 - lo) / rng) * (height - 2 * pad)
        base = (f'<line x1="{pad}" x2="{width - pad}" y1="{y1:.1f}" y2="{y1:.1f}" '
                f'stroke="var(--border)" stroke-width="1" stroke-dasharray="2 3"/>')
    title = f"净值 {v[0]:.3f} → {v[-1]:.3f}(共 {len(values)} 点)"
    return (f'<svg class="spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" '
            f'aria-label="{esc(title)}"><title>{esc(title)}</title>'
            f'<defs><linearGradient id="g{uid}" x1="0" x2="0" y1="0" y2="1">'
            f'<stop offset="0" stop-color="{col}" stop-opacity=".22"/>'
            f'<stop offset="1" stop-color="{col}" stop-opacity="0"/></linearGradient></defs>'
            f'{base}<path d="{area}" fill="url(#g{uid})"/>'
            f'<path d="{line}" fill="none" stroke="{col}" stroke-width="1.5" stroke-linejoin="round"/>'
            f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="2.2" fill="{col}"/></svg>')


def pill(text: str, kind: str = "idle") -> str:
    return f'<span class="pill {kind}"><i class="d"></i>{esc(text)}</span>'


def na(text: str = MISSING) -> str:
    return f'<span class="na">{esc(text)}</span>'


# ============================================================ 样式(自包含,无外链字体)
CSS = """
:root{
  --ground:#f1f3f5; --surface:#ffffff; --surface-2:#f6f8fa; --surface-3:#eef1f4;
  --border:#dde2e8; --border-2:#c9d1da;
  --ink:#11181f; --ink-dim:#39444f; --muted:#6a7581; --faint:#95a0ac;
  --accent:#1f6d90; --accent-soft:rgba(31,109,144,.10);
  --good:#1c8a61; --good-soft:rgba(28,138,97,.11);
  --neutral:#5b6b7c; --neutral-soft:rgba(91,107,124,.12);
  --warn:#96701d; --warn-soft:rgba(150,112,29,.12);
  --idle:#7d8894; --idle-soft:rgba(125,136,148,.10);
  --shadow:0 1px 2px rgba(17,24,31,.05), 0 4px 14px rgba(17,24,31,.05);
  --sans:system-ui,-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  --mono:ui-monospace,"SF Mono","Menlo","Consolas","Liberation Mono",monospace;
  --r:9px;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#0e1216; --surface:#151b21; --surface-2:#1b232b; --surface-3:#212b34;
    --border:#28323c; --border-2:#3a4754;
    --ink:#e5eaef; --ink-dim:#b2bdc8; --muted:#87939f; --faint:#5d6875;
    --accent:#56a3c7; --accent-soft:rgba(86,163,199,.14);
    --good:#43b285; --good-soft:rgba(67,178,133,.14);
    --neutral:#8394a6; --neutral-soft:rgba(131,148,166,.14);
    --warn:#c9a047; --warn-soft:rgba(201,160,71,.14);
    --idle:#75808c; --idle-soft:rgba(117,128,140,.12);
    --shadow:0 1px 2px rgba(0,0,0,.45), 0 6px 20px rgba(0,0,0,.28);
  }
}
:root[data-theme="dark"]{
  --ground:#0e1216; --surface:#151b21; --surface-2:#1b232b; --surface-3:#212b34;
  --border:#28323c; --border-2:#3a4754;
  --ink:#e5eaef; --ink-dim:#b2bdc8; --muted:#87939f; --faint:#5d6875;
  --accent:#56a3c7; --accent-soft:rgba(86,163,199,.14);
  --good:#43b285; --good-soft:rgba(67,178,133,.14);
  --neutral:#8394a6; --neutral-soft:rgba(131,148,166,.14);
  --warn:#c9a047; --warn-soft:rgba(201,160,71,.14);
  --idle:#75808c; --idle-soft:rgba(117,128,140,.12);
  --shadow:0 1px 2px rgba(0,0,0,.45), 0 6px 20px rgba(0,0,0,.28);
}

*{box-sizing:border-box;}
html{ -webkit-text-size-adjust:100%; }
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--sans); font-size:14px; line-height:1.55;
  -webkit-font-smoothing:antialiased;
}
.num{font-family:var(--mono); font-variant-numeric:tabular-nums;}
.na{color:var(--faint); font-family:var(--mono); font-size:12px;}
a{color:var(--accent);}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px;}

.wrap{max-width:1220px; margin:0 auto; padding:20px 24px 56px;}

/* ---------- header ---------- */
header.hd{background:var(--surface); border-bottom:1px solid var(--border);}
.hd-in{max-width:1220px; margin:0 auto; padding:14px 24px; display:flex; align-items:center; gap:14px; flex-wrap:wrap;}
.mark{width:30px; height:30px; border-radius:8px; background:var(--accent); color:#fff; flex:none;
  display:grid; place-items:center; font-family:var(--mono); font-weight:700; font-size:12px; letter-spacing:-.5px;}
.hd h1{font-size:16px; margin:0; letter-spacing:-.2px; font-weight:650;}
.hd .sub{font-size:11.5px; color:var(--muted); margin-top:1px;}
.hd .right{margin-left:auto; display:flex; align-items:center; gap:10px;}
.stamp{font-family:var(--mono); font-size:11px; color:var(--faint); text-align:right; line-height:1.45;}
.tbtn{width:34px; height:34px; display:grid; place-items:center; border-radius:var(--r);
  border:1px solid var(--border); background:var(--surface-2); color:var(--ink-dim); cursor:pointer;}
.tbtn:hover{border-color:var(--accent); color:var(--accent);}

/* ---------- pipeline ---------- */
.pipe{display:flex; gap:8px; flex-wrap:wrap; margin:18px 0 22px;}
.step{flex:1 1 150px; min-width:140px; background:var(--surface); border:1px solid var(--border);
  border-radius:var(--r); padding:10px 12px; box-shadow:var(--shadow); position:relative;}
.step .idx{font-family:var(--mono); font-size:10px; color:var(--faint);}
.step .nm{font-size:12.5px; font-weight:600; margin-top:1px;}
.step .st{font-family:var(--mono); font-size:11px; color:var(--muted); margin-top:3px; overflow-wrap:anywhere;}
.step::after{content:""; position:absolute; left:0; top:8px; bottom:8px; width:2px; border-radius:2px; background:var(--border-2);}
.step.on::after{background:var(--good);}
.step.run::after{background:var(--accent);}
.step.idle::after{background:var(--idle);}

/* ---------- section ---------- */
.sec{margin:26px 0 0;}
.sec-h{display:flex; align-items:baseline; gap:10px; margin:0 2px 10px; flex-wrap:wrap;}
.sec-h h2{font-size:14px; margin:0; letter-spacing:-.1px;}
.sec-h .n{font-family:var(--mono); font-size:11px; color:var(--muted);
  border:1px solid var(--border); border-radius:20px; padding:1px 8px;}
.sec-h .note{font-size:11.5px; color:var(--muted); margin-left:auto;}

/* ---------- kpi ---------- */
.kpis{display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:10px;}
.kpi{background:var(--surface); border:1px solid var(--border); border-radius:var(--r);
  padding:12px 14px; box-shadow:var(--shadow); position:relative; overflow:hidden;}
.kpi .lab{font-size:10.5px; color:var(--muted); letter-spacing:.4px;}
.kpi .big{font-family:var(--mono); font-size:22px; font-weight:600; margin-top:6px; letter-spacing:-.6px; overflow-wrap:anywhere;}
.kpi .foot{font-size:11px; color:var(--faint); margin-top:3px;}
.kpi::after{content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--border-2);}
.kpi.accent::after{background:var(--accent);} .kpi.accent .big{color:var(--accent);}
.kpi.good::after{background:var(--good);} .kpi.good .big{color:var(--good);}
.kpi.neutral::after{background:var(--neutral);} .kpi.neutral .big{color:var(--neutral);}
.kpi.warn::after{background:var(--warn);} .kpi.warn .big{color:var(--warn);}
.kpi.idle::after{background:var(--idle);} .kpi.idle .big{color:var(--idle); font-size:17px;}

.bar{height:4px; border-radius:3px; background:var(--surface-3); margin-top:8px; overflow:hidden;}
.bar i{display:block; height:100%; background:var(--accent);}

/* ---------- cards ---------- */
.grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:10px;}
.card{background:var(--surface); border:1px solid var(--border); border-radius:var(--r);
  padding:13px 15px; box-shadow:var(--shadow); min-width:0;}
.card h3{font-size:12.5px; margin:0 0 9px; color:var(--ink); font-weight:650;
  display:flex; align-items:center; gap:7px;}
.card h3 .c{font-family:var(--mono); font-size:11px; color:var(--muted); margin-left:auto;}
.kv{display:flex; justify-content:space-between; gap:12px; padding:4px 0; font-size:12px;
  border-bottom:1px solid var(--border);}
.kv:last-child{border-bottom:0;}
.kv .k{color:var(--muted); flex:none;}
.kv .v{font-family:var(--mono); color:var(--ink-dim); text-align:right; overflow-wrap:anywhere;}
.yrow{display:flex; align-items:center; gap:8px; font-size:11.5px; padding:2px 0;}
.yrow .y{font-family:var(--mono); color:var(--muted); width:34px; flex:none;}
.yrow .t{flex:1; height:8px; background:var(--surface-3); border-radius:2px; overflow:hidden; min-width:40px;}
.yrow .t i{display:block; height:100%; background:var(--accent-soft); border-right:2px solid var(--accent);}
.yrow .v{font-family:var(--mono); color:var(--ink-dim); width:64px; text-align:right; flex:none;}

/* ---------- pills ---------- */
.pill{font-family:var(--mono); font-size:11px; padding:2px 8px; border-radius:20px;
  display:inline-flex; align-items:center; gap:5px; white-space:nowrap;}
.pill .d{width:6px; height:6px; border-radius:50%; background:currentColor; flex:none;}
.pill.good{background:var(--good-soft); color:var(--good);}
.pill.neutral{background:var(--neutral-soft); color:var(--neutral);}
.pill.warn{background:var(--warn-soft); color:var(--warn);}
.pill.accent{background:var(--accent-soft); color:var(--accent);}
.pill.idle{background:var(--idle-soft); color:var(--idle);}

/* ---------- task cards ---------- */
.task{background:var(--surface); border:1px solid var(--border); border-radius:var(--r);
  padding:13px 15px; box-shadow:var(--shadow); min-width:0;}
.task .hd2{display:flex; align-items:center; gap:9px; flex-wrap:wrap;}
.task .tid{font-family:var(--mono); font-size:11px; color:var(--muted);}
.task .tnm{font-size:13.5px; font-weight:650;}
.task .q{margin:9px 0 0; padding:0 0 0 14px; font-size:12.5px; color:var(--ink-dim);}
.task .q li{margin:3px 0;}
.task .meta{margin-top:9px; font-size:11px; color:var(--faint); font-family:var(--mono); overflow-wrap:anywhere;}
details{margin-top:9px; border-top:1px solid var(--border); padding-top:8px;}
summary{cursor:pointer; font-size:11.5px; color:var(--muted); list-style:none;}
summary::-webkit-details-marker{display:none;}
summary::before{content:"▸ "; color:var(--faint);}
details[open] summary::before{content:"▾ ";}
details .body{margin-top:8px; font-size:12px; color:var(--ink-dim); line-height:1.6;}
details pre{margin:0; padding:10px 12px; background:var(--surface-2); border:1px solid var(--border);
  border-radius:6px; font-family:var(--mono); font-size:11px; color:var(--ink-dim);
  white-space:pre-wrap; overflow-wrap:anywhere; max-height:380px; overflow:auto;}

/* ---------- table ---------- */
.twrap{background:var(--surface); border:1px solid var(--border); border-radius:var(--r);
  box-shadow:var(--shadow); max-width:100%; overflow-x:auto;}
table{border-collapse:collapse; width:100%; min-width:900px;}
thead th{text-align:right; font-size:10.5px; font-weight:600; color:var(--muted); letter-spacing:.3px;
  padding:10px 12px; border-bottom:1px solid var(--border); white-space:nowrap;}
thead th.l{text-align:left;}
tbody td{padding:11px 12px; border-bottom:1px solid var(--border); text-align:right;
  white-space:nowrap; font-size:12.5px;}
tbody tr:last-child td{border-bottom:0;}
tbody tr:hover{background:var(--surface-2);}
td.l{text-align:left;}
td .fname{font-family:var(--mono); font-size:12px; color:var(--ink);}
td .fsig{font-family:var(--mono); font-size:10.5px; color:var(--faint); margin-top:2px;}
.spark{width:112px; height:30px; display:block;}
.v-pos{color:var(--good);} .v-neg{color:var(--neutral);}

/* ---------- empty state ---------- */
.empty{background:var(--surface); border:1px dashed var(--border-2); border-radius:var(--r);
  padding:20px 18px; text-align:left;}
.empty .t{font-size:13px; font-weight:650; color:var(--ink-dim);}
.empty .s{font-size:12px; color:var(--muted); margin-top:6px; line-height:1.65;}
.empty ul{margin:8px 0 0; padding-left:18px; font-size:11.5px; color:var(--faint); font-family:var(--mono);}
.empty li{margin:2px 0; overflow-wrap:anywhere;}

/* ---------- disclosure ---------- */
.disc{background:var(--surface-2); border:1px solid var(--border); border-radius:var(--r); padding:14px 16px;}
.disc ol{margin:0; padding-left:20px;}
.disc li{font-size:12px; color:var(--ink-dim); margin:5px 0; line-height:1.6;}
.disc .hl{color:var(--warn); font-weight:600;}
.foot{margin-top:24px; padding-top:14px; border-top:1px solid var(--border);
  display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap;
  font-size:11px; color:var(--faint); font-family:var(--mono);}

@media (max-width:760px){
  .wrap{padding:16px 16px 44px;}
  .hd-in{padding:12px 16px;}
  .kpis{grid-template-columns:repeat(auto-fit,minmax(140px,1fr));}
  .kpi .big{font-size:19px;}
  .stamp{display:none;}
  .sec-h .note{margin-left:0; flex-basis:100%;}
}
@media print{ .tbtn{display:none;} body{background:#fff;} }
"""

THEME_JS = """
(function(){
  var root=document.documentElement, btn=document.getElementById('themebtn');
  var KEY='ashare-forge-dash-theme';
  function get(){ try{ return localStorage.getItem(KEY); }catch(e){ return null; } }
  function set(v){ try{ v? localStorage.setItem(KEY,v) : localStorage.removeItem(KEY); }catch(e){} }
  function apply(v){ if(v){ root.setAttribute('data-theme',v); } else { root.removeAttribute('data-theme'); }
    if(btn){ btn.title='主题:'+(v||'跟随系统'); } }
  apply(get());
  if(btn){ btn.addEventListener('click',function(){
    var cur=get(), next = cur==='light' ? 'dark' : (cur==='dark' ? null : 'light');
    set(next); apply(next);
  }); }
})();
"""


# ============================================================ 渲染:各区块
def _pipeline(dl: dict, tasks: list[dict], trials: dict, ct: dict) -> str:
    n_engine = len([p for p in glob.glob(os.path.join(ENGINE_DIR, "*.py"))
                    if not os.path.basename(p).startswith("__")])
    n_factors = len([p for p in glob.glob(os.path.join(FACTORS_DIR, "*.py"))
                     if not os.path.basename(p).startswith("__")])
    n_codes, n_stocks = dl.get("n_daily_codes"), dl.get("n_stocks")
    cov = (n_codes / n_stocks) if (n_codes and n_stocks) else None
    n_open = sum(1 for t in tasks if t["status"] == "open")
    cur = [r for r in trials.get("current", []) if r["verdict"] != MISSING]
    n_pass = sum(1 for r in cur if r["verdict"] == "PASS")
    n_judged = len(cur)
    n_old = trials.get("n_superseded", 0)
    n_unlock = len(dl.get("holdout_unlocks") or [])

    steps = [
        ("01", "合同", f'{ct["contract_id"]} v{ct.get("version")}' if ct["exists"] else MISSING,
         "on" if ct["exists"] else "idle"),
        ("02", "数据层", (f'{fmt_int(n_codes)}/{fmt_int(n_stocks)} 只 · {fmt_pct(cov)}'
                        if cov is not None else MISSING),
         "run" if (cov is not None and cov < 0.999) else ("on" if cov else "idle")),
        ("03", "截面引擎", f"{n_engine} 模块 · {n_factors} 因子插件" if n_engine else MISSING,
         "on" if n_engine else "idle"),
        ("04", "任务卡", f"{len(tasks)} 张 · open {n_open}" if tasks else MISSING,
         "on" if tasks else "idle"),
        ("05", "因子裁定", (f'当前 {n_judged} 因子 · PASS {n_pass} · 历史 {n_old}') if trials["rows"] else f"{MISSING}(台账未建)",
         "on" if trials["rows"] else "idle"),
        ("06", "保留段", f'{ct["holdout_cutoff"]} · 解锁记录 {n_unlock} 条',
         "on" if ct["holdout_cutoff"] != MISSING else "idle"),
    ]
    out = ['<div class="pipe">']
    for idx, nm, st, cls in steps:
        out.append(f'<div class="step {cls}"><div class="idx">{esc(idx)}</div>'
                   f'<div class="nm">{esc(nm)}</div><div class="st">{esc(st)}</div></div>')
    out.append("</div>")
    return "".join(out)


def _kpis(dl: dict, trials: dict, ct: dict) -> str:
    n_codes, n_stocks = dl.get("n_daily_codes"), dl.get("n_stocks")
    cov = (n_codes / n_stocks) if (n_codes and n_stocks) else None
    rows = trials["rows"]
    # 分母 = 每个因子最近一次裁定(不含草稿;更早批次含已撤回的另计,不混入)
    cur = [r for r in trials.get("current", []) if r["verdict"] != MISSING]
    n_gate_total = len(cur)
    n_gate_pass = sum(1 for r in cur if r["verdict"] == "PASS")
    n_gate_draft = trials.get("n_superseded", 0)
    last = dl.get("last_sync")
    cards = []

    cards.append(
        f'<div class="kpi accent"><div class="lab">已同步股票</div>'
        f'<div class="big">{fmt_int(n_codes)}<span style="font-size:13px;color:var(--faint)"> / {fmt_int(n_stocks)}</span></div>'
        f'<div class="foot">股池覆盖 {fmt_pct(cov)} · 按代码顺序同步</div>'
        + (f'<div class="bar"><i style="width:{min(100.0, cov * 100):.1f}%"></i></div>' if cov is not None else "")
        + "</div>")

    avg = (dl["n_daily_rows"] / n_codes) if (dl.get("n_daily_rows") and n_codes) else None
    cards.append(
        f'<div class="kpi"><div class="lab">日线行数</div><div class="big">{fmt_int(dl.get("n_daily_rows"))}</div>'
        f'<div class="foot">均 {fmt_int(round(avg)) if avg else MISSING} 行/只</div></div>')

    dmin, dmax = dl.get("date_min"), dl.get("date_max")
    cards.append(
        f'<div class="kpi"><div class="lab">库内日期范围</div>'
        f'<div class="big" style="font-size:15px">{esc(dmin or MISSING)}<br>{esc(dmax or MISSING)}</div>'
        f'<div class="foot">研究加载止于保留段 {esc(ct["holdout_cutoff"])}</div></div>')

    cards.append(
        f'<div class="kpi neutral"><div class="lab">股池消失股(防幸存者偏差)</div>'
        f'<div class="big">{fmt_int(dl.get("n_vanished"))}</div>'
        f'<div class="foot">已取证退市 {fmt_int(dl.get("n_delisted"))} · 差额=已知消失未取证</div></div>')

    cards.append(
        f'<div class="kpi"><div class="lab">最近同步(UTC)</div>'
        f'<div class="big" style="font-size:14px">{esc((last or MISSING).replace("T", " ").replace("+00:00", ""))}</div>'
        f'<div class="foot">未解决失败 {fmt_int(dl.get("n_failures"))} 只</div></div>')

    gate_cls = "good" if (n_gate_total and n_gate_pass) else ("idle" if not n_gate_total else "neutral")
    gate_big = f"{n_gate_pass} / {n_gate_total}" if n_gate_total else "0 / 0"
    cards.append(
        f'<div class="kpi {gate_cls}"><div class="lab">门槛通过 / 已裁定</div>'
        f'<div class="big">{esc(gate_big)}</div>'
        f'<div class="foot">{"台账未建,无裁定条目" if not rows else ("零校准为主判据" + (f" · 更早批次裁定 {n_gate_draft} 条未计入" if n_gate_draft else ""))}</div></div>')

    return '<div class="kpis">' + "".join(cards) + "</div>"


def _data_cards(dl: dict, ct: dict) -> str:
    cards = []

    kv = [
        ("股票池(stocks)", fmt_int(dl.get("n_stocks"))),
        ("板块分布", esc(" · ".join(f"{b or '?'} {n}" for b, n in (dl.get("board_dist") or []))) or MISSING),
        ("缺复权因子股数", fmt_int(dl.get("n_missing_adj"))),
        ("停牌行占比", fmt_pct(dl.get("suspended_ratio"), 2)),
        ("ST 行占比", fmt_pct(dl.get("st_ratio"), 2)),
        ("股池快照", (f'{fmt_int(dl.get("n_snapshots"))} 张 · {esc(dl.get("snap_min") or "")}→{esc(dl.get("snap_max") or "")}'
                  if dl.get("n_snapshots") else MISSING)),
    ]
    cards.append('<div class="card"><h3>数据完整性</h3>'
                 + "".join(f'<div class="kv"><span class="k">{esc(k)}</span><span class="v">{v}</span></div>'
                           for k, v in kv) + "</div>")

    idx_rows = dl.get("index_rows") or []
    bench_names = {"sh.000300": "沪深300", "sh.000905": "中证500", "sh.000852": "中证1000",
                   "sh.000001": "上证综指", "sz.399006": "创业板指"}
    body = "".join(
        f'<div class="kv"><span class="k">{esc(bench_names.get(c, c))} <span class="na">{esc(c)}</span></span>'
        f'<span class="v">{fmt_int(n)} 行 · {esc(a)}→{esc(b)}</span></div>'
        for c, n, a, b in idx_rows) or f'<div class="kv"><span class="k">指数日线</span><span class="v">{MISSING}</span></div>'
    cards.append(f'<div class="card"><h3>基准指数<span class="c">主基准 中证500</span></h3>{body}</div>')

    ry = dl.get("rows_per_year") or []
    mx = max((n for _, n in ry), default=0) or 1
    bars = "".join(
        f'<div class="yrow"><span class="y">{esc(y)}</span>'
        f'<span class="t"><i style="width:{n / mx * 100:.1f}%"></i></span>'
        f'<span class="v">{fmt_int(n)}</span></div>' for y, n in ry) or na()
    cards.append(f'<div class="card"><h3>每年日线行数<span class="c">库内含保留段</span></h3>{bars}'
                 f'<div class="kv" style="margin-top:8px"><span class="k">保留段截止</span>'
                 f'<span class="v">{esc(ct["holdout_cutoff"])} 起 loader 默认不返回</span></div></div>')

    by_task = dict(dl.get("last_sync_by_task") or [])
    stat: dict[str, list[str]] = {}
    for t, s, n in (dl.get("sync_status") or []):
        stat.setdefault(t, []).append(f"{s} {n}")
    log_rows = "".join(
        f'<div class="kv"><span class="k">{esc(t)} <span class="na">{esc(" · ".join(stat.get(t, [])))}</span></span>'
        f'<span class="v">{esc((ts or MISSING).replace("T", " ").replace("+00:00", ""))}</span></div>'
        for t, ts in sorted(by_task.items())) or na()
    meta = dl.get("meta") or {}
    extra = "".join(
        f'<div class="kv"><span class="k">{esc(k)}</span><span class="v">{esc(str(v)[:60])}</span></div>'
        for k, v in meta.items() if k in ("universe_range", "delisted_threshold"))
    cards.append(f'<div class="card"><h3>同步日志(按任务)<span class="c">UTC</span></h3>{log_rows}{extra}</div>')

    return '<div class="grid">' + "".join(cards) + "</div>"


def _tasks(tasks: list[dict]) -> str:
    if not tasks:
        return (f'<div class="empty"><div class="t">任务卡:{MISSING}</div>'
                f'<div class="s">tasks/ 下没有任务卡 yaml。</div></div>')
    out = ['<div class="grid">']
    for t in tasks:
        st = t["status"]
        cls = {"open": "accent", "closed": "neutral", "done": "good"}.get(st, "idle")
        q = "".join(f"<li>{esc(x)}</li>" for x in t["core_question"]) or f"<li>{MISSING}</li>"
        det = []
        if t.get("rationale"):
            det.append(f'<details><summary>经济逻辑与对立假设</summary>'
                       f'<div class="body">{esc(t["rationale"])}</div></details>')
        if t.get("hypotheses"):
            hyp = "".join(f"<li>{esc(h)}</li>" for h in t["hypotheses"])
            fals = (f'<div class="body" style="margin-top:6px"><b>证伪条件</b>:{esc(t["falsification"])}</div>'
                    if t.get("falsification") else "")
            det.append(f'<details><summary>假设(拉数据前写死,{len(t["hypotheses"])} 条)</summary>'
                       f'<div class="body"><ol style="margin:0;padding-left:18px">{hyp}</ol>{fals}</div></details>')
        if t.get("dead_ends"):
            de = "".join(f"<li>{esc(x)}</li>" for x in t["dead_ends"])
            det.append(f'<details><summary>已知死路 / 不可采信来源</summary>'
                       f'<div class="body"><ul style="margin:0;padding-left:18px">{de}</ul></div></details>')
        for nf in t.get("notes_files", []):
            if not nf["text"]:
                continue
            det.append(
                f'<details><summary>中期记录原文 · {esc(nf["path"])}</summary>'
                f'<div class="body"><div class="pill neutral" style="margin-bottom:7px">'
                f'<i class="d"></i>非台账来源 · 不计入裁定表与 KPI</div>'
                f'<pre>{esc(nf["text"])}</pre></div></details>')
        mod = ""
        if t.get("factor_module"):
            exists = os.path.exists(os.path.join(ROOT, t["factor_module"]))
            mod = f' · 因子插件 {esc(t["factor_module"])}' + ("" if exists else f"({MISSING})")
        out.append(
            f'<div class="task"><div class="hd2"><span class="tid">{esc(t["task_id"])}</span>'
            f'{pill(st, cls)}'
            + (f'<span class="na">{esc(t.get("category") or "")}</span>' if t.get("category") else "")
            + f'</div><div class="tnm">{esc(t["name"])}</div>'
              f'<ul class="q">{q}</ul>'
              f'<div class="meta">{esc(t["path"])}{mod}</div>'
            + "".join(det) + "</div>")
    out.append("</div>")
    return "".join(out)


def _trials_table(trials: dict) -> str:
    rows = trials.get("current") or []
    if not rows:
        checked = "".join(f"<li>{esc(p)}</li>" for p in dict.fromkeys(trials["checked"]))
        return (
            f'<div class="empty"><div class="t">因子裁定台账:{MISSING}</div>'
            f'<div class="s">未找到 factor_trials 台账,因此本区不显示任何指标——'
            f'任务卡的中期记录不折算成台账条目,KPI 的"门槛通过"保持 0/0。<br>'
            f'裁定流程落地后按下列字段写入任一位置,本看板即自动读出并画净值 sparkline:'
            f'<br><span class="na">factor · task_id · signature · leakage_passed · ic_mean · icir · '
            f'excess_sharpe · null_passed/null_metrics · verdict · n_trials · nav_json(或 nav_path) · ts</span>'
            f'</div><ul>{checked}</ul></div>')

    def cell_num(v, nd=4, signed=False):
        if v is None:
            return na()
        cls = " v-pos" if v >= 0 else " v-neg"
        return f'<span class="num{cls}">{fmt_num(v, nd, sign=signed)}</span>'

    body = []
    for i, r in enumerate(rows):
        leak = r["leakage"]
        leak_html = pill("通过", "good") if leak is True else (
            pill("未通过", "neutral") if leak is False else na("未测"))
        vd = r["verdict"]
        vd_html = pill("PASS", "good") if vd == "PASS" else (
            pill("FAIL", "neutral") if vd == "FAIL" else na(vd))
        nullc = ""
        if r["null_passed"] is not None and r["null_total"]:
            nullc = " v-pos" if r["null_passed"] >= r["null_total"] else " v-neg"
        ts_short = (r["ts"] or "")[:10]
        body.append(
            f'<tr><td class="l"><div class="fname">{esc(r["factor"])}</div>'
            f'<div class="fsig">{esc(r["signature"])}{(" · " + esc(ts_short)) if ts_short else ""}</div></td>'
            f'<td class="l"><span class="na">{esc(r["task_id"])}</span></td>'
            f'<td>{leak_html}</td>'
            f'<td>{cell_num(r["ic"], 4, True)}</td>'
            f'<td>{cell_num(r["icir"], 3, True)}</td>'
            f'<td>{cell_num(r["excess_sharpe"], 2, True)}</td>'
            f'<td><span class="num{nullc}">{esc(r["null_txt"])}</span></td>'
            f'<td>{vd_html}</td>'
            f'<td><span class="num">{fmt_int(r["n_trials"]) if r["n_trials"] is not None else MISSING}</span></td>'
            f'<td class="l">{sparkline(r["nav"], uid=str(i))}</td></tr>')

    src = " · ".join(trials["sources"]) or MISSING
    return ('<div class="twrap"><table><thead><tr>'
            '<th class="l">因子 / 机制签名</th><th class="l">任务卡</th><th>泄漏测试</th>'
            '<th>RankIC</th><th>ICIR</th><th>超额夏普</th><th>零校准</th><th>门槛裁定</th>'
            '<th>试验次数</th><th class="l">净值</th>'
            f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
            f'<div class="foot" style="margin-top:8px;border:0;padding:0">'
            f'<span>台账来源:{esc(src)}</span>'
            f'<span>RankIC 对齐可交易收益(t+1→t+2 开盘);超额基准 = 中证500</span></div>')


def _disclosure(dl: dict, ct: dict, limits: list[str], trials: dict) -> str:
    n_codes, n_stocks = dl.get("n_daily_codes"), dl.get("n_stocks")
    cov = (n_codes / n_stocks) if (n_codes and n_stocks) else None
    items = [
        f'<span class="hl">样本覆盖率 {fmt_pct(cov, 2)}</span>'
        f'(已同步 {fmt_int(n_codes)} 只 / 股池 {fmt_int(n_stocks)} 只)。'
        f'覆盖率接近满格之前,任何指标只用于机械验证,不足以支撑研究结论。',
        '<span class="hl">已同步子集不是随机样本</span>:全量同步按股票代码顺序推进,'
        '当前已同步的部分集中在代码靠前的区段,存在系统性偏倚;'
        '以该子集算出的 IC 与超额收益不可外推到全市场。',
        f'<span class="hl">保留段锁定于 {esc(ct["holdout_cutoff"])}</span>:loader 默认不返回该日及以后的数据,'
        f'解锁需显式 unlock_holdout=True 并写 meta 台账;当前解锁记录 '
        f'{len(dl.get("holdout_unlocks") or [])} 条。',
        '<span class="hl">本页只读</span>:数据库以只读模式打开,不建表、不写 meta、不触发同步;'
        '页面无外链、无 CDN、无网络请求,仅供本地查看。',
    ]
    if not trials["rows"]:
        items.append(f'<span class="hl">因子裁定台账未建立</span>:裁定表显示{MISSING},'
                     f'"门槛通过"KPI 保持 0/0,不以任务卡中期记录或任何历史运行结果充数。')
    items += [esc(x) for x in limits]
    items += [esc(x) for x in (ct.get("disclosure") or [])]

    unlocks = dl.get("holdout_unlocks") or []
    det = ""
    if unlocks:
        nl = chr(10)
        det = ('<details><summary>保留段解锁台账原文(meta.holdout_unlocked_log)</summary>'
               f'<div class="body"><pre>{esc(nl.join(unlocks))}</pre></div></details>')
    return '<div class="disc"><ol>' + "".join(f"<li>{x}</li>" for x in items) + "</ol>" + det + "</div>"


# ============================================================ 渲染:整页
def render_page(dl: dict, tasks: list[dict], trials: dict, ct: dict, limits: list[str]) -> str:
    now_local = dt.datetime.now().astimezone()
    now_utc = dt.datetime.now(dt.timezone.utc)
    db_note = (f'{os.path.relpath(dl["db_path"], ROOT)} · '
               f'{(dl.get("db_bytes") or 0) / 1048576:.0f} MB') if dl.get("db_exists") else f'数据库{MISSING}'
    n_open = sum(1 for t in tasks if t["status"] == "open")
    head = (
        '<header class="hd"><div class="hd-in">'
        '<div class="mark">AF</div>'
        '<div><h1>AShare Forge · 研究看板</h1>'
        f'<div class="sub">A股截面因子研究平台 · 只读本地快照 · {esc(db_note)}</div></div>'
        '<div class="right">'
        f'<div class="stamp">生成 {now_local.strftime("%Y-%m-%d %H:%M")} 本地<br>'
        f'{now_utc.strftime("%Y-%m-%dT%H:%M")}Z</div>'
        '<button class="tbtn" id="themebtn" type="button" title="切换主题" aria-label="切换明暗主题">'
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">'
        '<path d="M13 8.6A5 5 0 1 1 7.4 3 4 4 0 0 0 13 8.6Z" stroke="currentColor" '
        'stroke-width="1.4" stroke-linejoin="round"/></svg></button>'
        '</div></div></header>')

    body = [
        head, '<div class="wrap">',
        _pipeline(dl, tasks, trials, ct),
        '<section class="sec"><div class="sec-h"><h2>数据层状态</h2>'
        f'<span class="n">{esc(os.path.relpath(dl["db_path"], ROOT))}</span>'
        '<span class="note">口径与 sync.data_quality_report 一致;只读快照,不触发同步</span></div>',
        _kpis(dl, trials, ct), '<div style="height:10px"></div>', _data_cards(dl, ct), '</section>',

        '<section class="sec"><div class="sec-h"><h2>任务卡</h2>'
        f'<span class="n">{len(tasks)} 张 · open {n_open}</span>'
        '<span class="note">假设在拉数据之前写死;负结果同样是交付物</span></div>',
        _tasks(tasks), '</section>',

        '<section class="sec"><div class="sec-h"><h2>因子裁定</h2>'
        f'<span class="n">当前 {len(trials.get("current") or [])} 个因子 · 台账共 {len(trials["rows"])} 条</span>'
        '<span class="note">每个因子取最近一次裁定;更早批次(含已撤回)留在台账并计入试验次数</span></div>',
        _trials_table(trials), '</section>',

        '<section class="sec"><div class="sec-h"><h2>诚实披露</h2>'
        '<span class="note">限制条目取自 docs/PHASE1-DATA-LAYER.md 与合同 disclosure</span></div>',
        _disclosure(dl, ct, limits, trials), '</section>',

        '<div class="foot"><span>AShare Forge · 本地研究看板 · 研究辅助,非投资建议</span>'
        f'<span>只用公开免费数据 · 不含任何非公开数据 · 生成于 {now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")}</span></div>',
        '</div>',
        f'<script>{THEME_JS}</script>',
    ]
    return ('<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<title>AShare Forge 研究看板</title>\n'
            '<meta name="description" content="A股截面因子研究平台的本地工作台:数据层状态、任务卡、因子裁定台账与诚实披露。">\n'
            f'<style>{CSS}</style>\n</head>\n<body>\n' + "\n".join(body) + '\n</body>\n</html>\n')


# ============================================================ 组装与 CLI
def build(db_path: str = DB_PATH) -> str:
    dl = collect_data_layer(db_path)
    tasks = collect_tasks()
    trials = collect_trials()
    ct = collect_contract()
    limits = collect_limitations()
    return render_page(dl, tasks, trials, ct, limits)


def _selftest() -> int:
    """用合成台账走通"有裁定条目"的渲染路径(真实台账为空时该路径不会被真实数据覆盖)。"""
    import tempfile
    fake = {
        **_with_current([
            _normalize_trial({"factor": "demo_factor_a", "task": "task99_demo",
                              "mechanism_signature": "sig:demo:a1b2", "leakage": True,
                              "ic_mean": 0.0123, "icir": 0.31, "excess_sharpe": -0.94,
                              "n_null_passed": 5, "n_null_metrics": 5, "verdict": False,
                              "n_trials": 3, "ts": "2026-01-02T03:04:05+00:00",
                              "nav": json.dumps([1.0, 1.02, 0.97, 0.93, 0.95, 0.88])}, "selftest"),
            _normalize_trial({"factor_name": "demo_factor_b", "task_id": "task99_demo",
                              "leakage_passed": 0, "rank_ic": -0.004, "icir": -0.08,
                              "long_excess_sharpe": 0.41, "null_pass": "2/5",
                              "overall_pass": True, "trials": 1,
                              "nav_json": {"2026-01-02": 1.0, "2026-01-03": 1.05, "2026-01-04": 1.11}},
                             "selftest"),
            _normalize_trial({"factor": "demo_factor_c"}, "selftest"),     # 全缺失 → 应全显示"未运行"
            _normalize_trial({"factor": "demo_factor_a", "verdict": True,   # 更早一次裁定 → 应被取代
                              "ts": "2025-06-01T00:00:00+00:00", "ic_mean": 0.9}, "selftest"),
        ]),
        "sources": ["selftest(4)"], "checked": ["selftest"],
    }
    dl = collect_data_layer(DB_PATH)
    html_txt = render_page(dl, collect_tasks(), fake, collect_contract(), collect_limitations())
    checks = [
        ("裁定表表头", "机制签名" in html_txt),
        ("合成条目 a", "demo_factor_a" in html_txt),
        ("合成条目 c 缺失占位", html_txt.count(MISSING) >= 3),
        ("sparkline 已生成", html_txt.count('class="spark"') >= 2),
        ("FAIL 用中性色", 'pill neutral">' in html_txt or "pill neutral" in html_txt),
        ("零校准 2/5 原样展示", ">2/5<" in html_txt),
        ("无外链", ("http://" not in html_txt) and ("https://" not in html_txt)),
        ("旧裁定被取代", fake["n_superseded"] == 1 and "0.9000" not in html_txt),
        ("KPI 分母=当前因子数", '<div class="big">1 / 2</div>' in html_txt),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html_txt)
        tmp = f.name
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    print(f"  合成渲染输出:{tmp}({len(html_txt):,} 字符)")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 AShare Forge 本地研究看板(单文件自包含网页)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出路径(默认 reports/dashboard.html)")
    ap.add_argument("--db", default=DB_PATH, help="数据库路径(只读打开)")
    ap.add_argument("--selftest", action="store_true", help="用合成台账验证渲染路径,不写输出文件")
    a = ap.parse_args(argv)

    if a.selftest:
        return _selftest()

    page = build(a.db)
    out = a.out if os.path.isabs(a.out) else os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)

    dl = collect_data_layer(a.db)
    trials = collect_trials()
    print(f"[dashboard] 已写出 {out}({len(page):,} 字符,{os.path.getsize(out) / 1024:.0f} KB)")
    print(f"[dashboard] 数据层:{fmt_int(dl.get('n_daily_codes'))}/{fmt_int(dl.get('n_stocks'))} 只 · "
          f"{fmt_int(dl.get('n_daily_rows'))} 行 · {dl.get('date_min')}→{dl.get('date_max')}")
    print(f"[dashboard] 任务卡 {len(collect_tasks())} 张 · 台账条目 {len(trials['rows'])} 条"
          f"{'(台账未建,裁定区显示未运行)' if not trials['rows'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
