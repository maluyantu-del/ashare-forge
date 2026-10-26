"""收口驱动:把一个因子串过完整流水线,出一份可复核的裁定。

流程(顺序不可调换,每一步的先后都有原因):
    加载面板/三面具/基准 → 机制签名 + 查重(草稿即注册) → **泄漏测试**
    → T+1 截面回测 → 零假设校准 → 分层门槛 → 记账 → 裁定

它修的坑:
1. **泄漏测试放在回测之前**。先回测再查泄漏,等于先看见漂亮数字再判有没有作弊,
   人会舍不得砍。本驱动泄漏不过直接判 LEAK_FAIL、记账、**跳过回测**,不产生任何业绩数字。
2. **IC 与回测同口径**。所有指标一律由 engine.backtest 产出(t 日收盘算因子 → t+1 开盘
   建仓 → 吃 t+1→t+2 开盘收益)。本文件不自行算任何"今收→明收"的 IC——那个口径实测
   会给出 IC 0.88 / 夏普 -3.6 的荒谬组合。
3. **成立判据用零分布,不用拍脑袋阈值;但两者并用**。实测出现过"零校准 5/5 超越 q95
   却年亏 20%"(随机因子在同一流水线下夏普约 -2.2,赢了随机不等于赚钱),因此合同
   gates_draft 的固定阈值仍作为硬门由 engine.gates 一并评估,且本驱动在这种组合出现时
   自动在成绩单里写明告警。
4. **多重检验计数**。DSR 的 n_trials 从台账里数出来(同 universe+family 的历次尝试),
   不是每次都填 1。
5. **样本代表性与成绩单同框**。股票数/全市场股票数、日均在池只数一并打印,覆盖率不足时
   成绩单顶部带"不可外推"的告警——避免"流水线机械通过"被当成研究结论。

用法:
    .venv/bin/python -m ashare_forge.pipeline --task all --n-null 50 --top-pct 0.1
    .venv/bin/python -m ashare_forge.pipeline --task accumulation_breakout --n-null 20
"""
from __future__ import annotations

import argparse
import importlib
import pkgutil
import sqlite3
import time
import traceback
import warnings

import pandas as pd

from .data import loader
from .engine import backtest, gates, leakage, null_calib, registry

DEFAULT_START = "2015-01-01"
BENCH_CODE = "sh.000905"          # 合同 benchmarks.primary = 中证500
NULL_Q = 0.95

# 参与零假设校准的指标。核心四项(ic_mean/icir/sharpe/excess_sharpe)是 gates 的主判据,
# 其余几项只作披露——多列一项不会放松判据,但能看出"赢在哪一维"。
NULL_KEYS: tuple[str, ...] = (
    "ic_mean", "icir", "t_stat", "monotonicity", "sharpe", "ann_return",
    "excess_sharpe", "excess_ann",
)

STAT_ORDER: tuple[str, ...] = (
    "n_days", "ic_mean", "icir", "icir_ann", "t_stat", "positive_ratio", "monotonicity",
    "ann_return", "ann_vol", "sharpe", "excess_ann", "excess_sharpe", "bench_ann",
    "max_drawdown", "avg_turnover", "avg_holdings", "total_cost", "ann_cost",
)

SUMMARY_COLS: tuple[str, ...] = (
    "name", "task_id", "family", "signature", "verdict", "leak_passed", "n_days", "n_codes",
    "ic_mean", "icir", "t_stat", "monotonicity", "ann_return", "sharpe", "excess_ann",
    "excess_sharpe", "max_drawdown", "avg_turnover", "avg_holdings", "null_passed",
    "n_trials", "seconds", "note",
)

__all__ = [
    "load_context", "run_factor", "run_all", "discover_specs", "scorecard",
    "summary_frame", "stats_tsv", "to_tsv", "final_pool",
]


# ----------------------------------------------------------------------- 小工具
def _say(on: bool, msg: str, end: str = "\n") -> None:
    if on:
        print(msg, end=end, flush=True)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "" if v != v else f"{v:.4f}"
    return str(v)


def _num(stats: dict | None, key: str):
    """取一个浮点指标;缺失/NaN/非数一律返回 None(不把 NaN 混进汇总表)。"""
    if not stats:
        return None
    v = stats.get(key)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def _count_stocks(conn: sqlite3.Connection) -> int | None:
    """库中已登记的股票总数(用于算样本覆盖率);表不存在则返回 None,而不是假装 0。"""
    try:
        cur = conn.cursor()
        cur.row_factory = None
        cur.execute("SELECT COUNT(*) FROM stocks")
        row = cur.fetchone()
        return int(row[0]) if row else None
    except sqlite3.Error:
        return None


# ----------------------------------------------------------------------- 数据上下文
class DataIncompleteError(RuntimeError):
    """数据完整性闸门未通过:研究运行拒绝在不完整的数据上开跑。"""


def _codes_missing_adj(conn: sqlite3.Connection, ref: pd.DataFrame) -> list[str]:
    """面板内有行情、但复权因子表里一行都没有的股票。

    loader.apply_adjust 对"整只股票无因子"按 1.0 处理且不告警——送转/拆股日会变成假暴跌。
    2026-09-21 实测:复权补拉因 baostock 登录限流失败,154 只 2015-2023 退市股(恰是为防幸存者
    偏差补入的那批)无因子,而下游批跑照常开跑。此闸门把"缺失即按 1.0"从静默变为显式。
    """
    try:
        have = {r[0] for r in conn.execute("SELECT DISTINCT code FROM adj_factor")}
    except sqlite3.Error:
        have = set()
    live = ref.columns[ref.notna().any(axis=0)]
    return sorted(c for c in live if c not in have)


def load_context(conn: sqlite3.Connection, start: str = DEFAULT_START, end: str | None = None,
                 bench_code: str = BENCH_CODE, fields: list[str] | None = None,
                 adj_policy: str = "raise") -> dict:
    """加载面板 + 三面具 + 基准,并算好样本代表性描述(一次加载供多个因子复用)。

    保留段由 loader 在数据层强制下沉,这里不传 unlock_holdout——本驱动无权解锁。
    adj_policy:面板内有股票完全缺复权因子时 "raise"(缺省,拒绝开跑)或 "note"(记入限制声明)。
    """
    if adj_policy not in ("raise", "note"):
        raise ValueError(f"adj_policy 只能是 raise/note,收到 {adj_policy!r}")
    panel = loader.load_panel(conn, start, end, fields=fields)
    ref = panel.get("adj_close")
    if ref is None or ref.empty:
        raise ValueError(f"面板为空:[{start}, {end or '∞'}) 内没有任何日线数据(库还没同步?)")

    notes: list[str] = []
    missing_adj = _codes_missing_adj(conn, ref)
    if missing_adj:
        msg = (f"复权因子缺失:{len(missing_adj)} 只有行情的股票在 adj_factor 表中无任何记录"
               f"(如 {missing_adj[:5]}),其后复权价将等于原始价,送转/拆股日呈假暴跌。"
               f"先运行 `python -m ashare_forge.data.sync --adj-only --codes ...` 补齐。")
        if adj_policy == "raise":
            raise DataIncompleteError(msg)
        notes.append("⚠ " + msg)
    masks = loader.tradable_masks(panel)
    masks["delisted"] = loader.delisted_mask(conn, ref.index, ref.columns)

    bench = loader.index_series(conn, bench_code, start=start, end=end)
    if bench.empty:
        bench = None
        notes.append(f"基准 {bench_code} 在库中无数据:本次只出绝对收益,超额指标缺失")
    else:
        bench = bench.reindex(ref.index).ffill()
        if bool(bench.isna().all()):
            bench = None
            notes.append(f"基准 {bench_code} 与面板日历无交集:本次只出绝对收益")
    bench_open = None
    if bench is not None:
        bo = loader.index_series(conn, bench_code, start=start, end=end, field="open")
        bench_open = bo.reindex(ref.index).ffill() if not bo.empty else None
        if bench_open is None or bool(bench_open.isna().all()):
            bench_open = None
            notes.append(f"基准 {bench_code} 无开盘价:超额退化为收盘→收盘旧口径(与组合时段错位)")
    # 全收益基准(仅披露口径,不进门槛、不参与零校准):合同 benchmarks 中主基准的 total_return 项。
    # 全收益指数只有收盘价,不能直接当 bench_code(会静默退回收盘口径);这里以价格指数开→开
    # 收益为骨架叠加分红补偿,得到与组合同一时段的序列。拿不到则为 None 并在 notes 说明。
    from .contract import load_contract
    bench_tr_ret, tr_code = None, None
    for b in load_contract().get("benchmarks") or []:
        if b.get("code") == bench_code and isinstance(b.get("total_return"), dict):
            tr_code = b["total_return"].get("code")
    if tr_code is None:
        notes.append(f"基准 {bench_code} 未在合同中配置全收益序列:excess_*_tr 披露列缺失")
    elif bench is None or bench_open is None:
        notes.append(f"全收益基准 {tr_code} 需要价格指数 {bench_code} 的开盘与收盘价:excess_*_tr 披露列缺失")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # 库中无该序列时 index_series 会告警,改由 notes 说明
            tr_close = loader.index_series(conn, tr_code, start=start, end=end)
        if tr_close.empty:
            notes.append(f"全收益基准 {tr_code} 在库中无数据:excess_*_tr 披露列缺失,主基准与门槛不受影响"
                         f"(`python -m ashare_forge.data.sync --index-tr` 可补齐)")
        else:
            bench_tr_ret = loader.total_return_open_to_open(bench_open, bench, tr_close, index=ref.index)
            n_miss = int(bench_tr_ret.attrs.get("n_comp_missing", 0))
            if n_miss:
                notes.append(f"全收益基准 {tr_code} 在面板日历内有 {n_miss} 个步长缺值,"
                             f"对应分红补偿按 0 处理(仅影响 excess_*_tr 披露列)")
    if bench is not None:
        # 指数收盘价也放进面板:需要市场代理的因子(如特质波动率)按其模块约定优先取
        # bench_close,拿不到才退化为"截面等权平均",而当前库只是全市场的一部分,
        # 等权代理会被样本偏倚污染。指数只用当日及以前,不引入前视。
        panel["bench_close"] = bench

    n_dates, n_codes = ref.shape
    avg_universe = float(masks["in_universe"].sum(axis=1).mean())
    total = _count_stocks(conn)
    coverage = (n_codes / total) if (total and total > 0) else None
    if coverage is not None and coverage < 0.9:
        notes.append(
            f"⚠ 样本覆盖率 {coverage:.1%}({n_codes}/{total} 只):当前库只是全市场的一部分,"
            f"且同步按代码顺序进行(非随机抽样),存在系统性偏倚。本轮结果只能作机械验证,"
            f"**不足以支撑任何研究结论**,任何引用须同时引用本行。")
    return {
        "panel": panel, "masks": masks, "bench": bench, "bench_open": bench_open, "bench_code": bench_code,
        "bench_tr_ret": bench_tr_ret, "bench_tr_code": tr_code,
        "start": start, "end": end, "n_dates": int(n_dates), "n_codes": int(n_codes),
        "first_date": str(ref.index[0].date()), "last_date": str(ref.index[-1].date()),
        "avg_universe": avg_universe, "total_stocks": total, "coverage": coverage,
        "n_missing_adj": len(missing_adj), "notes": notes,
    }


def _sample_line(s: dict) -> str:
    cov = (f",覆盖率 {s['coverage']:.1%}(全市场 {s['total_stocks']} 只)"
           if s.get("coverage") is not None else "")
    avg = s.get("avg_universe")
    avg_txt = f";日均在池 {avg:.1f} 只" if isinstance(avg, float) else ""
    return (f"{s.get('n_codes')} 只 × {s.get('n_dates')} 天"
            f"({s.get('first_date')}→{s.get('last_date')}){cov}{avg_txt}")


# ----------------------------------------------------------------------- spec 校验与因子
REQUIRED_SPEC_KEYS = ("name", "family", "operators", "direction", "horizon_days",
                      "columns", "build_factor")


def _check_spec(spec: dict) -> None:
    missing = [k for k in REQUIRED_SPEC_KEYS if not spec.get(k)]
    if missing:
        raise ValueError(f"spec 缺字段 {missing};约定字段:{list(REQUIRED_SPEC_KEYS)} "
                         f"+ 可选 task_id/thesis/params/universe")
    if not callable(spec["build_factor"]):
        raise TypeError("spec['build_factor'] 必须是 build_factor(panel) -> DataFrame 的可调用对象")


def _aligned_factor(build_factor, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """跑一次因子并对齐到面板的日历与股票集(列缺失补 NaN,不静默改变形状)。"""
    f = build_factor(panel)
    if not isinstance(f, pd.DataFrame):
        raise TypeError(f"build_factor 必须返回 DataFrame(日期×股票),得到 {type(f).__name__}")
    ref = panel["adj_close"]
    if len(f) != len(ref):
        raise ValueError(f"因子行数 {len(f)} ≠ 面板行数 {len(ref)}:因子必须与面板同日历")
    f = f.reindex(index=ref.index, columns=ref.columns).astype("float64")
    if not bool(f.notna().to_numpy().any()):
        raise ValueError("因子全为 NaN(滚动窗口比样本还长?),无法回测")
    return f


# ----------------------------------------------------------------------- 主流程
def provenance_snapshot(specs: list[dict]) -> dict:
    """进程启动时一次性锁定代码出处:git HEAD + 各因子模块源码 sha。

    审查发现(P1):原实现在写台账那一刻现读 HEAD 与源码——批次运行 4.5 小时期间若有新提交
    或改动因子文件,记下的是"写账时磁盘上的代码",而非"本进程实际加载并运行的代码"
    (实测:16:48 启动的进程在 22:03 写入的条目记成 21:48 的新提交)。
    """
    return {
        "git_commit": registry.git_commit(),
        "code_sha": {str(s.get("name")): registry.factor_code_sha(s["build_factor"]) for s in specs},
        "taken_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _provenance(spec: dict, result: dict, bt: dict, gv: dict, snapshot: dict | None = None) -> dict:
    """证据包:让每个数字都能追溯到"哪版代码 × 哪份数据 × 哪次校准"。"""
    nv = result.get("null_verdict") or {}
    name = str(spec.get("name"))
    now_commit, now_sha = registry.git_commit(), registry.factor_code_sha(spec["build_factor"])
    if snapshot is None:
        snapshot = {"git_commit": now_commit, "code_sha": {name: now_sha}, "taken_at": None}
    snap_commit = snapshot.get("git_commit")
    snap_sha = (snapshot.get("code_sha") or {}).get(name, now_sha)
    drift = None
    if snap_commit != now_commit or snap_sha != now_sha:
        drift = {"git_commit_at_log": now_commit, "code_sha_at_log": now_sha,
                 "note": "运行期间代码出处发生变化;以启动时快照为准,本条结果对应快照版本"}
    tbl = nv.get("table")
    stats = {k: v for k, v in (bt.get("stats") or {}).items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {
        "code_sha": snap_sha,
        "git_commit": snap_commit,
        "snapshot_taken_at": snapshot.get("taken_at"),
        "provenance_drift": drift,
        "leak": result.get("leak"),
        "null": {
            "mode": (result.get("null") or {}).get("mode"),
            "n": (result.get("null") or {}).get("n"),
            "q": nv.get("q"),
            "n_passed": nv.get("n_passed"), "n_metrics": nv.get("n_metrics"),
            "passed": gv.get("null_passed"),
            "table": tbl.to_dict(orient="records") if tbl is not None and hasattr(tbl, "to_dict") else None,
            "validity": nv.get("validity"),
        },
        "gates": {"overall_pass": gv.get("overall_pass"), "rows": gv.get("rows"),
                  "notes": gv.get("notes"), "n_trials": gv.get("n_trials")},
        "stats": stats,
        "data": result.get("sample"),
        "params": spec.get("params") or {},
    }


def run_factor(conn: sqlite3.Connection, spec: dict, *, n_null: int = 50, top_pct: float = 0.1,
               start: str = DEFAULT_START, end: str | None = None, ctx: dict | None = None,
               seed: int = 7, leak_points: int | None = None, q: float = NULL_Q,
               progress: bool = False, run_ref: str = "pipeline",
               strict: bool = False, snapshot: dict | None = None) -> dict:
    """把一个因子串过完整流水线,返回成绩单字典。

    spec 约定:{name, task_id, family, operators, direction, horizon_days, columns,
               build_factor, thesis, params}(universe 可选,缺省 ashare_cs)
    返回      {name, signature, leak, stats, null_verdict, gate_verdict, verdict, ...}
    verdict   PASS / FAIL / LEAK_FAIL / ERROR

    ctx 可传入 load_context 的结果以便多个因子复用同一份面板;不传则自行加载。
    strict=True 时异常直接抛出(单因子调试用);缺省 False 把异常记成 ERROR 一行,
    以免一个坏因子打断整批——但错误照实进成绩单与台账,不静默吞掉。
    """
    t0 = time.time()
    name = str(spec.get("name") or "unnamed")
    if snapshot is None:                       # 单因子调用:在开跑时(而非写账时)锁定出处
        snapshot = provenance_snapshot([spec])
    result: dict = {
        "name": name, "task_id": spec.get("task_id"), "family": None, "signature": None,
        "stage": "start", "leak": None, "stats": None, "null_verdict": None,
        "gate_verdict": None, "verdict": "ERROR", "n_trials": None, "notes": [],
        "sample": {},
    }
    sig = canon = None
    try:
        _check_spec(spec)
        registry.ensure_tables(conn)
        if ctx is None:
            _say(progress, f"  [0/4] 加载面板 {start}→{end or '今天'} …")
            ctx = load_context(conn, start=start, end=end)
        for k in ("panel", "masks"):
            if k not in ctx:
                raise KeyError(f"ctx 缺 {k}(需 load_context 的返回或同形状字典)")
        panel, masks, bench = ctx["panel"], ctx["masks"], ctx.get("bench")
        bench_open = ctx.get("bench_open")
        result["sample"] = {k: ctx.get(k) for k in
                            ("n_dates", "n_codes", "first_date", "last_date", "avg_universe",
                             "coverage", "total_stocks", "bench_code")}
        result["notes"].extend(list(ctx.get("notes", [])))

        # ---- 1. 机制签名 + 查重 + 草稿注册(在看任何业绩数字之前) ----
        result["stage"] = "signature"
        sig, canon = registry.signature_from_spec(spec)
        result["signature"], result["family"], result["canon"] = sig, canon["family"], canon
        dedup = registry.dedup_report(conn, sig)
        result["dedup_known"] = bool(dedup["known_mechanism"])
        result["dedup_verdicts"] = dedup["verdict_counts"]
        if dedup["known_mechanism"]:
            result["notes"].append(
                f"撞车预警:同签名机制已有前科 {dedup['verdict_counts'] or '{}'};{dedup['advice']}")
        # 行为指纹:声明签名只看"因子自称用了什么算子",换个名字即可绕过(已知弱点,见 tests/test_registry.py);
        # 这里再在固定合成面板上量一次"它实际算出什么",行为高度相似即判撞车。
        try:
            fp = registry.behavior_fingerprint(spec["build_factor"])
            behav = registry.find_behavioral_duplicates(conn, fp, exclude=sig)
            result["behavioral_duplicates"] = behav
            if behav:
                top = behav[0]
                result["notes"].append(
                    f"行为撞车:与既有机制 {top['signature'][:8]} 相似度 {top['similarity']}"
                    f"(申报签名不同但实际算出的东西几乎一样);{top.get('thesis') or ''}"[:300])
            registry.save_fingerprint(conn, sig, fp)
        except Exception as e:  # noqa: BLE001 —— 指纹失败不应阻断主流程,但要留痕
            result["notes"].append(f"行为指纹未能计算({type(e).__name__}: {e});本次仅依赖声明签名查重")

        registry.register_mechanism(conn, sig, canon, str(spec.get("thesis", ""))[:2000])
        registry.log_trial(conn, sig, canon, "draft", run_ref=run_ref, name=name,
                           task_id=str(spec.get("task_id") or ""), params=spec.get("params") or {},
                           notes="流水线草稿注册(尚未看业绩)")

        missing_cols = [c for c in spec["columns"] if c not in panel]
        if missing_cols:
            result["notes"].append(
                f"spec 声明的列 {missing_cols} 不在面板字段中:签名与实际用料可能不符")
        if float(spec["horizon_days"]) != 1:
            result["notes"].append(
                f"spec 声明持有期 {spec['horizon_days']} 日,但本回测口径固定为日度调仓 + "
                f"t+1→t+2 可交易收益;持有期仅用于签名分桶,不改变评估口径")

        # ---- 2a. 静态扫描:因子不得绕开传入面板自行取数(前缀重算对此无效) ----
        result["stage"] = "source_scan"
        scan_hits = leakage.scan_factor_source(spec["build_factor"])
        result["source_scan"] = scan_hits
        if scan_hits:
            detail = "; ".join(f"第{h['line']}行 {h['name']}({h['reason']})" for h in scan_hits[:5])
            result["leak"] = {"passed": False, "n_points": 0, "reason": f"静态扫描命中: {detail}"}
            registry.log_trial(conn, sig, canon, "abandoned", run_ref=run_ref, name=name,
                               task_id=str(spec.get("task_id") or ""),
                               notes=f"静态扫描拒绝(不跑回测): {detail}"[:400])
            result["verdict"] = "LEAK_FAIL"
            result["notes"].append(f"静态扫描拒绝:因子代码含外部取数 —— {detail}")
            result["stage"] = "done"
            result["seconds"] = round(time.time() - t0, 1)
            return result

        # ---- 2b. 泄漏测试(不过则就地终止,不产生任何业绩数字) ----
        result["stage"] = "leakage"
        _say(progress, "  [1/4] 泄漏测试(前缀重算)…", end="")
        leak = leakage.prefix_recompute_test(spec["build_factor"], panel, n_points=leak_points)
        result["leak"] = {"passed": bool(leak["passed"]),
                          "n_points_tested": int(leak["n_points_tested"]),
                          "failures": leak["failures"][:3], "note": leak["note"]}
        _say(progress, f" {'通过' if leak['passed'] else '未通过'}"
                       f"({leak['n_points_tested']} 切点)")
        if not leak["passed"]:
            registry.log_trial(conn, sig, canon, "abandoned", run_ref=run_ref, name=name,
                               task_id=str(spec.get("task_id") or ""),
                               params=spec.get("params") or {},
                               n_days=result["sample"].get("n_dates"),
                               n_codes=result["sample"].get("n_codes"),
                               notes=f"泄漏测试失败,未进回测:{leak['failures'][:1]}"[:400])
            result["verdict"] = "LEAK_FAIL"
            result["notes"].append("因子用到了当日之后的数据,已在回测之前终止;业绩数字一律不产生")
            result["stage"] = "done"
            result["seconds"] = round(time.time() - t0, 1)
            return result

        # ---- 3. T+1 截面回测 ----
        result["stage"] = "backtest"
        _say(progress, "  [2/4] T+1 截面回测 …", end="")
        factor = _aligned_factor(spec["build_factor"], panel)
        bt = backtest.run_backtest(factor, panel, masks, bench=bench, top_pct=top_pct,
                                   bench_open=bench_open, bench_tr_ret=ctx.get("bench_tr_ret"))
        result["backtest"], result["stats"] = bt, bt["stats"]
        _say(progress, f" 完成(IC均值 {_fmt(_num(bt['stats'], 'ic_mean'))} / "
                       f"夏普 {_fmt(_num(bt['stats'], 'sharpe'))})")

        # ---- 4. 零假设校准(逐日截面打乱,穿同一条流水线) ----
        if n_null and int(n_null) > 0:
            result["stage"] = "null"
            n_null = int(n_null)
            counter = {"i": 0}

            def _run_fn(f: pd.DataFrame) -> dict:
                counter["i"] += 1
                _say(progress, f"\r  [3/4] 零假设校准 {counter['i']}/{n_null} …", end="")
                return backtest.run_backtest(f, panel, masks, bench=bench,
                                             top_pct=top_pct, bench_open=bench_open)["stats"]

            null = null_calib.null_distribution(factor, _run_fn, n=n_null, seed=seed,
                                                keys=NULL_KEYS)
            result["null"] = null
            result["null_verdict"] = null_calib.calibrated_verdict(bt["stats"], null, q=q)
            nvld = null_calib.null_validity(bt["stats"], null)
            result["null_verdict"]["validity"] = nvld
            if nvld.get("valid") is False:
                result["notes"].append(nvld["note"])
            _say(progress, f"\r  [3/4] 零假设校准 {n_null}/{n_null} 完成:超越 q{int(q * 100)} "
                           f"{result['null_verdict']['n_passed']}/"
                           f"{result['null_verdict']['n_metrics']} 项        ")
        else:
            result["notes"].append("n_null=0:未做零假设校准,固定阈值仅作参考,不足以支撑成立结论")

        # ---- 5. 分层门槛(试验次数计入 DSR) ----
        result["stage"] = "gates"
        # DSR 的试验次数 = 全研究域已消耗的检验次数 + 本卡申报的参数档数(至少计本次 1 次)
        n_hist = registry.trial_count(conn, canon["universe"], scope="universe")
        n_decl = registry.declared_trials(spec.get("task_id"))
        n_trials = n_hist + max(1, n_decl)
        result["n_trials_breakdown"] = {"history": n_hist, "declared": n_decl, "total": n_trials}
        result["n_trials"] = n_trials
        gv = gates.evaluate(bt, result["null_verdict"], n_trials=n_trials,
                            n_trials_breakdown=result["n_trials_breakdown"])
        result["gate_verdict"] = gv
        _say(progress, f"  [4/4] 分层门槛(n_trials={n_trials}): "
                       f"{'通过' if gv['overall_pass'] else '未通过'}")

        # "赢了随机 ≠ 赚钱":零校准通过但绝对/超额收益为负时,成绩单必须自己说出来
        prim_ann = _num(bt["stats"], "excess_ann")
        if prim_ann is None:
            prim_ann = _num(bt["stats"], "ann_return")
        if (result["null_verdict"] and result["null_verdict"]["n_passed"] > 0
                and prim_ann is not None and prim_ann < 0):
            result["notes"].append(
                f"零假设校准有 {result['null_verdict']['n_passed']} 项超越 q{int(q * 100)},"
                f"但主收益为负({prim_ann:.1%}/年)——赢了随机不等于赚钱,不得据此判成立")

        # ---- 6. 记账 ----
        primary = bt.get("excess_ret")
        if primary is None or not len(pd.Series(primary).dropna()):
            primary = bt["daily_ret"]
        primary = pd.Series(primary).astype("float64")
        cm = gates.crisis_mask(pd.DatetimeIndex(primary.index)).reindex(primary.index).fillna(False)
        sr_calm = gates.ann_sharpe(primary[~cm])
        sr_crisis = gates.ann_sharpe(primary[cm]) if bool(cm.any()) else None
        dsr_value = next((r["value"] for r in gv["rows"] if str(r["gate"]).startswith("dsr")), None)
        registry.log_trial(
            conn, sig, canon, "pass" if gv["overall_pass"] else "fail", run_ref=run_ref, name=name,
            task_id=str(spec.get("task_id") or ""), params=spec.get("params") or {},
            ic_mean=_num(bt["stats"], "ic_mean"), icir=_num(bt["stats"], "icir"),
            sr_calm=sr_calm, sr_crisis=sr_crisis,
            dsr_value=dsr_value if isinstance(dsr_value, (int, float)) else None,
            n_days=int(len(primary)), n_codes=result["sample"].get("n_codes"),
            notes=str(spec.get("thesis", ""))[:400],
            provenance=_provenance(spec, result, bt, gv, snapshot))
        result["sr_calm"], result["sr_crisis"] = sr_calm, sr_crisis
        result["verdict"] = "PASS" if gv["overall_pass"] else "FAIL"
        result["stage"] = "done"
        result["seconds"] = round(time.time() - t0, 1)
        return result

    except Exception as e:  # noqa: BLE001
        if strict:
            raise
        result["verdict"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        result["trace"] = traceback.format_exc()[-1200:]
        result["seconds"] = round(time.time() - t0, 1)
        if sig is not None and canon is not None:      # 错误照样入账:失败是路标
            try:
                registry.log_trial(conn, sig, canon, "abandoned", run_ref=run_ref, name=name,
                                   task_id=str(spec.get("task_id") or ""),
                                   notes=f"流水线在 {result['stage']} 阶段出错:{result['error']}"[:400])
            except sqlite3.Error as db_err:
                result["notes"].append(f"台账写入也失败了:{db_err}")
        return result


def run_all(conn: sqlite3.Connection, specs: list[dict], *, n_null: int = 50, top_pct: float = 0.1,
            start: str = DEFAULT_START, end: str | None = None, ctx: dict | None = None,
            seed: int = 7, leak_points: int | None = None, q: float = NULL_Q,
            progress: bool = False, run_ref: str = "pipeline") -> pd.DataFrame:
    """批量跑并返回汇总 DataFrame(df.attrs['results'] 里是每个因子的完整成绩单)。

    面板只加载一次供全部因子复用——加载全市场面板比回测本身还贵。
    出处快照在加载面板之前锁定,整批共用。
    """
    snapshot = provenance_snapshot(specs)
    if ctx is None:
        _say(progress, f"加载面板 {start}→{end or '今天'} …", end="")
        ctx = load_context(conn, start=start, end=end)
        _say(progress, f" {ctx['n_codes']} 只 × {ctx['n_dates']} 天")
        for n in ctx["notes"]:
            _say(progress, f"  {n}")
    results = []
    for i, spec in enumerate(specs, 1):
        _say(progress, f"\n[{i}/{len(specs)}] {spec.get('name')} —— 开跑")
        r = run_factor(conn, spec, n_null=n_null, top_pct=top_pct, start=start, end=end, ctx=ctx,
                       seed=seed, leak_points=leak_points, q=q, progress=progress, run_ref=run_ref,
                       snapshot=snapshot)
        results.append(r)
        _say(progress, "\n" + scorecard(r))
    df = summary_frame(results)
    df.attrs["results"] = results
    df.attrs["context"] = {k: v for k, v in ctx.items() if k not in ("panel", "masks", "bench", "bench_open")}
    return df


# ----------------------------------------------------------------------- 成绩单
def stats_tsv(stats: dict | None) -> str:
    if not stats:
        return "指标\t数值\n(无)"
    lines = ["指标\t数值"]
    for k in STAT_ORDER:
        if k in stats:
            lines.append(f"{k}\t{_fmt(stats[k])}")
    return "\n".join(lines)


def _indent(text: str, pad: str = "    ") -> str:
    return "\n".join(pad + ln for ln in str(text).splitlines())


def scorecard(result: dict) -> str:
    """单个因子的成绩单(其中每个 TSV 块可直接粘进 Excel)。"""
    lines = [f"=== {result['name']} [{result.get('family') or '?'}] 签名 "
             f"{result.get('signature') or '-'}"
             + (f" | 任务 {result['task_id']}" if result.get("task_id") else "") + " ==="]
    head = f"裁定: {result['verdict']}"
    if result.get("n_trials"):
        head += f" | 同族累计试验 n_trials={result['n_trials']}"
    if result.get("seconds") is not None:
        head += f" | 用时 {result['seconds']}s"
    lines.append(head)
    if result.get("sample"):
        lines.append("样本: " + _sample_line(result["sample"]))

    leak = result.get("leak")
    if leak is not None:
        lines.append(f"泄漏测试(前缀重算): {'通过' if leak['passed'] else '未通过'}"
                     f"({leak['n_points_tested']} 个切点)")
        for f in (leak.get("failures") or [])[:2]:
            lines.append(f"  失败点: {f}")
    if result["verdict"] == "ERROR":
        lines.append(f"错误: {result.get('error')}")
        if result.get("trace"):
            lines.append(_indent(result["trace"], "  "))

    if result.get("stats"):
        lines.append("指标(TSV):")
        lines.append(_indent(stats_tsv(result["stats"])))
    if result.get("null_verdict"):
        nv = result["null_verdict"]
        lines.append(f"零假设校准(超越 q{int(nv['q'] * 100)} "
                     f"{nv['n_passed']}/{nv['n_metrics']} 项,TSV):")
        lines.append(_indent(null_calib.to_tsv(nv)))
    if result.get("gate_verdict"):
        lines.append("分层门槛(TSV):")
        lines.append(_indent(gates.to_tsv(result["gate_verdict"])))
    for n in result.get("notes", []):
        lines.append(f"备注: {n}")
    return "\n".join(lines)


def summary_frame(results: list[dict]) -> pd.DataFrame:
    """把多个成绩单压成一张可粘进 Excel 的汇总表。"""
    rows = []
    for r in results:
        st, nv = r.get("stats") or {}, r.get("null_verdict")
        note = "; ".join([str(x) for x in r.get("notes", [])][:2])
        if r.get("error"):
            note = r["error"] + ("; " + note if note else "")
        rows.append({
            "name": r.get("name"), "task_id": r.get("task_id"), "family": r.get("family"),
            "signature": r.get("signature"), "verdict": r.get("verdict"),
            "leak_passed": (r.get("leak") or {}).get("passed"),
            "n_days": st.get("n_days"), "n_codes": (r.get("sample") or {}).get("n_codes"),
            "ic_mean": _num(st, "ic_mean"), "icir": _num(st, "icir"), "t_stat": _num(st, "t_stat"),
            "monotonicity": _num(st, "monotonicity"), "ann_return": _num(st, "ann_return"),
            "sharpe": _num(st, "sharpe"), "excess_ann": _num(st, "excess_ann"),
            "excess_sharpe": _num(st, "excess_sharpe"), "max_drawdown": _num(st, "max_drawdown"),
            "avg_turnover": _num(st, "avg_turnover"), "avg_holdings": _num(st, "avg_holdings"),
            "null_passed": (f"{nv['n_passed']}/{nv['n_metrics']}" if nv else None),
            "n_trials": r.get("n_trials"), "seconds": r.get("seconds"), "note": note,
        })
    return pd.DataFrame(rows, columns=list(SUMMARY_COLS))


def to_tsv(df: pd.DataFrame) -> str:
    """可直接粘进 Excel 的汇总块。"""
    return df.to_csv(sep="\t", index=False, float_format="%.4f")


def final_pool(df: pd.DataFrame) -> list[str]:
    """最终池 = 泄漏测试通过且分层门槛全过的因子。"""
    if df.empty:
        return []
    return df.loc[df["verdict"] == "PASS", "name"].astype(str).tolist()


# ----------------------------------------------------------------------- spec 发现
def _params(mod, names: tuple[str, ...]) -> dict:
    """从因子模块读参数(而不是在这里抄一份),避免两处漂移。"""
    return {n: getattr(mod, n, None) for n in names}


def _fb_accumulation_breakout(mod) -> dict:
    return {
        "task_id": "task01_accumulation_breakout", "family": "pattern",
        "operators": ["rolling_max", "rolling_min", "rolling_mean", "hl_range",
                      "intensity_norm", "breakout", "cs_rank"],
        "direction": "long_only", "horizon_days": 1,
        "columns": ["adj_close", "high", "low", "volume"],
        "thesis": ("横盘缩量吸筹后放量突破箱顶,代表新增资金介入、价格脱离箱体后有惯性;"
                   "对立假设是它只是动量的换皮,且突破日多为涨停,T+1 下根本买不进。"),
        "params": _params(mod, ("BOX_WIN", "LONG_WIN", "VOL_CAP", "WEIGHTS")),
    }


def _fb_short_reversal(mod) -> dict:
    return {
        "task_id": "task02_short_reversal", "family": "reversal",
        "operators": ["pct_change", "cs_rank"], "direction": "long_only", "horizon_days": 1,
        "columns": ["adj_close", "tradestatus", "volume"],
        "thesis": ("过去一个月跌幅越大、后续反弹越强(流动性提供者获得的反转补偿);"
                   "对立假设是反转收益被 T+1 时滞与交易成本吃光。"),
        "params": _params(mod, ("WINDOW", "SKIP_RECENT", "MAX_STALE_DAYS", "MIN_TRADED_RATIO")),
    }


def _fb_idio_volatility(mod) -> dict:
    return {
        "task_id": "task03_idio_volatility", "family": "volatility",
        "operators": ["pct_change", "rolling_std", "capm_residual", "cs_rank"],
        "direction": "long_only", "horizon_days": 1,
        "columns": ["adj_close", "tradestatus", "volume", "bench_close"],
        "thesis": ("特质波动率越低、预期收益越高(低波异象);对立假设是它只是市值因子"
                   "换皮,或低波组合由停牌/价格冻结的标的堆出来。"),
        "params": _params(mod, ("WINDOW", "MODE", "MAX_STALE_DAYS", "MIN_OBS_RATIO")),
    }


def _fb_illiquidity(mod) -> dict:
    return {
        "task_id": "task04_illiquidity", "family": "liquidity",
        "operators": ["abs_return", "rolling_mean", "intensity_norm"],
        "direction": "long_only", "horizon_days": 1,
        "columns": ["amount", "pctChg"],
        "thesis": ("Amihud 非流动性越高、投资者要求的补偿越高(非流动性溢价);"
                   "对立假设是它只是小市值换皮,或收益被冲击成本吃光。"),
        "params": _params(mod, ("WIN", "MIN_OBS", "AMOUNT_UNIT", "DIRECTION", "RETURN_SOURCE")),
    }


def _fb_earnings_yield(mod) -> dict:
    return {
        "task_id": "task05_earnings_yield", "family": "value",
        "operators": ["inverse", "shift"], "direction": "long_only", "horizon_days": 1,
        "columns": ["peTTM"],
        "thesis": ("盈利收益率 E/P 越高、预期收益越高(价值溢价),取倒数以修负盈利的排序;"
                   "对立假设是估值字段的 PIT 切换时点不明,收益来自尚未公开的盈利信息。"),
        "params": _params(mod, ("VALUATION_LAG_DAYS", "DIRECTION", "DROP_NEGATIVE",
                                "WINSOR_Q", "MAIN_FIELD")),
    }


# 过渡方案:因子模块尚未自带 SPEC 声明,而签名字段(family/operators/direction/…)不能
# 由驱动猜——猜错会让查重失效。因此这里按各模块的文档与任务卡显式写死一份(参数从模块读),
# 只登记每个模块的主口径 build_factor;对照口径(如市值中性化档、B/P 档)由任务卡在模块里
# 声明 SPECS 后自动接管——模块一旦有 SPEC/SPECS,本表即被忽略,对应条目可删。
FALLBACK_SPEC_BUILDERS = {
    "accumulation_breakout": _fb_accumulation_breakout,
    "short_reversal": _fb_short_reversal,
    "idio_volatility": _fb_idio_volatility,
    "illiquidity": _fb_illiquidity,
    "earnings_yield": _fb_earnings_yield,
}


def discover_specs(names: list[str] | None = None, include_optional: bool = False) -> list[dict]:
    """扫描 ashare_forge.factors:模块自带 SPEC/SPECS 则用之,否则查过渡表,再否则跳过。

    不为没有 SPEC 的模块臆造签名字段——签名猜错等于查重失效。

    spec 带 batch_default=False 的是"按名点跑"的申报档(任务卡已申报、只需运行一次的稳健性档):
    批量发现(names 为空)与按任务卡号选择时都不包含,以免每次批跑平白多记一次试验;
    只有按因子名显式点名,或 include_optional=True(测试、诊断、--list)时才包含。
    """
    from . import factors as pkg

    out: list[dict] = []
    for m in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda x: x.name):
        if m.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{pkg.__name__}.{m.name}")
        if hasattr(mod, "SPECS"):
            specs = [dict(s) for s in mod.SPECS]
        elif hasattr(mod, "SPEC"):
            specs = [dict(mod.SPEC)]
        elif m.name in FALLBACK_SPEC_BUILDERS:
            specs = [FALLBACK_SPEC_BUILDERS[m.name](mod)]
        else:
            warnings.warn(f"因子模块 {m.name} 既无 SPEC 也无过渡条目,已跳过"
                          f"(请在模块里声明 SPEC = {{name, family, operators, direction,"
                          f" horizon_days, columns, thesis, params}})", stacklevel=2)
            continue
        for s in specs:
            s.setdefault("name", m.name)
            s.setdefault("build_factor", getattr(mod, "build_factor", None))
            s["module"] = f"{pkg.__name__}.{m.name}"
            out.append(s)
    want = {str(n) for n in names} if names else set()
    if not include_optional:
        out = [s for s in out if s.get("batch_default", True) is not False or str(s.get("name")) in want]
    if names:
        out = [s for s in out if str(s.get("name")) in want or str(s.get("task_id")) in want]
    return out


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    from .config import DB_PATH
    from .data import db

    ap = argparse.ArgumentParser(description="AShare Forge 截面因子收口驱动")
    ap.add_argument("--task", default="all", help="all 或因子名/任务卡号(逗号分隔可多选)")
    ap.add_argument("--n-null", type=int, default=50, help="零假设校准的打乱次数(默认50;0=跳过)")
    ap.add_argument("--top-pct", type=float, default=0.1, help="多头组比例(默认前10%%)")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--leak-points", type=int, default=None,
                    help="泄漏测试的切点数(默认按面板长度自适应)")
    ap.add_argument("--list", action="store_true", help="只列出可跑的因子,不跑")
    a = ap.parse_args(argv)

    names = None if a.task == "all" else [x.strip() for x in a.task.split(",") if x.strip()]
    specs = discover_specs(names)
    if not specs:
        print(f"没有可跑的因子(--task {a.task});ashare_forge/factors 下的模块需声明 SPEC。")
        return 1
    if a.list:
        print("name\ttask_id\tfamily\tmodule\tbatch_default")
        for s in discover_specs(names, include_optional=True):
            print(f"{s.get('name')}\t{s.get('task_id') or ''}\t{s.get('family')}\t{s.get('module')}"
                  f"\t{s.get('batch_default', True)}")
        return 0

    conn = db.connect(a.db)
    try:
        print(f"因子 {len(specs)} 个 | 零校准 n_null={a.n_null} | 多头 {a.top_pct:.0%} | "
              f"区间 {a.start}→{a.end or '今天'} | 库 {a.db}")
        print("保留段由数据层强制下沉,本驱动无权解锁。")
        df = run_all(conn, specs, n_null=a.n_null, top_pct=a.top_pct, start=a.start, end=a.end,
                     seed=a.seed, leak_points=a.leak_points, progress=True, run_ref="cli")
    finally:
        conn.close()

    print("\n=== 汇总(TSV,可粘进 Excel) ===")
    print(to_tsv(df))
    pool = final_pool(df)
    print("=== 最终池(泄漏测试通过 + 分层门槛全过) ===")
    if pool:
        print("\t".join(pool))
        print(to_tsv(df[df["verdict"] == "PASS"]))
    else:
        print("(空)本轮没有因子通过全部门槛——负结果照实记录,不降门槛凑数。")
    for n in df.attrs.get("context", {}).get("notes", []):
        print(f"限制声明: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
