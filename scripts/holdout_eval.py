#!/usr/bin/env python
"""保留段一次性执行器:按已提交的票据,对样本内全部门槛通过的候选做唯一一次保留段评估。

协议:docs/HOLDOUT_PROTOCOL.md;票据格式:tasks/holdout_tickets/README.md。

两种模式,必须二选一(防止漏写 --dry-run 而误消耗保留段):
  --dry-run              只做全部检查。数据库以只读 URI(mode=ro)打开,不解锁、不写库、不写文件。
  --confirm <ticket_id>  真实执行。检查全过后先在 holdout_uses 登记 started,再以
                         unlock_holdout=True 加载 [预热起点, 数据末尾],同口径回测,
                         只在保留段上计统计量与标签,写结果文件并把票据标记为 completed。

检查项(dry-run 与真实执行共用,任何一项不过即拒绝,不解锁):
  ticket_parse / ticket_location   票据格式、ticket_id 与文件名一致、位于 tasks/holdout_tickets/
  ticket_committed                 票据已在 HEAD 中且与已提交版本一致(暂存未提交同样拒绝)
  worktree_clean                   git status --porcelain 为空(含未跟踪文件)
  code_commit                      票据 code.git_commit 为 HEAD 或其祖先,且二者之间评估相关路径无改动
  contract                         合同 cutoff 与票据一致,合同文件 sha256 与票据一致
  candidates                       因子可解析、模块源码 sha 与参数与票据一致
  in_sample                        台账条目 verdict=pass 且门槛全过、泄漏与零校准通过、数值与票据一致
  single_use                       该票据未用过;保留段未被任何票据消耗(中止的使用须以 supersedes 接续)
  prior_unlocks                    披露 meta.holdout_unlocked_log 中的既往解锁记录(信息项)
  warmup                           预热充分性(只用 cutoff 之前的数据):W 与 2W 预热下末日因子值一致

退出码:0 = 检查全过(dry-run)或执行完成;2 = 检查未过或参数不全,已拒绝(未解锁、未写库);
        3 = 已登记 started 后执行出错(保留段按已消耗处理,见协议第 7 节)。

用法:
  .venv/bin/python scripts/holdout_eval.py --ticket tasks/holdout_tickets/HT001.yaml --db data/ashare.db --dry-run
  .venv/bin/python scripts/holdout_eval.py --ticket tasks/holdout_tickets/HT001.yaml --db data/ashare.db --confirm HT001
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import re
import sqlite3
import subprocess
import sys
import traceback
import urllib.parse

from typing import Callable

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ashare_forge import pipeline  # noqa: E402
from ashare_forge.config import CONTRACT_PATH, DB_PATH  # noqa: E402
from ashare_forge.contract import load_contract  # noqa: E402
from ashare_forge.data import db, loader  # noqa: E402
from ashare_forge.engine import backtest, gates, metrics, registry  # noqa: E402

PPY = gates.PERIODS_PER_YEAR
TICKET_SCHEMA = "ashare_holdout_ticket_v1"
RESULT_SCHEMA = "ashare_holdout_result_v1"
TICKET_DIR_REL = os.path.join("tasks", "holdout_tickets")
# 评估相关路径:票据 code.git_commit 与 HEAD 之间这些路径不得有任何改动。
# 执行器本身在列:标签规则的实现也不得在看过样本内结果之后再改。
EVAL_PATHS: tuple[str, ...] = ("ashare_forge", "contracts", "scripts/holdout_eval.py", "requirements.txt")
LABEL_OK, LABEL_BAD = "holdout_consistent", "holdout_inconsistent"
CALENDAR_CODE = "sh.000001"
SEGMENT_MIN_SHARPE_DAYS = 60          # 分段样本少于此数只报累计超额,不报夏普
SENSITIVITY_EXCLUDE = "policy_rally"  # 敏感性披露:剔除该段后的超额夏普(不进标签)
DEFAULT_SEGMENTS: tuple[dict, ...] = (
    {"name": "pre_policy", "start": "2024-09-01", "end": "2024-09-23"},
    {"name": "policy_rally", "start": "2024-09-24", "end": "2024-10-08"},
    {"name": "post_rally", "start": "2024-10-09", "end": None},
)
DEFAULT_EVAL: dict = {
    "end": None, "warmup_trading_days": 250, "warmup_min_coverage": 0.5, "top_pct": 0.1,
    "bench_code": "sh.000905", "ci_level": 0.90, "decay_alpha": 0.10, "expected_holdout_days": None,
}
_TOP_KEYS = {"schema", "ticket_id", "created_utc", "supersedes", "code", "evaluation", "candidates", "notes"}
_EVAL_KEYS = set(DEFAULT_EVAL) | {"cutoff", "segments"}
_CAND_KEYS = {"factor_name", "task_id", "module", "params", "factor_module_sha", "in_sample"}
_INS_KEYS = {"trial_id", "calm_excess_sharpe", "calm_n_days"}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{2,63}$")
_HEX_RE = re.compile(r"^[0-9a-f]+$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
USE_STATUSES = ("started", "aborted", "completed")
USES_DDL = """
CREATE TABLE IF NOT EXISTS holdout_uses (
  ticket_id     TEXT PRIMARY KEY,
  cutoff        TEXT NOT NULL,
  status        TEXT NOT NULL CHECK (status IN ('started','aborted','completed')),
  supersedes    TEXT,
  ticket_sha256 TEXT NOT NULL,
  ticket_commit TEXT,
  head_commit   TEXT NOT NULL,
  started_at    TEXT NOT NULL,
  finished_at   TEXT,
  data_end      TEXT,
  error         TEXT,
  result_json   TEXT
);
"""


# =============================================================================== 统计:标准误、标签、零假设概率
def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def sharpe_se_ann(sr_ann: float, n: int, skew: float = 0.0, kurt: float = 3.0, ppy: int = PPY) -> float:
    """年化夏普的标准误(Mertens 2002;与 gates.psr 的分母同式),独立同分布假设。

    skew=0、kurt=3(正态)时退化为 sqrt((1 + SR_d²/2)/(n-1))·sqrt(ppy)。
    """
    if not _finite(sr_ann) or n is None or int(n) < 3:
        return float("nan")
    srd = float(sr_ann) / math.sqrt(ppy)
    v = (1.0 - float(skew) * srd + (float(kurt) - 1.0) / 4.0 * srd * srd) / (int(n) - 1)
    return math.sqrt(max(v, 1e-18)) * math.sqrt(ppy)


def se_null(n: int, ppy: int = PPY) -> float:
    """真实夏普为 0 时年化夏普估计的标准误(与偏度峰度无关)。"""
    return math.sqrt(ppy / (int(n) - 1)) if n is not None and int(n) >= 3 else float("nan")


def sample_sharpe(r, ppy: int = PPY) -> dict:
    """样本年化夏普、标准误(含偏度与峰度修正)与描述量。偏度/峰度口径与 gates.psr 相同。"""
    v = pd.Series(r, dtype="float64").dropna().to_numpy()
    n = int(v.size)
    out = {"n": n, "mean_ann": float("nan"), "vol_ann": float("nan"), "sr_ann": float("nan"),
           "se_ann": float("nan"), "skew": float("nan"), "kurt": float("nan")}
    if n < 3:
        return out
    m, s = float(v.mean()), float(v.std(ddof=1))
    out["mean_ann"], out["vol_ann"] = m * ppy, s * math.sqrt(ppy)
    if s <= 0:
        return out
    z = (v - m) / s
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    sr = m / s * math.sqrt(ppy)
    out.update(sr_ann=sr, skew=skew, kurt=kurt, se_ann=sharpe_se_ann(sr, n, skew, kurt, ppy))
    return out


def consistency_threshold(s_is: float, n_is: int, n_h: int, decay_alpha: float = 0.10,
                          ppy: int = PPY) -> dict:
    """事前可算的判定门限 tau(只依赖样本内夏普、两段样本量与检验水平,不依赖保留段收益)。

    se_is    样本内夏普标准误(正态 IID,在 s_is 处取值)
    se_h_ref 若真实夏普等于 s_is,保留段夏普估计的标准误(正态 IID)
    margin   单侧衰减检验的容许幅度 = z_{1-decay_alpha}·sqrt(se_is² + se_h_ref²)
    tau      max(0, s_is − margin):保留段夏普须严格大于 tau 才判 holdout_consistent
    """
    se_is = sharpe_se_ann(s_is, n_is, ppy=ppy)
    se_h_ref = sharpe_se_ann(s_is, n_h, ppy=ppy)
    z = gates.norm_ppf(1.0 - float(decay_alpha))
    margin = z * math.sqrt(se_is ** 2 + se_h_ref ** 2) if _finite(se_is) and _finite(se_h_ref) else float("nan")
    tau = max(0.0, float(s_is) - margin) if _finite(margin) else float("nan")
    return {"se_is": se_is, "se_h_ref": se_h_ref, "z": z, "margin": margin, "tau": tau,
            "decay_alpha": float(decay_alpha)}


def holdout_label(s_h: float, tau: float) -> str:
    """两值标签。保留段夏普不可得(NaN)时无法确立一致性,判 inconsistent。"""
    if _finite(s_h) and _finite(tau) and float(s_h) > float(tau):
        return LABEL_OK
    return LABEL_BAD


def p_consistent_given_true(s_true: float, tau: float, n_h: int, ppy: int = PPY) -> float:
    """真实年化夏普为 s_true 时,保留段被判 consistent 的概率(正态近似)。"""
    se = sharpe_se_ann(s_true, n_h, ppy=ppy)          # s_true=0 时即 se_null(n_h)
    if not (_finite(se) and _finite(tau)) or se <= 0:
        return float("nan")
    return 1.0 - gates.norm_cdf((float(tau) - float(s_true)) / se)


def p_consistent_null_selected(g: float, n_is: int, n_h: int, decay_alpha: float = 0.10,
                               ppy: int = PPY, n_grid: int = 4001) -> float:
    """零技能策略在样本内纯凭运气越过入选线 g 之后,保留段仍被判 consistent 的概率。

    样本内夏普估计 ~ N(0, se_null(n_is)²) 截断于 [g, ∞);对每个入选值按其 tau 计算
    P(N(0, se_null(n_h)²) > tau),再按截断正态密度积分(梯形法)。
    """
    se0 = se_null(n_is, ppy)
    if not (_finite(se0) and _finite(g)):
        return float("nan")
    xs = np.linspace(float(g), float(g) + 12.0 * se0, int(n_grid))
    w = np.exp(-0.5 * (xs / se0) ** 2)
    p = np.array([p_consistent_given_true(0.0, consistency_threshold(x, n_is, n_h, decay_alpha, ppy)["tau"],
                                          n_h, ppy) for x in xs])
    den = float(np.trapezoid(w, xs))
    return float(np.trapezoid(w * p, xs) / den) if den > 0 else float("nan")


def _solve_line(base: float, z: float, n: int, ppy: int = PPY) -> float:
    """解 S = base + z·SE(S)(SE 取正态 IID),不动点迭代。"""
    s = base + z * se_null(n, ppy)
    for _ in range(50):
        s_new = base + z * sharpe_se_ann(s, n, ppy=ppy)
        if abs(s_new - s) < 1e-12:
            break
        s = s_new
    return float(s)


def selection_lines(n_is: int, n_trials: int | None, ppy: int = PPY) -> dict:
    """样本内门槛对平静期超额夏普隐含的入选线:合同绝对线、PSR 线、DSR 线,取最大者为有效线。"""
    g = load_contract().get("gates_draft", {}) or {}
    lines: dict = {}
    if g.get("long_excess_sharpe_min") is not None:
        lines["contract_excess_sharpe_min"] = float(g["long_excess_sharpe_min"])
    if g.get("psr_min") is not None:
        lines["psr_line"] = _solve_line(0.0, gates.norm_ppf(float(g["psr_min"])), n_is, ppy)
    if n_trials is not None and int(n_trials) >= 1 and g.get("use_dsr", True):
        base = gates.expected_max_sharpe_ann(int(n_trials), int(n_is), ppy)
        lines["dsr_line"] = _solve_line(base, gates.norm_ppf(0.95), n_is, ppy)
        lines["dsr_n_trials"] = int(n_trials)
    cand = [v for k, v in lines.items() if k != "dsr_n_trials"]
    lines["effective"] = max(cand) if cand else 0.0
    return lines


def rule_preview(s_is: float, n_is: int, n_h: int, decay_alpha: float, n_trials: int | None) -> dict:
    """在不接触保留段的前提下,给出门限与判别力(dry-run 用;n_h 取票据申报的预计天数)。"""
    thr = consistency_threshold(s_is, n_is, n_h, decay_alpha)
    lines = selection_lines(n_is, n_trials)
    return {
        "n_h": int(n_h), "tau": thr["tau"], "margin": thr["margin"],
        "p_consistent_if_true_equals_is": p_consistent_given_true(s_is, thr["tau"], n_h),
        "p_consistent_null_given_is": p_consistent_given_true(0.0, thr["tau"], n_h),
        "selection_line": lines["effective"],
        "p_consistent_null_selected": p_consistent_null_selected(lines["effective"], n_is, n_h, decay_alpha),
    }


# =============================================================================== 票据
class Check:
    """一项检查的结果。ok=None 表示信息项,或因前序失败而跳过。"""

    __slots__ = ("name", "ok", "detail")

    def __init__(self, name: str, ok: bool | None, detail: str):
        self.name, self.ok, self.detail = name, ok, detail

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def _norm_date(x) -> str | None:
    if x is None:
        return None
    s = x.isoformat() if isinstance(x, (_dt.date, _dt.datetime)) else str(x)
    s = s[:10]
    if not _DATE_RE.match(s):
        raise ValueError(f"日期格式须为 YYYY-MM-DD,收到 {x!r}")
    _dt.date.fromisoformat(s)
    return s


def load_ticket(path: str) -> tuple[dict | None, list[str]]:
    """读取并校验票据,返回 (规范化票据, 错误列表)。错误非空时票据不可用。"""
    errs: list[str] = []
    try:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as e:
        return None, [f"无法读取票据:{type(e).__name__}: {e}"]
    if not isinstance(raw, dict):
        return None, ["票据顶层须为映射"]
    if raw.get("schema") != TICKET_SCHEMA:
        errs.append(f"schema 须为 {TICKET_SCHEMA},收到 {raw.get('schema')!r}")
    tid = str(raw.get("ticket_id") or "")
    stem = os.path.splitext(os.path.basename(path))[0]
    if not _ID_RE.match(tid):
        errs.append(f"ticket_id 须匹配 {_ID_RE.pattern},收到 {tid!r}")
    elif tid != stem:
        errs.append(f"ticket_id({tid})须与文件名({stem})一致")
    if not path.endswith(".yaml"):
        errs.append("票据文件扩展名须为 .yaml")

    code = raw.get("code") or {}
    gc = str(code.get("git_commit") or "").strip().lower()
    if not (7 <= len(gc) <= 40 and _HEX_RE.match(gc)):
        errs.append(f"code.git_commit 须为 7–40 位十六进制提交号,收到 {code.get('git_commit')!r}")
    cs = str(code.get("contract_sha256") or "").strip().lower()
    if not (len(cs) == 64 and _HEX_RE.match(cs)):
        errs.append("code.contract_sha256 须为 64 位十六进制")

    unknown = sorted(set(raw) - _TOP_KEYS)
    if unknown:
        errs.append(f"未知的顶层字段 {unknown}(拼写错误会让字段静默取缺省值,故一律拒绝)")
    ev_raw = raw.get("evaluation") or {}
    if not isinstance(ev_raw, dict):
        errs.append("evaluation 须为映射")
        ev_raw = {}
    unknown = sorted(set(ev_raw) - _EVAL_KEYS)
    if unknown:
        errs.append(f"evaluation 中未知字段 {unknown}")
    ev = dict(DEFAULT_EVAL)
    ev.update({k: v for k, v in ev_raw.items() if k not in ("segments", "cutoff", "end")})
    ev["cutoff"], ev["end"] = None, None
    for key in ("cutoff", "end"):
        try:
            ev[key] = _norm_date(ev_raw.get(key))
        except ValueError as e:
            errs.append(f"evaluation.{key}:{e}")
    if ev["cutoff"] is None and "evaluation.cutoff" not in " ".join(errs):
        errs.append("evaluation.cutoff 必填")
    if ev["cutoff"] and ev["end"] and ev["end"] <= ev["cutoff"]:
        errs.append(f"evaluation.end({ev['end']})须晚于 cutoff({ev['cutoff']})")
    try:
        if not 0.0 < float(ev["decay_alpha"]) < 0.5:
            errs.append("evaluation.decay_alpha 须在 (0, 0.5)")
        if not 0.5 < float(ev["ci_level"]) < 1.0:
            errs.append("evaluation.ci_level 须在 (0.5, 1)")
        if not 0.0 < float(ev["top_pct"]) <= 1.0:
            errs.append("evaluation.top_pct 须在 (0, 1]")
        if int(ev["warmup_trading_days"]) < 20:
            errs.append("evaluation.warmup_trading_days 须 ≥ 20")
        if not 0.0 <= float(ev["warmup_min_coverage"]) <= 1.0:
            errs.append("evaluation.warmup_min_coverage 须在 [0, 1]")
        if ev.get("expected_holdout_days") is not None and int(ev["expected_holdout_days"]) < 3:
            errs.append("evaluation.expected_holdout_days 须 ≥ 3")
    except (TypeError, ValueError) as e:
        errs.append(f"evaluation 数值字段:{e}")
    raw_segs = ev_raw.get("segments")
    if raw_segs is None:
        raw_segs = DEFAULT_SEGMENTS
    segs = []
    if not isinstance(raw_segs, (list, tuple)) or not all(isinstance(s, dict) for s in raw_segs):
        errs.append("evaluation.segments 须为映射列表")
    else:
        segs = [dict(s) for s in raw_segs]
    for s in segs:
        try:
            s["start"], s["end"] = _norm_date(s.get("start")), _norm_date(s.get("end"))
            if not s.get("name") or s["start"] is None:
                raise ValueError("每段须有 name 与 start")
            if ev.get("cutoff") and s["start"] < ev["cutoff"]:
                raise ValueError(f"段 {s['name']} 起点 {s['start']} 早于 cutoff:分段只能划分保留段")
            if s["end"] is not None and s["end"] < s["start"]:
                raise ValueError(f"段 {s['name']} 终点早于起点")
        except ValueError as e:
            errs.append(f"evaluation.segments:{e}")
    ev["segments"] = segs

    cands = raw.get("candidates")
    if not isinstance(cands, list) or not cands:
        errs.append("candidates 须为非空列表")
        cands = []
    names = []
    norm_cands = []
    for i, c in enumerate(cands):
        if not isinstance(c, dict):
            errs.append(f"candidates[{i}] 须为映射")
            continue
        unknown = sorted(set(c) - _CAND_KEYS)
        if unknown:
            errs.append(f"candidates[{i}] 中未知字段 {unknown}")
        nm = str(c.get("factor_name") or "")
        if not nm:
            errs.append(f"candidates[{i}].factor_name 必填")
        names.append(nm)
        sha = str(c.get("factor_module_sha") or "").strip().lower()
        if not (len(sha) == 16 and _HEX_RE.match(sha)):
            errs.append(f"candidates[{i}].factor_module_sha 须为 16 位十六进制(registry.factor_code_sha)")
        if not isinstance(c.get("params", {}), dict):
            errs.append(f"candidates[{i}].params 须为映射")
        ins = c.get("in_sample") or {}
        if isinstance(ins, dict) and set(ins) - _INS_KEYS:
            errs.append(f"candidates[{i}].in_sample 中未知字段 {sorted(set(ins) - _INS_KEYS)}")
        try:
            ins = {"trial_id": int(ins["trial_id"]), "calm_excess_sharpe": float(ins["calm_excess_sharpe"]),
                   "calm_n_days": int(ins["calm_n_days"])}
            if ins["calm_n_days"] < 30:
                errs.append(f"candidates[{i}].in_sample.calm_n_days 过小")
        except (KeyError, TypeError, ValueError):
            errs.append(f"candidates[{i}].in_sample 须含 trial_id / calm_excess_sharpe / calm_n_days")
            ins = {}
        norm_cands.append({"factor_name": nm, "task_id": c.get("task_id"), "module": c.get("module"),
                           "params": c.get("params") or {}, "factor_module_sha": sha, "in_sample": ins})
    if len(set(names)) != len(names):
        errs.append("candidates 中因子名重复")

    sup = raw.get("supersedes")
    sup = [] if sup in (None, "", []) else ([str(sup)] if isinstance(sup, str) else [str(x) for x in sup])
    tk = {"schema": raw.get("schema"), "ticket_id": tid, "created_utc": raw.get("created_utc"),
          "supersedes": sup, "code": {"git_commit": gc, "contract_sha256": cs},
          "evaluation": ev, "candidates": norm_cands, "notes": raw.get("notes")}
    return tk, errs


# =============================================================================== git 与出处
def _git(root: str, *args: str) -> subprocess.CompletedProcess:
    """git 调用;git 不可用或超时一律视为失败(returncode≠0),由调用方判拒绝,不抛出。"""
    try:
        return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return subprocess.CompletedProcess(["git", *args], 127, "", f"{type(e).__name__}: {e}")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance_snapshot(repo_root: str, ticket_path: str, specs: dict[str, dict]) -> dict:
    """锁定出处:HEAD、工作区状态指纹、合同/执行器/票据 sha256、各候选因子模块 sha。"""
    head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    status = _git(repo_root, "status", "--porcelain").stdout
    fsha = {}
    for name, spec in specs.items():
        try:
            fsha[name] = registry.factor_code_sha(spec["build_factor"])
        except Exception as e:  # noqa: BLE001
            fsha[name] = f"error:{type(e).__name__}"
    return {
        "head": head,
        "worktree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "contract_sha256": _sha256_file(CONTRACT_PATH),
        "executor_sha256": _sha256_file(os.path.abspath(__file__)),
        "ticket_sha256": _sha256_file(ticket_path) if os.path.exists(ticket_path) else None,
        "factor_module_sha": fsha,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "taken_at": _now(),
    }


def provenance_drift(start: dict, end: dict) -> list[str]:
    keys = ("head", "worktree_status_sha256", "contract_sha256", "executor_sha256", "ticket_sha256",
            "factor_module_sha")
    return [k for k in keys if start.get(k) != end.get(k)]


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# =============================================================================== 数据库
def connect_readonly(path: str) -> sqlite3.Connection:
    """只读打开(URI mode=ro):任何写入都会被 SQLite 拒绝,dry-run 的"不写库"由此强制。"""
    ap = os.path.abspath(path)
    if not os.path.exists(ap):
        raise FileNotFoundError(f"数据库不存在:{ap}")
    return sqlite3.connect(f"file:{urllib.parse.quote(ap)}?mode=ro", uri=True, timeout=30.0)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def read_uses(conn: sqlite3.Connection) -> list[dict]:
    if not _table_exists(conn, "holdout_uses"):
        return []
    cur = conn.cursor()
    cur.row_factory = None
    cur.execute("SELECT ticket_id, cutoff, status, supersedes, started_at, finished_at FROM holdout_uses")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def prior_unlocks(conn: sqlite3.Connection) -> list[dict]:
    """meta.holdout_unlocked_log 的逐行解析(解析失败的行原样保留)。"""
    if not _table_exists(conn, "meta"):
        return []
    row = conn.execute("SELECT value FROM meta WHERE key=?", (loader.HOLDOUT_LOG_KEY,)).fetchone()
    out = []
    for line in (row[0].splitlines() if row and row[0] else []):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({"raw": line})
    return out


def build_context(conn: sqlite3.Connection, start: str, end: str | None, bench_code: str,
                  unlock_holdout: bool) -> dict:
    """与 pipeline.load_context 同口径的面板 + 三面具 + 基准,唯一差别是可传 unlock_holdout。

    pipeline.load_context 不接受解锁参数(驱动无权解锁),故此处复刻;同口径由
    tests/test_holdout.py::test_context_parity_with_pipeline 在 cutoff 之前的窗口上逐项比对。
    """
    panel = loader.load_panel(conn, start, end, unlock_holdout=unlock_holdout)
    ref = panel.get("adj_close")
    if ref is None or ref.empty:
        raise ValueError(f"面板为空:[{start}, {end or '∞'}]")
    missing = pipeline._codes_missing_adj(conn, ref)
    if missing:
        raise pipeline.DataIncompleteError(
            f"复权因子缺失:{len(missing)} 只有行情的股票在 adj_factor 中无记录(如 {missing[:5]})")
    masks = loader.tradable_masks(panel)
    masks["delisted"] = loader.delisted_mask(conn, ref.index, ref.columns)
    bench = loader.index_series(conn, bench_code, start=start, end=end, unlock_holdout=unlock_holdout)
    if bench.empty:
        raise ValueError(f"基准 {bench_code} 在 [{start}, {end or '∞'}] 内无数据:超额口径不可得")
    bench = bench.reindex(ref.index).ffill()
    bo = loader.index_series(conn, bench_code, start=start, end=end, unlock_holdout=unlock_holdout, field="open")
    bench_open = bo.reindex(ref.index).ffill() if not bo.empty else None
    if bench_open is None or bool(bench_open.isna().all()):
        raise ValueError(f"基准 {bench_code} 无开盘价:无法与组合同为开盘→次日开盘口径")
    panel["bench_close"] = bench
    return {"panel": panel, "masks": masks, "bench": bench, "bench_open": bench_open,
            "first_date": str(ref.index[0].date()), "last_date": str(ref.index[-1].date()),
            "n_dates": int(ref.shape[0]), "n_codes": int(ref.shape[1])}


def pre_cutoff_calendar(conn: sqlite3.Connection) -> pd.DatetimeIndex:
    """cutoff 之前的交易日历(经 loader,受保留段约束,不触及保留段)。"""
    s = loader.index_series(conn, CALENDAR_CODE, start="2005-01-01")
    return pd.DatetimeIndex(s.index)


# =============================================================================== 检查
def _params_equal(a, b) -> bool:
    a = json.loads(json.dumps(a, default=str, sort_keys=True))
    b = json.loads(json.dumps(b, default=str, sort_keys=True))

    def eq(x, y) -> bool:
        if isinstance(x, bool) or isinstance(y, bool):
            return x is y or x == y
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return float(x) == float(y)
        if isinstance(x, dict) and isinstance(y, dict):
            return x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x)
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(eq(u, v) for u, v in zip(x, y))
        return x == y
    return eq(a, b)


def default_resolver(name: str) -> list[dict]:
    return [s for s in pipeline.discover_specs(include_optional=True) if str(s.get("name")) == name]


def _resolve_commit(root: str, ref: str) -> str | None:
    r = _git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def run_checks(ticket_path: str, conn: sqlite3.Connection, repo_root: str,
               resolver: Callable[[str], list[dict]]) -> tuple[list[Check], dict]:
    """执行全部检查。返回 (检查列表, 上下文)。上下文含解析后的票据、因子 spec、台账摘要等。"""
    checks: list[Check] = []
    ctx: dict = {"ticket": None, "specs": {}, "trials": {}, "prior_unlocks": [], "code_commit": None}

    def add(name, ok, detail):
        checks.append(Check(name, ok, detail))
        return ok

    tk, errs = load_ticket(ticket_path)
    if not add("ticket_parse", not errs, "格式与字段通过" if not errs else "; ".join(errs)):
        for n in ("ticket_location", "ticket_committed", "worktree_clean", "code_commit", "contract",
                  "candidates", "in_sample", "single_use", "warmup"):
            add(n, None, "跳过:票据无法解析")
        return checks, ctx
    ctx["ticket"] = tk
    ev = tk["evaluation"]

    # ---- 位置 ----
    rp_ticket = os.path.realpath(ticket_path)
    rp_dir = os.path.realpath(os.path.join(repo_root, TICKET_DIR_REL))
    in_dir = os.path.dirname(rp_ticket) == rp_dir
    add("ticket_location", in_dir,
        "位于 tasks/holdout_tickets/" if in_dir else f"票据须直接位于 {rp_dir},实际 {rp_ticket}")
    rel = os.path.relpath(rp_ticket, os.path.realpath(repo_root))

    # ---- 已提交 ----
    in_head = bool(_git(repo_root, "ls-tree", "--name-only", "HEAD", "--", rel).stdout.strip())
    st = _git(repo_root, "status", "--porcelain", "--", rel).stdout.strip()
    if not in_head:
        add("ticket_committed", False, "票据未提交:不在 HEAD 中(未跟踪或仅暂存)")
    elif st:
        add("ticket_committed", False, f"票据在工作区/暂存区有未提交改动:{st}")
    else:
        lg = _git(repo_root, "log", "-1", "--format=%H %cI", "--", rel).stdout.strip()
        ctx["ticket_commit"] = lg
        add("ticket_committed", True, f"已提交且与 HEAD 一致(最近提交 {lg})")

    # ---- 工作区干净 ----
    porcelain = _git(repo_root, "status", "--porcelain")
    dirty = porcelain.stdout.strip().splitlines()
    if porcelain.returncode != 0:
        add("worktree_clean", False, f"git status 失败:{porcelain.stderr.strip()}")
    else:
        add("worktree_clean", not dirty,
            "工作区干净" if not dirty else f"工作区不干净({len(dirty)} 项):" + " | ".join(dirty[:5]))

    # ---- 代码提交 ----
    head = _resolve_commit(repo_root, "HEAD")
    c_full = _resolve_commit(repo_root, tk["code"]["git_commit"])
    ctx["head"], ctx["code_commit"] = head, c_full
    if head is None:
        add("code_commit", False, "无法解析 HEAD")
    elif c_full is None:
        add("code_commit", False, f"票据 code.git_commit={tk['code']['git_commit']} 在仓库中不存在")
    elif _git(repo_root, "merge-base", "--is-ancestor", c_full, head).returncode != 0:
        add("code_commit", False, f"票据提交 {c_full[:12]} 不是 HEAD {head[:12]} 的祖先")
    else:
        diff = _git(repo_root, "diff", "--name-only", c_full, head, "--", *EVAL_PATHS).stdout.splitlines()
        if diff:
            add("code_commit", False, f"票据提交与 HEAD 之间评估相关路径有改动:{diff[:8]}")
        else:
            add("code_commit", True, ("HEAD 即票据提交" if c_full == head else
                                      f"票据提交 {c_full[:12]} 为 HEAD {head[:12]} 的祖先,评估相关路径无改动"))

    # ---- 合同 ----
    cutoff_contract = str(load_contract()["holdout"]["cutoff"])
    csha = _sha256_file(CONTRACT_PATH)
    probs = []
    if ev["cutoff"] != cutoff_contract:
        probs.append(f"票据 cutoff {ev['cutoff']} ≠ 合同 {cutoff_contract}")
    if csha != tk["code"]["contract_sha256"]:
        probs.append(f"合同 sha256 {csha[:16]}… ≠ 票据 {tk['code']['contract_sha256'][:16]}…")
    add("contract", not probs, "; ".join(probs) if probs else f"cutoff={cutoff_contract},sha256 一致")

    # ---- 候选因子 ----
    probs = []
    for c in tk["candidates"]:
        nm = c["factor_name"]
        try:
            found = resolver(nm)
        except Exception as e:  # noqa: BLE001
            probs.append(f"{nm}: 解析失败 {type(e).__name__}: {e}")
            continue
        if len(found) != 1:
            probs.append(f"{nm}: 找到 {len(found)} 个同名 spec(须恰好 1 个)")
            continue
        spec = found[0]
        if c.get("module") and str(spec.get("module")) != str(c["module"]):
            probs.append(f"{nm}: 模块 {spec.get('module')} ≠ 票据 {c['module']}")
        sha = registry.factor_code_sha(spec["build_factor"])
        if sha != c["factor_module_sha"]:
            probs.append(f"{nm}: 模块源码 sha {sha} ≠ 票据 {c['factor_module_sha']}")
        if not _params_equal(spec.get("params") or {}, c["params"]):
            probs.append(f"{nm}: 参数与票据不一致(spec={spec.get('params')})")
        ctx["specs"][nm] = spec
    add("candidates", not probs, "; ".join(probs) if probs else f"{len(tk['candidates'])} 个候选与票据一致")

    # ---- 样本内资格 ----
    probs, notes = [], []
    if not _table_exists(conn, "forge_trials"):
        probs.append("库中无 forge_trials 台账")
    else:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(forge_trials)")}
        need = {"name", "verdict", "sr_calm", "code_sha", "git_commit", "provenance_json",
                "leak_passed", "null_passed"}
        if not need.issubset(cols):
            probs.append(f"forge_trials 缺列 {sorted(need - cols)}(旧库,出处不可核)")
        else:
            for c in tk["candidates"]:
                nm, ins = c["factor_name"], c["in_sample"]
                cur = conn.cursor()
                cur.row_factory = None
                cur.execute("SELECT name, verdict, sr_calm, code_sha, git_commit, provenance_json, leak_passed,"
                            " null_passed FROM forge_trials WHERE id=?", (ins.get("trial_id"),))
                row = cur.fetchone()
                if row is None:
                    probs.append(f"{nm}: 台账无 id={ins.get('trial_id')}")
                    continue
                t_name, verdict, sr_calm, code_sha, t_commit, pj, leak_p, null_p = row
                pv = json.loads(pj) if pj else {}
                gv = pv.get("gates") or {}
                rows = gv.get("rows") or []
                calm_n = next((r.get("value") for r in rows
                               if r.get("layer") == "calm" and r.get("gate") == "n_days"), None)
                data = pv.get("data") or {}
                p = []
                if t_name != nm:
                    p.append(f"台账条目名 {t_name} ≠ {nm}")
                if verdict != "pass" or gv.get("overall_pass") is not True:
                    p.append(f"样本内未全部通过门槛(verdict={verdict}, overall_pass={gv.get('overall_pass')})")
                if leak_p != 1 or null_p != 1:
                    p.append(f"泄漏/零校准未通过(leak_passed={leak_p}, null_passed={null_p})")
                if code_sha != c["factor_module_sha"]:
                    p.append(f"台账 code_sha {code_sha} ≠ 票据 {c['factor_module_sha']}")
                tc = str(t_commit or "")
                if not tc or tc.endswith("-dirty") or tc == "unknown":
                    p.append(f"台账 git_commit={tc!r}:样本内运行的代码未入库,不可溯源")
                elif c_full is None or _resolve_commit(repo_root, tc) != c_full:
                    p.append(f"台账 git_commit {tc} ≠ 票据 code.git_commit")
                if pv.get("provenance_drift"):
                    p.append("样本内运行期间出处漂移")
                if not _finite(sr_calm) or abs(float(sr_calm) - ins["calm_excess_sharpe"]) > 1e-6 * max(
                        1.0, abs(ins["calm_excess_sharpe"])):
                    p.append(f"台账 sr_calm={sr_calm} ≠ 票据 {ins['calm_excess_sharpe']}")
                if calm_n is None or int(calm_n) != ins["calm_n_days"]:
                    p.append(f"台账平静期天数 {calm_n} ≠ 票据 {ins['calm_n_days']}")
                if data.get("bench_code") and data["bench_code"] != ev["bench_code"]:
                    p.append(f"台账基准 {data['bench_code']} ≠ 票据 {ev['bench_code']}")
                ld = str(data.get("last_date") or "")
                if ld and ld >= ev["cutoff"]:
                    p.append(f"样本内末日 {ld} 已进入保留段")
                if not _params_equal(pv.get("params") or {}, c["params"]):
                    p.append("台账参数与票据不一致")
                if p:
                    probs.append(f"{nm}: " + "; ".join(p))
                ctx["trials"][nm] = {"trial_id": ins.get("trial_id"), "n_trials": gv.get("n_trials"),
                                     "last_date": ld or None, "first_date": data.get("first_date")}
                notes.append(f"{nm}: trial {ins.get('trial_id')}, n_trials={gv.get('n_trials')}, 样本内末日 {ld}")
    add("in_sample", not probs, "; ".join(probs) if probs else "; ".join(notes))

    # ---- 一次性 ----
    uses = [u for u in read_uses(conn) if u.get("cutoff") == ev["cutoff"]]
    ctx["uses"] = uses
    probs = []
    mine = [u for u in uses if u["ticket_id"] == tk["ticket_id"]]
    if mine:
        probs.append(f"票据 {tk['ticket_id']} 已被使用(状态 {mine[0]['status']})")
    done = [u["ticket_id"] for u in uses if u["status"] == "completed"]
    if done:
        probs.append(f"保留段已被票据 {done} 消耗;新评估须使用其后新增的数据(协议第 8 节)")
    pending = {u["ticket_id"] for u in uses if u["status"] in ("started", "aborted")}
    if pending - set(tk["supersedes"]):
        probs.append(f"存在未完成的使用 {sorted(pending - set(tk['supersedes']))},新票据须以 supersedes 接续并披露")
    if set(tk["supersedes"]) - pending:
        probs.append(f"supersedes 指向的 {sorted(set(tk['supersedes']) - pending)} 不是未完成的使用")
    res_path = os.path.join(os.path.dirname(rp_ticket), f"{tk['ticket_id']}.result.json")
    if os.path.exists(res_path):
        probs.append(f"结果文件已存在:{os.path.basename(res_path)}")
    add("single_use", not probs, "; ".join(probs) if probs else "票据未用过,保留段未被消耗")

    # ---- 既往解锁(信息项) ----
    pu = prior_unlocks(conn)
    ctx["prior_unlocks"] = pu
    if pu:
        ts = [str(r.get("ts_utc", "?")) for r in pu]
        add("prior_unlocks", None, f"meta 中有 {len(pu)} 条既往解锁记录(最早 {ts[0]},最近 {ts[-1]}),结果中须披露")
    else:
        add("prior_unlocks", None, "meta 中无既往解锁记录")

    # ---- 预热(只用 cutoff 之前的数据;前序全过才做,避免无谓加载) ----
    if not all(ch.ok for ch in checks if ch.ok is not None):
        add("warmup", None, "跳过:前序检查未通过")
    else:
        try:
            ok, detail, info = warmup_check(conn, tk, ctx["specs"])
            ctx["warmup"] = info
            add("warmup", ok, detail)
        except Exception as e:  # noqa: BLE001
            add("warmup", False, f"预热检查出错:{type(e).__name__}: {e}")
    return checks, ctx


def warmup_check(conn: sqlite3.Connection, tk: dict, specs: dict[str, dict]) -> tuple[bool, str, dict]:
    """以 W 与 2W 两种预热长度构造因子,比较 cutoff 前最后一个交易日的取值(全部为样本内数据)。"""
    ev = tk["evaluation"]
    cal = pre_cutoff_calendar(conn)
    if len(cal) == 0:
        return False, f"交易日历 {CALENDAR_CODE} 在 cutoff 前为空", {}
    w = int(ev["warmup_trading_days"])
    load_start = cal[-w] if len(cal) >= w else cal[0]
    start_2w = cal[-2 * w] if len(cal) >= 2 * w else cal[0]
    last_pre = cal[-1]
    info = {"load_start": str(load_start.date()), "start_2w": str(start_2w.date()),
            "last_pre_cutoff_day": str(last_pre.date()), "calendar_short": len(cal) < 2 * w}
    ctx = build_context(conn, str(start_2w.date()), str(last_pre.date()), ev["bench_code"], unlock_holdout=False)
    panel, inu = ctx["panel"], ctx["masks"]["in_universe"]
    sub = {k: v.loc[load_start:] for k, v in panel.items()}
    keep = sub["adj_close"].notna().any(axis=0)
    sub = {k: (v.loc[:, keep[keep].index] if isinstance(v, pd.DataFrame) else v) for k, v in sub.items()}
    probs, notes = [], []
    for nm, spec in specs.items():
        f2 = pipeline._aligned_factor(spec["build_factor"], panel)
        f1 = pipeline._aligned_factor(spec["build_factor"], sub)
        a, b = f1.iloc[-1], f2.iloc[-1].reindex(f1.columns)
        both_nan = a.isna() & b.isna()
        diff = (a - b).abs()[~both_nan]
        mismatch = int((diff.isna() | (diff > 1e-9 * np.maximum(1.0, b.abs()[~both_nan]))).sum())
        last_inu = inu.iloc[-1]
        n_inu = int(last_inu.sum())
        cov = float(f2.iloc[-1][last_inu[last_inu].index].notna().mean()) if n_inu else float("nan")
        if mismatch:
            probs.append(f"{nm}: W={w} 与 2W 预热下末日因子值有 {mismatch} 处不一致(预热不足)")
        if not _finite(cov) or cov < float(ev["warmup_min_coverage"]):
            probs.append(f"{nm}: 末日在池股票的因子覆盖率 {cov:.1%} < {float(ev['warmup_min_coverage']):.0%}")
        notes.append(f"{nm}: 覆盖率 {cov:.1%},W/2W 不一致 {mismatch} 处")
        info[nm] = {"coverage_last_pre_day": cov, "mismatch_w_vs_2w": mismatch}
    tail = f"(预热起点 {info['load_start']},末个样本内交易日 {info['last_pre_cutoff_day']}" + (
        ";日历短于 2W,比较退化为同一面板)" if info["calendar_short"] else ")")
    return (not probs), ("; ".join(probs) if probs else "; ".join(notes)) + tail, info


# =============================================================================== 保留段评估
def evaluate_holdout(bt: dict, cutoff: str, end: str | None, segments: list[dict], s_is: float, n_is: int,
                     n_trials: int | None, decay_alpha: float, ci_level: float) -> dict:
    """在同口径回测结果上切出保留段,计算统计量、标签与零假设概率。

    观测口径:只计收益区间完全落在 [cutoff, 末尾] 的观测。组合第 d 行 = d 开盘→次日开盘;
    面板末行的次日开盘不可得,剔除。IC 按收益区间对齐(ic.shift(1)),与组合行一一对应。
    """
    net = pd.Series(bt["daily_ret"], dtype="float64")
    idx = pd.DatetimeIndex(net.index)
    excess = pd.Series(bt.get("excess_ret"), dtype="float64").reindex(idx) if bt.get(
        "excess_ret") is not None else pd.Series(np.nan, index=idx)
    h = np.asarray(idx >= pd.Timestamp(cutoff))
    if end:
        h &= np.asarray(idx <= pd.Timestamp(end))
    if len(h):
        h[-1] = False
    ic_ret = pd.Series(bt.get("ic"), dtype="float64").reindex(idx).shift(1)
    z_ci = gates.norm_ppf(0.5 + ci_level / 2.0)

    e = excess[h]
    sh = sample_sharpe(e)
    n_h = sh["n"]
    net_s = sample_sharpe(net[h])
    ics = metrics.ic_stats(ic_ret[h].dropna())
    ic_se = ics["ic_std"] / math.sqrt(ics["n_days"]) if ics["n_days"] >= 2 and _finite(ics["ic_std"]) else float("nan")
    stats = {
        "n_days": n_h,
        "first_day": str(idx[h][0].date()) if h.any() else None,
        "last_day": str(idx[h][-1].date()) if h.any() else None,
        "excess_sharpe": sh["sr_ann"], "excess_sharpe_se": sh["se_ann"],
        "excess_sharpe_ci": [sh["sr_ann"] - z_ci * sh["se_ann"], sh["sr_ann"] + z_ci * sh["se_ann"]],
        "excess_ann": sh["mean_ann"], "excess_vol": sh["vol_ann"],
        "excess_skew": sh["skew"], "excess_kurt": sh["kurt"],
        "excess_psr_vs_0": gates.psr(e) if n_h >= 30 else float("nan"),
        "excess_max_drawdown": gates.max_drawdown((1.0 + e.fillna(0.0)).cumprod()) if n_h else float("nan"),
        "net_ann": net_s["mean_ann"], "net_sharpe": net_s["sr_ann"],
        "ic_n_days": ics["n_days"], "ic_mean": ics["ic_mean"],
        "ic_mean_ci": [ics["ic_mean"] - z_ci * ic_se, ics["ic_mean"] + z_ci * ic_se],
        "icir": ics["icir"], "ic_t": ics["t_stat"],
        "avg_turnover": float(pd.Series(bt["turnover"]).reindex(idx)[h].mean()) if n_h else float("nan"),
        "avg_holdings": float(pd.Series(bt["n_holdings"]).reindex(idx)[h].mean()) if n_h else float("nan"),
        "ann_cost": float(pd.Series(bt["cost"]).reindex(idx)[h].mean()) * PPY if n_h else float("nan"),
    }

    segs = []
    for s in segments:
        m = h & np.asarray(idx >= pd.Timestamp(s["start"]))
        if s.get("end"):
            m &= np.asarray(idx <= pd.Timestamp(s["end"]))
        es = excess[m].dropna()
        ss = sample_sharpe(es)
        segs.append({
            "name": s["name"], "start": s["start"], "end": s.get("end"), "n_days": int(es.size),
            "cum_excess": float((1.0 + es).prod() - 1.0) if es.size else float("nan"),
            "excess_ann": ss["mean_ann"],
            "excess_sharpe": ss["sr_ann"] if es.size >= SEGMENT_MIN_SHARPE_DAYS else None,
            "ic_mean": float(ic_ret[m].dropna().mean()) if ic_ret[m].notna().any() else float("nan"),
        })
    sens = None
    ex_seg = next((s for s in segments if s["name"] == SENSITIVITY_EXCLUDE), None)
    if ex_seg is not None:
        m = np.asarray(idx >= pd.Timestamp(ex_seg["start"]))
        if ex_seg.get("end"):
            m &= np.asarray(idx <= pd.Timestamp(ex_seg["end"]))
        ss = sample_sharpe(excess[h & ~m])
        sens = {"excluded_segment": SENSITIVITY_EXCLUDE, "n_days": ss["n"], "excess_sharpe": ss["sr_ann"],
                "note": "仅披露,不进标签"}

    thr = consistency_threshold(s_is, n_is, max(n_h, 3), decay_alpha) if n_h >= 3 else {
        "se_is": sharpe_se_ann(s_is, n_is), "se_h_ref": float("nan"), "z": gates.norm_ppf(1 - decay_alpha),
        "margin": float("nan"), "tau": float("nan"), "decay_alpha": decay_alpha}
    label = holdout_label(sh["sr_ann"], thr["tau"])
    d = sh["sr_ann"] - s_is
    se_d = math.sqrt(thr["se_is"] ** 2 + sh["se_ann"] ** 2) if _finite(sh["se_ann"]) and _finite(
        thr["se_is"]) else float("nan")
    lines = selection_lines(n_is, n_trials)
    null = {
        "given_observed_in_sample": p_consistent_given_true(0.0, thr["tau"], n_h) if n_h >= 3 else float("nan"),
        "selected_at_effective_line": (p_consistent_null_selected(lines["effective"], n_is, n_h, decay_alpha)
                                       if n_h >= 3 else float("nan")),
        "selected_at_contract_line": (p_consistent_null_selected(lines.get("contract_excess_sharpe_min", 0.0),
                                                                 n_is, n_h, decay_alpha)
                                      if n_h >= 3 else float("nan")),
        "selection_lines": lines,
    }
    return {
        "holdout": stats,
        "segments": segs,
        "sensitivity": sens,
        "rule": {**thr, "s_is": s_is, "n_is": n_is, "n_h": n_h,
                 "statement": "holdout_consistent ⇔ 保留段超额夏普 > tau = max(0, S_is − z·sqrt(se_is² + se_h_ref²))"},
        "label": label,
        "decay": {"d": d, "se_obs": se_d, "z_obs": d / se_d if _finite(se_d) and se_d > 0 else float("nan"),
                  "ci": [d - z_ci * se_d, d + z_ci * se_d]},
        "p_consistent_null": null,
        "p_consistent_if_true_equals_is": p_consistent_given_true(s_is, thr["tau"], n_h) if n_h >= 3 else float("nan"),
    }


# =============================================================================== 执行
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, _dt.date)):
        return o.isoformat()
    return o


def _checks_tsv(checks: list[Check]) -> str:
    lines = ["check\tstatus\tdetail"]
    for c in checks:
        st = "PASS" if c.ok is True else ("FAIL" if c.ok is False else "INFO")
        lines.append(f"{c.name}\t{st}\t{c.detail}")
    return "\n".join(lines)


def _f(x, nd: int = 4) -> str:
    return f"{float(x):.{nd}f}" if _finite(x) else ""


def summary_tsv(result: dict) -> str:
    lines = ["factor\tn_days\texcess_sharpe\tci_low\tci_high\ts_is\ttau\tlabel\tp_null_selected"
             "\tp_null_given_is\tp_if_true_eq_is\tic_mean"]
    for c in result.get("candidates", []):
        hs, r, pn = c["holdout"], c["rule"], c["p_consistent_null"]
        ci = hs["excess_sharpe_ci"]
        lines.append("\t".join([
            c["factor_name"], str(hs["n_days"]), _f(hs["excess_sharpe"]), _f(ci[0]), _f(ci[1]), _f(r["s_is"]),
            _f(r["tau"]), str(c["label"]), _f(pn["selected_at_effective_line"]),
            _f(pn["given_observed_in_sample"]), _f(c["p_consistent_if_true_equals_is"]), _f(hs["ic_mean"])]))
    return "\n".join(lines)


class RegistrationError(RuntimeError):
    """登记 started 失败(并发占用、库锁等)。此时尚未解锁,保留段未被消耗。"""


def _record_start(conn, tk, ticket_sha, ticket_commit, head) -> None:
    """在同一个写事务里复核"一次性"条件并登记 started,防止两个进程同时通过检查后各自解锁。"""
    try:
        conn.executescript(USES_DDL)
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute("SELECT ticket_id, status FROM holdout_uses WHERE cutoff=?",
                            (tk["evaluation"]["cutoff"],)).fetchall()
        sup = set(tk["supersedes"])
        blocking = [(r[0], r[1]) for r in rows if r[0] == tk["ticket_id"] or r[1] == "completed"
                    or (r[1] in ("started", "aborted") and r[0] not in sup)]
        if blocking:
            conn.rollback()
            raise RegistrationError(f"登记时发现占用 {blocking}")
        conn.execute(
            "INSERT INTO holdout_uses (ticket_id, cutoff, status, supersedes, ticket_sha256, ticket_commit,"
            " head_commit, started_at) VALUES (?,?,?,?,?,?,?,?)",
            (tk["ticket_id"], tk["evaluation"]["cutoff"], "started", ",".join(tk["supersedes"]) or None,
             ticket_sha, ticket_commit, head, _now()))
        conn.commit()
    except sqlite3.Error as e:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        raise RegistrationError(f"{type(e).__name__}: {e}") from e


def _record_end(conn, ticket_id: str, status: str, data_end=None, error=None, result=None) -> None:
    conn.execute("UPDATE holdout_uses SET status=?, finished_at=?, data_end=?, error=?, result_json=?"
                 " WHERE ticket_id=?",
                 (status, _now(), data_end, error,
                  json.dumps(_jsonable(result), ensure_ascii=False, sort_keys=True) if result is not None else None,
                  ticket_id))
    conn.commit()


def execute(conn: sqlite3.Connection, tk: dict, specs: dict[str, dict], cctx: dict, repo_root: str,
            ticket_path: str, snap0: dict, out_path: str) -> dict:
    """真实执行:登记 started → 解锁加载 → 同口径回测 → 保留段统计与标签 → 出处比对 → 写结果。"""
    ev = tk["evaluation"]
    tid = tk["ticket_id"]
    prior = list(cctx.get("prior_unlocks") or [])
    _record_start(conn, tk, snap0["ticket_sha256"], cctx.get("ticket_commit"), snap0["head"])
    loader._log_holdout_unlock(conn, "holdout_eval", ticket_id=tid, phase="begin")
    try:
        load_start = (cctx.get("warmup") or {}).get("load_start")
        if not load_start:
            raise RuntimeError("预热起点缺失(warmup 检查未产出)")
        hctx = build_context(conn, load_start, ev["end"], ev["bench_code"], unlock_holdout=True)
        cands = []
        for c in tk["candidates"]:
            spec = specs[c["factor_name"]]
            fac = pipeline._aligned_factor(spec["build_factor"], hctx["panel"])
            bt = backtest.run_backtest(fac, hctx["panel"], hctx["masks"], bench=hctx["bench"],
                                       top_pct=float(ev["top_pct"]), bench_open=hctx["bench_open"])
            n_trials = (cctx.get("trials", {}).get(c["factor_name"]) or {}).get("n_trials")
            res = evaluate_holdout(bt, ev["cutoff"], ev["end"], ev["segments"],
                                   c["in_sample"]["calm_excess_sharpe"], c["in_sample"]["calm_n_days"],
                                   n_trials, float(ev["decay_alpha"]), float(ev["ci_level"]))
            cands.append({"factor_name": c["factor_name"], "in_sample": c["in_sample"],
                          "trial": cctx.get("trials", {}).get(c["factor_name"]), **res})
            del bt, fac
        snap1 = provenance_snapshot(repo_root, ticket_path, specs)
        drift = provenance_drift(snap0, snap1)
        valid = not drift
        if not valid:
            for c in cands:
                c["label"] = None
        p_sel = [c["p_consistent_null"]["selected_at_effective_line"] for c in cands]
        fam = (1.0 - float(np.prod([1.0 - p for p in p_sel]))) if all(_finite(p) for p in p_sel) else float("nan")
        result = {
            "schema": RESULT_SCHEMA, "ticket_id": tid, "valid": valid,
            "status": "completed" if valid else "invalid_provenance_drift",
            "cutoff": ev["cutoff"], "load_start": load_start, "data_end": hctx["last_date"],
            "ticket_commit": cctx.get("ticket_commit"), "code_commit": cctx.get("code_commit"),
            "supersedes": tk["supersedes"], "evaluation": ev,
            "prior_unlocks": prior,
            "provenance": {"start": snap0, "end": snap1, "drift": drift},
            "candidates": cands,
            "family": {"k": len(cands), "p_any_consistent_null_selected_upper": fam,
                       "note": "各候选零技能下被判一致的概率按独立合成;候选正相关时为上界"},
            "disclaimer": ("标签只有 holdout_consistent / holdout_inconsistent 两值。"
                           "一致≠证明有效:见 p_consistent_null(零技能策略被事后挑中后仍被判一致的概率)。"),
        }
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(result), fh, ensure_ascii=False, indent=2, sort_keys=True)
        _record_end(conn, tid, "completed", data_end=hctx["last_date"], result=result)
        loader._log_holdout_unlock(conn, "holdout_eval", ticket_id=tid, phase="end", status=result["status"])
        return result
    except Exception as e:
        _record_end(conn, tid, "aborted", error=f"{type(e).__name__}: {e}"[:2000])
        loader._log_holdout_unlock(conn, "holdout_eval", ticket_id=tid, phase="end", status="aborted")
        raise


def _pre_resolve(ticket_path: str, resolver: Callable[[str], list[dict]]) -> dict[str, dict]:
    """启动时解析候选因子(只为锁定出处;解析失败的候选留给 run_checks 报错)。"""
    tk, _ = load_ticket(ticket_path)
    out: dict[str, dict] = {}
    for c in (tk or {}).get("candidates", []):
        try:
            found = resolver(c["factor_name"])
        except Exception:  # noqa: BLE001
            continue
        if len(found) == 1:
            out[c["factor_name"]] = found[0]
    return out


def run_ticket(ticket_path: str, db_path: str, *, dry_run: bool, confirm: str | None = None,
               repo_root: str | None = None, resolver: Callable[[str], list[dict]] | None = None,
               out_path: str | None = None, echo: Callable[[str], None] = print) -> tuple[int, dict]:
    """执行器主体(CLI 与测试共用)。repo_root/resolver 为测试注入口,不经 CLI 暴露。"""
    repo_root = os.path.realpath(repo_root or ROOT)
    resolver = resolver or default_resolver
    ticket_path = os.path.abspath(ticket_path)
    if not dry_run and not confirm:
        echo("拒绝:真实执行须显式 --confirm <ticket_id>;只做检查请用 --dry-run。")
        return 2, {"refused": "no_mode"}
    if not os.path.exists(os.path.abspath(db_path)):
        echo(f"拒绝:数据库不存在 {os.path.abspath(db_path)}(不会新建空库)")
        return 2, {"refused": "no_db"}

    # 出处在进程启动时、任何数据加载之前锁定;结束时与之比对(见 execute)
    snap0 = provenance_snapshot(repo_root, ticket_path, _pre_resolve(ticket_path, resolver))
    conn = connect_readonly(db_path) if dry_run else db.connect(db_path)
    try:
        checks, cctx = run_checks(ticket_path, conn, repo_root, resolver)
        tk = cctx.get("ticket")
        echo(_checks_tsv(checks))
        failed = [c.name for c in checks if c.ok is False]
        report = {"checks": [c.as_dict() for c in checks], "failed": failed, "snapshot": snap0}
        if tk is not None and tk["evaluation"].get("expected_holdout_days"):
            n_h = int(tk["evaluation"]["expected_holdout_days"])
            prev = []
            for c in tk["candidates"]:
                ins = c.get("in_sample") or {}
                if ins:
                    nt = (cctx.get("trials", {}).get(c["factor_name"]) or {}).get("n_trials")
                    prev.append({"factor_name": c["factor_name"],
                                 **rule_preview(ins["calm_excess_sharpe"], ins["calm_n_days"], n_h,
                                                float(tk["evaluation"]["decay_alpha"]), nt)})
            report["rule_preview"] = prev
            if prev:
                echo("规则预估(不接触保留段;n_h 取票据申报值)\nfactor\tn_h\ttau\tp_if_true_eq_is"
                     "\tp_null_given_is\tselection_line\tp_null_selected")
                for p in prev:
                    echo(f"{p['factor_name']}\t{p['n_h']}\t{_f(p['tau'])}\t{_f(p['p_consistent_if_true_equals_is'])}"
                         f"\t{_f(p['p_consistent_null_given_is'])}\t{_f(p['selection_line'])}"
                         f"\t{_f(p['p_consistent_null_selected'])}")
        if failed:
            echo(f"拒绝:{len(failed)} 项检查未通过 {failed};未解锁保留段,未写库。")
            return 2, report
        if dry_run:
            echo("dry-run:全部检查通过。未解锁保留段,未写库,未写文件。")
            return 0, report
        if confirm != tk["ticket_id"]:
            echo(f"拒绝:--confirm {confirm!r} 与票据 ticket_id {tk['ticket_id']!r} 不一致;未解锁。")
            report["refused"] = "confirm_mismatch"
            return 2, report
        out = out_path or os.path.join(os.path.dirname(ticket_path), f"{tk['ticket_id']}.result.json")
        try:
            result = execute(conn, tk, cctx["specs"], cctx, repo_root, ticket_path, snap0, out)
        except RegistrationError as e:
            echo(f"拒绝:登记 started 失败({e});未解锁保留段。")
            report["refused"] = "registration"
            return 2, report
        except Exception as e:  # noqa: BLE001
            echo(f"执行出错(保留段按已消耗处理,票据状态 aborted):{type(e).__name__}: {e}")
            echo(traceback.format_exc()[-1500:])
            report["error"] = f"{type(e).__name__}: {e}"
            return 3, report
        report["result"] = result
        report["result_path"] = out
        echo("=== 保留段结果(TSV) ===")
        echo(summary_tsv(result))
        if not result["valid"]:
            echo(f"出处漂移 {result['provenance']['drift']}:本次不出标签,保留段已消耗。")
        echo(f"结果文件:{out}(须原样提交入库)")
        return 0, report
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="保留段一次性执行器(协议见 docs/HOLDOUT_PROTOCOL.md)")
    ap.add_argument("--ticket", required=True, help="tasks/holdout_tickets/<id>.yaml")
    ap.add_argument("--db", default=DB_PATH)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只做全部检查,只读打开数据库,不解锁不写库")
    mode.add_argument("--confirm", default=None, metavar="TICKET_ID", help="真实执行;须与票据 ticket_id 一致")
    ap.add_argument("--out", default=None, help="结果文件路径(缺省 tasks/holdout_tickets/<id>.result.json)")
    a = ap.parse_args(argv)
    code, _ = run_ticket(a.ticket, a.db, dry_run=a.dry_run, confirm=a.confirm, out_path=a.out)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
