"""机制签名 + 试验台账(SQLite 版,截面场景)。

它修的坑(每条对应一种常见的研究自欺方式):
1. **换皮**:同一个机制换名字、加平滑、加封顶、外面套一层截面秩,再跑一次当成新发现。
   本模块的签名不按代码文本算,而按**规范化后的机制**算:装饰性算子(clip/smooth/
   cs_rank/gate…)被剔除、同义算子归并,换皮变体会碰撞到同一签名——碰撞正是设计目标。
2. **多重检验不计数**:DSR 的 n_trials 必须数出来。每次尝试记一行 trial,
   `trial_count` 直接喂给 `gates.evaluate`,而不是每次都填 1。
3. **草稿不入账的竞态**:机制在跑之前先以 draft 注册,并发/跨会话立刻可见,
   避免两条线各自"新发现"同一个东西。
4. **失败被删掉**:失败是路标不是垃圾。abandoned/fail 一样入账,`dedup_report`
   把同签名的全部前科连同死因笔记还给下一个研究者。

研究域:试验计数与查重以**股票池 universe** 为边界(缺省 `ashare_cs` = A股全市场截面);
算子词表按量价截面场景定义。

台账与行情同库(缺省 data/ashare.db),表 `forge_mechanisms` / `forge_trials`,
建表幂等;所有写入都是单行小事务——全量同步在跑时不会长时间独占写锁。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone

DDL = """
CREATE TABLE IF NOT EXISTS forge_mechanisms (
  signature  TEXT PRIMARY KEY,
  universe   TEXT NOT NULL,
  family     TEXT NOT NULL,
  canon_json TEXT NOT NULL,
  thesis     TEXT,
  first_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forge_trials (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  signature   TEXT NOT NULL,
  universe    TEXT NOT NULL,
  family      TEXT NOT NULL,
  task_id     TEXT,
  name        TEXT,
  params_json TEXT,
  run_ref     TEXT,
  verdict     TEXT NOT NULL,
  ic_mean     REAL,
  icir        REAL,
  sr_calm     REAL,
  sr_crisis   REAL,
  dsr_value   REAL,
  n_days      INTEGER,
  n_codes     INTEGER,
  notes       TEXT,
  created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trials_sig ON forge_trials(signature);
CREATE INDEX IF NOT EXISTS idx_trials_domain ON forge_trials(universe, family);
"""

# ---- 受控词表 ----------------------------------------------------------------
# 信号族(截面场景):同族内部换参数不算新机制,跨族才算。
FAMILIES = {
    "value",        # 估值
    "growth",       # 成长
    "quality",      # 质量/盈利
    "momentum",     # 动量/持续
    "reversal",     # 反转
    "volatility",   # 波动率/风险
    "liquidity",    # 流动性/换手
    "size",         # 规模
    "pattern",      # 量价形态(箱体突破、缺口、形态识别…)
    "sentiment",    # 情绪/关注度
    "seasonal",     # 季节性/日历
    "other",
}

# 方向:因子值与预期收益的约定 + 组合构建方式。A股融券受限,截面默认 long_only。
DIRECTIONS = {"long_only", "long_short", "signed", "reversion", "continuation"}

# 装饰性算子:只改形态不改经济机制 → 从签名中剔除(反"换皮"的关键)。
# 注意 neutralize(行业/市值中性化)**不在此列**:它改变了因子在赌什么,算核心算子。
DECOR_OPS = {
    "clip", "cap", "winsorize", "smooth", "ema", "sma_smooth", "gate", "threshold",
    "rank", "cs_rank", "pct_rank", "sign", "scale", "normalize_output", "standardize_cs",
    "lag_confirm", "shift", "fillna", "dropna", "weighted_sum", "clip_upper", "clip_lower",
}

# 核心算子同义词归并
OP_ALIASES = {
    "z": "zscore", "z_score": "zscore", "rolling_zscore": "zscore", "ts_zscore": "zscore",
    "ma": "rolling_mean", "sma": "rolling_mean", "mean": "rolling_mean",
    "rolling_avg": "rolling_mean", "ts_mean": "rolling_mean",
    "std": "rolling_std", "ts_std": "rolling_std", "vol": "rolling_std",
    "max": "rolling_max", "ts_max": "rolling_max", "roll_max": "rolling_max",
    "min": "rolling_min", "ts_min": "rolling_min", "roll_min": "rolling_min",
    "sum": "rolling_sum", "windowed_sum": "rolling_sum",
    "ret": "pct_change", "return": "pct_change", "momentum": "pct_change",
    "mom": "pct_change", "reversal": "pct_change",
    "ratio": "imbalance", "imb": "imbalance",
    "intensity": "intensity_norm", "relative_intensity": "intensity_norm",
    "vol_ratio": "intensity_norm", "volume_ratio": "intensity_norm",
    "turnover_ratio": "intensity_norm",
    "spread": "diff_ab", "basis": "diff_ab",
    "range": "hl_range", "amplitude": "hl_range",
    "breakout_pct": "breakout", "box_breakout": "breakout",
}

VERDICTS = {"draft", "pass", "fail", "downgraded", "abandoned"}
DEFAULT_UNIVERSE = "ashare_cs"

__all__ = [
    "FAMILIES", "DIRECTIONS", "DECOR_OPS", "OP_ALIASES", "VERDICTS", "DEFAULT_UNIVERSE",
    "ensure_tables", "mechanism_signature", "signature_from_spec", "register_mechanism",
    "log_trial", "dedup_report", "trial_count", "recent_trials",
]


# ---- 签名 --------------------------------------------------------------------

def _canon_universe(universe: str | None) -> str:
    u = str(universe or DEFAULT_UNIVERSE).strip().lower()
    return u or DEFAULT_UNIVERSE


def _bucket_horizon(horizon_days: float) -> str:
    if horizon_days < 1:
        return "intraday"
    if horizon_days <= 3:
        return "short"
    if horizon_days <= 15:
        return "medium"
    return "long"


def _canon_ops(operators) -> list[str]:
    out = set()
    for op in operators or []:
        op = str(op).strip().lower()
        op = OP_ALIASES.get(op, op)
        if op in DECOR_OPS:
            continue
        if op:
            out.add(op)
    return sorted(out)


def mechanism_signature(family: str, operators, direction: str, horizon_days: float,
                        columns, universe: str = DEFAULT_UNIVERSE) -> tuple[str, dict]:
    """返回 (16 位签名, 规范化描述)。签名相同 = 同一机制家族。

    family       FAMILIES 之一
    operators    核心算子列表(装饰算子剔除、同义词归并)
    direction    DIRECTIONS 之一
    horizon_days 持有期(天),按桶归并(intraday/short/medium/long)
    columns      用到的数据列(panel 字段名)
    universe     研究域,缺省 'ashare_cs'(A股全市场截面)
    """
    family = str(family).strip().lower()
    if family not in FAMILIES:
        raise ValueError(f"family 必须是 {sorted(FAMILIES)} 之一,收到: {family!r}")
    direction = str(direction).strip().lower()
    if direction not in DIRECTIONS:
        raise ValueError(f"direction 必须是 {sorted(DIRECTIONS)} 之一,收到: {direction!r}")
    canon = {
        "universe": _canon_universe(universe),
        "family": family,
        "operators": _canon_ops(operators),
        "direction": direction,
        "horizon_bucket": _bucket_horizon(float(horizon_days)),
        "columns": sorted({str(c).strip().lower() for c in (columns or []) if str(c).strip()}),
    }
    js = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(js.encode("utf-8")).hexdigest()[:16], canon


def signature_from_spec(spec: dict) -> tuple[str, dict]:
    """从因子 spec 取签名字段(pipeline 用)。缺字段直接抛错,不猜。"""
    missing = [k for k in ("family", "operators", "direction", "horizon_days", "columns")
               if k not in spec]
    if missing:
        raise ValueError(f"spec 缺签名字段 {missing}(name={spec.get('name')!r})")
    return mechanism_signature(spec["family"], spec["operators"], spec["direction"],
                               spec["horizon_days"], spec["columns"],
                               spec.get("universe", DEFAULT_UNIVERSE))


# ---- 台账 --------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cursor(conn: sqlite3.Connection) -> sqlite3.Cursor:
    """取元组游标:调用方可能把 conn.row_factory 设成 sqlite3.Row,这里不受影响。"""
    cur = conn.cursor()
    cur.row_factory = None
    return cur


def ensure_tables(conn: sqlite3.Connection) -> None:
    """建表(幂等)。重复调用无副作用。"""
    conn.executescript(DDL)
    conn.commit()


def register_mechanism(conn: sqlite3.Connection, signature: str, canon: dict,
                       thesis: str = "") -> bool:
    """草稿即注册。已存在返回 False(= 撞车预警),新签名返回 True。"""
    cur = _cursor(conn)
    cur.execute("SELECT 1 FROM forge_mechanisms WHERE signature = ?", (signature,))
    if cur.fetchone():
        return False
    cur.execute(
        "INSERT INTO forge_mechanisms (signature, universe, family, canon_json, thesis, first_seen)"
        " VALUES (?,?,?,?,?,?)",
        (signature, canon["universe"], canon["family"],
         json.dumps(canon, ensure_ascii=False, sort_keys=True), thesis, _now()),
    )
    conn.commit()
    return True


def log_trial(conn: sqlite3.Connection, signature: str, canon: dict, verdict: str,
              run_ref: str = "", name: str = "", task_id: str = "",
              params: dict | None = None, ic_mean: float | None = None,
              icir: float | None = None, sr_calm: float | None = None,
              sr_crisis: float | None = None, dsr_value: float | None = None,
              n_days: int | None = None, n_codes: int | None = None,
              notes: str = "", provenance: dict | None = None) -> int:
    """记一次尝试,返回行 id。verdict 必须是 VERDICTS 之一(拼错直接抛错,不静默入库)。

    provenance:证据包(因子代码哈希、git 提交、泄漏测试、零校准明细、全部统计量、数据快照)。
    审查发现(P1,"最致命的追问"):原台账只存 IC 与几个夏普,泄漏结论、零校准结果、超额收益、
    代码版本全部不存,头条数字只活在手写 markdown 里——被问"你怎么证明这些数字是这套代码
    跑出来的"时答不上来。现把证据整包入库,标量关键项另存列以便查询。
    """
    _ensure_provenance_columns(conn)
    if verdict not in VERDICTS:
        raise ValueError(f"verdict 必须是 {sorted(VERDICTS)} 之一,收到 {verdict!r}")

    def _f(x):
        if x is None:
            return None
        x = float(x)
        return None if x != x else x            # NaN → NULL(SQL 里 IS NULL 查得到)

    pv = provenance or {}
    st = pv.get("stats") or {}

    def _b(x):
        return None if x is None else int(bool(x))

    cur = _cursor(conn)
    cur.execute(
        "INSERT INTO forge_trials (signature, universe, family, task_id, name, params_json,"
        " run_ref, verdict, ic_mean, icir, sr_calm, sr_crisis, dsr_value, n_days, n_codes,"
        " notes, created_at, excess_ann, excess_sharpe, monotonicity, avg_turnover,"
        " leak_passed, null_passed, code_sha, git_commit, provenance_json)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (signature, canon["universe"], canon["family"], task_id or None, name or None,
         json.dumps(params or {}, ensure_ascii=False, sort_keys=True, default=str),
         run_ref, verdict, _f(ic_mean), _f(icir), _f(sr_calm), _f(sr_crisis), _f(dsr_value),
         None if n_days is None else int(n_days), None if n_codes is None else int(n_codes),
         notes, _now(),
         _f(st.get("excess_ann")), _f(st.get("excess_sharpe")), _f(st.get("monotonicity")),
         _f(st.get("avg_turnover")),
         _b((pv.get("leak") or {}).get("passed")), _b((pv.get("null") or {}).get("passed")),
         pv.get("code_sha"), pv.get("git_commit"),
         json.dumps(pv, ensure_ascii=False, sort_keys=True, default=_json_default) if pv else None),
    )
    conn.commit()
    return int(cur.lastrowid)


def recent_trials(conn: sqlite3.Connection, signature: str, limit: int = 5) -> list[dict]:
    cur = _cursor(conn)
    cur.execute(
        "SELECT created_at, verdict, name, ic_mean, sr_calm, sr_crisis, run_ref, notes"
        " FROM forge_trials WHERE signature = ? ORDER BY id DESC LIMIT ?",
        (signature, int(limit)),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def dedup_report(conn: sqlite3.Connection, signature: str) -> dict:
    """同签名前科:是否已知、判定分布、最近几次的死因笔记。写代码前必查。"""
    cur = _cursor(conn)
    cur.execute("SELECT signature, universe, family, canon_json, thesis, first_seen"
                " FROM forge_mechanisms WHERE signature = ?", (signature,))
    row = cur.fetchone()
    mech = None
    if row is not None:
        cols = [d[0] for d in cur.description]
        mech = dict(zip(cols, row))
    cur.execute("SELECT verdict, COUNT(*) FROM forge_trials WHERE signature = ? GROUP BY verdict",
                (signature,))
    counts = {str(v): int(n) for v, n in cur.fetchall()}
    return {
        "known_mechanism": mech is not None,
        "mechanism": mech,
        "verdict_counts": counts,
        "recent_trials": recent_trials(conn, signature),
        "advice": ("同签名机制已有前科:先读失败笔记,换机制而不是换参数换名字"
                   if mech is not None else "新机制:注册后继续"),
    }


def trial_count(conn: sqlite3.Connection, universe: str, family: str | None = None,
                scope: str = "universe") -> int:
    """累计尝试次数 → gates.evaluate 的 n_trials(DSR 的多重检验代价)。

    口径:draft 不计数(还没消耗一次检验);pass/fail/downgraded/abandoned 计数
    ——泄漏被毙掉的那次同样消耗了一次"看数据的机会",不豁免。

    审查发现(P1):原按 (研究域, 信号族) 计数,实测只有 2~3,而整个研究程序已跑 7 个因子、
    每个还有窗口/方向/中性化等选择,真实试验数在 30~100 量级——DSR 的惩罚差一个数量级。
    现默认 scope="universe":**同一研究域内所有族的试验都计入**(多重检验针对的是
    "研究者一共看了多少次数据",不以因子族为界)。scope="family" 保留旧口径作对照。
    """
    cur = _cursor(conn)
    if scope == "family":
        cur.execute(
            "SELECT COUNT(*) FROM forge_trials WHERE universe = ? AND family = ?"
            " AND verdict IN ('pass','fail','downgraded','abandoned')",
            (_canon_universe(universe), str(family).strip().lower()),
        )
    elif scope == "universe":
        cur.execute(
            "SELECT COUNT(*) FROM forge_trials WHERE universe = ?"
            " AND verdict IN ('pass','fail','downgraded','abandoned')",
            (_canon_universe(universe),),
        )
    else:
        raise ValueError(f"scope 只能是 'universe' 或 'family',收到 {scope!r}")
    return int(cur.fetchone()[0])


def declared_trials(task_id: str | None, tasks_dir: str | None = None) -> int:
    """读任务卡的 trials_declared(申报的参数档数)。

    审查发现:任务卡早已申报 trials_declared,但流水线从未读取,申报了等于没申报。
    参数档是"研究者为这张卡试过的变体",必须计入多重检验代价。
    """
    import os
    import yaml
    if not task_id:
        return 0
    root = tasks_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "tasks")
    path = os.path.join(root, f"{task_id}.yaml")
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        card = yaml.safe_load(fh) or {}
    td = card.get("trials_declared") or []
    if isinstance(td, list):
        return len(td)
    if isinstance(td, dict):
        # 分类申报(如 {preregistered: 6, post_hoc: 2}):各类档数相加,事后追加同样计费
        return sum(int(v or 0) for v in td.values())
    return int(td or 0)


def main() -> None:
    from ..config import DB_PATH
    from ..data import db

    ap = argparse.ArgumentParser(description="AShare Forge 试验台账")
    ap.add_argument("--db", default=DB_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report", help="按签名查前科")
    p.add_argument("signature")
    p = sub.add_parser("count", help="按(研究域,信号族)数尝试次数")
    p.add_argument("family")
    p.add_argument("--universe", default=DEFAULT_UNIVERSE)
    sub.add_parser("list", help="列出全部已注册机制")
    a = ap.parse_args()

    conn = db.connect(a.db)
    try:
        ensure_tables(conn)
        if a.cmd == "report":
            print(json.dumps(dedup_report(conn, a.signature), ensure_ascii=False, indent=2, default=str))
        elif a.cmd == "count":
            print(trial_count(conn, a.universe, a.family))
        else:
            cur = _cursor(conn)
            cur.execute("SELECT signature, universe, family, first_seen FROM forge_mechanisms"
                        " ORDER BY first_seen")
            print("signature\tuniverse\tfamily\tfirst_seen")
            for r in cur.fetchall():
                print("\t".join(str(x) for x in r))
    finally:
        conn.close()


if __name__ == "__main__":
    main()


# ==================== 行为指纹(对"申报算子名"签名的补强) ====================
# 声明签名的已知弱点(tests/test_registry.py 如实钉住):机制签名建立在**自我申报的算子名**上,
# 换个表外名字(znorm/zscore_robust/standard_score…)即可绕过查重;
# 反过来,算子集合化又会把 zscore∘mean 与 mean∘zscore 误判为同一机制。
# 本节的补强思路:不看因子"自称是什么",看它**实际算出什么**——
# 在一张固定的合成面板上运行因子,取若干日的截面排序向量作为指纹,
# 两个因子的指纹相关性高 = 行为上是同一个东西,无论它们怎么申报。
#
# 它仍不能防什么(必须诚实说明):
#   - 行为指纹只在合成面板上取样,真实数据上的差异可能更大;
#   - 线性变换后的同一因子(如取负)相关性为 -1,本实现按绝对值判重复,
#     因此"因子与其反向版本"会被判为同一机制——这是有意的(方向在签名里另记);
#   - 对依赖真实财务字段(peTTM 等)的因子,合成面板只能给出近似行为。

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FP_DATES, FP_STOCKS, FP_SAMPLE_ROWS = 400, 80, 12
BEHAVIOR_DUP_THRESHOLD = 0.95


def _structured_fundamentals(rng, close, shape, f) -> dict:
    """有结构的合成基本面:盈利与净资产跨股票正相关,约 12% 的股票亏损(负 PE)。"""
    n_days, n_stocks = shape
    quality = rng.normal(size=(1, n_stocks))                         # 公司"质地"潜变量
    eps_lvl = 0.6 + 0.5 * quality + 0.2 * rng.normal(size=(1, n_stocks))
    eps = eps_lvl * np.exp(np.cumsum(rng.normal(0, 0.004, shape), axis=0))
    bps = np.abs(6.0 + 3.0 * quality + 1.5 * rng.normal(size=(1, n_stocks))) + 0.5
    bps = bps * np.exp(np.cumsum(rng.normal(0, 0.002, shape), axis=0))
    sps = np.abs(eps) * rng.uniform(4.0, 12.0, size=(1, n_stocks))
    cfps = eps * rng.uniform(0.6, 1.6, size=(1, n_stocks))
    px = close.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"peTTM": f(px / eps), "pbMRQ": f(px / bps),
                "psTTM": f(px / sps), "pcfNcfTTM": f(px / cfps)}


def _fixed_panel(seed: int = 42) -> dict:
    """一张固定的合成面板(同 seed 必得同结果),用于给因子"量行为"。

    字段覆盖 loader 输出的常用列,数值量级贴近 A 股(价格 5~50 元、换手 0.1~5%)。
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=FP_DATES)
    cols = [f"sz.{i:06d}" for i in range(FP_STOCKS)]
    shape = (FP_DATES, FP_STOCKS)
    ret = rng.normal(0.0, 0.02, shape)
    close = pd.DataFrame(10.0 * np.cumprod(1.0 + ret, axis=0), index=idx, columns=cols)
    intraday = np.abs(rng.normal(0.0, 0.01, shape))
    vol = pd.DataFrame(rng.lognormal(15.0, 0.6, shape), index=idx, columns=cols)
    turn = pd.DataFrame(rng.uniform(0.1, 5.0, shape), index=idx, columns=cols)
    f = lambda v: pd.DataFrame(v, index=idx, columns=cols)  # noqa: E731
    return {
        "close": close, "adj_close": close,
        "open": close * (1.0 + rng.normal(0, 0.005, shape)),
        "high": close * (1.0 + intraday), "low": close * (1.0 - intraday),
        "preclose": close.shift(1).bfill(),
        "volume": vol, "amount": vol * close, "turn": turn,
        "tradestatus": f(np.ones(shape)), "isST": f(np.zeros(shape)),
        # 审查发现(P1):原先基本面字段是**各自独立的随机数**,E/P 与 B/P 在合成面板上毫无关系,
        # 于是任何基于基本面的因子指纹必然"互不相关"——实测估值两因子指纹相似度 0.007
        # 而真实面板截面秩相关 +0.43,"无重复挖矿"的结论对它们是随机数造出来的。
        # 现按现实结构生成:每股有持久的盈利水平与净资产(二者正相关),估值 = 价格 / 基本面。
        **_structured_fundamentals(rng, close, shape, f),
        # 审查发现:合成面板缺列会让行为指纹静默失效(实测 illiquidity 因缺 pctChg 退化为仅声明签名查重)。
        # 面板字段须与 loader.load_panel 的默认输出保持一致。
        "pctChg": f(ret * 100.0),                        # baostock 口径:百分数
        "float_mcap": close * vol * 100.0,
        "listing_age": f(np.tile(np.arange(FP_DATES).reshape(-1, 1), (1, FP_STOCKS)) + 500.0),
    }


def behavior_fingerprint(build_factor, seed: int = 42) -> np.ndarray:
    """在固定合成面板上运行因子,返回若干日截面**百分位秩**拼成的行为指纹向量。

    用秩而非原值:消除量纲与单调变换(x → 2x+1 行为相同)的影响。
    """
    panel = _fixed_panel(seed)
    fac = build_factor(panel)
    if not isinstance(fac, pd.DataFrame):
        raise TypeError(f"因子必须返回 DataFrame,得到 {type(fac).__name__}")
    ranks = fac.rank(axis=1, pct=True)
    rows = np.linspace(len(ranks) // 3, len(ranks) - 1, FP_SAMPLE_ROWS).astype(int)
    v = ranks.iloc[rows].to_numpy(dtype="float64").ravel()
    return np.nan_to_num(v, nan=0.5)      # 未取值处填中位秩,保证长度一致可比


def fingerprint_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """两条行为指纹的相关性绝对值(取绝对值:因子与其反向版本视为同一机制)。"""
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    if a.shape != b.shape or a.size < 2:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(abs(np.corrcoef(a, b)[0, 1]))


def _fp_encode(fp: np.ndarray) -> str:
    return json.dumps([round(float(x), 4) for x in fp])


def _fp_decode(s: str) -> np.ndarray:
    return np.asarray(json.loads(s), dtype="float64")


def save_fingerprint(conn: sqlite3.Connection, signature: str, fp: np.ndarray) -> None:
    """把行为指纹挂到机制上(表已存在则忽略列缺失,兼容旧库)。"""
    conn.execute("CREATE TABLE IF NOT EXISTS forge_fingerprints ("
                 "signature TEXT PRIMARY KEY, fp_json TEXT NOT NULL, created_at TEXT NOT NULL)")
    conn.execute("INSERT OR REPLACE INTO forge_fingerprints (signature, fp_json, created_at) "
                 "VALUES (?,?,?)",
                 (signature, _fp_encode(fp), datetime.now(timezone.utc).isoformat(timespec="seconds")))
    conn.commit()


def find_behavioral_duplicates(conn: sqlite3.Connection, fp: np.ndarray, exclude: str | None = None,
                               threshold: float = BEHAVIOR_DUP_THRESHOLD) -> list[dict]:
    """在台账里找行为上与之高度相似的既有机制(**不看它们怎么申报**)。"""
    conn.execute("CREATE TABLE IF NOT EXISTS forge_fingerprints ("
                 "signature TEXT PRIMARY KEY, fp_json TEXT NOT NULL, created_at TEXT NOT NULL)")
    out = []
    for sig, js in conn.execute("SELECT signature, fp_json FROM forge_fingerprints"):
        if exclude and sig == exclude:
            continue
        s = fingerprint_similarity(fp, _fp_decode(js))
        if s == s and s >= threshold:
            row = conn.execute("SELECT family, thesis FROM forge_mechanisms WHERE signature=?",
                               (sig,)).fetchone()
            out.append({"signature": sig, "similarity": round(s, 4),
                        "family": row[0] if row else None,
                        "thesis": (row[1] if row else None)})
    return sorted(out, key=lambda x: -x["similarity"])

# ==================== 证据包(provenance) ====================
_PROV_COLUMNS = {
    "excess_ann": "REAL", "excess_sharpe": "REAL", "monotonicity": "REAL", "avg_turnover": "REAL",
    "leak_passed": "INTEGER", "null_passed": "INTEGER",
    "code_sha": "TEXT", "git_commit": "TEXT", "provenance_json": "TEXT",
}


def _ensure_provenance_columns(conn: sqlite3.Connection) -> None:
    """幂等补列(兼容旧库,不改建表 DDL)。"""
    have = {r[1] for r in conn.execute("PRAGMA table_info(forge_trials)")}
    for c, t in _PROV_COLUMNS.items():
        if c not in have:
            conn.execute(f"ALTER TABLE forge_trials ADD COLUMN {c} {t}")   # 列名来自内部常量
    conn.commit()


def _json_default(o):
    import numpy as _np
    if isinstance(o, (_np.floating, _np.integer)):
        v = o.item()
        return None if isinstance(v, float) and v != v else v
    if isinstance(o, _np.bool_):
        return bool(o)
    if hasattr(o, "to_dict"):
        return o.to_dict(orient="records") if hasattr(o, "columns") else o.to_dict()
    return str(o)


def factor_code_sha(build_factor) -> str:
    """因子**所在模块**源码的 sha256 前 16 位:同一哈希 ⇒ 同一版代码。

    取模块而非函数本体:因子常依赖模块内的常量与辅助函数,只哈希函数会漏掉它们的改动。
    """
    import inspect
    try:
        mod = inspect.getmodule(build_factor)
        src = inspect.getsource(mod) if mod is not None else inspect.getsource(build_factor)
    except (OSError, TypeError):
        return "unavailable"
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def git_commit(repo_root: str | None = None) -> str:
    """当前 git HEAD(工作区有未提交改动时加 -dirty 后缀,如实反映"这份结果对应的代码未入库")。"""
    import os
    import subprocess
    root = repo_root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        head = subprocess.run(["git", "-C", root, "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "ashare_forge"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (head + ("-dirty" if dirty else "")) if head else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"
