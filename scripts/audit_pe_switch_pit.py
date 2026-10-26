#!/usr/bin/env python
"""估值字段 peTTM / pbMRQ 切换时点的 PIT 审计(只读;不读收益,不消耗试验次数)。

问题
----
baostock 日线的 peTTM(滚动市盈率)与 pbMRQ(市净率)是逐日快照字段,字段本身不带财报发布日。
若它们在**报告期末**而非**首次披露日**切换到新一期财报,则两日之间的因子值含有当时尚未公开的
盈利/净资产数字,构成数据层前视。本脚本抽样(股票, 报告期)对,逐日观察隐含每股收益
e = close / peTTM 与隐含每股净资产 b = close / pbMRQ 在何日跳变,并与首次披露日比对。

判据(预注册:下列常量随本文件首次提交写定,运行后不改;如需偏离,在 docs/PIT-VALUATION-AUDIT.md
单列"偏离预注册"并说明理由)
------------------------------------------------------------------------------------------
1. 剔除日(该日的跳变不计入):
   a. 复权因子变动日:adj_factor 表中 backAdjustFactor 相对前一条记录变化的日期(除权除息/送转)。
   b. 股本变动日:相邻两行 e 与 s = close / psTTM 同比例变化,即 |ln(e_t/e_{t-1}) − ln(s_t/s_{t-1})|
      ≤ SAME_PROP_TOL 且 |ln(e_t/e_{t-1})| > REL_JUMP_TOL;b 与 s 同比例者同样计入(取并集),
      该日对 e、b 两个序列都剔除。股本变化按同一比例改变每股收益、每股净资产与每股营收,
      新财报则各项变化比例不同。
   c. 全市场大面积陈旧日及其后一交易日(见第 5 条的"精确口径";EXCLUDE_STALE_DAYS 控制)。
2. 跳变:同一序列相邻有效行 |x_t − x_p| / max(|x_t|, |x_p|) > REL_JUMP_TOL(可处理穿零)。
3. 首次披露日 P = min(query_profit_data.pubDate, query_performance_express_report 中同一报告期的
   performanceExpPubDate),非交易日顺延至下一交易日(交易日历取 index_daily 中 sh.000001 的日期)。
4. 早跳:报告期末 S ≤ d < P 内存在未剔除、且不能归因于他期披露的跳变。
   "归因于他期披露":存在其他报告期的首次披露事件 E(上一报告期的正式报告 pubDate、窗口内其他
   报告期的业绩快报 pubDate,均顺延到交易日)满足 E ≤ d 且 d − E ≤ EXPLAIN_LAG_TD 个交易日。
   上一报告期正式报告的 pubDate 按需获取:一季度对(年报常在 S 之后披露)一律获取;
   其他季度仅当 [S, P) 内出现未剔除跳变时获取(节省调用,不影响判定)。
   判定:peTTM 的早跳比例 ≤ EARLY_RATIO_MAX → 通过(peTTM 的切换时点可当 PIT 使用);否则不通过。
   pbMRQ 按同一口径另行判定。比例的分母为"可判定"对:有披露日、披露日早于保留段、报告期末前
   BASE_LOOKBACK_TD 个交易日内有有效基准行、[S, P) 内交易日至少 MIN_PRE_COVERAGE 有有效行
   (长期停牌无行时无法观察切换日)、且 P 当日或之后有有效行。按点估计判定,
   另报 Clopper–Pearson 95% 区间。
5. 另报(不参与判定):
   - 跳变日 J − 首次披露日 P 的交易日差分布(J = 第一个早跳;无早跳时取 [P, 窗口末] 内第一个
     未剔除跳变;窗口末 = max(P, 正式报告 pubDate) + POST_TD 个交易日,不越过保留段);
     同时报相对正式报告 pubDate 的差。
   - 隐含 EPS(J 日新值)与 query_profit_data.epsTTM 的相对误差分布;早跳中新值与本期 epsTTM
     吻合(相对误差 ≤ EPS_MATCH_TOL)的占比,作为"早跳是否真是提前切到本期"的直接证据。
   - 早跳中属于"回摆"(REVERT_ROWS 行内回到跳前水平 ±REL_JUMP_TOL)的占比,提示数据毛刺。
   - 全市场大面积陈旧日:
       宽口径(任务原定义):当日有前值的股票中 > STALE_SHARE 的 peTTM 与前一行完全相同;
       精确口径:仅计当日正常交易(tradestatus=1)且收盘价较前一行变化的股票,> STALE_SHARE 的
       peTTM 与前一行完全相同(排除停牌与平盘股的合法不变)。剔除日(1c)用精确口径。
       CHECK_DATES 中的日期无论是否达标都列出。

抽样(固定种子,拉数据前写定)
----------------------------
沪主板(sh.60)/ 深主板(sz.00)/ 创业板(sz.30)/ 科创板(sh.68)× 报告期所在年 2015–2024。
2024 年只取 Q1/Q2(Q3 起披露日落入保留段);科创板自 2019Q3 起。每格 N_PER_CELL[板块] 对,
其中 DELISTED_PER_CELL 对取自已退市股(stocks.out_date 非空)——若该格有合格退市股。
合格:ipo_date < S 且(未退市或 out_date > S + MIN_ALIVE_AFTER_STAT_DAYS 自然日),且库中
[S − 30 自然日, S] 有日线行。同格内股票不重复。处理顺序按"各格轮转",中途停止时覆盖仍分层。

数据与约束
----------
- 行情只读:sqlite3 URI mode=ro,只取 HOLDOUT_CUTOFF 之前的日线(保留段不碰)。不写库。
- baostock:单进程串行;每对 2 次(本期利润表 + 快报区间)+ 按需 1 次(上期利润表);
  总调用(含登录与重试)上限 MAX_API_CALLS。error_code=10002007 按 BACKOFF 指数退避并重登,
  退避耗尽即停止取数,用已取部分出结果并写明覆盖率。响应缓存于 data/cache/pit_audit/(已被
  .gitignore),重跑命中缓存不再计调用,可断点续传。快报查询区间截止于保留段之前。
- 不可测:数值是否为首次披露值(追溯重述)。本审计只检验切换时点,不检验数值版本。

用法
----
  .venv/bin/python scripts/audit_pe_switch_pit.py --selftest
  .venv/bin/python scripts/audit_pe_switch_pit.py --db /abs/path/data/ashare.db
  .venv/bin/python scripts/audit_pe_switch_pit.py --db /abs/path/data/ashare.db --offline   # 只用缓存
输出:reports/pit_audit_pairs.tsv / _summary.tsv / _lag.tsv / _relerr.tsv / _stale_days.tsv
"""
from __future__ import annotations

import argparse
import bisect
import contextlib
import datetime as _dt
import hashlib
import io
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# ============================================================================ 预注册判据
SEED = 20260927
HOLDOUT_CUTOFF = "2024-09-01"          # 只读此前的日线;保留段不碰
YEARS = tuple(range(2015, 2025))       # 报告期所在年
LAST_QUARTER_2024 = 2                  # 2024 年只取 Q1/Q2
STAR_FIRST_STAT = "2019-09-30"         # 科创板首个可用报告期(2019-07-22 开板)
STRATA = (                             # (标签, 代码前缀, stocks.board)
    ("沪主板", "sh.60", "main"),
    ("深主板", "sz.00", "main"),
    ("创业板", "sz.30", "chinext"),
    ("科创板", "sh.68", "star"),
)
N_PER_CELL = {"沪主板": 6, "深主板": 6, "创业板": 6, "科创板": 5}   # 30×6 + 6×5 = 210 对
DELISTED_PER_CELL = 1
MIN_ALIVE_AFTER_STAT_DAYS = 150
SAMPLE_ROW_CHECK_DAYS = 30             # 抽样时要求 [S−30 自然日, S] 有日线
MAX_REDRAW = 60

REL_JUMP_TOL = 1e-3                    # 跳变阈值(对称相对变化 0.1%)
SAME_PROP_TOL = 1e-3                   # e 与 s 同比例(对数差 ≤ 0.1%)→ 股本变动日
EXCLUDE_STALE_DAYS = True              # 剔除精确口径陈旧日及其后一交易日
STALE_SHARE = 0.5
CHECK_DATES = ("2020-02-19",)
EXPLAIN_LAG_TD = 5                     # 他期披露后 0–5 个交易日内的跳变归因他期
POST_TD = 20                           # 披露后观察窗(交易日)
BASE_LOOKBACK_TD = 20                  # S 之前须有有效基准行的回看窗(交易日)
MIN_PRE_COVERAGE = 0.8                 # [S, P) 内交易日须有 ≥ 80% 存在有效行,否则该对不可判定
EARLY_RATIO_MAX = 0.01                 # 判定线:早跳比例 ≤ 1% → 通过
EPS_MATCH_TOL = 0.02                   # 辅助:新值与 epsTTM 相对误差 ≤ 2% 视为吻合
REVERT_ROWS = 3                        # 辅助:回摆判定行数
DIAG_TOL_ALT = 0.01                    # 事后敏感性(运行后追加,不参与判定):跳变阈值放宽到 1%
MAX_API_CALLS = 590                    # 含登录与重试
EXPRESS_RANGE_DAYS = (-120, 300)       # 快报查询区间(相对 S 的自然日;截止于保留段之前)
RATE_LIMIT_CODE = "10002007"
BACKOFF = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
CALENDAR_INDEX = "sh.000001"
# ============================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT, "reports")
CACHE_DIR = os.path.join(ROOT, "data", "cache", "pit_audit")
QEND = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}


def _log(msg: str) -> None:
    print(f"# [{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _add_days(d: str, n: int) -> str:
    return (_dt.date.fromisoformat(d) + _dt.timedelta(days=n)).isoformat()


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


# ============================================================================ 交易日历
class Calendar:
    def __init__(self, dates: list[str]):
        self.d = sorted(dates)

    def roll(self, day: str | None) -> str | None:
        """≥ day 的第一个交易日;越过日历末端(保留段)返回 None。"""
        if not day:
            return None
        i = bisect.bisect_left(self.d, day)
        return self.d[i] if i < len(self.d) else None

    def pos(self, day: str) -> int:
        return bisect.bisect_left(self.d, day)

    def td(self, a: str, b: str) -> int:
        """b − a 的交易日差(均按日历位置;非交易日按顺延位置计)。"""
        return self.pos(b) - self.pos(a)

    def shift(self, day: str, n: int) -> str:
        i = min(max(self.pos(day) + n, 0), len(self.d) - 1)
        return self.d[i]

    def next_of(self, day: str) -> str | None:
        i = bisect.bisect_right(self.d, day)
        return self.d[i] if i < len(self.d) else None


# ============================================================================ 纯逻辑(可自测)
@dataclass
class Jump:
    date: str
    prev_date: str
    x_prev: float
    x_cur: float
    rel: float
    flags: tuple = ()
    revert: bool = False


def ratio_series(close, denom) -> list[float]:
    """x = close / denom;denom 为 0/缺失/非有限 → NaN。"""
    out = []
    for c, v in zip(close, denom):
        if _finite(c) and _finite(v) and v != 0.0:
            out.append(float(c) / float(v))
        else:
            out.append(float("nan"))
    return out


def share_change_days(dates: list[str], e: list[float], b: list[float], s: list[float]) -> set[str]:
    """股本变动日:相邻两行 e(或 b)与 s 同比例变化且变化本身超过跳变阈值。

    股本变化同时按同一比例改变每股收益、每股净资产与每股营收;新财报则各项变化比例不同。
    预注册口径以 e 与 s 为准;pe 缺失时以 b 与 s 补判(取并集)。
    """
    out: set[str] = set()
    for i in range(1, len(dates)):
        sp, sc = s[i - 1], s[i]
        if not (_finite(sp) and _finite(sc)) or sp * sc <= 0:
            continue
        ls = math.log(sc / sp)
        for x in (e, b):
            xp, xc = x[i - 1], x[i]
            if not (_finite(xp) and _finite(xc)) or xp * xc <= 0:
                continue
            lx = math.log(xc / xp)
            if abs(lx) > REL_JUMP_TOL and abs(lx - ls) <= SAME_PROP_TOL:
                out.add(dates[i])
                break
    return out


def find_jumps(dates: list[str], x: list[float], adj_days: set[str], share_days: set[str],
               stale_days: set[str]) -> list[Jump]:
    """相邻有效行的跳变,附剔除标记(adj/share/stale)与回摆标记。

    回摆(诊断用,不参与判定):跳变后 REVERT_ROWS 行内回到跳前水平;去程与返程两条腿都标记。
    """
    valid = [i for i in range(len(dates)) if _finite(x[i])]
    out: list[Jump] = []
    glitch_until = ""
    for k in range(1, len(valid)):
        ip, ic = valid[k - 1], valid[k]
        xp, xc = x[ip], x[ic]
        rel = abs(xc - xp) / max(abs(xc), abs(xp))
        if rel <= REL_JUMP_TOL:
            continue
        d = dates[ic]
        flags = []
        if d in adj_days:
            flags.append("adj")
        if d in share_days:
            flags.append("share")
        if d in stale_days:
            flags.append("stale")
        revert = d <= glitch_until            # 前一回摆的返程腿
        if not revert:
            for kk in range(k + 1, min(k + 1 + REVERT_ROWS, len(valid))):
                xv = x[valid[kk]]
                if abs(xv - xp) / max(abs(xv), abs(xp)) <= REL_JUMP_TOL:
                    revert = True
                    glitch_until = dates[valid[kk]]
                    break
        out.append(Jump(d, dates[ip], xp, xc, rel, tuple(flags), revert))
    return out


@dataclass
class PairResult:
    early: bool = False
    switch_date: str | None = None
    lag_td: int | None = None
    lag_profit_td: int | None = None
    x_new: float | None = None
    n_pre_unexplained: int = 0
    n_pre_explained: int = 0
    pre_dates: str = ""
    early_revert: bool = False
    excl: Counter = field(default_factory=Counter)
    pre_unexpl: list = field(default_factory=list)      # 诊断用:[S,P) 内未归因跳变
    pre_excluded: list = field(default_factory=list)    # 诊断用:[S,P) 内被剔除的跳变


def classify(jumps: list[Jump], S: str, P: str, P_profit: str | None, w_end: str,
             other_events: list[str], cal: Calendar) -> PairResult:
    """按预注册口径把一个序列在 [S, w_end] 内的跳变分类。"""
    r = PairResult()
    inwin = [j for j in jumps if S <= j.date <= w_end]
    for j in inwin:
        for f in j.flags:
            r.excl[f] += 1
    cand = [j for j in inwin if not j.flags]
    pre_unexpl = []
    for j in cand:
        if j.date >= P:
            continue
        explained = any(E <= j.date and cal.td(E, j.date) <= EXPLAIN_LAG_TD
                        for E in other_events if E and E < P)
        if explained:
            r.n_pre_explained += 1
        else:
            pre_unexpl.append(j)
    r.n_pre_unexplained = len(pre_unexpl)
    r.pre_unexpl = pre_unexpl
    r.pre_excluded = [j for j in inwin if j.flags and j.date < P]
    r.pre_dates = ";".join(j.date for j in pre_unexpl)
    if pre_unexpl:
        J = pre_unexpl[0]
        r.early = True
        r.early_revert = J.revert
    else:
        post = [j for j in cand if j.date >= P]
        J = post[0] if post else None
    if J is not None:
        r.switch_date = J.date
        r.lag_td = cal.td(P, J.date)
        r.lag_profit_td = cal.td(P_profit, J.date) if P_profit else None
        r.x_new = J.x_cur
    return r


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """二项比例的精确 95% 区间(二分求解,不依赖 scipy)。"""
    if n <= 0:
        return (float("nan"), float("nan"))

    def cdf(kk: int, p: float) -> float:   # P(X ≤ kk | p),随 p 单调递减
        if kk < 0:
            return 0.0
        if kk >= n:
            return 1.0
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(kk + 1))

    def bisect_p(pred) -> float:   # pred(p) 在 [0,1] 上由 True 变 False 的分界点
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if pred(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else bisect_p(lambda p: 1 - cdf(k - 1, p) < alpha / 2)
    upper = 1.0 if k == n else bisect_p(lambda p: cdf(k, p) > alpha / 2)
    return (lower, upper)


# ============================================================================ 自测(合成数据)
def selftest() -> None:
    days = [f"2020-{m:02d}-{d:02d}" for m in (9, 10, 11) for d in range(1, 29)]
    cal = Calendar(days)
    S, P, W = "2020-09-28", "2020-10-20", "2020-11-20"
    n = len(days)
    nan = [float("nan")] * n
    close = [10.0 + 0.1 * (i % 5) for i in range(n)]

    def per_share(v):            # 每股量 → 对应估值倍数(close / 每股量)
        return [close[i] / v[i] for i in range(n)]

    def jumps(eps, rev, adj=(), stale=(), pe=None):
        e = ratio_series(close, pe if pe is not None else per_share(eps))
        s = ratio_series(close, per_share(rev)) if rev is not None else nan
        sh = share_change_days(days, e, nan, s)
        return find_jumps(days, e, set(adj), sh, set(stale))

    step = lambda d0, a, b: [a if d < d0 else b for d in days]   # noqa: E731

    # 1) 在 P 当日切换 → 不早跳,lag=0,新值 0.6
    r = classify(jumps(step(P, 0.5, 0.6), step(P, 5.0, 5.5)), S, P, None, W, [], cal)
    assert not r.early and r.lag_td == 0 and abs(r.x_new - 0.6) < 1e-9, r
    # 2) 报告期末后切换(P 之前)→ 早跳
    r = classify(jumps(step("2020-10-02", 0.5, 0.6), step("2020-10-02", 5.0, 5.5)), S, P, None, W, [], cal)
    assert r.early and r.switch_date == "2020-10-02" and r.lag_td < 0, r
    # 3) 股本变动(e 与 s 同比例)→ 剔除;真实切换在 P 后一日
    eps = [v * (1.3 if d >= "2020-10-21" else 1.0) for v, d in zip(step("2020-10-05", 0.5, 0.4), days)]
    rev = [v * (1.1 if d >= "2020-10-21" else 1.0) for v, d in zip(step("2020-10-05", 5.0, 4.0), days)]
    r = classify(jumps(eps, rev), S, P, None, W, [], cal)
    assert not r.early and r.excl["share"] == 1 and r.switch_date == "2020-10-21" and r.lag_td == 1, r
    # 4) 复权日跳变 → 剔除
    r = classify(jumps(step("2020-10-06", 0.5, 0.45), [5.0] * n, adj={"2020-10-06"}), S, P, None, W, [], cal)
    assert not r.early and r.excl["adj"] == 1, r
    # 5) 他期披露后 2 个交易日内的跳变 → 归因他期;10 个交易日以上 → 计早跳
    js = jumps(step("2020-10-08", 0.5, 0.7), [5.0] * n)
    r = classify(js, S, P, None, W, ["2020-10-06"], cal)
    assert not r.early and r.n_pre_explained == 1, r
    r = classify(js, S, P, None, W, ["2020-09-20"], cal)
    assert r.early, r
    #    他期事件在 P 之后不参与归因
    r = classify(js, S, "2020-10-07", None, W, ["2020-10-08"], cal)
    assert not r.early and r.switch_date == "2020-10-08", r
    # 6) 陈旧日:peTTM 与前一日相同而价格变动 → 当日与次日跳变;列入剔除后不早跳
    pe = per_share([0.5] * n)
    k = days.index("2020-10-09")
    pe[k] = pe[k - 1]
    assert len(jumps(None, [5.0] * n, pe=pe)) == 2
    r = classify(jumps(None, [5.0] * n, stale={"2020-10-09", cal.next_of("2020-10-09")}, pe=pe),
                 S, P, None, W, [], cal)
    assert not r.early, r
    #    陈旧同时波及 psTTM 时,同比例规则本身即可剔除
    ps = per_share([5.0] * n)
    ps[k] = ps[k - 1]
    e, s = ratio_series(close, pe), ratio_series(close, ps)
    assert share_change_days(days, e, nan, s) == {"2020-10-09", "2020-10-10"}
    # 7) 穿零(由盈转亏)可识别;回摆标记
    r = classify(jumps(step("2020-10-12", 0.01, -0.02), None), S, P, None, W, [], cal)
    assert r.early and r.switch_date == "2020-10-12", r
    eps = [0.5] * n
    eps[days.index("2020-10-13")] = 0.55
    js = jumps(eps, None)
    assert len(js) == 2 and all(j.revert for j in js) and js[0].date == "2020-10-13", js
    # 8) 0.1% 以下的变动不算跳变
    assert not jumps(step("2020-10-12", 0.5, 0.5004), None)
    # 9) 日历顺延与 Clopper–Pearson
    assert cal.roll("2020-10-29") == "2020-11-01" and cal.roll("2020-12-01") is None
    lo, hi = clopper_pearson(0, 200)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** (1 / 200))) < 1e-6, (lo, hi)
    lo, hi = clopper_pearson(2, 200)
    assert 0.0010 < lo < 0.0015 and 0.034 < hi < 0.037, (lo, hi)
    lo, hi = clopper_pearson(200, 200)
    assert hi == 1.0 and abs(lo - 0.025 ** (1 / 200)) < 1e-6, (lo, hi)
    print("selftest OK")


# ============================================================================ baostock(缓存 + 预算 + 退避)
class BudgetExhausted(RuntimeError):
    pass


class PersistentFailure(RuntimeError):
    pass


class Api:
    def __init__(self, max_calls: int, offline: bool):
        self.max_calls = max_calls
        self.offline = offline
        self.calls = 0            # 实际网络调用(含登录、重试)
        self.cache_hits = 0
        self.logged_in = False
        self.rate_limited = 0
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _spend(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExhausted(f"已达调用上限 {self.max_calls}")
        self.calls += 1

    def _quiet(self, fn, *a):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a)

    def login(self) -> None:
        import baostock as bs
        self._spend()
        lg = self._quiet(bs.login)
        if lg is None or lg.error_code != "0":
            raise PersistentFailure(f"login 失败: {getattr(lg, 'error_code', None)} {getattr(lg, 'error_msg', None)}")
        self.logged_in = True

    def logout(self) -> None:
        if self.logged_in:
            import baostock as bs
            with contextlib.suppress(Exception):
                self._quiet(bs.logout)
            self.logged_in = False

    @staticmethod
    def _key(name: str, args: tuple) -> str:
        raw = name + "|" + "|".join(str(a) for a in args)
        safe = raw.replace("|", "__").replace(".", "_")
        return os.path.join(CACHE_DIR, f"{safe}_{hashlib.sha1(raw.encode()).hexdigest()[:8]}.json")

    def query(self, name: str, *args) -> tuple[list[str], list[list[str]]]:
        path = self._key(name, args)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
            self.cache_hits += 1
            return obj["fields"], obj["rows"]
        if self.offline:
            raise BudgetExhausted("offline 模式且缓存未命中")
        import baostock as bs
        fn = getattr(bs, name)
        if not self.logged_in:
            self.login()
        last = ""
        for attempt in range(len(BACKOFF) + 1):
            if attempt:
                self.rate_limited += 1
                wait = BACKOFF[attempt - 1]
                _log(f"{name}{args} 失败({last}),退避 {wait:.0f}s 后重登重试")
                time.sleep(wait)
                self.logout()
                with contextlib.suppress(PersistentFailure):
                    self.login()
            self._spend()
            try:
                rs = self._quiet(fn, *args)
            except Exception as e:  # noqa: BLE001
                last = f"exception={e!r}"
                continue
            if rs is None:
                raise PersistentFailure(f"{name}{args} 返回 None(参数错误)")
            if rs.error_code != "0":
                last = f"error_code={rs.error_code} {rs.error_msg}"
                if rs.error_code.startswith("10004"):
                    raise PersistentFailure(f"{name}{args} 参数错误: {last}")
                continue
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rs.error_code != "0":
                last = f"翻页中断 error_code={rs.error_code}"
                continue
            fields = list(rs.fields)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"fields": fields, "rows": rows, "args": [str(a) for a in args],
                           "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")},
                          f, ensure_ascii=False)
            return fields, rows
        raise PersistentFailure(f"{name}{args} 退避耗尽: {last}")


# ============================================================================ 只读数据库
def connect_ro(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise SystemExit(f"数据库不存在: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = 1")
    return conn


def load_calendar(conn) -> Calendar:
    rows = conn.execute("SELECT date FROM index_daily WHERE code=? AND date<? ORDER BY date",
                        (CALENDAR_INDEX, HOLDOUT_CUTOFF)).fetchall()
    if not rows:
        raise SystemExit("index_daily 中无 sh.000001:无法建立交易日历(是否传错了 --db?)")
    return Calendar([r[0] for r in rows])


def stale_days(conn) -> list[dict]:
    """逐日统计 peTTM 与前一行完全相同的股票占比(宽口径与精确口径)。结果缓存。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cpath = os.path.join(CACHE_DIR, f"stale_days_{HOLDOUT_CUTOFF}.json")
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            return json.load(f)
    _log("统计全市场陈旧日(一次 SQL 聚合,约数分钟)……")
    sql = """
    SELECT date,
      SUM(ppe IS NOT NULL AND pe IS NOT NULL)                                   AS raw_base,
      SUM(ppe IS NOT NULL AND pe IS NOT NULL AND pe = ppe)                      AS raw_same,
      SUM(ppe IS NOT NULL AND pe IS NOT NULL AND tradestatus = 1
          AND pclose IS NOT NULL AND close <> pclose)                           AS ref_base,
      SUM(ppe IS NOT NULL AND pe IS NOT NULL AND tradestatus = 1
          AND pclose IS NOT NULL AND close <> pclose AND pe = ppe)              AS ref_same,
      SUM(ppb IS NOT NULL AND pb IS NOT NULL AND tradestatus = 1
          AND pclose IS NOT NULL AND close <> pclose)                           AS ref_base_pb,
      SUM(ppb IS NOT NULL AND pb IS NOT NULL AND tradestatus = 1
          AND pclose IS NOT NULL AND close <> pclose AND pb = ppb)              AS ref_same_pb
    FROM (
      SELECT date, tradestatus, close, peTTM AS pe, pbMRQ AS pb,
             LAG(close) OVER w AS pclose, LAG(peTTM) OVER w AS ppe, LAG(pbMRQ) OVER w AS ppb
      FROM daily WHERE date < ?
      WINDOW w AS (PARTITION BY code ORDER BY date)
    ) GROUP BY date ORDER BY date
    """
    t0 = time.time()
    out = []
    for d, rb, rs, fb, fs, fbp, fsp in conn.execute(sql, (HOLDOUT_CUTOFF,)):
        out.append({"date": d, "raw_base": rb or 0, "raw_same": rs or 0, "ref_base": fb or 0,
                    "ref_same": fs or 0, "ref_base_pb": fbp or 0, "ref_same_pb": fsp or 0})
    _log(f"陈旧日统计完成,{len(out)} 个日期,用时 {time.time() - t0:.0f}s")
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def _share(a: int, b: int) -> float:
    return a / b if b else float("nan")


# ============================================================================ 抽样
@dataclass
class Pair:
    stratum: str
    code: str
    year: int
    quarter: int
    delisted: int
    draw: int

    @property
    def stat(self) -> str:
        return f"{self.year}-{QEND[self.quarter]}"


def draw_sample(conn) -> list[Pair]:
    rng = np.random.default_rng(SEED)
    stocks = conn.execute("SELECT code, board, ipo_date, out_date FROM stocks ORDER BY code").fetchall()
    cells: dict[tuple[str, int], list[Pair]] = {}
    for label, prefix, board in STRATA:
        uni = [s for s in stocks if s[0].startswith(prefix) and s[1] == board and s[2]]
        for y in YEARS:
            qs = [q for q in (1, 2, 3, 4) if not (y == 2024 and q > LAST_QUARTER_2024)]
            if label == "科创板":
                qs = [q for q in qs if f"{y}-{QEND[q]}" >= STAR_FIRST_STAT]
            if not qs:
                continue
            picked: list[Pair] = []
            used_codes: set[str] = set()
            for k in range(N_PER_CELL[label]):
                want_delisted = k < DELISTED_PER_CELL
                for _ in range(MAX_REDRAW):
                    q = int(rng.choice(qs))
                    S = f"{y}-{QEND[q]}"
                    alive_to = _add_days(S, MIN_ALIVE_AFTER_STAT_DAYS)
                    pool = [s for s in uni if s[2] < S and (s[3] is None or s[3] > alive_to)
                            and s[0] not in used_codes]
                    if want_delisted:
                        dpool = [s for s in pool if s[3] is not None]
                        if dpool:
                            pool = dpool
                        else:
                            want_delisted = False
                    if not pool:
                        continue
                    s = pool[int(rng.integers(len(pool)))]
                    has = conn.execute(
                        "SELECT 1 FROM daily WHERE code=? AND date>=? AND date<=? AND date<? LIMIT 1",
                        (s[0], _add_days(S, -SAMPLE_ROW_CHECK_DAYS), S, HOLDOUT_CUTOFF)).fetchone()
                    if not has:
                        continue
                    picked.append(Pair(label, s[0], y, q, int(s[3] is not None), k))
                    used_codes.add(s[0])
                    break
            cells[(label, y)] = picked
    # 各格轮转:所有格的第 0 抽,再所有格的第 1 抽 ……
    order: list[Pair] = []
    kmax = max(N_PER_CELL.values())
    for k in range(kmax):
        for key in sorted(cells):
            if k < len(cells[key]):
                order.append(cells[key][k])
    return order


# ============================================================================ 单对处理
def _prev_quarter(y: int, q: int) -> tuple[int, int]:
    return (y, q - 1) if q > 1 else (y - 1, 4)


def _parse_profit(fields, rows, stat: str) -> dict | None:
    for r in rows:
        d = dict(zip(fields, r))
        if d.get("statDate") == stat:
            return d
    return None


def _num(v) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def load_stock_window(conn, code: str, start: str, end: str):
    rows = conn.execute(
        "SELECT date, close, peTTM, pbMRQ, psTTM FROM daily WHERE code=? AND date>=? AND date<=? AND date<? "
        "ORDER BY date", (code, start, end, HOLDOUT_CUTOFF)).fetchall()
    adj = conn.execute(
        "SELECT date, backAdjustFactor FROM adj_factor WHERE code=? AND date<? ORDER BY date",
        (code, HOLDOUT_CUTOFF)).fetchall()
    adj_days, prev = set(), None
    for d, f in adj:
        if prev is not None and f is not None and f != prev:
            adj_days.add(d)
        if f is not None:
            prev = f
    return rows, adj_days


def process_pair(p: Pair, api: Api, conn, cal: Calendar, stale_set: set[str]) -> dict:
    S = p.stat
    out = {"stratum": p.stratum, "code": p.code, "delisted": p.delisted, "year": p.year,
           "quarter": p.quarter, "stat_date": S, "status": "", "api_prev_fetched": 0}
    # --- 本期利润表 + 快报区间
    f, rows = api.query("query_profit_data", p.code, p.year, p.quarter)
    prof = _parse_profit(f, rows, S)
    ex_end = min(_add_days(S, EXPRESS_RANGE_DAYS[1]), _add_days(HOLDOUT_CUTOFF, -1))
    f2, rows2 = api.query("query_performance_express_report", p.code,
                          _add_days(S, EXPRESS_RANGE_DAYS[0]), ex_end)
    express = [dict(zip(f2, r)) for r in rows2]
    pub_profit = prof.get("pubDate") if prof else None
    eps_ttm = _num(prof.get("epsTTM")) if prof else None
    ex_same = sorted(e.get("performanceExpPubDate") for e in express
                     if e.get("performanceExpStatDate") == S and e.get("performanceExpPubDate"))
    ex_other = sorted({e.get("performanceExpPubDate") for e in express
                       if e.get("performanceExpStatDate") != S and e.get("performanceExpPubDate")})
    out.update({"profit_pub": pub_profit or "", "express_pub": ex_same[0] if ex_same else "",
                "eps_ttm": eps_ttm if eps_ttm is not None else ""})
    cands = [d for d in [pub_profit] + ex_same[:1] if d]
    if not cands:
        out["status"] = "无披露记录"
        return out
    first = min(cands)
    out["first_pub"] = first
    if first < S:
        out["status"] = "披露日早于报告期末(数据异常)"
        return out
    P = cal.roll(first)
    if P is None or first >= HOLDOUT_CUTOFF:
        out["status"] = "披露日落入保留段"
        return out
    out["first_pub_td"] = P
    P_profit = cal.roll(pub_profit) if pub_profit and pub_profit < HOLDOUT_CUTOFF else None
    w_end = cal.shift(max(P, P_profit or P), POST_TD)
    out["window_end"] = w_end
    out["window_censored"] = int(cal.pos(max(P, P_profit or P)) + POST_TD >= len(cal.d))
    # --- 行情窗
    base_from = cal.shift(S, -BASE_LOOKBACK_TD)
    rows_d, adj_days = load_stock_window(conn, p.code, base_from, w_end)
    dates = [r[0] for r in rows_d]
    close = [r[1] for r in rows_d]
    e = ratio_series(close, [r[2] for r in rows_d])
    b = ratio_series(close, [r[3] for r in rows_d])
    s = ratio_series(close, [r[4] for r in rows_d])
    stale_for = stale_set if EXCLUDE_STALE_DAYS else set()
    share_days = share_change_days(dates, e, b, s)
    res = {}
    pre_td = [d for d in cal.d if S <= d < P]
    for name, x in (("eps", e), ("bps", b)):
        has_base = any(d < S and _finite(v) for d, v in zip(dates, x))
        has_cov = any(d >= P and _finite(v) for d, v in zip(dates, x))
        pre_rows = {d for d, v in zip(dates, x) if S <= d < P and _finite(v)}
        pre_cov = len(pre_rows) / len(pre_td) if pre_td else 1.0
        out[f"{name}_pre_coverage"] = round(pre_cov, 3)
        res[name] = (has_base, has_cov, pre_cov, find_jumps(dates, x, adj_days, share_days, stale_for))
    out["s_missing_share"] = round(sum(not _finite(v) for v in s) / len(s), 4) if s else ""
    other = [cal.roll(d) for d in ex_other if d < HOLDOUT_CUTOFF]

    def run(other_events):
        return {k: classify(v[3], S, P, P_profit, w_end, other_events, cal) for k, v in res.items()}

    cls = run(other)
    need_prev = p.quarter == 1 or any(c.n_pre_unexplained for c in cls.values())
    if need_prev:
        py, pq = _prev_quarter(p.year, p.quarter)
        f3, rows3 = api.query("query_profit_data", p.code, py, pq)
        out["api_prev_fetched"] = 1
        pprof = _parse_profit(f3, rows3, f"{py}-{QEND[pq]}")
        if pprof and pprof.get("pubDate") and pprof["pubDate"] < HOLDOUT_CUTOFF:
            other = other + [cal.roll(pprof["pubDate"])]
            out["prev_pub"] = pprof["pubDate"]
            out["prev_eps_ttm"] = _num(pprof.get("epsTTM")) if _num(pprof.get("epsTTM")) is not None else ""
        cls = run(other)
    out["other_events"] = ";".join(sorted({d for d in other if d}))
    ok_any = False
    for name, c in cls.items():
        has_base, has_cov, pre_cov, _ = res[name]
        pre = f"{name}_"
        if not (has_base and has_cov and pre_cov >= MIN_PRE_COVERAGE):
            out[pre + "status"] = ("无基准行" if not has_base else "披露日后无有效行" if not has_cov
                                   else "[S,P) 行覆盖不足")
            continue
        ok_any = True
        out[pre + "status"] = "可判定"
        out[pre + "early"] = int(c.early)
        out[pre + "switch"] = c.switch_date or ""
        out[pre + "lag_td"] = c.lag_td if c.lag_td is not None else ""
        out[pre + "lag_profit_td"] = c.lag_profit_td if c.lag_profit_td is not None else ""
        out[pre + "new"] = round(c.x_new, 6) if c.x_new is not None else ""
        out[pre + "n_pre_unexpl"] = c.n_pre_unexplained
        out[pre + "n_pre_expl"] = c.n_pre_explained
        out[pre + "pre_dates"] = c.pre_dates
        out[pre + "early_revert"] = int(c.early_revert) if c.early else ""
        out[pre + "excl_adj"] = c.excl.get("adj", 0)
        out[pre + "excl_share"] = c.excl.get("share", 0)
        out[pre + "excl_stale"] = c.excl.get("stale", 0)
        # ---- 事后诊断(不参与预注册判定)
        out[pre + "early_nonrevert"] = int(any(not j.revert for j in c.pre_unexpl))
        out[pre + "early_tol1pct"] = int(any(j.rel > DIAG_TOL_ALT for j in c.pre_unexpl))
        out[pre + "excl_pre"] = len(c.pre_excluded)
        if name == "eps" and eps_ttm not in (None, 0.0):
            def _hit(js):
                return int(any(abs(j.x_cur - eps_ttm) / abs(eps_ttm) <= EPS_MATCH_TOL for j in js))
            out["eps_early_match_q"] = _hit(c.pre_unexpl)
            out["eps_excl_pre_match_q"] = _hit(c.pre_excluded)
    if eps_ttm not in (None, 0.0) and cls["eps"].x_new is not None:
        out["eps_relerr"] = round(abs(cls["eps"].x_new - eps_ttm) / abs(eps_ttm), 6)
    prev_eps = out.get("prev_eps_ttm")
    if isinstance(prev_eps, float) and prev_eps != 0.0 and cls["eps"].x_new is not None:
        out["eps_relerr_prev"] = round(abs(cls["eps"].x_new - prev_eps) / abs(prev_eps), 6)
    out["status"] = "可判定" if ok_any else "行情不足"
    return out


# ============================================================================ 汇总与输出
PAIR_COLS = ["stratum", "code", "delisted", "year", "quarter", "stat_date", "status",
             "profit_pub", "express_pub", "first_pub", "first_pub_td", "window_end", "window_censored",
             "prev_pub", "other_events", "api_prev_fetched",
             "eps_status", "eps_pre_coverage", "eps_early", "eps_switch", "eps_lag_td", "eps_lag_profit_td", "eps_new",
             "eps_ttm", "eps_relerr", "prev_eps_ttm", "eps_relerr_prev",
             "eps_n_pre_unexpl", "eps_n_pre_expl", "eps_pre_dates", "eps_early_revert",
             "eps_excl_adj", "eps_excl_share", "eps_excl_stale",
             "bps_status", "bps_pre_coverage", "bps_early", "bps_switch", "bps_lag_td", "bps_lag_profit_td", "bps_new",
             "bps_n_pre_unexpl", "bps_n_pre_expl", "bps_pre_dates", "bps_early_revert",
             "bps_excl_adj", "bps_excl_share", "bps_excl_stale", "s_missing_share",
             "eps_early_nonrevert", "eps_early_tol1pct", "eps_early_match_q", "eps_excl_pre",
             "eps_excl_pre_match_q", "bps_early_nonrevert", "bps_early_tol1pct", "bps_excl_pre"]


def write_tsv(path: str, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join("" if r.get(c) is None else str(r.get(c, "")) for c in cols) + "\n")


def _quant(vals: list[float], qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> list[float]:
    if not vals:
        return [float("nan")] * len(qs)
    a = np.asarray(vals, dtype=float)
    return [float(np.quantile(a, q)) for q in qs]


LAG_BINS = [(-10 ** 6, -21, "≤−21"), (-20, -6, "−20~−6"), (-5, -2, "−5~−2"), (-1, -1, "−1"),
            (0, 0, "0"), (1, 1, "+1"), (2, 2, "+2"), (3, 5, "+3~+5"), (6, 10, "+6~+10"),
            (11, 20, "+11~+20"), (21, 10 ** 6, "≥+21")]


def summarize(results: list[dict], sampled: int, api: Api | None, stale: list[dict], stop_reason: str):
    summary: list[dict] = []

    def add(k, v, note=""):
        summary.append({"item": k, "value": v, "note": note})

    status = Counter(r.get("status", "未取") for r in results)
    add("抽样对数", sampled, f"种子 {SEED};各格轮转处理")
    add("已处理对数", len(results), stop_reason or "全部处理")
    for k, v in sorted(status.items()):
        add(f"状态:{k}", v)
    if api is not None:
        add("baostock 网络调用(含登录/重试)", api.calls, f"上限 {api.max_calls}")
        add("缓存命中", api.cache_hits)
        add("退避重试次数", api.rate_limited)
    verdicts = {}
    for name, label in (("eps", "peTTM"), ("bps", "pbMRQ")):
        ok = [r for r in results if r.get(f"{name}_status") == "可判定"]
        n = len(ok)
        k = sum(int(r.get(f"{name}_early", 0) or 0) for r in ok)
        lo, hi = clopper_pearson(k, n) if n else (float("nan"), float("nan"))
        ratio = k / n if n else float("nan")
        verdict = "通过" if n and ratio <= EARLY_RATIO_MAX else ("不通过" if n else "无可判定样本")
        verdicts[label] = verdict
        add(f"{label} 可判定对数", n)
        add(f"{label} 早跳对数", k)
        add(f"{label} 早跳比例", f"{ratio:.4f}" if n else "", f"95% CI [{lo:.4f}, {hi:.4f}](Clopper–Pearson)")
        add(f"{label} 判定(早跳比例 ≤ {EARLY_RATIO_MAX:.0%})", verdict)
        early = [r for r in ok if int(r.get(f"{name}_early", 0) or 0)]
        add(f"{label} 早跳中回摆(毛刺)数", sum(int(r.get(f"{name}_early_revert") or 0) for r in early))
        if name == "eps":
            m = [r for r in early if isinstance(r.get("eps_relerr"), float) and r["eps_relerr"] <= EPS_MATCH_TOL]
            add("peTTM 早跳中新值与本期 epsTTM 吻合数", len(m), f"相对误差 ≤ {EPS_MATCH_TOL:.0%}")
        found = [r for r in ok if r.get(f"{name}_switch")]
        add(f"{label} 窗口内找到切换日的对数", len(found))
        add(f"{label} 窗口内未见跳变的对数", n - len(found), "新期数值变化低于阈值或数据缺失")
        for flag in ("adj", "share", "stale"):
            add(f"{label} 窗口内被剔除跳变:{flag}", sum(int(r.get(f"{name}_excl_{flag}") or 0) for r in ok))
        add(f"{label} [S,P) 内归因他期披露的跳变数", sum(int(r.get(f"{name}_n_pre_expl") or 0) for r in ok))
        for label2, _, _ in STRATA:
            sub = [r for r in ok if r["stratum"] == label2]
            kk = sum(int(r.get(f"{name}_early", 0) or 0) for r in sub)
            add(f"{label} 早跳/可判定:{label2}", f"{kk}/{len(sub)}")
        sub = [r for r in ok if r["delisted"] == 1]
        add(f"{label} 早跳/可判定:已退市股", f"{sum(int(r.get(f'{name}_early', 0) or 0) for r in sub)}/{len(sub)}")
    # 事后敏感性(运行后追加;不改变上面的预注册判定)
    for name, label in (("eps", "peTTM"), ("bps", "pbMRQ")):
        ok = [r for r in results if r.get(f"{name}_status") == "可判定"]
        n = len(ok)
        if not n:
            continue
        variants = [("去掉回摆跳变(单日毛刺)后", f"{name}_early_nonrevert"),
                    (f"跳变阈值放宽到 {DIAG_TOL_ALT:.0%}", f"{name}_early_tol1pct")]
        if name == "eps":
            variants.append(("内容口径:早跳新值与本期 epsTTM 吻合", "eps_early_match_q"))
        for lab, col in variants:
            k = sum(int(r.get(col) or 0) for r in ok)
            lo, hi = clopper_pearson(k, n)
            add(f"[事后敏感性] {label} {lab}", f"{k}/{n} = {k / n:.4f}", f"95% CI [{lo:.4f}, {hi:.4f}];不改变判定")
        add(f"[事后诊断] {label} [S,P) 内被剔除的跳变数", sum(int(r.get(f"{name}_excl_pre") or 0) for r in ok))
        if name == "eps":
            add("[事后诊断] peTTM [S,P) 内被剔除跳变中新值与本期 epsTTM 吻合的对数",
                sum(int(r.get("eps_excl_pre_match_q") or 0) for r in ok), "检查剔除规则是否掩盖了真实早切换")
        exf = [r for r in ok if r.get("express_pub") and (not r.get("profit_pub") or r["express_pub"] < r["profit_pub"])]
        lag0 = sum(1 for r in exf if r.get(f"{name}_lag_td") in (0, 1))
        lagp0 = sum(1 for r in exf if r.get(f"{name}_lag_profit_td") == 0)
        add(f"[事后诊断] {label} 快报先于正式报告的对:切换在快报日(0/+1 交易日)/ 在正式报告日",
            f"{lag0} / {lagp0}", f"共 {len(exf)} 对")
    # 相对误差
    rel = [r["eps_relerr"] for r in results
           if r.get("eps_status") == "可判定" and isinstance(r.get("eps_relerr"), float)]
    relerr_rows = []
    qs = _quant(rel)
    relerr_rows.append({"series": "隐含EPS(切换日新值) vs 本期 epsTTM", "n": len(rel),
                        "p10": qs[0], "p25": qs[1], "p50": qs[2], "p75": qs[3], "p90": qs[4],
                        "share_le_0.5pct": _share(sum(v <= 0.005 for v in rel), len(rel)),
                        "share_le_2pct": _share(sum(v <= 0.02 for v in rel), len(rel)),
                        "share_le_5pct": _share(sum(v <= 0.05 for v in rel), len(rel))})
    relp = [r["eps_relerr_prev"] for r in results
            if r.get("eps_status") == "可判定" and isinstance(r.get("eps_relerr_prev"), float)]
    qs = _quant(relp)
    relerr_rows.append({"series": "隐含EPS(切换日新值) vs 上期 epsTTM(仅取了上期的对)", "n": len(relp),
                        "p10": qs[0], "p25": qs[1], "p50": qs[2], "p75": qs[3], "p90": qs[4],
                        "share_le_0.5pct": _share(sum(v <= 0.005 for v in relp), len(relp)),
                        "share_le_2pct": _share(sum(v <= 0.02 for v in relp), len(relp)),
                        "share_le_5pct": _share(sum(v <= 0.05 for v in relp), len(relp))})
    # 滞后分布
    lag_rows = []
    for name, label in (("eps", "peTTM"), ("bps", "pbMRQ")):
        ok = [r for r in results if r.get(f"{name}_status") == "可判定"]
        for ref, col in (("首次披露日", f"{name}_lag_td"), ("正式报告 pubDate", f"{name}_lag_profit_td")):
            vals = [r[col] for r in ok if isinstance(r.get(col), int)]
            for lo, hi, lab in LAG_BINS:
                c = sum(lo <= v <= hi for v in vals)
                lag_rows.append({"field": label, "reference": ref, "bin_td": lab, "count": c,
                                 "share": round(_share(c, len(vals)), 4) if vals else ""})
            lag_rows.append({"field": label, "reference": ref, "bin_td": "未见跳变",
                             "count": len(ok) - len(vals) if ref == "首次披露日" else "",
                             "share": ""})
    # 陈旧日
    stale_rows = []
    for d in stale:
        raw = _share(d["raw_same"], d["raw_base"])
        ref = _share(d["ref_same"], d["ref_base"])
        refpb = _share(d["ref_same_pb"], d["ref_base_pb"])
        if (raw > STALE_SHARE) or (ref > STALE_SHARE) or d["date"] in CHECK_DATES:
            stale_rows.append({"date": d["date"], "raw_share_pe_same": round(raw, 4), "raw_base": d["raw_base"],
                               "ref_share_pe_same": round(ref, 4) if ref == ref else "",
                               "ref_base": d["ref_base"],
                               "ref_share_pb_same": round(refpb, 4) if refpb == refpb else "",
                               "stale_raw": int(raw > STALE_SHARE), "stale_ref": int(ref > STALE_SHARE),
                               "in_check_list": int(d["date"] in CHECK_DATES)})
    add("陈旧日(宽口径 > 50%)数", sum(r["stale_raw"] for r in stale_rows))
    add("陈旧日(精确口径 > 50%)数", sum(r["stale_ref"] for r in stale_rows),
        ";".join(r["date"] for r in stale_rows if r["stale_ref"]))
    return summary, relerr_rows, lag_rows, stale_rows, verdicts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", help="主仓库 data/ashare.db 的绝对路径(只读打开)")
    ap.add_argument("--selftest", action="store_true", help="只跑合成数据自测")
    ap.add_argument("--offline", action="store_true", help="只用缓存,不联网")
    ap.add_argument("--max-calls", type=int, default=MAX_API_CALLS)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 对(调试)")
    ap.add_argument("--stale-only", action="store_true", help="只统计陈旧日")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.db:
        raise SystemExit("须显式传 --db(worktree 内的 data/ashare.db 为空库)")
    os.makedirs(REPORT_DIR, exist_ok=True)
    conn = connect_ro(a.db)
    cal = load_calendar(conn)
    stale = stale_days(conn)
    stale_ref = [d["date"] for d in stale if _share(d["ref_same"], d["ref_base"]) > STALE_SHARE]
    stale_set = set(stale_ref) | {cal.next_of(d) for d in stale_ref if cal.next_of(d)}
    if a.stale_only:
        _, _, _, stale_rows, _ = summarize([], 0, None, stale, "")
        write_tsv(os.path.join(REPORT_DIR, "pit_audit_stale_days.tsv"), list(stale_rows[0].keys()) if stale_rows
                  else ["date"], stale_rows)
        return
    sample = draw_sample(conn)
    if a.limit:
        sample = sample[: a.limit]
    _log(f"抽样 {len(sample)} 对;陈旧日(精确口径)剔除集 {len(stale_set)} 天")
    api = Api(a.max_calls, a.offline)
    results: list[dict] = []
    stop_reason = ""
    try:
        for i, p in enumerate(sample, 1):
            try:
                results.append(process_pair(p, api, conn, cal, stale_set))
            except BudgetExhausted as e:
                stop_reason = f"停止于第 {i} 对:{e}"
                break
            except PersistentFailure as e:
                msg = str(e)
                if "参数错误" in msg or "返回 None" in msg:
                    results.append({"stratum": p.stratum, "code": p.code, "delisted": p.delisted,
                                    "year": p.year, "quarter": p.quarter, "stat_date": p.stat,
                                    "status": "接口参数错误"})
                    continue
                stop_reason = f"停止于第 {i} 对:持续失败/限流 {msg}"
                break
            if i % 20 == 0:
                _log(f"{i}/{len(sample)} 对;网络调用 {api.calls},缓存命中 {api.cache_hits}")
    finally:
        api.logout()
    summary, relerr_rows, lag_rows, stale_rows, verdicts = summarize(results, len(sample), api, stale, stop_reason)
    write_tsv(os.path.join(REPORT_DIR, "pit_audit_pairs.tsv"), PAIR_COLS, results)
    write_tsv(os.path.join(REPORT_DIR, "pit_audit_summary.tsv"), ["item", "value", "note"], summary)
    write_tsv(os.path.join(REPORT_DIR, "pit_audit_relerr.tsv"), list(relerr_rows[0].keys()), relerr_rows)
    write_tsv(os.path.join(REPORT_DIR, "pit_audit_lag.tsv"), ["field", "reference", "bin_td", "count", "share"],
              lag_rows)
    write_tsv(os.path.join(REPORT_DIR, "pit_audit_stale_days.tsv"),
              ["date", "raw_share_pe_same", "raw_base", "ref_share_pe_same", "ref_base", "ref_share_pb_same",
               "stale_raw", "stale_ref", "in_check_list"], stale_rows)
    for r in summary:
        print(f"{r['item']}\t{r['value']}\t{r['note']}")
    _log(f"判定:{verdicts}")


if __name__ == "__main__":
    main()
