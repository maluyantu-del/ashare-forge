#!/usr/bin/env python
"""批次诊断:回答任务卡规定、但台账证据包不直接存的问题。

只读(不写台账、不改门槛),输出 TSV 块,可原样粘进 Excel / 结论文档。
诊断不是判据:裁定仍以 pipeline 的零校准 + 分层门槛为准,这里给的是"为什么"。

  A  task01 H3   —— 不可买的一字涨停若可买,超额会高多少(衰减是否过半)
  B  task02      —— 反转:毛/净收益与成本拆分(降换手能否救)、IC 分段(2017 前后)
  C  task03      —— 特质波动 vs 总波动 vs 纯低换手:逐日 IC 配对差(含 Newey-West t)
  D  task04      —— 非流动性市值中性化前后:IC、超额、与市值的截面秩相关
  E  全部        —— 平静期 / 危机期分层的超额夏普与天数

用法: .venv/bin/python scripts/batch_diagnostics.py [--start 2015-01-01] [--end 2024-08-31]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ashare_forge import pipeline  # noqa: E402
from ashare_forge.config import DB_PATH  # noqa: E402
from ashare_forge.data import db  # noqa: E402
from ashare_forge.engine import backtest, gates, metrics  # noqa: E402

PPY = 243


def _log(msg: str) -> None:
    print(f"# [{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _ann_sharpe(x: pd.Series) -> float:
    x = x.dropna()
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(PPY)) if len(x) > 2 and sd > 0 else float("nan")


def _nw_t(x: pd.Series, lags: int = 5) -> float:
    """均值的 Newey-West t 统计量(Bartlett 核)。逐日 IC 有自相关,朴素 t 会高估显著性。"""
    x = x.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 20:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, lags + 1):
        s += 2.0 * (1.0 - k / (lags + 1.0)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else float("nan")


def _tsv(rows: list[dict], title: str) -> None:
    print(f"\n### {title}")
    if not rows:
        print("(无)")
        return
    print(pd.DataFrame(rows).to_csv(sep="\t", index=False, float_format="%.4f"), end="")


def _forgone_share(factor, masks, top_pct) -> float:
    """目标想买但因 can_buy=False 买不进的权重,占目标总权重的比例(日均)。"""
    tgt = backtest.to_weights(factor, masks["in_universe"], top_pct=top_pct).shift(1).fillna(0.0)
    blocked = tgt.where(~masks["can_buy"].reindex_like(tgt).fillna(False).astype(bool), 0.0)
    tot = tgt.sum(axis=1)
    share = (blocked.sum(axis=1) / tot.where(tot > 0))
    return float(share.mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default=pipeline.DEFAULT_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--top-pct", type=float, default=0.1)
    ap.add_argument("--db", default=DB_PATH)
    a = ap.parse_args(argv)

    specs = {s["name"]: s for s in pipeline.discover_specs(include_optional=True)}
    conn = db.connect(a.db)
    _log("加载面板(保留段由数据层强制截断)")
    ctx = pipeline.load_context(conn, start=a.start, end=a.end)
    panel, masks, bench = ctx["panel"], ctx["masks"], ctx["bench"]
    print(f"# 样本 {panel['close'].index.min().date()} → {panel['close'].index.max().date()} | "
          f"{panel['close'].shape[1]} 只 | 多头 {a.top_pct:.0%} | 基准 {ctx.get('bench_code')}")

    cache: dict[str, dict] = {}

    def run(name: str, m=None, **kw) -> dict:
        key = name if m is None and not kw else None
        if key and key in cache:
            return cache[key]
        f = specs[name]["build_factor"](panel)
        bt = backtest.run_backtest(f, panel, m or masks, bench=bench, top_pct=a.top_pct,
                                   bench_open=ctx.get("bench_open"), bench_tr_ret=ctx.get("bench_tr_ret"), **kw)
        bt["factor"] = f
        if key:
            cache[key] = bt
        return bt

    # ---------------- A · task01 H3 ----------------
    _log("A 横盘突破:涨停不可买的代价")
    rows = []
    if "accumulation_breakout" in specs:
        real = run("accumulation_breakout")
        relaxed_masks = {**masks, "can_buy": masks["in_universe"]}   # 仅去掉"一字涨停不可买"
        ideal = run("accumulation_breakout", m=relaxed_masks)
        for tag, bt in (("实际(一字涨停不可买)", real), ("假设涨停可买", ideal)):
            st = bt["stats"]
            rows.append({"口径": tag, "超额年化": st.get("excess_ann"), "超额夏普": st.get("excess_sharpe"),
                         "年化收益": st["ann_return"], "日均换手": st["avg_turnover"],
                         "日均现金": st["avg_cash"]})
        ea, eb = real["stats"].get("excess_ann"), ideal["stats"].get("excess_ann")
        rows.append({"口径": "目标权重被涨停挡住占比", "超额年化": _forgone_share(real["factor"], masks, a.top_pct)})
        if ea is not None and eb is not None:
            rows.append({"口径": "H3 判读", "超额年化": ea - eb,
                         "超额夏普": float("nan"),
                         "年化收益": float("nan")})
    _tsv(rows, "A · task01 H3:一字涨停不可买的代价(末行=实际−假设可买)")

    # ---------------- B · task02 ----------------
    _log("B 短期反转:成本拆分与 IC 分段")
    rows, seg = [], []
    for nm in ("short_reversal", "short_reversal_skip0", "short_reversal_hold5", "short_reversal_hold20"):
        if nm not in specs:
            continue
        bt = run(nm)
        st = bt["stats"]
        gross_ann = float(bt["gross_ret"].mean()) * PPY
        rows.append({"因子": nm, "IC均值": st.get("ic_mean"), "毛年化": gross_ann,
                     "成本年化": st["ann_cost"], "净年化": st["ann_return"],
                     "超额年化": st.get("excess_ann"), "超额夏普": st.get("excess_sharpe"),
                     "日均换手": st["avg_turnover"],
                     "买不进占比": _forgone_share(bt["factor"], masks, a.top_pct)})
        ic = bt["ic"].dropna()
        pre, post = ic[ic.index < "2017-01-01"], ic[ic.index >= "2017-01-01"]
        seg.append({"因子": nm, "IC_2017前": pre.mean(), "天数_前": len(pre),
                    "IC_2017后": post.mean(), "天数_后": len(post),
                    "后/前": post.mean() / pre.mean() if pre.mean() else float("nan"),
                    "H3(后≥前/2)": bool(post.mean() >= 0.5 * pre.mean()) if len(pre) and len(post) else None})
    _tsv(rows, "B1 · task02 反转:毛收益 − 成本 = 净收益(降换手能否救)")
    _tsv(seg, "B2 · task02 H3:IC 以 2017-01-01 分段")
    if "short_reversal" in cache:
        ic = cache["short_reversal"]["ic"].dropna()
        yr = ic.groupby(ic.index.year).agg(["mean", "count"]).reset_index()
        yr.columns = ["年份", "IC均值", "天数"]
        _tsv(yr.to_dict(orient="records"), "B3 · short_reversal 逐年 IC")

    # ---------------- C · task03 ----------------
    _log("C 特质波动三方对照")
    trio = [n for n in ("idio_volatility", "idio_volatility_total", "low_turnover_control") if n in specs]
    rows = []
    for nm in trio:
        st = run(nm)["stats"]
        rows.append({"因子": nm, "IC均值": st.get("ic_mean"), "ICIR年化": st.get("icir_ann"),
                     "单调性": st.get("monotonicity"), "超额年化": st.get("excess_ann"),
                     "超额夏普": st.get("excess_sharpe"), "日均换手": st["avg_turnover"]})
    _tsv(rows, "C1 · task03 三方并排")
    pairs = []
    if len(trio) == 3:
        base = cache["idio_volatility"]
        for other in trio[1:]:
            d = (base["ic"] - cache[other]["ic"]).dropna()
            fa, fb = base["factor"], cache[other]["factor"]
            common = fa.index.intersection(fb.index)[::20]           # 每 20 日取一截面,省时
            rho = [pd.concat([fa.loc[t], fb.loc[t]], axis=1).dropna().corr(method="spearman").iloc[0, 1]
                   for t in common]
            ex = (base["excess_ret"] - cache[other]["excess_ret"]).dropna()
            pairs.append({"对比": f"idio − {other}", "IC差均值": d.mean(), "IC差NW_t": _nw_t(d),
                          "超额差年化": ex.mean() * PPY, "超额差NW_t": _nw_t(ex),
                          "截面秩相关均值": float(np.nanmean(rho))})
    _tsv(pairs, "C2 · task03 配对差(|NW t|<2 视为无法区分)")

    # ---------------- D · task04 ----------------
    _log("D 非流动性市值中性化")
    rows = []
    fmc = panel.get("float_mcap")
    for nm in ("illiquidity_amihud", "illiquidity_amihud_size_neutral"):
        if nm not in specs:
            continue
        bt = run(nm)
        st = bt["stats"]
        rho = float("nan")
        if fmc is not None:
            f = bt["factor"]
            idx = f.index.intersection(fmc.index)[::20]
            rho = float(np.nanmean([pd.concat([f.loc[t], fmc.loc[t]], axis=1).dropna()
                                    .corr(method="spearman").iloc[0, 1] for t in idx]))
        rows.append({"因子": nm, "IC均值": st.get("ic_mean"), "超额年化": st.get("excess_ann"),
                     "超额夏普": st.get("excess_sharpe"), "与流通市值秩相关": rho,
                     "日均换手": st["avg_turnover"]})
    _tsv(rows, "D · task04 市值中性化前后")

    # ---------------- E · 分层 ----------------
    _log("E 平静期/危机期分层")
    rows = []
    for nm in specs:
        bt = run(nm)
        ex = bt.get("excess_ret")
        if ex is None:
            continue
        cm = gates.crisis_mask(pd.DatetimeIndex(ex.index)).reindex(ex.index).fillna(False)
        calm, cri = ex[~cm], ex[cm]
        rows.append({"因子": nm, "平静_天数": int(calm.dropna().size), "平静_超额年化": calm.mean() * PPY,
                     "平静_超额夏普": _ann_sharpe(calm), "危机_天数": int(cri.dropna().size),
                     "危机_超额年化": cri.mean() * PPY, "危机_超额夏普": _ann_sharpe(cri)})
    _tsv(rows, "E · 平静期 / 危机期分层(危机窗口为合同事前写定的四段)")
    conn.close()
    _log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
