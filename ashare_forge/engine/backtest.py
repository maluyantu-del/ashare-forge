"""T+1 截面回测:合同的 A 股交易规则由代码执行,不靠"注意事项"。

它修的坑(每条都是A股回测最常见的虚高来源):
1. **T+1 执行**:t 日收盘算因子 → t+1 开盘调仓 → 收益从 t+1 开盘起算。
   用"t 日收盘到 t+1 收盘"是偷看(t 日收盘时你还没建上仓)。
2. **涨跌停不可成交**:一字涨停买不进、一字跌停卖不掉。忽略它会让"追涨停"类因子
   凭空盈利。本模块对买不进的标的保持原权重、卖不掉的强制留仓。
3. **停牌**:停牌股不能调仓,权重顺延。
4. **成本按合同**:印花税(卖出单边)+ 佣金(双边)+ 冲击。
5. **前瞻收益用后复权价**:原始价在除权日会有假跳空。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract import load_contract, stamp_tax_series
from . import metrics


def forward_returns(adj_close: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """t 行 = 从 t 到 t+horizon 的收益(后复权)。末尾 horizon 行为 NaN。"""
    a = adj_close.astype("float64")
    return a.shift(-horizon) / a - 1.0


def open_to_open_returns(adj_open: pd.DataFrame) -> pd.DataFrame:
    """t 行 = 从 t 开盘到 t+1 开盘的收益。持仓在 t 已建立时吃到的就是这段。"""
    a = adj_open.astype("float64")
    return a.shift(-1) / a - 1.0


def tradable_returns(adj_open: pd.DataFrame) -> pd.DataFrame:
    """**t 日决策真正能吃到的收益** = 从 t+1 开盘到 t+2 开盘。

    这是本引擎最容易被忽略的口径问题:IC 若用"今收→明收"衡量,而 T+1 下你明天开盘
    才建得上仓,那段涨跌已经错过——会出现 "IC 0.88 但夏普 -3.6" 的荒谬组合
    (本模块自测实测)。所有截面指标一律对齐这段可交易收益。
    """
    return open_to_open_returns(adj_open).shift(-1)


def to_weights(factor: pd.DataFrame, in_universe: pd.DataFrame,
               top_pct: float = 0.10, min_names: int = 5) -> pd.DataFrame:
    """每日取因子最大的 top_pct 只(仅限在池标的),等权。返回目标权重(行和为1或0)。"""
    f = factor.astype("float64").where(
        in_universe.reindex(index=factor.index, columns=factor.columns).fillna(False).astype(bool))
    w = pd.DataFrame(0.0, index=f.index, columns=f.columns)
    prev = None
    for d in f.index:
        row = f.loc[d].dropna()
        k = max(min_names, int(np.ceil(len(row) * top_pct))) if len(row) else 0
        if row.empty or len(row) < k:
            # 审查发现(P1):原实现当日可选不足时 continue → 权重全 0 → 次日再全额重建,
            # 凭空制造一次全清仓+全建仓的换手(实测稀疏因子 5.1% 的交易日如此)。
            # 现改为维持前一日目标,可选不足本身不是调仓理由。
            if prev is not None:
                w.loc[d] = prev
            continue
        pick = row.nlargest(k).index
        w.loc[d, pick] = 1.0 / len(pick)
        prev = w.loc[d].copy()
    return w


def _apply_tradability(target: pd.DataFrame, can_buy: pd.DataFrame,
                       can_sell: pd.DataFrame, ret: pd.DataFrame | None = None,
                       delisted: pd.DataFrame | None = None, haircut: float = 1.0,
                       ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """把"目标权重"变成"实际可达到的权重",并显式记现金。

    审查发现(P1):原实现只做"能卖就卖、能买就买",权重从不归一化——
      · 卖不掉时旧仓仍占位而新买照买 → 实测单日最高 1.61 倍敞口(凭空加杠杆)
      · 买不进时资金既不留旧仓也不按比例补足 → 凭空空仓
    两者方向相反,不是保守估计;实测使 illiquidity 的超额 +3.53% 中有一半来自此。

    现模型:总资本恒为 1。先执行允许的卖出释放现金,再用**可用现金**上限内的资金买入;
    现金不足时所有买入按比例缩减。返回 (权重表, 现金序列, 交易前权重表),权重行和 + 现金 ≡ 1。

    审查发现(P2,成本低估):原实现把"昨日目标权重"当作今日交易前持仓,等于假设隔夜
    价格变动不改变权重——等权组合每天回到等权所需的"漂移再平衡"交易被漏记,
    成本系统性偏低。现传入 ret(t 行 = t 开盘→t+1 开盘收益)时,交易前权重按实际涨跌
    漂移:pre[t+1] = held[t]·(1+r_t) / (held[t]·(1+r_t) + cash[t])。卖不掉的股票
    保持漂移后的权重,而不是冻结在旧目标上。ret=None 时退化为旧口径(仅供对照)。

    合同 v2(退市):delisted[t, j]=True(已过摘牌日)而仍持有 j 时,于 t 日开盘前强制清算,
    按 haircut 比例计损失(写入第 4 个返回值 loss,单位=组合净值比例),其余头寸按剩余净值
    重新归一。v1 把这类头寸永久冻结在最后价格上(等价零损失,最乐观)。
    返回 (权重表, 现金序列, 交易前权重表, 退市清算损失序列)。
    """
    cb = can_buy.reindex_like(target).fillna(False).to_numpy(dtype=bool)
    cs = can_sell.reindex_like(target).fillna(False).to_numpy(dtype=bool)
    tgt = target.to_numpy(dtype="float64")
    n_rows, n_cols = tgt.shape
    held = np.zeros_like(tgt)
    pre = np.zeros_like(tgt)
    cash = np.zeros(n_rows, dtype="float64")
    prev = np.zeros(n_cols, dtype="float64")
    loss = np.zeros(n_rows, dtype="float64")
    dl_arr = (delisted.reindex_like(target).fillna(False).to_numpy(dtype=bool)
              if delisted is not None else None)
    if ret is not None:
        # 停牌/缺失收益按 0 计(权重随组合其余部分被动稀释,不虚构涨跌)
        r_arr = np.nan_to_num(ret.reindex_like(target).to_numpy(dtype="float64"),
                              nan=0.0, posinf=0.0, neginf=0.0)
    else:
        r_arr = None

    for i in range(n_rows):
        want, cur = tgt[i], prev
        if dl_arr is not None:
            gone = dl_arr[i] & (cur > 0)
            if gone.any():
                written_off = float(cur[gone].sum()) * float(haircut)
                cur = cur.copy()
                cur[gone] = 0.0                      # 头寸按 (1-haircut) 变现为现金,损失 written_off
                remain = 1.0 - written_off
                loss[i] = written_off
                cur = cur / remain if remain > 1e-12 else np.zeros(n_cols, dtype="float64")
        pre[i] = cur
        new = cur.copy()
        # 1) 先卖(只卖得掉的),释放现金
        dec = (want < cur) & cs[i]
        new[dec] = want[dec]
        cash_avail = 1.0 - new.sum()
        # 2) 再买(只买得进的),受可用现金约束
        inc = (want > new) & cb[i]
        need = np.where(inc, want - new, 0.0)
        total_need = need.sum()
        if total_need > 1e-12:
            scale = min(1.0, max(0.0, cash_avail) / total_need)   # 现金不足则等比缩减
            new = new + need * scale
        held[i] = new
        cash[i] = max(0.0, 1.0 - new.sum())
        if r_arr is not None:
            grown = new * (1.0 + r_arr[i])
            total = grown.sum() + cash[i]
            prev = grown / total if total > 1e-12 else np.zeros(n_cols, dtype="float64")
        else:
            prev = new

    w = pd.DataFrame(held, index=target.index, columns=target.columns)
    c = pd.Series(cash, index=target.index, name="cash")
    p = pd.DataFrame(pre, index=target.index, columns=target.columns)
    return w, c, p, pd.Series(loss, index=target.index, name="delist_loss")


def run_backtest(factor: pd.DataFrame, panel: dict[str, pd.DataFrame],
                 masks: dict[str, pd.DataFrame], bench: pd.Series | None = None,
                 top_pct: float = 0.10, horizon: int = 1,
                 periods_per_year: int = 243, drift: bool = True,
                 bench_open: pd.Series | None = None,
                 bench_tr_ret: pd.Series | None = None) -> dict:
    """完整 T+1 截面回测。返回指标、净值、超额净值与逐日明细。

    执行口径:factor.loc[t] 用 t 日收盘信息 → t+1 开盘调仓 → 吃 t+1→t+2 开盘收益。

    bench_tr_ret(可选,仅披露):全收益基准的逐日收益,第 t 行 = t 开盘 → t+1 开盘
    (由 loader.total_return_open_to_open 派生)。给出时另算 excess_ann_tr / excess_sharpe_tr /
    bench_tr_ann 与 out["excess_ret_tr"];不改变主基准的 excess_* 指标,不进门槛,
    零假设校准也不传入。缺省 None 时行为与未引入该参数前完全一致。
    """
    rules = load_contract()["tradability_rules"]
    c = rules["costs"]
    cost_buy = float(c["commission_each_side"]) + float(c["impact_each_side"])
    haircut = float(rules.get("delisting_haircut", 1.0))

    adj_close = panel["adj_close"]
    # 开盘后复权价:用 close 的复权比例近似(同日同因子),避免再查一次因子表
    ratio = (adj_close / panel["close"].replace(0.0, np.nan)).ffill()
    adj_open = panel["open"] * ratio

    # T+1:信号滞后一日才能建仓
    target = to_weights(factor, masks["in_universe"], top_pct=top_pct).shift(1).fillna(0.0)
    ret_oo = open_to_open_returns(adj_open).reindex_like(target)
    held, cash, pre, delist_loss = _apply_tradability(
        target, masks["can_buy"], masks["can_sell"], ret=ret_oo if drift else None,
        delisted=masks.get("delisted"), haircut=haircut)
    # 卖出印花税按日期取值(合同 v2:2023-08-28 前 0.1%,之后 0.05%)
    cost_sell = cost_buy + stamp_tax_series(held.index).to_numpy()

    gross = (held * ret_oo).sum(axis=1)

    # 成交量 = 交易后权重 − 交易前(已漂移)权重;含每日回到等权的再平衡交易
    dw = held - pre
    cost = dw.clip(lower=0).sum(axis=1) * cost_buy + (-dw.clip(upper=0)).sum(axis=1) * cost_sell
    cost = cost + delist_loss                         # 退市清算损失并入扣减项(另列 delist_loss 披露)
    trade_turnover = 0.5 * dw.abs().sum(axis=1)
    net = (gross - cost).fillna(0.0)

    nav = (1.0 + net).cumprod()
    out: dict = {
        "daily_ret": net, "gross_ret": gross, "cost": cost, "nav": nav, "cash": cash,
        "weights": held, "pre_trade_weights": pre, "turnover": trade_turnover,
        "n_holdings": (held > 0).sum(axis=1),
    }

    # IC/分组一律对齐"t 日决策真正吃得到的收益",与回测口径严格一致
    fwd = tradable_returns(adj_open) if horizon == 1 else forward_returns(adj_close, horizon)
    ic = metrics.rank_ic(factor, fwd, masks["in_universe"])
    dec = metrics.decile_returns(factor, fwd, masks["in_universe"])
    ann = float(net.mean()) * periods_per_year
    vol = float(net.std(ddof=1)) * np.sqrt(periods_per_year) if net.std(ddof=1) > 0 else np.nan
    stats = {
        **metrics.ic_stats(ic, periods_per_year),
        "monotonicity": metrics.monotonicity(dec),
        "ann_return": ann,
        "ann_vol": vol,
        "sharpe": ann / vol if vol and vol == vol and vol > 0 else np.nan,
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()) if len(nav) else np.nan,
        "avg_turnover": float(out["turnover"].mean()),
        "avg_holdings": float(out["n_holdings"].mean()),
        "avg_cash": float(cash.mean()),                 # 因买不进而闲置的资金占比
        "min_holdings": float(out["n_holdings"][out["n_holdings"] > 0].min())
        if bool((out["n_holdings"] > 0).any()) else 0.0,
        "max_gross_exposure": float(held.sum(axis=1).max()),   # 必须 ≤ 1(无杠杆)
        "total_cost": float(cost.sum()),
        "ann_cost": float(cost.mean()) * periods_per_year,
        "total_delist_loss": float(delist_loss.sum()),
        "cost_model": "drift_rebalance" if drift else "target_diff_legacy",
    }

    # 基准须与组合同一时段。组合第 t 行 = t 开盘→t+1 开盘(ret_oo);审查发现(P0,度量缺陷):
    # 原基准用收盘价算 t-1 收盘→t 收盘,两段只在 t 日盘中重叠,隔夜段错位——以中证1000 当
    # "组合"对中证500 实测,跟踪误差 16.3% 里约 82% 的方差来自这一错位,超额夏普被系统性压低。
    # 现:传入 bench_open 时基准同为开盘→次日开盘;旧口径另存 *_cc_legacy 仅作对照。
    b_ret, alignment = None, None
    if bench_open is not None:
        bo = bench_open.reindex(net.index).astype("float64")
        b_ret = (bo.shift(-1) / bo - 1.0).fillna(0.0)
        alignment = "open_to_open"
    elif bench is not None:
        b = bench.reindex(net.index).astype("float64")
        b_ret = (b / b.shift(1) - 1.0).fillna(0.0)
        alignment = "close_to_close_legacy"
    if b_ret is not None:
        excess = net - b_ret
        out["excess_ret"] = excess
        out["excess_nav"] = (1.0 + excess).cumprod()
        e_ann = float(excess.mean()) * periods_per_year
        e_vol = float(excess.std(ddof=1)) * np.sqrt(periods_per_year) if excess.std(ddof=1) > 0 else np.nan
        stats["excess_ann"] = e_ann
        stats["excess_sharpe"] = e_ann / e_vol if e_vol and e_vol == e_vol and e_vol > 0 else np.nan
        stats["bench_ann"] = float(b_ret.mean()) * periods_per_year
        stats["bench_alignment"] = alignment
        # 披露(不进门槛):相对"可投域等权"的超额。可投域取 t-1 日 in_universe(t-1 收盘已知),
        # 收益与组合同为 t 开盘→t+1 开盘、同含分红(后复权),剥离"中证500 vs 全市场等权"的
        # 规模/分红差,回答"超额里有多少只是买了可投域的平均股票"。
        inu = masks["in_universe"].reindex_like(held).fillna(False).astype(bool).shift(1, fill_value=False)
        ew = ret_oo.where(inu).mean(axis=1).fillna(0.0)
        ex_ew = net - ew
        v_ew = float(ex_ew.std(ddof=1)) * np.sqrt(periods_per_year)
        stats["ew_universe_ann"] = float(ew.mean()) * periods_per_year
        stats["excess_vs_ew_ann"] = float(ex_ew.mean()) * periods_per_year
        stats["excess_vs_ew_sharpe"] = (stats["excess_vs_ew_ann"] / v_ew
                                        if v_ew and v_ew == v_ew and v_ew > 0 else np.nan)
        if bench is not None and bench_open is not None:
            b = bench.reindex(net.index).astype("float64")
            ex_cc = net - (b / b.shift(1) - 1.0).fillna(0.0)
            v_cc = float(ex_cc.std(ddof=1)) * np.sqrt(periods_per_year)
            stats["excess_ann_cc_legacy"] = float(ex_cc.mean()) * periods_per_year
            stats["excess_sharpe_cc_legacy"] = (stats["excess_ann_cc_legacy"] / v_cc
                                                if v_cc and v_cc == v_cc and v_cc > 0 else np.nan)

    # 披露(不进门槛、不参与零校准):相对全收益基准的超额。主基准为价格指数(不含分红),
    # 而组合用后复权价(含分红),主口径超额中含约一个指数股息率;此列把该部分扣回。
    # 时段与组合一致(第 t 行 = t 开盘 → t+1 开盘),缺失按 0 处理,与主基准同一约定。
    if bench_tr_ret is not None:
        btr = pd.Series(bench_tr_ret).reindex(net.index).astype("float64").fillna(0.0)
        ex_tr = net - btr
        v_tr = float(ex_tr.std(ddof=1)) * np.sqrt(periods_per_year)
        out["excess_ret_tr"] = ex_tr
        stats["bench_tr_ann"] = float(btr.mean()) * periods_per_year
        stats["excess_ann_tr"] = float(ex_tr.mean()) * periods_per_year
        stats["excess_sharpe_tr"] = (stats["excess_ann_tr"] / v_tr
                                     if v_tr and v_tr == v_tr and v_tr > 0 else np.nan)

    out["ic"] = ic
    out["decile"] = dec
    out["stats"] = stats
    return out
