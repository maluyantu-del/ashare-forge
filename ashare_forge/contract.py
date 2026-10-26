"""合同加载:单一事实源 contracts/ashare.yaml。"""
from __future__ import annotations
import functools
import yaml
from .config import CONTRACT_PATH

@functools.lru_cache(maxsize=1)
def load_contract(path: str = CONTRACT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def board_of(code: str) -> str:
    """'sz.000001'→main, 'sz.300xxx'→chinext, 'sh.688xxx'→star, 'bj.xxx'→bse。"""
    ex, num = code.split(".")
    if ex == "bj": return "bse"
    if ex == "sh" and num.startswith("68"): return "star"
    if ex == "sz" and num.startswith("30"): return "chinext"
    return "main"

def is_stock_code(code: str) -> bool:
    """只保留股票:sh.60/68, sz.00/30, bj.4/8;排除指数(sh.000/sz.399)与基金(sh.5/sz.15,16,18)。"""
    try:
        ex, num = code.split(".")
    except ValueError:
        return False
    if ex == "sh": return num.startswith(("60", "68"))
    if ex == "sz": return num.startswith(("00", "30"))
    if ex == "bj": return num.startswith(("4", "8"))
    return False

def limit_pct(code: str, is_st: bool = False, date: str | None = None) -> float:
    """涨跌停幅度。date(YYYY-MM-DD)早于创业板改革日时,创业板取改革前 10%(合同 v2 修正)。"""
    lp = load_contract()["tradability_rules"]["limit_pct"]
    b = board_of(code)
    if b == "main" and is_st: return float(lp["st_main_board"])
    if b == "chinext" and date is not None and str(date) < str(lp.get("chinext_reform_date", "")):
        return float(lp.get("chinext_pre_reform", lp["chinext"]))
    return float(lp[{"main": "main_board", "chinext": "chinext", "star": "star", "bse": "bse"}[b]])


def stamp_tax_series(dates) -> "pd.Series":
    """按合同 stamp_tax_sell_schedule 逐日取卖出印花税率(v2 修正:2023-08-28 前为 0.1%)。"""
    import pandas as pd
    c = load_contract()["tradability_rules"]["costs"]
    idx = pd.DatetimeIndex(dates)
    sched = c.get("stamp_tax_sell_schedule")
    if not sched:
        return pd.Series(float(c["stamp_tax_sell"]), index=idx)
    out = pd.Series(float("nan"), index=idx)
    for item in sorted(sched, key=lambda x: str(x["from"])):
        out[idx >= pd.Timestamp(str(item["from"]))] = float(item["rate"])
    return out.bfill().fillna(float(c["stamp_tax_sell"]))
