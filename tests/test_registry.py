"""台账与机制查重测试。重点:行为指纹能否堵住"换个算子名绕过查重"的洞。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_forge.data import db
from ashare_forge.engine import registry as reg


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(str(tmp_path / "reg.db"))
    reg.ensure_tables(c)
    yield c
    c.close()


def _rev(win):
    def f(p):
        a = p["adj_close"]
        return -(a / a.shift(win) - 1.0)
    return f


def _vol(win):
    def f(p):
        a = p["adj_close"]
        return -(a / a.shift(1) - 1.0).rolling(win).std()
    return f


# ---------- 声明签名 ----------

def test_signature_stable_and_bucketed():
    s1, c1 = reg.mechanism_signature("reversal", ["rolling_return"], "reversion", 20, ["adj_close"])
    s2, _ = reg.mechanism_signature("reversal", ["rolling_return"], "reversion", 22, ["adj_close"])
    assert s1 == s2, "同一持有期桶内应得同签名"
    s3, _ = reg.mechanism_signature("volatility", ["rolling_std"], "reversion", 20, ["adj_close"])
    assert s1 != s3


def test_declared_signature_can_be_bypassed_by_renaming():
    """如实记录声明签名的已知弱点:换个算子名就能得到不同签名(这正是行为指纹要补的洞)。"""
    a, _ = reg.mechanism_signature("reversal", ["rolling_return"], "reversion", 20, ["adj_close"])
    b, _ = reg.mechanism_signature("reversal", ["pct_change_ret"], "reversion", 20, ["adj_close"])
    assert a != b


# ---------- 行为指纹(补强声明签名:看因子实际算出什么,而非它自称用了什么算子) ----------

def test_behavior_fingerprint_catches_renamed_duplicate():
    """同一机制的两种等价写法,行为指纹必须判为同一个(相似度≥阈值)。"""
    f1 = reg.behavior_fingerprint(_rev(20))

    def rev20_alt(p):                      # 完全等价的另一种写法
        return -p["adj_close"].pct_change(20)

    f2 = reg.behavior_fingerprint(rev20_alt)
    s = reg.fingerprint_similarity(f1, f2)
    assert s >= reg.BEHAVIOR_DUP_THRESHOLD, f"等价写法未被识别为同一机制(相似度 {s})"


def test_behavior_fingerprint_separates_real_different_mechanisms():
    s = reg.fingerprint_similarity(reg.behavior_fingerprint(_rev(20)),
                                   reg.behavior_fingerprint(_vol(20)))
    assert s < 0.5, f"反转与波动率被误判为同一机制(相似度 {s})"


def test_fingerprint_is_deterministic():
    assert np.allclose(reg.behavior_fingerprint(_rev(20)), reg.behavior_fingerprint(_rev(20)))


def test_fingerprint_invariant_to_monotone_transform():
    """x → 2x+1 不改变截面排序,行为指纹必须不变(用秩而非原值的意义)。"""
    def scaled(p):
        return 2.0 * _rev(20)(p) + 1.0
    s = reg.fingerprint_similarity(reg.behavior_fingerprint(_rev(20)),
                                   reg.behavior_fingerprint(scaled))
    assert s == pytest.approx(1.0, abs=1e-9)


def test_find_behavioral_duplicates_roundtrip(conn):
    sig_a, canon_a = reg.mechanism_signature("reversal", ["rolling_return"], "reversion", 20, ["adj_close"])
    reg.register_mechanism(conn, sig_a, canon_a, "20日反转")
    reg.save_fingerprint(conn, sig_a, reg.behavior_fingerprint(_rev(20)))

    def rev20_alt(p):
        return -p["adj_close"].pct_change(20)

    fp_b = reg.behavior_fingerprint(rev20_alt)
    dups = reg.find_behavioral_duplicates(conn, fp_b)
    assert dups and dups[0]["signature"] == sig_a, "换名写法未在台账中找到行为撞车"

    far = reg.find_behavioral_duplicates(conn, reg.behavior_fingerprint(_vol(20)))
    assert not far, "不同机制被误报为行为撞车"


# ---------- 台账 ----------

def test_trial_count_excludes_draft(conn):
    sig, canon = reg.mechanism_signature("reversal", ["rolling_return"], "reversion", 20, ["adj_close"])
    reg.register_mechanism(conn, sig, canon, "t")
    u, fam = canon["universe"], canon["family"]
    reg.log_trial(conn, sig, canon, "draft", task_id="task02", name="x")
    assert reg.trial_count(conn, u, fam) == 0, "草稿不应计入试验次数(会虚增DSR惩罚)"
    reg.log_trial(conn, sig, canon, "fail", task_id="task02", name="x")
    assert reg.trial_count(conn, u, fam) == 1


def test_dedup_report_surfaces_prior_failures(conn):
    sig, canon = reg.mechanism_signature("reversal", ["rolling_return"], "reversion", 20, ["adj_close"])
    reg.register_mechanism(conn, sig, canon, "t")
    reg.log_trial(conn, sig, canon, "fail", name="x", notes="超额夏普为负")
    r = reg.dedup_report(conn, sig)
    assert r["known_mechanism"] and r["verdict_counts"].get("fail") == 1


def test_fixed_panel_covers_loader_default_fields():
    """回归:合成面板缺字段会让行为指纹**静默失效**,退化为仅靠声明签名查重。

    实测:illiquidity 因面板缺 pctChg 而指纹计算抛错,流水线只记了一条备注就放过去了。
    """
    from ashare_forge.data.loader import DEFAULT_FIELDS
    missing = sorted(set(DEFAULT_FIELDS) - set(reg._fixed_panel().keys()))
    assert not missing, f"合成面板缺少 loader 默认字段 {missing},行为指纹会对用到这些列的因子失效"


def test_all_shipped_factors_get_a_fingerprint():
    """所有随仓库发布的因子都必须能算出行为指纹(否则查重形同虚设)。

    直接复用 pipeline 的因子发现逻辑,测的就是真实路径。
    """
    from ashare_forge import pipeline

    specs = pipeline.discover_specs(include_optional=True)
    assert specs, "未发现任何因子"
    failed = []
    for sp in specs:
        fn = sp.get("build_factor")
        if not callable(fn):
            failed.append(f"{sp.get('name')}: spec 未解析出 build_factor")
            continue
        try:
            reg.behavior_fingerprint(fn)
        except Exception as e:  # noqa: BLE001
            failed.append(f"{sp.get('name')}: {type(e).__name__}: {e}")
    assert not failed, "以下因子无法计算行为指纹:\n" + "\n".join(failed)


def test_trial_provenance_is_persisted(conn):
    """回归(审查"最致命的追问"):每次裁定必须把证据包入库,数字可追溯到代码版本。"""
    import json as _json
    sig, canon = reg.mechanism_signature("reversal", ["rolling_return"], "reversion", 20, ["adj_close"])
    reg.register_mechanism(conn, sig, canon, "t")
    pv = {"code_sha": "abc123", "git_commit": "deadbeef",
          "leak": {"passed": True, "n_points": 40},
          "null": {"mode": "fixed", "n": 50, "passed": False, "n_passed": 1, "n_metrics": 4},
          "stats": {"excess_ann": -0.05, "excess_sharpe": -0.4, "monotonicity": 0.9, "avg_turnover": 0.2}}
    reg.log_trial(conn, sig, canon, "fail", name="x", provenance=pv)
    row = conn.execute("SELECT code_sha, git_commit, leak_passed, null_passed, excess_sharpe, "
                       "provenance_json FROM forge_trials WHERE verdict='fail'").fetchone()
    assert row[0] == "abc123" and row[1] == "deadbeef", "代码版本未入库"
    assert row[2] == 1 and row[3] == 0, "泄漏/零校准结论未入库"
    assert row[4] == pytest.approx(-0.4), "超额夏普未入库"
    assert _json.loads(row[5])["null"]["mode"] == "fixed", "零校准模式未入库(新旧零分布不可混)"


def test_factor_code_sha_changes_with_code():
    """同一因子函数哈希稳定;不同模块哈希不同。"""
    from ashare_forge.factors import idio_volatility, short_reversal
    a = reg.factor_code_sha(short_reversal.build_factor)
    assert a == reg.factor_code_sha(short_reversal.build_factor) and len(a) == 16
    assert a != reg.factor_code_sha(idio_volatility.build_factor)


def test_declared_trials_accepts_categorized_mapping(tmp_path):
    """分类申报 {preregistered, post_hoc} 须相加计费;列表与整数写法保持兼容。"""
    (tmp_path / "t_map.yaml").write_text("trials_declared:\n  preregistered: 6\n  post_hoc: 2\n",
                                         encoding="utf-8")
    (tmp_path / "t_list.yaml").write_text("trials_declared:\n  - a\n  - b\n  - c\n", encoding="utf-8")
    (tmp_path / "t_int.yaml").write_text("trials_declared: 4\n", encoding="utf-8")
    assert reg.declared_trials("t_map", tasks_dir=str(tmp_path)) == 8
    assert reg.declared_trials("t_list", tasks_dir=str(tmp_path)) == 3
    assert reg.declared_trials("t_int", tasks_dir=str(tmp_path)) == 4
    assert reg.declared_trials("t_missing", tasks_dir=str(tmp_path)) == 0


def test_every_task_card_with_variants_declares_trials():
    """一张卡下挂了 k 个因子变体,申报档数不得少于 k——否则跑了的变体没有被计费。"""
    from collections import Counter
    from ashare_forge import pipeline
    per_task = Counter(str(s.get("task_id") or "") for s in pipeline.discover_specs(include_optional=True))
    for task_id, k in per_task.items():
        if not task_id:
            continue
        assert reg.declared_trials(task_id) >= k, f"{task_id} 挂 {k} 个变体但只申报 {reg.declared_trials(task_id)} 档"
