"""面板版泄漏测试:因子只能用"当日及以前"的数据。

为什么用前缀重算而不用"扰动未来":
  - 常见做法"只扰动 t 之后的数据、看 t 之前信号变不变"有结构性假阴性——
    切点从样本后半段起会让前半段零守卫,单点单步前视只有约 2.5% 命中率;
    且正向乘性扰动对"用全样本 max/min/分位归一化"的前视不敏感。
  - 根治 = **前缀重算**:factor[t] 只依赖过去 ⟺ 用前缀 panel[:t+1] 重算的末行
    等于全量面板的第 t 行。去掉未来 = 最彻底的扰动。

截面场景的额外坑(本模块特有):
  - 因子常做"截面标准化/分位化",这只用当日数据、不构成前视;但一旦对**时序**
    做全样本标准化(如除以全样本 std),就是前视——前缀重算能抓到。
  - 面板是 {字段: 宽表} 的字典,前缀要对每张表同时截断,不能只截一张。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _prefix(panel: dict[str, pd.DataFrame], t: int) -> dict[str, pd.DataFrame]:
    """把面板里每张宽表都截到前 t+1 行(同一日历)。"""
    return {k: v.iloc[: t + 1] for k, v in panel.items()}


def prefix_recompute_test(factor_fn, panel: dict[str, pd.DataFrame],
                          n_points: int | None = None, warmup: int = 20,
                          atol: float = 1e-9, seed: int = 7) -> dict:
    """前缀重算检验:对采样的若干个 t,比较 factor_fn(panel[:t+1]) 末行与全量第 t 行。

    factor_fn(panel) -> DataFrame(index 与 panel 一致, columns=股票)
    返回 {passed, n_points_tested, failures, note}
    """
    any_df = next(iter(panel.values()))
    n = len(any_df)
    if n < warmup + 10:
        raise ValueError(f"面板太短({n} 行),无法做前缀重算检验")

    full = factor_fn(panel)
    if not isinstance(full, pd.DataFrame) or len(full) != n:
        raise ValueError(f"factor_fn 必须返回与面板等长的 DataFrame(得到 {type(full).__name__})")

    rng = np.random.default_rng(seed)
    # 审查发现(P0):原默认 max(12, n//30) 在 2351 行上仅约 78 切点(覆盖 3.3%),
    # "只在 2% 日期偷看"的因子抓获率约 10%——单步前视必须恰好采样到那一天。
    # 计算:偷看 2% 日期 ≈ 47 个泄漏日,全部漏掉的概率 = (1-覆盖率)^47:
    #   覆盖 3.3% → 21% 漏网;覆盖 12.5% → 0.2% 漏网。故提高密度至 n//8。
    k = n_points if n_points is not None else max(40, n // 8)
    k_uniform = k // 2
    grid = np.linspace(warmup, n - 1, max(k_uniform, 2)).astype(int)
    # 另一半按"截面排序突变幅度"加权抽样:注入未来信息常造成当日排序骤变
    ranks = full.rank(axis=1, pct=True)
    # pandas 3.0 写时复制下 .to_numpy() 返回只读视图,下一行原地赋值会抛错(自测实测)
    jump = ranks.diff().abs().mean(axis=1).fillna(0.0).to_numpy(copy=True)
    jump[:warmup] = 0.0
    if jump.sum() > 0:
        w = jump / jump.sum()
        weighted = rng.choice(n, size=k - k_uniform, replace=True, p=w)
    else:
        weighted = rng.integers(warmup, n, size=k - k_uniform)
    points = sorted(set(int(x) for x in np.concatenate([grid, weighted]) if int(x) >= warmup))

    failures = []
    for t in points:
        try:
            pv = factor_fn(_prefix(panel, t))
        except Exception as e:  # noqa: BLE001
            failures.append({"pos": t, "reason": f"前缀重算抛错(疑访问未来): {type(e).__name__}: {e}"})
            continue
        if len(pv) != t + 1:
            failures.append({"pos": t, "reason": f"前缀因子长度 {len(pv)} ≠ {t+1}"})
            continue
        a = full.iloc[t].reindex(pv.columns).to_numpy(dtype="float64")
        b = pv.iloc[-1].to_numpy(dtype="float64")
        both_nan = np.isnan(a) & np.isnan(b)
        ok = both_nan | np.isclose(a, b, atol=atol, rtol=0.0)
        bad = np.flatnonzero(~ok)
        if bad.size:
            j = int(bad[0])
            failures.append({"pos": t, "date": str(full.index[t]),
                             "code": str(pv.columns[j]),
                             "full_value": None if np.isnan(a[j]) else float(a[j]),
                             "prefix_value": None if np.isnan(b[j]) else float(b[j]),
                             "n_diff": int(bad.size)})

    return {
        "passed": not failures,
        "n_points_tested": len(points),
        "failures": failures[:10],
        "note": ("前缀重算:factor[t] 只依赖过去 ⟺ 用 panel[:t+1] 重算的末行等于全量第 t 行。"
                 "这是无前视的必要条件强化检验,非穷举证明;数据层面的发布时滞仍由合同管辖。"),
    }

# ==================== 静态扫描:因子不得绕开面板自行取数 ====================
# 审查发现(P0):前缀重算只截断 panel 字典里的表。因子若自己开数据库/读文件取未来数据,
# 截断对它无效——实测一个自开 sqlite 读未来 5 日收益的因子零告警通过,真实面板 IC 0.40/t 53.8。
# 静态扫描是必要补充:因子只准用传进来的 panel,任何外部 I/O 一律拒绝。
import ast as _ast  # noqa: E402
import inspect as _inspect  # noqa: E402

# 按语法树检查**真实代码**(导入与调用),文档字符串/注释天然不受影响。
# 首版用正则扫源码,把"baostock 的 peTTM 是日频快照"这类说明文字也当成取数,4 个因子全被误拦。
FORBIDDEN_MODULES = {
    "sqlite3": "直连数据库", "requests": "网络访问", "urllib": "网络访问", "httpx": "网络访问",
    "http": "网络访问", "socket": "网络访问", "akshare": "直连行情源", "baostock": "直连行情源",
    "tushare": "直连行情源", "subprocess": "执行外部命令", "importlib": "动态导入",
    "pymysql": "直连数据库", "sqlalchemy": "直连数据库",
}
FORBIDDEN_CALLS = {
    "open": "打开文件", "eval": "动态执行代码", "exec": "动态执行代码", "__import__": "动态导入",
    "read_sql": "pandas 读数据库", "read_sql_query": "pandas 读数据库", "read_sql_table": "pandas 读数据库",
    "read_csv": "读外部文件", "read_parquet": "读外部文件", "read_excel": "读外部文件",
    "read_json": "读外部文件", "read_pickle": "读外部文件", "read_hdf": "读外部文件",
    "read_feather": "读外部文件", "load_panel": "绕过传入面板自行加载数据",
    "connect": "建立外部连接", "urlopen": "网络访问",
}


def _call_name(f) -> str | None:
    return f.id if isinstance(f, _ast.Name) else (f.attr if isinstance(f, _ast.Attribute) else None)


def scan_factor_source(build_factor) -> list[dict]:
    """用语法树扫描因子代码,返回命中的禁用项(空列表 = 通过)。

    扫描范围(两次修正后的口径):
      ① **模块级导入全部检查**——导入会进入因子的命名空间,无论在哪个函数里用到。
         (只扫函数本体时,`from akshare import ...` 零告警漏网)
      ② **函数调用只检查因子函数及其在同模块内实际调用到的辅助函数(传递闭包)**。
         (整文件扫描时,同文件测试夹具/`__main__` 调试块里的合法 I/O 被误算到因子头上)
    无法取得源码的因子一律拒绝(无法审计的因子不得入库)。
    """
    try:
        path = _inspect.getsourcefile(build_factor)
        if path:
            with open(path, encoding="utf-8") as fh:        # 扫描器自身读文件,非因子代码
                src = fh.read()
        else:
            src = _inspect.getsource(build_factor)
        tree = _ast.parse(src)
    except (OSError, TypeError, SyntaxError) as e:
        return [{"kind": "source", "name": "<source>", "line": 0,
                 "reason": f"无法取得或解析源码({type(e).__name__}),拒绝:无法审计的因子不得入库"}]

    hits = []
    # ① 模块级导入(含 if/try 块里的模块级导入)
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            continue
        if isinstance(node, _ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    hits.append({"kind": "import", "name": a.name, "line": node.lineno,
                                 "reason": FORBIDDEN_MODULES[root]})
        elif isinstance(node, _ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_MODULES:
                hits.append({"kind": "import", "name": node.module, "line": node.lineno,
                             "reason": FORBIDDEN_MODULES[root]})

    # ② 因子函数的调用闭包
    funcs = {n.name: n for n in _ast.walk(tree)
             if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
    root_name = getattr(build_factor, "__name__", None)
    if root_name not in funcs:
        return hits + [{"kind": "source", "name": str(root_name), "line": 0,
                        "reason": "在源文件中找不到因子函数定义,无法审计,拒绝"}]
    seen, stack = set(), [root_name]
    while stack:
        fn = stack.pop()
        if fn in seen or fn not in funcs:
            continue
        seen.add(fn)
        for node in _ast.walk(funcs[fn]):
            if isinstance(node, _ast.Call):
                name = _call_name(node.func)
                if name in FORBIDDEN_CALLS:
                    hits.append({"kind": "call", "name": name, "line": node.lineno,
                                 "reason": FORBIDDEN_CALLS[name], "in": fn})
                if name in funcs and name not in seen:
                    stack.append(name)
            elif isinstance(node, _ast.Name) and node.id in funcs and node.id not in seen:
                stack.append(node.id)                        # 函数被当作值传递也纳入闭包
            elif isinstance(node, _ast.Attribute):
                if node.attr == "environ" and isinstance(node.value, _ast.Name) and node.value.id == "os":
                    hits.append({"kind": "attr", "name": "os.environ", "line": node.lineno,
                                 "reason": "读环境变量", "in": fn})
            # 函数内部的导入同样检查
            elif isinstance(node, (_ast.Import, _ast.ImportFrom)):
                mods = [a.name for a in node.names] if isinstance(node, _ast.Import) else [node.module or ""]
                for m in mods:
                    if m.split(".")[0] in FORBIDDEN_MODULES:
                        hits.append({"kind": "import", "name": m, "line": node.lineno,
                                     "reason": FORBIDDEN_MODULES[m.split(".")[0]], "in": fn})
    return hits
