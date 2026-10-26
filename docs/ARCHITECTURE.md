# AShare Forge 架构与模块接口(第一期:数据层)

目标:A股截面因子研究平台。设计原则:交易规则与研究边界写进合同、由代码执行,而不是写成"注意事项"靠人记;
每个结论须可复现、可追溯,负结果与撤回同样留痕。本地运行、免 token、只用公开数据、可公开。
第一期只建**数据层**,但接口要为第二期(截面引擎)、第三期(任务卡)、第四期(网页)留好。

> 本文是第一期施工时的接口规格,保留原貌以便对照。此后的规则修订——涨跌停价按分取整、创业板改革前 10%、
> 按真实上市日期计上市天数、复权因子全历史同步与完整性闸门等——以 `contracts/ashare.yaml`(合同 v2)、
> 代码与 README 的"验证机制"一节为准。

## 单一事实源
- 合同:`contracts/ashare.yaml`(`ashare_forge.contract.load_contract()`),规则由代码执行。
- 表结构:`ashare_forge/data_schema.sql`;数据库 `data/ashare.db`(SQLite,git 忽略)。
- 路径:`ashare_forge/config.py`。
- 环境:`.venv`(pandas 3.0 / numpy 2.5 / baostock 0.9.3 / akshare 1.18.96 / pyyaml)。**pandas 3.0 注意**:默认 Copy-on-Write、字符串列为 `str` dtype、`fillna(method=)` 已移除、`groupby` 默认 `observed=True`。

## 模块与接口(并行施工按此对齐,签名不得改)

### `ashare_forge/data/db.py`
```python
def connect(path: str = DB_PATH) -> sqlite3.Connection        # PRAGMA journal_mode=WAL, synchronous=NORMAL, row_factory=sqlite3.Row
def ensure_schema(conn) -> None                                # 执行 data_schema.sql(幂等)
def upsert_many(conn, table: str, rows: list[dict]) -> int     # INSERT OR REPLACE,按 rows[0].keys() 生成列;空列表返回0;分批 executemany
def last_date(conn, table: str, code: str) -> str | None       # SELECT MAX(date) WHERE code=?
def get_meta(conn, key: str, default=None) -> str | None
def set_meta(conn, key: str, value: str) -> None
def log_sync(conn, task: str, code: str | None, rows: int, status: str, message: str = "") -> None   # 写 sync_log,ts=UTC ISO
```

### `ashare_forge/data/baostock_source.py`
```python
class BaostockSource:                                          # 上下文管理器:__enter__ login, __exit__ logout;重连:任何查询 error_code!="0" 或异常 → 指数退避重试3次(1s/3s/9s),重试前 relogin
    DAILY_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
    def fetch_daily(self, code: str, start: str, end: str) -> pd.DataFrame
        # 不复权(adjustflag="3"),按自然年分块查询后拼接;数值列 to_numeric(errors="coerce");空串→NaN;
        # 返回列 = daily 表列(去掉 adjustflag);date 为 'YYYY-MM-DD' 字符串;tradestatus/isST 为 int
    def fetch_adjust_factors(self, code: str, start: str, end: str) -> pd.DataFrame   # 列 code,date(dividOperateDate→date),foreAdjustFactor,backAdjustFactor,adjustFactor
    def fetch_all_stock(self, day: str) -> pd.DataFrame          # 列 code,tradestatus(int),name;不过滤(过滤由调用方用 contract.is_stock_code)
    def fetch_industry(self) -> pd.DataFrame                     # 列 updateDate,code,code_name,industry,industryClassification
    def fetch_index_daily(self, code: str, start: str, end: str) -> pd.DataFrame   # 列 code,date,open,high,low,close,volume,amount,pctChg
```
所有 fetch_* 在失败重试耗尽后**抛 RuntimeError**(不返回空表冒充成功),消息含 code/查询/error_msg。

### `ashare_forge/data/sync.py`(CLI:`python -m ashare_forge.data.sync ...`)
```python
def discover_universe(src, conn, start: str, end: str, step_months: int = 3) -> int
    # 从 start 到 end 每 step_months 取一次 query_all_stock 快照(含 end 当日),写 universe_snapshot;
    # 用 contract.is_stock_code 过滤后 upsert 进 stocks(first_seen/last_seen/board/name);返回股票总数
def sync_daily(src, conn, codes: list[str] | None, since: str, limit: int | None, end: str | None = None) -> dict
    # codes 为空则取 stocks 全表(按 code 排序);对每只:从 max(last_date+1, since) 增量拉到 end(缺省今天);
    # 断点续传;每50只打印进度与速率(只/分钟,行/秒)、预计剩余;每只写 sync_log;单只失败记录并继续;返回 {codes, rows, failed:[...], seconds}
def sync_adj(src, conn, codes, since, limit) -> dict           # 复权因子,同上口径
def sync_index(src, conn, codes: list[str] | None = None, since: str = "2005-01-01") -> dict   # 缺省取 contract.benchmarks
def sync_industry(src, conn) -> int                            # 写 stocks.industry/industry_src/industry_asof
def mark_delisted(conn, as_of: str, gap_days: int = 60) -> int  # last K 行早于 as_of-gap_days 的标 delisted=1
def data_quality_report(conn) -> dict                          # 股票数/已退市数/日线行数/日期范围/停牌比例/ST比例/缺复权因子股数/每年行数/最近同步日期/失败清单
```
CLI 参数:`--db --since(缺省2015-01-01) --end --limit --codes a,b --universe-only --daily-only --adj-only --index-only --industry --report --snapshots-from --snapshots-to`。
默认全流程顺序:discover_universe → sync_index → sync_industry → sync_daily → sync_adj → mark_delisted → report。

### `ashare_forge/data/loader.py`
```python
def load_panel(conn, start: str, end: str | None = None, fields: list[str] | None = None,
               unlock_holdout: bool = False) -> dict[str, pd.DataFrame]
    # 返回 {field: 宽表}(index=DatetimeIndex 升序, columns=code 升序),fields 缺省 = daily 全部数值列 + 'adj_close' + 'float_mcap';
    # **holdout 强制**:contract.holdout.cutoff 及之后的日期一律不返回,除非 unlock_holdout=True(此时 set_meta 追加 'holdout_unlocked_log' 一条含 UTC 时间的记录)
def apply_adjust(close_wide: pd.DataFrame, adj_long: pd.DataFrame) -> pd.DataFrame   # 后复权:backAdjustFactor 按 (code,date) 前向填充到每日,缺失视为1.0;adj_close = close*factor
def derive_float_mcap(close: pd.DataFrame, volume: pd.DataFrame, turn: pd.DataFrame) -> pd.DataFrame   # close*volume/(turn/100),turn<=0 或 NaN → NaN
def listing_age(daily_dates_by_code: dict[str, pd.DatetimeIndex] | pd.DataFrame) -> pd.DataFrame        # 每 (date,code) 的已上市交易日数(从该股首个K行起算)
def tradable_masks(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]
    # 返回 {'can_buy','can_sell','in_universe'} 三张布尔宽表:
    #  in_universe = 非停牌 & 非ST & 上市≥new_listing_days & float_mcap≥min_float_mcap_yuan & 当日有K行
    #  can_buy  = in_universe & 非一字涨停(open >= preclose*(1+limit)-1e-6 为不可买)
    #  can_sell = 非停牌 & 非一字跌停(open <= preclose*(1-limit)+1e-6 为不可卖)   # 卖出不受 ST/微盘限制
    #  limit 用 contract.limit_pct(code, is_st)
def index_series(conn, code: str = "sh.000905", start: str = "2005-01-01", end: str | None = None, unlock_holdout: bool = False) -> pd.Series  # close,同样受 holdout 约束
```

### `tests/test_data_layer.py`(pytest,不联网)
用 `ensure_schema` 建临时库,插入合成数据覆盖:一只正常股、一只ST股、一只次新(不足120日)、一只微盘(流通市值<20亿)、一只有停牌日、一只有一字涨停日与一字跌停日、一只已退市(最后K行早于as_of);一条复权因子生效日。断言:
- `tradable_masks` 三张表在每个构造点上的真/假;
- `apply_adjust` 在生效日前后因子正确前向填充、缺失为1.0;
- `derive_float_mcap` 与手算一致,turn=0 → NaN;
- `load_panel` 默认**不含** holdout.cutoff 及之后日期;unlock 后含且 meta 有日志;
- `mark_delisted` 正确标记;`data_quality_report` 键齐全;
- `upsert_many` 重复写入不重复计数(INSERT OR REPLACE)。
`tests/test_live_smoke.py`:仅当环境变量 `ASHARE_LIVE=1` 才跑,拉 sz.000001 近20天与 sh.000905 指数,断言非空与列齐。

## 约定
- 日期一律 'YYYY-MM-DD' 字符串入库,加载后再转 DatetimeIndex。
- 代码一律 baostock 形式 'sz.000001'。
- 不打印/不落盘任何密钥(本项目无密钥)。所有网络失败必须显式报错,不静默为空。
- 每个模块顶部 docstring 用中文说明"它修了常见做法的哪个坑"。
