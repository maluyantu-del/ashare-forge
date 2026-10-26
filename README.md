# AShare Forge · A 股截面因子研究平台

面向 A 股日频截面因子的本地研究平台。设计目标不是"找到赚钱的因子",而是**让每一个结论可复现、可追溯、难以自欺**:任何因子须依次通过泄漏检测、T+1 可交易回测、零假设校准与分层门槛,结果连同代码指纹写入台账。

- 数据:仅用公开免费数据(baostock 的日线、复权因子、股池快照、上市退市日期与指数日线),无需 token,仓库不含数据文件。
- 研究辅助,非投资建议。负结果照实记录;工程通过不等于策略有效。

## 流水线

```
合同(contracts/ashare.yaml)          交易规则、成本、保留段、危机窗口、门槛 —— 单一事实源
  → 任务卡(tasks/*.yaml)             假设与对立假设在拉数据前写定,申报参数档数;方法修订只追加记录
  → 因子插件(ashare_forge/factors)    纯函数:panel → 因子矩阵
  → 查重                             声明签名 + 行为指纹(固定合成面板上的截面秩向量)
  → 泄漏检测                          语法树静态扫描 → 前缀重算(不过即终止,不产出业绩数字)
  → T+1 回测                         涨跌停/停牌不可成交、现金账户、漂移再平衡成本、退市清算;基准同为开盘→次日开盘
  → 零假设校准                        固定 donor 零分布;核心指标 p ≤ 0.05,零样本换手须为真实的 0.8–1.25 倍
  → 分层门槛                          平静期超额夏普/PSR/DSR、危机期超额夏普、平均持仓;须与零校准同时通过
  → 台账(SQLite)                     代码 sha256、git 提交(进程启动时快照)、泄漏与零校准明细、数据快照
```

## 验证机制与其修复的问题

每条机制均对应开发中实测发现的偏差来源,修复前后数据见提交记录与 `tasks/BATCH02_FINDINGS.md`。

| 机制 | 防范的偏差 | 实测证据 |
|---|---|---|
| 全历史复权因子同步 | baostock 后复权因子为上市以来累计值,截断同步产生假跳变 | 截断时单只股票出现 +1139% 伪收益 |
| 复权完整性闸门 | 整只股票缺复权因子时按 1.0 静默处理,送转日呈假暴跌 | 154 只 2015–2023 年退市股因接口限流无因子,批跑照常开跑;现发现"有行情而无任何因子"的股票即拒绝开跑(`0b9ed72`) |
| 收益对齐 T+1 可交易区间 | t 日收盘信号只能在 t+1 开盘成交,IC 须对齐 t+1→t+2 开盘收益 | 未对齐时合成因子 IC 0.876 而回测夏普 −3.57 |
| 基准与组合同时段 | 组合为开盘→次日开盘,基准原为收盘→收盘,两段只在盘中重叠,超额方差虚增、夏普被压低 | 中证1000 对中证500:跟踪误差 16.3% → 7.0%,超额方差 81.5% 来自错位(`4ab2a8c`);旧口径另存 `*_cc_legacy` |
| 真实上市/退市日期(point-in-time 股池) | 幸存者偏差;以库内首根 K 线计上市天数导致样本初期全市场被误判为次新股 | 补入 2015–2023 年退市股 154 只;2015 上半年日均在池 0 → 1686 只 |
| 合同 v2:涨跌停价按分取整 | 未取整的 preclose×(1+limit) 在取整向下时漏判一字板,偏乐观 | 2019 年主板一字涨停漏判约 41%(`a15a404`) |
| 合同 v2:创业板改革前 10% | 全样本按 20% 判定,2020-08-24 前创业板涨跌停全部漏判 | 改革日前取 10% |
| 合同 v2:印花税按日期 | 全样本按 0.05% 低估成本 | 2023-08-28 前为 0.1%,覆盖样本 89.5% 的交易日 |
| 合同 v2:退市清算 | 退市持仓永久冻结在最后价格,等价于零损失 | 摘牌日仍未卖出者按 haircut=1.0(事前写定的保守上界)清算;BATCH02b 各因子退市清算损失合计 0–2.9% 净值 |
| 现金账户 | 卖不掉的旧仓占位而新仓照买,形成隐含杠杆 | 修复前单日敞口最高 1.61 倍 |
| 漂移再平衡成本 | 隔夜价格变动改变权重,回到等权的交易被漏记 | 手算回归:单股上涨 10% 时单边换手 1.1/2.1 − 0.5,旧口径为 0 |
| 零分布构造(三次迭代) | 零因子与真实因子换手不可比时,"超越零分布"比较的是手续费而非选股能力 | ① 逐日独立打乱:零样本日换手 0.85,真实因子 0.009(`1dffd28` 弃用)→ ② 整列置换 + 逐行恢复缺失:真实数据中上市/退市/停牌几乎每天改变缺失布局,退化为近似逐日打乱,零样本换手为真实的 2.2–13.4 倍(`a15a404` 修正)→ ③ 现行:按生命周期分组、组内循环错位的固定 donor,加换手有效性闸门 [0.8, 1.25]。2019–2021 验证 0.84–1.15 倍;BATCH02b 中 11 个因子为 0.88–1.17 倍,横盘突破为 0.23 倍、判零分布无效 |
| 零校准 p 值口径 | 有限次置换的 p 值不应为 0;插值 q95 判据偏宽 | n=50 时单指标假阳性约 6.8%;现取 p = (1+#)/(n+1) ≤ 0.05(Phipson–Smyth) |
| IC 族强制在场 | IC 为 NaN 时核心判据静默降级,零技能因子可凭收益指标通过 | 回归测试覆盖 |
| 前缀重算 + 稠密抽样 | 全样本归一化等隐蔽前视;稀疏偷看 | 旧采样在 2351 行上覆盖 3.3%,约 21% 概率漏网;现 max(40, n/8) 且半数按截面突变加权 |
| 语法树静态扫描 | 因子在面板外自行读库/读文件,前缀截断对其无效 | 攻击样例在修复前零告警、IC 0.40 |
| 行为指纹查重 | 更换算子名即可绕过声明签名查重 | 等价写法相似度 1.0;反转 vs 波动率 0.036 |
| DSR 全域试验计数 | 按信号族计数时试验数仅 2–3,多重检验代价被低估 | 计入全研究域历史检验与任务卡申报档数 |
| 分组市值中性化 | 线性残差无法去除非线性市值依赖 | 残差与市值 Spearman:线性 −0.209 → 十分组 −0.017 |
| 出处启动快照 | git 提交与代码指纹在写账时现读,长批次中途的提交会被记到已在跑的结果上 | 16:48 启动的进程,其 22:03 条目记成 21:48 的提交;现于进程启动时快照,写账时比对并记录漂移(`4ab2a8c`) |
| 保留段 | 2024-09-01 及以后的数据默认不可加载,解锁须留痕 | `load_panel` 强制执行 |

## 研究结果

BATCH02b 裁定(代码版本 `a15a40463312`;2015-01-05 → 2024-08-30,2351 个交易日,5298 只含退市股;零假设 50 次)。完整记录见 [`tasks/BATCH02_FINDINGS.md`](tasks/BATCH02_FINDINGS.md)。

- 十二个因子(含对照与事后变体)全部未通过分层门槛,最终池为空;门槛数值未作调整。
- 非流动性:平静期超额夏普 1.13,本批唯一越过 0.8 绝对线;DSR 0.875(n=28)与危机期 −0.46 未过。十分组市值中性化后降至 −0.02,超额几乎全部来自小市值暴露。
- 特质波动率:平静期 0.71、危机期 1.06、DSR 0.49。相对总波动率的 IC 差 Newey–West t 为 4.87;相对纯低换手,超额差 t 为 1.03,收益层面无法区分。
- 短期反转:IC 逐年为正(0.018–0.054),但成本年化 18.9% 高于毛年化 11.8%,净年化 −7.1%;skip=0 对照同样为负。
- 横盘突破:日均换手 0.73、年化成本 61%、平均持仓 7.7 只;放开"一字涨停不可买"仅改善 19.6 个百分点,仍深度为负。
- 估值(E/P、B/P):平静期 0.28 / 0.31,优于同换手随机组合约 2.3%/年,幅度远低于门槛。
- 零校准多为 8/8 而门槛未过:零校准回答"是否优于同换手的随机选股",门槛回答"扣费后是否足够赚钱",二者须并用。
- 任务卡06(价值 × 低波秩合成,BATCH02b 之后单独运行):等权合成平静期超额夏普 0.46、DSR 0.21(n=34),价值权重 0.7 档为 0.39;均未通过。合成低于低波单独(0.71),E/P 稀释了低波,"两者信息正交"的前提不成立,方向关闭。见 [`tasks/task06_FINDINGS.md`](tasks/task06_FINDINGS.md)。
- 首批结果(BATCH01)因零假设构造错误已撤回,不作引用,见 `tasks/BATCH01_FINDINGS.md` 顶部说明。

**保留段使用记录**:`meta.holdout_unlocked_log` 仅有一条记录——2026-09-20 15:14 UTC 的数据层冒烟(`load_panel`,2024-01-01 → 2026-09-18),当时库内约 30 只股票,未评估任何因子。此后无解锁。任何基于保留段的结论须同时披露此记录。

## 快速开始

```bash
# 1. 环境与离线自检(安装依赖需联网;其余命令不联网)
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests -q                           # 153 项已知答案/回归测试;1 项联网冒烟默认跳过(ASHARE_LIVE=1 才跑)
.venv/bin/python -m ashare_forge.pipeline --list               # 列出已登记因子与申报档(不读库)
.venv/bin/python scripts/build_dashboard.py --selftest         # 用合成台账验证看板渲染(不写文件)

# 2. 数据同步(需联网,baostock;全市场耗时数小时,支持断点续传)
.venv/bin/python -m ashare_forge.data.sync --limit 30          # 小样本走通全流程(股池、上市退市日期、指数、行业、30 只日线与复权)
.venv/bin/python -m ashare_forge.data.sync --universe-only --basic --index-only   # 股池、真实上市退市日期、基准指数
.venv/bin/python scripts/sync_all.py --workers 3 --since 2015-01-01    # 全市场日线并行同步
.venv/bin/python scripts/sync_all.py --workers 3 --retry-failed        # 补拉库内无日线的股票
.venv/bin/python -m ashare_forge.data.sync --adj-only                  # 全历史复权因子
.venv/bin/python -m ashare_forge.data.sync --report                    # 数据质量报告(不联网)

# 3. 研究流水线(需完成全量同步;BATCH02b 全量批跑约 2 小时)
.venv/bin/python -m ashare_forge.pipeline --task all --n-null 50 --top-pct 0.1
.venv/bin/python -m ashare_forge.pipeline --task illiquidity_amihud_w60 # 按名点跑任务卡已申报的稳健性档
.venv/bin/python scripts/build_dashboard.py                             # 生成本地看板 reports/dashboard.html
```

同步、流水线、诊断与看板脚本均接受 `--db` 指定数据库路径,`--help` 列出全部参数。

新增因子:在 `ashare_forge/factors/` 下写一个纯函数并登记 `SPECS`,在 `tasks/` 写任务卡(假设、对立假设、申报参数档数),运行流水线即可。任务卡已申报、只需运行一次的稳健性档登记为 `batch_default: False`,只在按名点跑时执行,不随全量批跑重复入账。

## 目录

```
contracts/          交易规则与门槛合同
ashare_forge/data   同步(baostock)、SQLite 存储、面板加载与可交易掩码
ashare_forge/engine 指标、回测、零假设校准、门槛、泄漏检测、台账与查重
ashare_forge/factors 因子插件
tasks/              任务卡与批次结论
scripts/            并行同步、批次诊断、看板生成
docs/               各期架构说明
tests/              已知答案测试与回归测试
```

## 已知局限

- 冲击成本为固定比例假设(单边 0.1%),未按成交额建模;小市值与低流动性因子的成本可能被低估。
- 基准为中证500 价格指数,不含分红;组合使用后复权价(含分红),超额中含约一个指数股息率。
- 行业数据未同步,行业中性化与行业集中度无法评估。
- 退市清算 haircut=1.0 为保守上界,0 与 0.5 的敏感性尚未单独报告。
- 行为指纹基于合成面板,对依赖真实数据特有结构的换皮仍可能漏判。
- 泄漏检测是强化的必要条件检验,不是穷举证明。
- 危机窗口为事前在合同中写定的四段,共 361 天,其中约 62% 来自 2018 年;危机期结论仅作披露。
- 仅日频;分钟级数据未接入(baostock 5 分钟线只覆盖近期)。

## 数据与来源声明

- 行情、复权因子、股池快照、上市退市日期与指数日线均来自 baostock 公开接口。仓库不含、也不分发任何原始数据;数据库 `data/ashare.db` 由使用者在本地同步生成,已被版本控制忽略。仓库中的裁定记录、诊断表与看板为由其派生的统计结果。
- 任务卡01(横盘吸筹 → 放量突破)的形态参考了公开项目 AB-Screener 的思路:借鉴其三层筛选的结构与参数取值,改写为连续打分的截面因子;未复制其代码(该项目未附 LICENSE)。

## 参考文献

仅列代码与文档中实际使用或引用的方法。

- Acklam, P. J. An algorithm for computing the inverse normal cumulative distribution function. 技术笔记。——`engine/gates.py` 的标准正态分位数近似。
- Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31–56. ——任务卡04 非流动性因子定义。
- Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). The cross-section of volatility and expected returns. *Journal of Finance*, 61(1), 259–299. ——任务卡03 特质波动率口径(本平台只扣市场单因子)。
- Bailey, D. H., & López de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk*, 15(2), 3–44. ——概率夏普比(PSR)。
- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management*, 40(5), 94–107. ——缩水夏普比(DSR)与 N 次试验的期望最大夏普。
- Basu, S. (1977). Investment performance of common stocks in relation to their price-earnings ratios: a test of the efficient market hypothesis. *Journal of Finance*, 32(3), 663–682. ——任务卡05 方向申报依据。
- Nelson, C. R., & Kang, H. (1981). Spurious periodicity in inappropriately detrended time series. *Econometrica*, 49(3), 741–751. ——说明去趋势序列上固定阈值判据的伪显著。
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708. ——批次诊断中逐日 IC 配对差的 t 统计量。
- Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39. ——零假设校准的 p 值口径。

## 协作声明

研究问题、假设、判据与裁定由作者设计和确定;代码在 AI 编码助手协助下实现,提交记录保留协作者标注。

本仓库为公开发布整理版,以单次提交发布。开发仓库的历史中曾误纳入本地数据库的临时文件,为不分发数据而未直接公开该历史;文中形如 `a15a404` 的提交号与台账中的 `git_commit` 均指开发仓库内的版本。

## 许可

MIT,见 [`LICENSE`](LICENSE)。
