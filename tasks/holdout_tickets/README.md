# 保留段票据

本目录存放保留段使用票据(`<ticket_id>.yaml`)及其结果文件(`<ticket_id>.result.json`)。协议见 `docs/HOLDOUT_PROTOCOL.md`,执行器为 `scripts/holdout_eval.py`。

截至本说明提交时,本目录下没有任何票据,保留段未被正式使用。

## 规则

1. 只有样本内全部门槛通过的候选才能写入票据:台账 `verdict = pass`,门槛全过,泄漏与零校准通过,无出处漂移。
2. 票据须先提交进 git,再执行;已提交的票据不得修改。
3. 以同一 cutoff 为界,只能有一张票据执行完成。多个候选须写进同一张票据。
4. 执行中止后,只能以新票据的 `supersedes` 字段接续,不得复用原票据。
5. 结果文件由执行器生成,须原样提交,不得编辑。

## 流程

**第 1 步:确认资格。** 台账与行情同库,以只读方式查询:

```bash
sqlite3 -readonly data/ashare.db \
  "SELECT id, name, verdict, sr_calm, code_sha, git_commit FROM forge_trials WHERE verdict='pass';"
```

**第 2 步:收集票据字段。**

```bash
# 因子模块 sha(须与台账 code_sha 相同)
.venv/bin/python -c "from ashare_forge import pipeline; from ashare_forge.engine import registry; \
s=[x for x in pipeline.discover_specs(include_optional=True) if x['name']=='<因子名>'][0]; \
print(registry.factor_code_sha(s['build_factor'])); print(s.get('params'))"
# 合同 sha256
shasum -a 256 contracts/ashare.yaml
# 平静期天数与试验数(台账证据包)
sqlite3 -readonly data/ashare.db "SELECT json_extract(provenance_json,'$.gates') FROM forge_trials WHERE id=<id>;"
```

**第 3 步:写票据并 dry-run。** 此时票据尚未提交,`ticket_committed` 与 `worktree_clean` 两项应显示未过,其余各项应通过。

```bash
.venv/bin/python scripts/holdout_eval.py --ticket tasks/holdout_tickets/<id>.yaml --db data/ashare.db --dry-run
```

**第 4 步:提交票据,再次 dry-run。** 全部检查应通过。

**第 5 步:真实执行。** 只执行一次:

```bash
.venv/bin/python scripts/holdout_eval.py --ticket tasks/holdout_tickets/<id>.yaml --db data/ashare.db --confirm <id>
```

**第 6 步:提交结果文件。**

## 模板

```yaml
schema: ashare_holdout_ticket_v1
ticket_id: HT001_example              # 须与文件名一致;字母数字、下划线、连字符
created_utc: "2026-01-01T00:00:00Z"
supersedes: null                      # 仅在接续中止的使用时填写其 ticket_id
code:
  git_commit: "0123456789ab"          # 样本内运行所用提交(= 台账 git_commit);须为 HEAD 或其祖先
  contract_sha256: "<64 位十六进制>"   # shasum -a 256 contracts/ashare.yaml
evaluation:
  cutoff: "2024-09-01"                # 须等于合同 holdout.cutoff
  end: null                           # null = 执行时的数据末尾;也可写定具体日期
  warmup_trading_days: 250            # 预热交易日数;须大于因子最长回看窗
  warmup_min_coverage: 0.5            # cutoff 前末日在池股票的因子覆盖率下限
  top_pct: 0.1                        # 须与样本内运行一致(台账未记录此项)
  bench_code: sh.000905               # 须与台账证据包的 bench_code 一致
  ci_level: 0.90                      # 报告区间水平
  decay_alpha: 0.10                   # 单侧衰减检验水平,决定门限 tau
  expected_holdout_days: 497          # 可选;仅用于 dry-run 预估门限与判别力
  segments:                           # 省略则用协议第 9 节的缺省分段;起点不得早于 cutoff
    - {name: pre_policy, start: "2024-09-01", end: "2024-09-23"}
    - {name: policy_rally, start: "2024-09-24", end: "2024-10-08"}
    - {name: post_rally, start: "2024-10-09", end: null}
candidates:
  - factor_name: <因子名>              # discover_specs 中的 name
    task_id: <任务卡号>                # 可选,仅作记录
    module: ashare_forge.factors.<模块> # 可选;填写则须与 spec 的 module 一致
    params: {}                        # 须与 spec 的 params 及台账参数逐项相等
    factor_module_sha: "<16 位十六进制>"
    in_sample:
      trial_id: 0                     # forge_trials.id
      calm_excess_sharpe: 0.0         # = 台账 sr_calm
      calm_n_days: 0                  # = 证据包 gates.rows 中 calm/n_days
notes: null
```

出现模板以外的字段即判票据格式不合格。拼错的字段会静默取缺省值,因此一律拒绝。
