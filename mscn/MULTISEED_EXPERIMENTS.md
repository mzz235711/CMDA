# Cardinality Estimation 多随机种子实验

## 实验协议

`run.py` 在设置 `--seed` 后采用以下协议：

1. 使用固定的 `split_seed` 将完整 target workload 划分为 adaptation candidate pool 和 evaluation set。
2. 默认将 20% target queries 放入 candidate pool，其余 80% 作为固定 evaluation set。
3. 每个实验 seed 只从 candidate pool 无放回抽取 `adaptation_budget` 条查询。
4. 同一 dataset、budget 和 seed 下的所有方法读取同一份 selection manifest。
5. 所有 seeds 和方法始终在相同的 evaluation set 上评估。
6. 每个方法、seed 的模型、预测、指标和日志分别保存，互不覆盖。

固定 split 默认参数：

```text
target_candidate_fraction = 0.2
split_seed = 20260923
adaptation_budget = 10
```

当前数据对应的固定划分为：

| Dataset | Target pool | Candidate pool | Evaluation set |
|---|---:|---:|---:|
| JOB-light | 3,897 | 779 | 3,118 |
| STATS | 1,145 | 229 | 916 |

## 方法名称

批量运行器目前支持：

| 名称 | Contrastive | Meta-reweighting |
|---|---|---|
| `adapt` | 否 | 否 |
| `cmda_c` | 是 | 否 |
| `cmda_m` | 否 | 是 |
| `cmda` | 是 | 是 |

## 正式运行

在 `mscn` 目录运行：

```bash
python3 run_multiseed.py \
  --datasets job-light-ranges,stats \
  --methods adapt,cmda_c,cmda_m,cmda \
  --seeds 0-19 \
  --output_root ../results/cardinality_multiseed \
  --adaptation_budget 10 \
  --target_candidate_fraction 0.2 \
  --split_seed 20260923 \
  --epochs 50 \
  --batch_size 512 \
  --device cuda \
  --num_samples 1000 \
  --alpha 0.8 \
  --lr 0.001 \
  --repeat 1
```

运行器默认顺序执行，防止多个训练进程争抢同一 GPU。已经存在完整 `metrics.json` 的运行会自动跳过；使用 `--force` 可强制重跑。

在 CPU 上测试时建议添加：

```bash
--device cpu --cpu_threads 1
```

只生成并检查 split/selection manifests，不训练模型：

```bash
python3 run_multiseed.py \
  --datasets job-light-ranges,stats \
  --methods adapt,cmda_c,cmda_m,cmda \
  --seeds 0-19 \
  --output_root ../results/cardinality_multiseed \
  --prepare_only \
  --device cpu
```

## 输出目录

```text
results/cardinality_multiseed/
├── experiment_plan.json
├── job-light-ranges/
│   ├── target_splits/
│   └── budget_10/
│       ├── selections/
│       ├── logs/
│       ├── adapt/seed_000/
│       ├── cmda_c/seed_000/
│       ├── cmda_m/seed_000/
│       └── cmda/seed_000/
└── stats/
```

每个方法和 seed 的目录包含：

- `manifest.json`：数据划分、adaptation IDs、超参数和 selection hash；
- `metrics.json`：workload-level q-error 指标；
- `predictions.csv`：每条 evaluation query 的预测、真实值和 q-error；
- 模型参数和编码结构。

## 统计分析

正式实验全部完成后运行：

```bash
python3 analyze_multiseed.py \
  --results_root ../results/cardinality_multiseed \
  --datasets job-light-ranges,stats \
  --methods adapt,cmda_c,cmda_m,cmda \
  --comparison_pairs adapt:cmda_c,adapt:cmda_m,adapt:cmda,cmda_c:cmda,cmda_m:cmda \
  --adaptation_budget 10 \
  --expected_seeds 20
```

分析脚本首先验证：

- 每种方法是否恰好包含预期数量的 seeds；
- 同一 seed 下所有方法是否使用相同 adaptation selection；
- 所有 seeds 是否使用同一 evaluation workload；
- 同一 seed 下的方法结果是否可以进行配对比较。

通过验证后生成：

- `analysis/per_seed_metrics.csv`：每个 seed 的原始 workload-level 指标；
- `analysis/summary.csv`：均值、样本标准差和 95% t confidence interval；
- `analysis/paired_wilcoxon_holm.csv`：双侧配对 Wilcoxon、Holm 校正后的 p-value、median relative improvement 和 matched-pairs rank-biserial effect size。

统计检验的实验单元是一次 target adaptation sample/seed，而不是单条 evaluation query，避免将同一模型下的查询错误地当成独立重复。
