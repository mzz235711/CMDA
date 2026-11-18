# Cost estimation 多随机种子实验

## 设计

- adaptation candidate pool：`adaptation_all_*_plans`。
- JOB-light 和 STATS 各自使用固定、独立的 prediction workload 评估，不从中抽 adaptation query。
- 每个 dataset / budget / seed 只生成一份共享 selection manifest；Adapt、CMDA-C、CMDA-M、CMDA 必须读取相同的 plan IDs。
- plan 与 label 按 `sql<ID>_plan.json` 的显式 ID 对齐。历史数据中仅位于末尾的多余 label 会记录在 manifest；缺失 label、缺失 plan ID 或重复 ID 会终止实验。
- 每次运行保存 manifest、逐查询 prediction、q-error metrics 和模型。

## 运行

在 `BaoForPostgreSQL/bao_server` 下执行：

```bash
python run_cost_multiseed.py \
  --datasets job-light-ranges,stats \
  --methods adapt,cmda_c,cmda_m,cmda \
  --seeds 0-19 \
  --adaptation_budget 10 \
  --epochs 100 \
  --batch_size 256 \
  --repeat 1 \
  --cpu_threads 1
```

仅生成并核查 selection manifests：

```bash
python run_cost_multiseed.py --seeds 0-2 --prepare_only
```

脚本默认按顺序运行并跳过已有 `metrics.json` 的任务，因此中断后可直接续跑。当前环境强制使用 CPU，以避免与 cardinality 长任务争抢 GPU。

## 统计分析

在仓库根目录执行：

```bash
python mscn/analyze_multiseed.py \
  --results_root ../results/cost_multiseed \
  --datasets job-light-ranges,stats \
  --expected_seeds 20
```

输出位于 `results/cost_multiseed/analysis/`：

- `per_seed_metrics.csv`
- `summary.csv`（mean、sample standard deviation、95% t confidence interval）
- `paired_wilcoxon_holm.csv`（配对 Wilcoxon、Holm 校正和 rank-biserial effect size）
