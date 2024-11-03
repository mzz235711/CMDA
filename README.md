# CMDA
Contrastive learning and Meta-learning based framework for few-shot workload Drift Adaptation

## Setup
You can refer to [MSCN](https://github.com/andreaskipf/learnedcardinalities) and [Bao](https://github.com/learnedsystems/BaoForPostgreSQL) for environmental setup. Notice that the environment requirements are different in MSCN and Bao, so you may need to create to environments.

## Running
Run MSCN
```
cd mscn
bash run.sh
```
You can modify `--dataset` as `job-light-ranges` or `stats` in `run.sh` to run different datasets.

Run Bao
```
cd BaoForPostgreSQL/bao_server
bash run.sh
```
You can modify `--dataset` as `job-light-ranges` or `stats` in `run.sh` to run different datasets.
