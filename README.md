# CMDA
Contrastive learning and Meta-learning based framework for few-shot workload Drift Adaptation

## Repository layout and reproducibility

- `mscn/`: cardinality estimation, seeded adaptation runners and controls.
- `BaoForPostgreSQL/`: plan-level cost estimation and PostgreSQL extension.
- JOB-light/STATS prepared workload inputs remain in each component's `workloads/` directory. Raw databases, trained models, environments and generated experiment results are excluded from new commits.
- Preserve the seed, adaptation query IDs, evaluation-set hash, configuration and predictions for each experiment. Prepared labels depend on the original database snapshot.

Use `python run_multiseed.py --help` from `mscn/`, or `python run_cost_multiseed.py --help` from `BaoForPostgreSQL/bao_server/`, for the repeated-experiment interfaces. Start with a single seed and one epoch in a separate output directory before formal runs.

Some optional JOB/DSB and external-baseline scripts still reference local data locations; configure these before using them on another machine. The historical ranking implementation retains self comparisons and non-strict distance ordering, which must be reflected when describing the executed objective.

For updating the existing GitHub remote, see [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md). The local `github_release/` folder is an ignored standalone export and is not required for updating this repository. Upstream licenses remain applicable; no new project-wide license is assumed.

## Setup
### One-command CPU smoke test

From a fresh clone, install Python 3.10 and its `venv` support, then run:

```bash
bash run_smoke.sh --setup
```

This creates `.venv`, installs `requirements.txt`, and runs one epoch of Adapt and CMDA on the bundled JOB-light inputs for both cardinality and cost estimation. Outputs go to `results/clone_smoke/`. Internet access is required for dependency installation; a GPU, running PostgreSQL, and raw database tables are not required for this prepared-data test. Completed runs are skipped on subsequent invocations.

For an existing compatible environment:

```bash
CMDA_PYTHON=/absolute/path/to/python bash run_smoke.sh
```

### Optional paths

Copy `config.env.example` to a local configuration file, edit it and source it before running optional experiments. `CMDA_JOB_DATA`, `CMDA_DSB_QUERIES`, and `CMDA_DSB_BITMAPS` configure optional MSCN data locations. `CMDA_SHIFTHANDLER_ROOT` configures the external baseline checkout; `--shift_handler_root` overrides it. Relative defaults refer to this repository's prepared workload layout. Historical collection scripts under `scripts/` are advanced workflows and may require additional external artifacts; they are not invoked by the smoke entry point.

You can refer to [MSCN](https://github.com/andreaskipf/learnedcardinalities) and [Bao](https://github.com/learnedsystems/BaoForPostgreSQL) for environmental setup. Notice that the environment requirements are different in MSCN and Bao, so you may need to create two environments.

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
