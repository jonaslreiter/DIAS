# DIAS

**DIAS** — _Differentiable Induction of Action Schemas._
Code accompanying our NeurIPS 2026 paper.

> **Status:** initial public release. Single-domain training and the
> per-checkpoint MLP / KR2024 baseline validation are supported.
> Multi-seed sweep reproduction is deferred — see [`TODO.md`](TODO.md).

## Installation

```bash
git clone <this-repo>
cd DIAS
pip install -e .
```

This installs the `dias` package (training/model/data code) and the
`dias-train` console entry point.

The 13 PDDL domains used in the paper are bundled under `pddl_files/`.
Additional instances can be regenerated with the upstream generators:
<https://github.com/...> <!-- TODO: link upstream pddl-generators -->

## Quick start

Train DIAS on a single domain/problem:

```bash
dias-train fit --config configs/main.yaml
```

The base config points at one `(domain_path, problem_path)` pair under
`pddl_files/`. Edit those two fields in
`configs/main.yaml` (or override on the CLI) to switch
domain. Training logs and checkpoints land under `logs/`.

### Noise ablation

Edge add/remove noise is exposed as two parameters in the data section
of the config:

```yaml
data:
  init_args:
    edge_remove_ratio: 0.1 #  paper sweeps 0.0–0.5
    edge_add_ratio: 0.1
```

Set both to `0.0` for the clean setup.

## Baseline validation

The shipped runners validate a single checkpoint (or learned PDDL
domain) on a single PDDL problem. Multi-seed aggregation lives in the
deferred sweep pipeline (see [`TODO.md`](TODO.md)).

```bash
# DIAS model on a problem
python scripts/validation/run_dias.py \
    --model-id <wandb-run-id> \
    --problem  pddl_files/satellite/train.pddl

# MLP baseline
python scripts/validation/run_mlp_baseline.py \
    --model-id <wandb-run-id> \
    --problem  pddl_files/satellite/train.pddl

# KR2024 baseline (compares a *learned* PDDL domain against the ground truth)
python scripts/validation/run_kr2024.py \
    --original-domain pddl_files/satellite/domain.pddl \
    --learned-domain  path/to/kr2024_learned/satellite.pddl \
    --problem         pddl_files/satellite/train.pddl
```

## Full paper sweep: all domains × all seeds

`scripts/run_all_domains.py` runs train + validate for every
`(domain, seed)` pair in `pddl_files/domain-settings.yaml`. Per-domain
training uses `problems[0]`; validation runs on the next three problems
(configurable via `configs/validation.yaml`). Each pair becomes one W&B
run; all pairs share a W&B group for easy aggregation. Validation
results are also written to
`logs/validation/<group>/<domain>/seed<seed>.json` for offline use.

```bash
# Locally — full 130-run grid (13 domains × 10 seeds), sequential
python scripts/run_all_domains.py --group paper-main-2026

# Just two domains and three seeds
python scripts/run_all_domains.py \
    --domains satellite hanoi \
    --seeds 0 1 2 \
    --group debug

# Slurm array — 130 parallel tasks, one per (domain, seed)
DIAS_GROUP=paper-main-2026 sbatch slurm/run_all_domains.sh
```

## Slurm (single-domain)

`slurm/train.sh` is a minimal cluster-agnostic template for one training
run — set `--partition` / `--account` / module loads for your site and
submit:

```bash
sbatch slurm/train.sh
```

## Repo layout

```
src/dias/
├── main.py              # LightningCLI entry (also `dias-train`)
├── models/              # DIAS model + components
├── data/                # PDDL datamodule + sample generation
├── utils/               # schedules, noise, adjacency↔PyG helpers
└── validation/          # ModelValidation classes (DIAS / MLP / KR2024)

configs/
└── main.yaml   # single-domain training config

scripts/
├── run_all_domains.py   # full sweep: train + validate for every (domain, seed)
└── validation/
    ├── run_dias.py
    ├── run_mlp_baseline.py
    └── run_kr2024.py

configs/
├── main.yaml   # single-domain training base
└── validation.yaml           # validation knobs + seed list

pddl_files/              # 13 paper domains + domain-settings.yaml
slurm/
├── train.sh             # single-domain sbatch template
└── run_all_domains.sh   # slurm array (13×10) over the full grid
```

## Citation

<!-- TODO: BibTeX once the camera-ready is out -->

## License

See [`LICENSE`](LICENSE).
