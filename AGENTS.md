# DIAS Agent Instructions

This file is the permanent repository-level guidance for AI coding agents (GitHub Copilot, Claude Code, and others) working in this workspace. Keep it up to date when the codebase, pipeline, naming, or experiment conventions change.

## Purpose

DIAS means Differentiable Induction of Action Schemas. The repository implements the training, validation, and experiment pipeline for learning lifted STRIPS-like action schemas from PDDL state transitions, matching the paper "Differentiable Learning of Lifted Action Schemas for Classical Planning".

The core problem setting is:

- Inputs are state-transition triplets `<s, a, s'>` sampled from a single planning problem.
- States are symbolic PDDL states represented through unary and binary predicates.
- Depending on the experiment, action labels may be full STRIPS actions, STRIPS+ actions, or action names only.
- The goal is to learn lifted preconditions and lifted add/delete effects for each action schema.

## Repository Truths

- The installable package is `dias`, defined via `pyproject.toml` with console entry point `dias-train = dias.main:main`.
- Main source code lives under `src/dias/`.
- PDDL assets live under `pddl/`.
- The per-domain manifest is `pddl/domain-settings.yaml` (paths + problem list only; it no longer holds `data_overrides`).
- Per-domain training overrides live in `pddl/<domain>/config.yaml` (a Lightning partial config composed after `configs/main.yaml`).
- Per-experiment overlays live in `configs/experiments/{mlp,noise}.yaml`.
- Validation sweep settings live in `configs/validation.yaml`.
- Per-domain Slurm resources live in `configs/slurm-resources.yaml` (training array) and `configs/slurm-resources-validation.yaml` (validation array), both consumed by `scripts/launch.py`. Same schema (`defaults` + per-domain overrides), resolved independently; validation runs on CPU (DIAS/noise validators are CPU-only, the MLP baseline falls back to CPU), so it requests no GPU and targets all partitions.
- Training GPU requests are per domain and live entirely in `configs/slurm-resources.yaml`; treat that file as the source of truth rather than hardcoding a partition or GPU size here. It currently requests GPU shards (`gres: shard:N`) sized per domain from a shared `defaults` block, with no partition pinning. A domain that needs a whole GPU or a specific partition gets it via its own entry (`gres: "gpu:1"` and/or `sbatch_extra: ["--partition=<name>"]`).
- The current single-domain base training config in this workspace is `configs/main.yaml`.
- Standardized filenames per domain folder are `domain.pddl`, `train.pddl`, `test-1.pddl`, `test-2.pddl`, `test-3.pddl`, `strips+.yaml`, and `config.yaml`.

## High-Level Pipeline

The end-to-end DIAS pipeline is:

1. Load one PDDL domain and one problem instance.
2. Build symbolic transition samples from random-walk or BFS-based state exploration.
3. Convert symbolic states into adjacency-tensor and PyG graph representations.
4. Train the DIAS model with Lightning using stratified or random batching.
5. Recover learned lifted action schemas from trained model parameters.
6. Validate learned schemas by comparing reachable successor states against the original domain on held-out problems.
7. Run this per `(domain, seed)` pair for sweep experiments and log results to W&B plus local JSON summaries.

## Training Data Semantics

When working on data generation or dataset code, preserve these assumptions unless a task explicitly changes them:

- Only unary and binary predicates are supported.
- Action binding is assumed injective.
- Pure STRIPS-style action schemas are the default target.
- Domain predicates are assumed known.
- Training samples come from a single problem instance per domain.
- For full sweep experiments, training uses `problems[0]` from `pddl/domain-settings.yaml`.
- Validation uses later problems according to `configs/validation.yaml`, currently offset from the manifest by `problem_offset` and limited by `num_problems`.

## Core Code Map

- `src/dias/main.py`: Lightning CLI entry point for training.
- `src/dias/data/datamodule.py`: PDDL loading, metadata extraction, sampling constraints, dataset construction, batching strategy, and optional adjacency noise (two `noise_model`s: `ratio` and `expected_count`).
- `src/dias/data/generate_samples.py`: symbolic sample generation utilities.
- `src/dias/models/dias.py`: main DIAS architecture.
- `src/dias/models/gnn.py`: graph encoder components.
- `src/dias/models/prediction.py`: `PredictionModule` — Sinkhorn slot selection plus learnable precondition/effect tensors. With `state_dependent_effects=True` the effect tensor is replaced by an MLP that predicts effects from the current state (the MLP baseline; consumed by `validation/mlp_baseline.py`).
- `src/dias/validation/dias_validate.py`: reconstruct learned schemas from checkpoints and compare successor-state behavior.
- `src/dias/validation/l1.py`: L1 validation logic.
- `src/dias/validation/mlp_baseline.py`: MLP-effects baseline validation.
- `src/dias/validation/transpiler.py`: transpilation from learned tensors to PDDL-like action schema output.
- `scripts/launch.py`: the single experiment launcher. Per domain it submits a Slurm training array (one task per seed) plus a `--dependency=aftercorr` validation array, or runs the same steps sequentially with `--backend local`. Selects the experiment via `--experiment-type {dias,mlp,noise}`.
- `scripts/validate_runs.py`: compute-side per-`(experiment, domain, seed)` validation; picks the validator by experiment type and writes the per-run JSON. Invoked by `scripts/slurm/validate.sh` and by `launch.py --backend local`.
- `scripts/collect_results.py`: aggregates per-run results (local JSON or the W&B API) into a `P / R / N_S / N_C / N_SC` table per domain.
- `scripts/validation/run_dias.py`, `run_l1.py`, `run_mlp_baseline.py`: single-checkpoint validation entry points.

## Training Conventions

- Prefer the installed CLI entry point `dias-train fit --config configs/main.yaml` for single-domain training unless a script explicitly uses `python -m dias.main`.
- Per-domain runs compose configs: `--config configs/main.yaml --config pddl/<domain>/config.yaml` (plus `--config configs/experiments/<type>.yaml` for MLP/noise). `scripts/launch.py` builds this chain and injects `domain_path` / `problem_path` / seed.
- The default logger is W&B with entity `team-jonas-reiter`, project name `dias`, and save directory `logs`.
- Sweep runs use a deterministic W&B run id `<experiment>-<domain>-s<seed>` (see `validate_runs.run_id_for`) so the validation array can locate each checkpoint without an id handoff.
- Training checkpoints are expected under `logs/dias/<run_id>/checkpoints/`.
- Validation code looks for a checkpoint matching `best-epoch*.ckpt`; preserve this naming unless the validation loader is updated in the same change.
- `src/dias/main.py` links datamodule metadata such as numbers of objects, actions, and relations into model init args through LightningCLI. Keep these interfaces synchronized.
- The default batching strategy in `configs/main.yaml` is `stratified`, which yields an effectively unbounded iterable dataset; training length is therefore controlled by `trainer.max_steps` and related trainer limits, not dataset exhaustion.
- Noise ablations: `noise_model: ratio` uses `edge_remove_ratio` / `edge_add_ratio`; `noise_model: expected_count` uses `expected_removed_atoms` / `expected_added_atoms` (Poisson flip budget, `noise_on_fluents_only`). The `noise` experiment overlay selects `expected_count`.
- The MLP baseline is trained by adding `model.init_args.state_dependent_effects: true` (the `configs/experiments/mlp.yaml` overlay).
- Model ablation switches live on `model.init_args` in `configs/main.yaml`; all default to the full model. Each toggles one component:
  - `use_slot_weights: false` — forces all M slots active (no learned slot activation); also fixes the slot weights to 1 in the precondition aux term.
  - `use_precondition_loss: false` — drops L_pre (the restrictive-precondition aux term).
  - `use_effect_loss: false` — drops L_eff (the minimal-effects aux term).
  - `use_pcgrad: false` — replaces PCGrad projection with a plain weighted sum `L_adj + α·L_aux` (same α).
  - `random_node_embeddings: false` — starts every object node from an all-zero embedding instead of seeding the first half of the feature dimension with `0.1 * randn`, so symmetric objects stay indistinguishable to the GNN.
  - Ready-made overlays live in `configs/experiments/ablation-*.yaml` (composed last on top of `main.yaml`): `ablation-no-slot-weights`, `ablation-no-precondition-loss`, `ablation-no-effect-loss`, `ablation-no-aux-loss` (drops both aux terms, i.e. α = 0), `ablation-no-pcgrad`, and `ablation-no-random-embeddings`.

## Validation Conventions

- Validation is behavior-based, not string-based. The important metric is whether the learned domain produces the same successor states as the original domain.
- For DIAS checkpoints, `ModelValidation` reconstructs learned lifted schemas from model logits and merges them with transformed original actions into temporary untyped PDDL files.
- Validation samples states from held-out problems and compares original versus learned reachable successor sets.
- Sweep validation writes local results to `logs/validation/<experiment>/<domain>/seed<seed>.json` and mirrors numeric metrics to the resumed W&B run under `validation/<problem_stem>/...`.
- If checkpoint layout, run directory layout, or logger naming changes, update validation loaders and sweep scripts together.

## Sweep Conventions

- `scripts/launch.py` is the authoritative sweep launcher. Per domain it submits a training array (`scripts/slurm/train.sh`, one task per seed) and a dependent validation array (`scripts/slurm/validate.sh`, `--dependency=aftercorr`). `--backend local` runs the same steps sequentially.
- One run corresponds to one `(domain, seed)` pair; seeds default to `0 .. num_runs-1` (`--num-runs`, default = `len(configs/validation.yaml seeds)`).
- All runs of one launch share an `--experiment` name (= W&B group = results directory). `--experiment-type {dias,mlp,noise}` picks the overlay and validator.
- `--ablation {no-slot-weights,no-precondition-loss,no-effect-loss,no-aux-loss,no-pcgrad,no-random-embeddings}` composes one `configs/experiments/ablation-*.yaml` overlay on top of the experiment type (orthogonal axis; always uses the DIAS validator). Default: none (full model).
- Domain-specific sample bounds come from `pddl/<domain>/config.yaml`; paths/problem lists from `pddl/domain-settings.yaml`; per-domain Slurm resources from `configs/slurm-resources.yaml`.
- Global validation settings such as number of validation problems and sampling counts come from `configs/validation.yaml`.
- `scripts/collect_results.py` aggregates a finished experiment (local JSON or W&B) into the paper table.
- The Slurm scripts under `scripts/slurm/` first `source ~/.zshrc` then `conda activate ddl`; they stay thin and defer experiment semantics to Python scripts and YAML configs.

## Paper-Level Modeling Notes

Use these as conceptual ground truth when reasoning about intended behavior:

- If the summary in this file is not enough and the workspace contains `local/dias-paper.pdf`, consult that local paper copy for additional modeling or experimental detail.
- DIAS learns lifted positive preconditions, lifted negative preconditions, add effects, and delete effects.
- Object selection is performed via a GNN plus query-slot matching with a rectangular Sinkhorn-style assignment.
- Slot activations determine learned action arity.
- Successor-state prediction is trained from observed state transitions.
- Auxiliary losses encourage minimal effects and restrictive preconditions, while PCGrad-style handling prevents auxiliary objectives from overriding the main prediction objective.
- Evaluation is based on soundness and completeness of induced successor-state behavior on held-out problems.

## Editing Rules For This Repository

- Prefer minimal, behavior-preserving changes.
- Preserve the distinction between single-domain training, per-checkpoint validation, and full sweep orchestration.
- Do not silently change experiment semantics stored in YAML manifests or config defaults without updating the relevant documentation and runners.
- Treat `pddl/domain-settings.yaml` as the source of truth for per-domain paths and problem lists, and `pddl/<domain>/config.yaml` as the source of truth for per-domain sampling/model overrides.
- Keep standardized filenames in `pddl/` intact unless there is an explicit migration.
- When changing path conventions, update code, configs, docs, and validation scripts in the same task.
- When changing checkpoint naming, W&B directory layout, or logger config, verify all validation entry points still resolve checkpoints correctly.
- When changing datamodule arguments, check both CLI config compatibility and checkpoint reload compatibility.
- Preserve compatibility with older checkpoints where practical; this repository already contains legacy-hparam filtering in validation code for that reason.

## Documentation Rules

- When documenting commands, prefer the commands that match the current workspace state, especially `configs/main.yaml` and `configs/validation.yaml`.
- If you encounter stale references to `configs/main.yaml`, either update them or clearly mark them as stale.
- Keep explanations aligned with the paper: symbolic states are fully observed in this repository pipeline, while action supervision varies by experiment.
- Prefer precise planning terminology: predicates, lifted action schemas, preconditions, add effects, delete effects, successor states, STRIPS, and STRIPS+.

## Maintenance Policy

Whenever working in this repository, update this file if any of the following change:

- main entry points
- config filenames or locations
- logging or checkpoint layout
- sweep procedure
- validation procedure
- PDDL dataset layout
- terminology or modeling assumptions that future code changes should respect
- file or folder or parameter names

If there is a conflict between this file and the code, fix the file after confirming the code path that is actually used.
