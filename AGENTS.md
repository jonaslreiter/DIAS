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
- `submodules/pddl-generators` is the upstream instance generator collection (<https://github.com/AI-Planning/pddl-generators>), added as a git submodule. Nothing in the main pipeline needs it; it exists for the scaling study (see below), which commits the instances it generates.
- `scaling/` is a self-contained one-off study of runtime and GPU memory versus instance size. It is intentionally *not* wired into `scripts/launch.py`; see "Instance-Size Scaling Study".

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
- `scripts/validate_runs.py`: compute-side per-`(experiment, domain, seed)` validation; picks the validator by experiment type and writes the per-run JSON. Invoked by `scripts/slurm/validate.sh` and by `launch.py --backend local`. `--threshold` / `--results-experiment` support post-hoc re-validation (see below).
- `scripts/launch_threshold_validation.py`: re-validates the checkpoints of a finished experiment at several binarization thresholds, into sibling `logs/validation/<experiment>-thr<T>/` directories. Validation only — no training.
- `scripts/collect_results.py`: aggregates per-run results (local JSON or the W&B API) into a `P / R / N_S / N_C / N_SC` table per domain; with `--sweep AXIS` into a domain × swept-value matrix per metric, with `--threshold-sweep` into a domain × threshold matrix, optionally written out as markdown tables (`--markdown`). Without a sweep, `--experiment` accepts several unrelated experiment names (one table each).
- `scripts/validation/run_dias.py`, `run_l1.py`, `run_mlp_baseline.py`: single-checkpoint validation entry points.
- `scaling/`: the one-off instance-size scaling study (`generate_instances.py`, `benchmark.py`, `run_scaling.py`, `launch_scaling.py`, `collect_scaling.py`, `configs/{sizes,scaling}.yaml`, `slurm/benchmark.sh`). Separate from the main pipeline on purpose — see "Instance-Size Scaling Study".

## Training Conventions

- Prefer the installed CLI entry point `dias-train fit --config configs/main.yaml` for single-domain training unless a script explicitly uses `python -m dias.main`.
- Per-domain runs compose configs: `--config configs/main.yaml --config pddl/<domain>/config.yaml` (plus `--config configs/experiments/<type>.yaml` for MLP/noise). `scripts/launch.py` builds this chain and injects `domain_path` / `problem_path` / seed.
- The default logger is W&B with entity `team-jonas-reiter`, project name `dias`, and save directory `logs`.
- Sweep runs use a deterministic W&B run id `<experiment>-<domain>-s<seed>` (see `validate_runs.run_id_for`) so the validation array can locate each checkpoint without an id handoff.
- Training checkpoints are expected under `logs/dias/<run_id>/checkpoints/`.
- Validation code looks for a checkpoint matching `best-epoch*.ckpt`; preserve this naming unless the validation loader is updated in the same change. When a run id is reused (a re-run of one seed), several such files can coexist because Lightning suffixes the clash with `-v1`, so `validation/utils.resolve_best_checkpoint` picks the newest by mtime; both `dias_validate.py` and `mlp_baseline.py` go through it.
- `src/dias/main.py` links datamodule metadata such as numbers of objects, actions, and relations into model init args through LightningCLI. Keep these interfaces synchronized.
- The default batching strategy in `configs/main.yaml` is `stratified`, which yields an effectively unbounded iterable dataset; training length is therefore controlled by `trainer.max_steps` and related trainer limits, not dataset exhaustion.
- Noise ablations: `noise_model: ratio` uses `edge_remove_ratio` / `edge_add_ratio`; `noise_model: expected_count` uses `expected_removed_atoms` / `expected_added_atoms` (Poisson flip budget, `noise_on_fluents_only`). The `noise` experiment overlay selects `expected_count`.
- `configs/experiments/noise.yaml` also carries the paper's noise-experiment training adjustments (Appendix A.4.2): `min_per_label_train_floor: 200` / `min_per_label_valid_floor: 100`, `batch_size: 50`, and a floor of 0.1 on the precondition-aggregation parameter τ. The matching schema-derivation threshold of 0.75 is validation-side (see Validation Conventions).
- τ (`gmean_temperature`) is the root degree of the geometric mean that stands in for the true precondition product; `model.init_args.gmean_offset` is *added* to the scheduled value, so it acts as a floor under τ. `main.yaml` schedules 1.0 → 0.1 (exponential, T = 500, decaying on toward 0) with offset 0; the noise overlay scales the schedule to 0.9 → 0.09 and sets `gmean_offset: 0.1`, so the effective τ still starts at 1.0 and decays at the same rate but bottoms out at 0.1.
- `min_per_label_train_floor` / `min_per_label_valid_floor` raise (never lower) the per-domain `min_per_label_*`. Experiment overlays are composed *after* `pddl/<domain>/config.yaml`, so a plain `min_per_label_train` there would also shrink the domains that sample more; the floors express "raise the smallest per-action minimum" instead. They are applied in `PddlGraphDataModule.__init__` before sampling-config validation, so `hparams` keep the raw per-domain values.
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
- Reading a schema off a checkpoint binarizes the slot activations and the add/delete/positive/negative tensors at a single threshold, `validation.utils.process_model_logits(threshold=0.5)`. 0.5 is what the paper reports for the main experiments; `validate_runs.py --threshold` overrides it (DIAS validator only — the MLP baseline has no schema to binarize and rejects a non-default value). The threshold applies uniformly: lowering it activates more slots (higher learned arity) *and* adds both effects and preconditions.
- The threshold of a launch is chosen per experiment type in `launch.SWEEP_AXES`' neighbour `VALIDATION_THRESHOLD` (`DEFAULT_VALIDATION_THRESHOLD = 0.5`, `noise` = 0.75 as in the paper), overridable with `launch.py --validation-threshold T`. It is exported to `scripts/slurm/validate.sh` as `THRESHOLD` only when it differs from 0.5, and passed to `validate_run(threshold=...)` on the local backend.
- `validate_runs.py --results-experiment NAME` redirects only the output directory (and W&B group) to `logs/validation/<NAME>/`, while the checkpoint is still located via `--experiment`. That is what keeps a re-validation from overwriting the original run's JSONs; the written JSON records the `threshold` it used.
- If checkpoint layout, run directory layout, or logger naming changes, update validation loaders and sweep scripts together.

### Threshold Re-Validation

- `scripts/launch_threshold_validation.py` re-runs *only* the validation step of a finished experiment once per threshold, on the checkpoints that experiment already produced. Per `(domain, threshold)` it submits one array over the seeds through the same `scripts/slurm/validate.sh` (no training, no `aftercorr` dependency); `--backend local` runs them sequentially.
- Results go to `logs/validation/<experiment>-thr<T>/<domain>/seed<seed>.json` (`launch_threshold_validation.threshold_experiment_name`, `%g` formatting, so `0.5` → `-thr0.5`). Each is an ordinary experiment directory downstream. The source `logs/validation/<experiment>/` is never written to.
- Default grid `THRESHOLD_VALUES = (0.1, 0.25, 0.5, 0.75, 0.9)`; 0.5 is re-run rather than reused so every column comes from the same code path and the same random state sampling.
- W&B logging is forced off (`NO_WANDB=1`): `validate_runs._log_to_wandb` mirrors metrics by *resuming* the training run, so a second threshold would overwrite the 0.5 numbers in that run's summary. The JSONs are the source of truth.
- The launcher refuses to submit when a `(domain, seed)` has no `best-epoch*.ckpt` (`--allow-missing` overrides). `--time` / `--mem` override the per-domain values from `configs/slurm-resources-validation.yaml` for every array.
- `scripts/slurm/validate.sh` reads the optional `THRESHOLD` / `RESULTS_EXPERIMENT` / `NO_WANDB` environment variables; unset means plain validation of a fresh run, so the normal pipeline is unaffected.
- Aggregate with `scripts/collect_results.py --experiment <experiment> --threshold-sweep [T ...]`, which rebuilds the column names with the same `threshold_experiment_name` (imported, not re-derived — keep it that way) and prints/writes the same domain × value matrices as `--sweep`.

## Sweep Conventions

- `scripts/launch.py` is the authoritative sweep launcher. Per domain it submits a training array (`scripts/slurm/train.sh`, one task per seed) and a dependent validation array (`scripts/slurm/validate.sh`, `--dependency=aftercorr`). `--backend local` runs the same steps sequentially.
- One run corresponds to one `(domain, seed)` pair; seeds default to `0 .. num_runs-1` (`--num-runs`, default = `len(configs/validation.yaml seeds)`).
- `--seeds` runs an explicit seed subset instead (single values and/or inclusive ranges, e.g. `--seeds 3 7-9`), which is how individual failed seeds are re-run. It is mutually exclusive with `--num-runs`. The Slurm array task id *is* the seed, so a subset becomes a sparse array spec (`--array=3,7-9`) and the `aftercorr` dependency still pairs the train/validate tasks of each seed. Re-running under the same `--experiment` reuses the run id `<experiment>-<domain>-s<seed>`, so the replacement run reuses that seed's checkpoint directory and overwrites its result JSON. Lightning does not delete the failed attempt's checkpoint — it writes the new one alongside with a `-v1` suffix — so validation resolves the *most recently written* `best-epoch*.ckpt` (see below) rather than an arbitrary glob hit.
- All runs of one launch share an `--experiment` name (= W&B group = results directory). `--experiment-type {dias,mlp,noise}` picks the overlay, the validator, and the validation threshold (0.75 for `noise`, else 0.5; `--validation-threshold` overrides).
- `--ablation {no-slot-weights,no-precondition-loss,no-effect-loss,no-aux-loss,no-pcgrad,no-random-embeddings}` composes one `configs/experiments/ablation-*.yaml` overlay on top of the experiment type (orthogonal axis; always uses the DIAS validator). Default: none (full model).
- `--sweep AXIS [--sweep-values V ...]` sweeps one hyperparameter: the whole domain × seed grid is submitted once per value, each into its own sub-experiment `<experiment>-<tag><value>` (`launch.sweep_experiment_name`, value formatted with `%g`, so `1.0` → `aux1`). The axes live in `launch.SWEEP_AXES` (`name → SweepAxis(param, tag, values, default, cast)`):
  - `aux-loss-weight` — `model.init_args.aux_loss_weight`, tag `aux`, grid `0.01, 0.03, 0.1, 0.3, 1.0`, no reusable default (tuned per domain).
  - `num-slots` — `model.init_args.num_slots`, tag `slots`, grid `5, 7, 9`, default `5`.
  - `gmean-temperature` — `model.init_args.loss_weight_schedule.gmean_temperature.T`, tag `gmeanT`, grid `0, 100, 500, 1000, 5000`, default `500`.
  - `dataset-size` — `data.init_args.min_per_label_train`, tag `n`, grid `20, 50, 100, 200, 500, 1000, 2000`, no reusable default (see below).
- The `dataset-size` axis is the one axis that moves several parameters at once (`SweepAxis.linked` → `launch._dataset_size_overrides`). For value N it pins `min_per_label_train = max_per_label_train = N` and `min_per_label_valid = max_per_label_valid = N/2`, which makes the sampled pool *exactly* `num_actions · 1.5N` transitions instead of a lower bound; the 2:1 train:valid weight then splits it into ≈N train / ≈N/2 valid per action (the ratio is enforced pool-wide, so individual actions drift a few percent). It also pins `min_per_label_train_floor` / `min_per_label_valid_floor` to the same values, which neutralizes the 200/100 floors in `configs/experiments/noise.yaml` — those are composed after the domain config and would otherwise clamp the 20/50/100 points back up under `--experiment-type noise`. No column reproduces a domain's default run (the per-domain configs pair a tuned minimum with a loose cap), hence `default=None` and `--reuse-default` is rejected. `--set` is rejected for *any* of the six keys, not just the axis parameter (`launch._axis_params`).
- Sweep overrides are appended last on the Lightning command line, so they beat the per-domain values in `pddl/<domain>/config.yaml`. `gmean-temperature` is the exception to plain `--key=value`: jsonargparse rebuilds a dict-typed argument from scratch when a sub-key is set on the command line (`--...loss_weight_schedule.gmean_temperature.T=100` would drop `type`/`v0`/`vT`), so the launcher sends the *whole* schedule read from `configs/main.yaml` with `T` substituted — and refuses to run if a selected domain config *or* a selected overlay overrides `loss_weight_schedule` (the noise overlay does, so `--experiment-type noise --sweep gmean-temperature` is rejected). It is serialized as a **YAML flow mapping, not JSON**, because `keypoints` is keyed by int step and JSON would stringify those keys (`LossWeightSchedule.__call__` then compares int to str and raises).
- `T = 0` on that axis is a special point: the exponential form would divide by `T`, so the launcher emits the constant schedule `{type: linear, keypoints: {0: 0.0}}` instead. That is both the T → 0 limit of the decay and the regime where the divisor `1 + τ·(n-1)` in `utils.gmean` becomes 1 — i.e. the *exact* precondition product from step 0 rather than a relaxation. It is the only point in the grid that reaches τ = 0: for any T > 0 the schedule bottoms out at `vT = 0.1`.
- `--reuse-default EXPERIMENT` skips the sweep point equal to the axis `default` and maps that column to an existing experiment (e.g. `dias-full`) instead of re-running it. The launcher rejects it when the axis has no default, when `configs/main.yaml` no longer holds that default, or when a selected domain config overrides the swept parameter — all cases where the reused runs would not actually sit at the default.
- A sweep is rejected together with `--set <swept param>=...`; the `aux-loss-weight` axis is additionally rejected with `--ablation no-aux-loss`. `--sweep-aux-loss-weight [ALPHA ...]` is a back-compat alias for `--sweep aux-loss-weight [--sweep-values ALPHA ...]`. Each sub-experiment is an ordinary experiment downstream (own W&B group, own `logs/validation/<sub-experiment>/` directory) — nothing in `validate_runs.py` or the Slurm scripts knows about sweeps.
- Domain-specific sample bounds come from `pddl/<domain>/config.yaml`; paths/problem lists from `pddl/domain-settings.yaml`; per-domain Slurm resources from `configs/slurm-resources.yaml`.
- Global validation settings such as number of validation problems and sampling counts come from `configs/validation.yaml`.
- `scripts/collect_results.py` aggregates a finished experiment (local JSON or W&B) into the paper table. `--sweep AXIS [--sweep-values V ...] [--reuse-default EXPERIMENT]` rebuilds the sub-experiment names with the same `launch.sweep_experiment_name` and `launch.SWEEP_AXES` (imported, not re-derived — keep it that way) and prints a domain × value matrix per metric plus an `n` matrix; `--markdown <path>` writes the same matrices as markdown tables (one table per metric, raw values). `--aux-sweep [ALPHA ...]` is the back-compat alias for `--sweep aux-loss-weight`.
- Without a sweep, `--experiment` takes any number of unrelated experiment names (e.g. a baseline plus its ablations). Each gets its own printed table; `--markdown <path>` then writes one document — a cross-experiment summary table over the pooled seeds, followed by one domain table per experiment. All tables span the union of the domains (`-` for a domain an experiment lacks), and the summary's `n` column reads `<collected>/<expected>`, where expected is the per-domain seed count of the most complete experiment in the set. A sweep still takes exactly one `--experiment` (its base name) and is rejected with several. This is what generates `local/ablations-results-summary.md`.
- The Slurm scripts under `scripts/slurm/` first `source ~/.zshrc` then `conda activate ddl`; they stay thin and defer experiment semantics to Python scripts and YAML configs.

## Instance-Size Scaling Study

`scaling/` is a self-contained one-off analysis: iterations per second and peak GPU memory as the *instance* grows, on `blocks-3`, `logistics`, `delivery` × 6 sizes × 3 seeds. It produces timings, not schemas — no checkpoints, no validation, no W&B. Documented for users in `scaling/README.md`.

- It measures the ordinary training path: `scaling/benchmark.py` instantiates `dias.main.CLI(run=False)` on the real config chain (`configs/main.yaml` → `pddl/<domain>/config.yaml` → `scaling/configs/scaling.yaml`) and attaches a timing callback. Nothing is stubbed or reimplemented; if the training step changes, this measures the new one.
- Runs are cut to `warmup_steps + measure_steps` (20 + 100, from `scaling/configs/sizes.yaml`), not the 10 000 steps of a real run. The warm-up steps are discarded, `torch.cuda.reset_peak_memory_stats()` is called right after them, and every step is bracketed by `torch.cuda.synchronize()`. `iterations_per_second` is `1 / median(step time)`.
- `scaling/configs/scaling.yaml` is the overlay that holds everything but the instance fixed: exactly 100 train / 50 valid transitions per action (pinned min *and* max, so delivery does not sample 10× more than blocks-3), batch size 200, and validation / logging / checkpointing off. Sampling time is timed separately by wrapping `datamodule.setup` — a Lightning `Callback.setup` hook runs *after* the datamodule's and would measure nothing.
- Ladder and Slurm resources live in `scaling/configs/sizes.yaml`. Instance ids encode the object count (`n0068`), which `scaling/generate_instances.py::OBJECT_COUNT` derives from the generator parameters and cross-checks against the parsed instance. Each size entry also carries a `target` — the shared ladder rung — because the domains scale in different units and cannot all hit the same count (logistics lands on 21/38/67/103/147). `run_scaling.ladder_for` pairs the two, and that pairing is what aligns the columns in `collect_scaling.py`; the exact counts get their own table.
- `scaling/instances/<domain>/n<objects>.pddl` plus `index.json` are generated once and **committed**; running the study needs neither the submodule nor the generator toolchain. `scaling/generate_instances.py` rewrites generator output into the repo's domain dialect (blocksworld: `- object` typing plus the reflexive `(eq b b)` facts the repo domain's `(not (eq ?x ?y))` preconditions require; logistics: `OBJ` → `object`, `in-city` → `loc`, added `(vehicle ?v)` facts; delivery: unchanged but renamed). The delivery generator needs `tarski` and runs from a gitignored venv at `scaling/.venv-delivery`.
- Layout mirrors the main pipeline: `launch_scaling.py` (submit) → `slurm/benchmark.sh` (array task) → `run_scaling.py` (one domain's ladder for one seed) → `benchmark.py` (one point) → `collect_scaling.py` (tables). It imports `_format_array` / `_sbatch_resource_opts` from `scripts/launch.py` rather than re-deriving the sbatch flags — keep it that way.
- Two deliberate departures from `scripts/launch.py`: an array task is one **(domain, seed)** and walks the whole ladder (so one curve is measured on one physical GPU), and it requests an **exclusive** GPU on a **pinned** partition (`gres: gpu:1`, `--partition=rleap_gpu_24gb`) instead of a `shard:N` slice — co-tenants on a shared GPU would land in the throughput numbers, and mixed GPU models would make a curve meaningless. Re-running on a different card belongs in a different `--experiment`.
- Each size runs as its own subprocess so a CUDA OOM costs one point, not the ladder. OOM is a recorded outcome (`status: "oom"`, exit 0), shown as `oom` by `collect_scaling.py`; a non-zero exit is a real failure. Memory is quadratic in objects, so the top of the `delivery` ladder is expected to exceed 24 GB.
- Results: `logs/scaling/<experiment>/<domain>/<instance>-s<seed>.json` (under the gitignored `logs/`).

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
