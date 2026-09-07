# DIAS Agent Instructions

Repository-level guidance for AI coding agents working in this workspace. Keep it current when the pipeline, layout, naming or conventions change; when it disagrees with the code, confirm which path actually runs and fix this file.

## What This Is

DIAS means Differentiable Induction of Action Schemas. It learns lifted STRIPS action schemas from PDDL state transitions, and implements the paper [Differentiable Learning of Lifted Action Schemas for Classical Planning](https://arxiv.org/abs/2605.13282). Consult the paper when this file is not enough.

The setting, which code changes should preserve unless a task says otherwise:

- Inputs are `<s, a, s'>` triplets sampled from a **single** problem instance per domain.
- States are symbolic PDDL states over **unary and binary predicates only**, fully observed.
- Action labels are full STRIPS actions, STRIPS+ actions, or names only, depending on the experiment.
- The output is, per action schema, lifted positive and negative preconditions and lifted add and delete effects.
- Action binding is assumed injective, and the domain's predicates are known.
- Objects are embedded by a relational GNN; slot queries plus a Sinkhorn-style rectangular assignment bind objects to schema parameters; slot activations give the learned arity.
- Auxiliary losses push toward minimal effects and restrictive preconditions, and PCGrad keeps them from overriding the prediction objective.
- Evaluation is **behavioural**: does the learned domain generate the same successor states as the real one, on held-out problems.

End to end: load a domain and one problem, sample transitions by random walk or BFS, encode them as adjacency tensors and PyG graphs, train with Lightning, read the lifted schemas off the trained parameters, score them against the original domain on the three held-out instances, and repeat per `(domain, seed)`.

## Layout

- `src/dias/` is the package. It is installed from a clone (`pip install -e .`, console script `dias`), never from PyPI, where the name is taken. `configs/`, `pddl/` and `experiments/` are **not** package data, so every documented command runs from inside the checkout. Dev tooling is the PEP 735 `[dependency-groups] dev` table (`pip install -e . --group dev`).
- `pddl/<domain>/` is one folder per domain, holding `domain.pddl`, `train.pddl`, `test-1.pddl` .. `test-3.pddl`, `strips+.yaml` (the STRIPS+ argument positions) and `config.yaml`. **The folder is the manifest**: nothing lists the domains or their files, adding a domain is adding a folder, and those filenames are load-bearing (`tests/test_pipeline.py` fails on a folder missing one).
- `configs/` holds the Lightning configs `dias fit` reads, flat: the base `main.yaml`, the experiment overlays `experiment-mlp.yaml` / `experiment-noise.yaml`, and six `ablation-*.yaml`. What makes one the base and the rest overlays is the order they are passed.
- `experiments/` is what you run: the launchers (`launch.py`, `launch_threshold_validation.py`), the validation steps (`validate_runs.py`, `validate_checkpoint.py`), the collector (`collect_results.py`), their shared `pipeline.py`, `slurm/`, the orchestration configs in `configs/`, the `scaling/` study, and `README.md`, the recipes users are pointed at.
- `tests/` mirrors the package: a test module is named after the module it covers, minus the package path (`tests/test_noise.py`, `tests/test_mlp_effects.py`).
- `logs/`, `local/` and `experiments/results/` are gitignored. Result tables are regenerated from `logs/`, so do not commit them; public docs must not point at `local/`.

**`experiments/pipeline.py` is the one place that resolves a path.** `EXPERIMENTS_ROOT` is its own folder and `PROJECT_ROOT` its parent; `SLURM_DIR`, `PIPELINE_CONFIGS` and `RESULTS_DIR` hang off the first, `CONFIGS_DIR` / `PDDL_DIR` / `LOGS_DIR` off the second, and the domain-folder layout (`domain_pddl`, `train_problem`, `validation_problems`, `domain_config`, `available_domains`, `select_domains`) lives there too. Nothing else joins one of those paths by hand. It imports neither torch nor pymimir, so submission works on a login node.

## Code Map

- `main.py`: the LightningCLI entry point. It links datamodule metadata (`num_objects`, `relation_arities`, `num_actions`) into the model's init args; keep the two interfaces in step.
- `data/`: `datamodule.py` (`PddlGraphDataModule`, composing the rest), `datasets.py` (`PddlTransitionDataset` plus its index-list and stratified-stream views), `generate_samples.py` (BFS and random-walk exploration; sample dicts carry primitives only, never a live pymimir object), `quota.py` (`SampleQuota`, the per-action bounds), `splitting.py` (the action-stratified split), `noise.py` (`Ratio` / `ExpectedCount`).
- `models/`: `dias.py` (the LightningModule), `encoder.py` (`RelationalEncoder`), `selection.py` (`resolve_num_slots`, `SlotSelector`), `preconditions.py` (`Preconditions`, `soft_conjunction`), `effects.py` (`StaticEffects` and `StateDependentEffects` behind `EffectSource`), `prediction.py` (`SuccessorPredictor`, composing them into `A' = A + pre·Δ`), `edges.py` (`EdgeTypeLayout`, the only place that knows what an edge type id means, and the builder of the edges), `masking.py` (which `(relation, slot, slot)` cells no lifted schema can express), `objectives.py` (`DiasObjective`, returning the auxiliary terms *beside* `L_adj`, never summed into it), `pcgrad.py` (`gradient_combiner` over `SingleObjective` / `WeightedSum` / `PCGrad`, taking the two scalar losses), `outputs.py` (`Channels`, `SchemaPrediction`, `SchemaLogits`), `checkpoints.py` (migration).
- `schema/`: `domain.py` (`DomainSchema`, the frozen relation/action/object metadata of one `(domain, problem)` pair, and the functions deciding what that metadata is) and `action.py` (`ActionSchema` / `LiftedAtom` and their PDDL serialization). Consumers needing only metadata use `DomainSchema` instead of building a datamodule.
- `pddl/`: `text.py` (pure PDDL string handling, imports nothing), `untype.py` (typed to untyped compilation), `transpiler.py` (`PDDLTranspiler`, binarized tensors to `ActionSchema`s), `states.py` (reading pymimir states and re-hosting one problem's state on another).
- `validation/`: one module per model scored, each holding one class: `dias.py: DiasValidator`, `mlp_effects.py: MlpEffectsValidator`, `l1.py: L1Validator` (a learned PDDL domain from another system). Plus `checkpoint.py` (`load_run`), `decoding.py` (`derive_schema`, the paper's Sec. 4.5 derivation), `successors.py` (state sampling and `score_successors`), `artifacts.py` (`PddlWorkspace`, since pymimir only parses from paths).
- `reporting/`: `labels.py`, `vega.py` (three Vega-Lite heatmap specifications), `charts.py`, `callback.py` (`SchemaChartCallback`).
- Top level: `adjacency.py` (the dense `[R, O, O]` representation and its PyG conversions), `schedules.py` (`Constant` / `Exponential` / `PiecewiseLinear`, pure functions of the global step, holding no state), `hparams.py` (round-tripping a dataclass-typed argument through hyper-parameters, which must hold primitives or `torch.load` refuses the checkpoint).

## Training

- The config chain is `dias fit --config configs/main.yaml --config pddl/<domain>/config.yaml`, plus an experiment and/or ablation overlay. **`main.yaml` names no domain**, so it is not runnable alone: a domain config opens with its own `domain_path` / `problem_path`, which is what completes the chain (`tests/test_configs.py` pins that they are the folder's own files).
- **Each layer states only what it changes about the layer below, and the bottom layer is the class signature.** `DIAS` and `PddlGraphDataModule` default to the paper's full model with no noise, so `main.yaml` carries the run itself (trainer, logger, callbacks) plus what a class cannot choose: `num_slots`, the τ schedule, the four sampling bounds. A per-domain config carries only its own two paths, `aux_loss_alpha`, and, for the six domains that sample differently, the `*_per_action_*` bounds. Do not restate a default to make it visible.
- **Every `__init__` argument of `DIAS` and `PddlGraphDataModule` carries a one-line description**, as a Google-style `Args:` block that `--model.help` / `--data.help` print. A new argument gets its line in the same commit.
- **Two arguments have no default anywhere**: `data.domain_path` and `data.problem_path`. Arguments defaulting to `None` mean "not set", not a value: `num_slots`, `precondition_tau`, `noise`, `data.seed`, and the four sampling bounds, which `SampleQuota` refuses as a group if nothing says how much to sample.
- **One seed per run.** `data.seed` defaults to `None` and `CLI._seed_the_data_from_the_run` fills it from `seed_everything`, so the saved config records the number the walk sampled under and an unseeded run stays reproducible after the fact. Launchers therefore pass `--seed_everything` alone.
- τ (`model.init_args.precondition_tau`) is the root degree of the relaxed precondition product, one of the schedule classes, configured by `class_path` / `init_args`. `Exponential` is `floor + (start - floor) * factor ** (step / every)`; rate and asymptote are independent.
- **One caveat on typed config values.** jsonargparse *merges* a configured value's fields onto the default's rather than replacing, so a schedule-typed argument must default to `None`, and switching the schedule *class* on the command line over a config that names another does not work. Keep overrides to fields.
- Ablation switches are `model.init_args` arguments defaulting to the full model, one overlay each: `use_slot_weights`, `use_precondition_loss`, `use_effect_loss`, `use_pcgrad`, `random_node_embeddings`, plus `ablation-no-aux-loss` (both auxiliary terms, α = 0).
- The MLP-effects model is `model.init_args.state_dependent_effects: true` (`configs/experiment-mlp.yaml`). Its effect MLP has SiLU activations between the linear layers and its output goes through the same arity mask as the static tensor; both are load-bearing, so **a checkpoint whose effect MLP predates them is refused on load** rather than migrated.
- Noise ablations set `data.init_args.noise` to one of the two dataclasses; unset means clean transitions. `configs/experiment-noise.yaml` also carries the paper's noise-experiment adjustments (Appendix A.4.2): the 200/100 per-action floors, `batch_size: 50`, and τ `floor: 0.1`. `min_per_action_*_floor` only ever **raises** the per-domain minimum, which is why the overlay states a floor rather than a value: overlays compose *after* the domain config.
- The default batching strategy is `stratified`, an effectively unbounded iterable dataset, so run length comes from `trainer.max_steps`, not dataset exhaustion.
- The default logger is W&B (`project: dias`, `entity: null`, meaning the logged-in account or `$WANDB_ENTITY`). No account appears anywhere in the code. Checkpoints are expected under `logs/dias/<run_id>/checkpoints/`, named `best-epoch*.ckpt`.
- **The last encoder convolution has no bias**, deliberately: nothing is normalised after it, so it reaches the assignment as a per-slot constant that Sinkhorn's row scaling removes exactly, and no gradient above float32 round-off ever reached it.
- **What reproducibility is promised.** A run repeats exactly only under `--trainer.deterministic=true` (Lightning's own flag; do not add a second). It is off by default because it raises on any operation without a deterministic implementation. On CPU the thread count decides the trajectory with or without it, which is why `tests/test_training_characterization.py` sets `torch.set_num_threads(1)` and compares at `TRAJECTORY_TOLERANCE` (1e-4) while the forward golden keeps 1e-6.

## Validation

- Validation is behavioural, never string-based. `DiasValidator` reconstructs the learned schemas, merges them into the original domain as separate actions, and compares reachable successor sets on `test-1..3.pddl`. Which problems those are is the folder layout; `experiments/configs/validation.yaml` holds only the sampling knobs and seeds.
- **A trained model's schema is read through `DIAS.schema_logits()` and nowhere else.** It returns the effect and precondition logits with the arity mask already applied, plus the slot-weight logits. Reading `model.predictor.*` directly is wrong: the raw parameters hold untouched noise at cells no lifted schema can express. `schema_logits().effects` is `None` for a state-dependent model, which is how readers learn there are no lifted effects.
- A caller that already knows the slot binding asks `DIAS.apply_schema(adjacency, selection, y_action)`, which deliberately does **not** re-check the preconditions: applicability was already decided by grounding. `DIAS.retarget(num_objects)` is how an instance size changes.
- Reading a schema off a checkpoint binarizes slot activations, effects and preconditions at one threshold, `derive_schema(threshold=0.5)`. The threshold applies uniformly: lowering it activates more slots *and* adds effects and preconditions. `launch.py` picks it per experiment type (0.75 for `noise`, else 0.5; `--validation-threshold` overrides); the MLP-effects model has no schema to binarize and rejects a non-default value.
- **Precision and recall are micro-averaged over per-action counts.** Every validator returns the same payload: `action_stats` plus the `tp`/`fp`/`fn` they sum to, and `collect_results` sums those across a seed's problems. `score_successors` is the one scorer, and it reports one measure, attributed to actions: a successor reached under the wrong operator is a false positive there and a false negative under the right one. There is deliberately no second union-over-actions count beside it; the state-level question survives as `matches` / `accuracy`. Soundness is precision, completeness is recall.
- **The sampled states are drawn freshly but always recorded.** `sample_seed` is null in the config, `successors.resolve_sample_seed` draws one per run, every validator stores it and the result JSON carries it, so passing it back reproduces exactly those states. The generator is the validator's own, never the global `random`.
- The MLP-effects validator is the only one with parallelism, because its per-state work is grounding plus a batched forward pass. DIAS and L1 validate 250 states in seconds, and a sweep already parallelises over array tasks.
- `resolve_best_checkpoint` picks the newest `best-epoch*.ckpt` by mtime, because a re-run seed reuses the run id and Lightning writes a `-v1` file beside the old one. Every validator reaches it through `load_run`.
- Sweep validation writes `logs/validation/<experiment>/<domain>/seed<seed>.json` and mirrors numeric metrics to the resumed W&B run.
- `validate_checkpoint.py --validator {dias,mlp,l1}` is the single-model entry point, and **the only way to reach the L1 baseline**, which scores a `--learned-domain` produced by another system and so has no checkpoint, run id or seed for a sweep to iterate over.

## Sweeps

- `experiments/launch.py` is the authoritative launcher, structured as `parse_args -> build_plan -> describe -> execute`; `LaunchPlan` is where every rejection happens, so nothing reaches sbatch unless the whole grid is valid. Per domain it submits a training array plus a `--dependency=aftercorr` validation array, or runs the same steps sequentially with `--backend local`.
- One run is one `(domain, seed)`; the array task id **is** the seed, so `--seeds 3 7-9` becomes a sparse array spec and the dependency still pairs each seed's two tasks. Run ids are `<experiment>-<domain>-s<seed>` (`validate_runs.run_id_for`), which is how the validation array finds the checkpoint with no id handoff. Re-running a seed under the same `--experiment` overwrites its result JSON.
- `--experiment-type {dias,mlp,noise}` picks the overlay, the validator and the threshold; `--ablation` composes one `ablation-*.yaml` on top (orthogonal, always the DIAS validator).
- `--sweep AXIS` submits the whole grid once per value into `<experiment>-<tag><value>`. Axes live in `launch.SWEEP_AXES`: `aux-loss-alpha` (tag `aux`), `num-slots` (`slots`), `tau` (`tau`, overriding the schedule's `every` field), `dataset-size` (`n`). Each sub-experiment is an ordinary experiment downstream; nothing in `validate_runs.py` or the Slurm scripts knows about sweeps. A sweep is rejected together with `--set` on any of its parameters, and `aux-loss-alpha` additionally with `--ablation no-aux-loss`.
- `dataset-size` is the one axis moving several parameters at once: it pins min and max train to N, min and max valid to N/2, and both floors to the same, which neutralises the noise overlay's 200/100 floors. No column reproduces a domain's default run.
- Sweep and `--set` overrides are appended last on the Lightning command line, so they beat the per-domain config.
- `collect_results.py` aggregates into the paper's `P / R / N_S / N_C / N_SC` table. `--sweep AXIS` or `--threshold-sweep` gives domain × value matrices, rebuilding the sub-experiment names by importing `launch.sweep_experiment_name` and `launch_threshold_validation.threshold_experiment_name` rather than re-deriving them; keep it that way. Without a sweep, `--experiment` takes several unrelated names and `--markdown` writes one document with a cross-experiment summary plus one table per experiment.
- `launch_threshold_validation.py` re-runs only the validation step of a finished experiment at several thresholds (default `0.1, 0.25, 0.5, 0.75, 0.9`), into sibling `logs/validation/<experiment>-thr<T>/` directories, never writing to the source. 0.5 is re-run rather than reused so every column comes from one code path. W&B is forced off there, since mirroring works by resuming the training run and a second threshold would overwrite the first's summary.
- `--no-wandb` covers both steps on both backends, training logger included; without a logger the run directory comes from `trainer.default_root_dir`, so `launch.run_directory_args` (mirrored in `experiments/slurm/train.sh`) names `<logs>/dias/<run_id>` explicitly.

## Slurm

- Per-domain resources are `experiments/configs/slurm-resources.yaml` (training) and `slurm-resources-validation.yaml` (validation), same schema (`defaults` plus per-domain overrides), resolved independently. Treat them as the source of truth rather than hardcoding a partition or GPU size anywhere. Training takes GPU shards with no partition pinning; validation is CPU-only.
- The scripts under `experiments/slurm/` open with a `── site configuration ──` block holding every line specific to one cluster, including `DIAS_ENV_SETUP`. Nothing below the end marker is site-specific, and the scripts defer all experiment semantics to the Python and YAML.
- Nothing in the Slurm path assumes the submission directory: Slurm runs a spool copy, so launchers pass absolute script and `--output` paths and export `DIAS_ROOT`.
- `train.sh` and the scaling benchmark export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, because co-tenants on a shared GPU see the *reserved* pool and that setting is what makes the per-domain shard reservations hold.

## Reporting and Logging

- **The library configures no logging.** Every module reports through `logging.getLogger(__name__)` and only an entry point installs a handler (`dias.main.main`, `pipeline.configure_logging`), which is what keeps `dias` importable inside another application and why `src/` contains no `print`. Scripts take `-v` / `-q`, and `DIAS_LOG_LEVEL` overrides both, which is how a Slurm job with a fixed command line is made verbose.
- **INFO is for state changes someone waits on, once each; anything per-step or per-batch is DEBUG.** Messages carry values as `%s` arguments, never f-strings, so a suppressed line costs nothing.
- `pipeline.run_command(..., capture=True)` for sbatch, which needs the job id; `capture=False` for a local training run, which needs to inherit the terminal.
- The learned schema is reported by `SchemaChartCallback`, registered in `configs/main.yaml`, not by the model. Six panels go out as `wandb.Table`s bound to the Vega-Lite specifications in `reporting/vega.py`, every `log_every_n_validations` validation epochs and once at `on_fit_end`. The selection panel re-runs the forward pass inside `torch.random.fork_rng`, because a diagnostic must not shift the run's random stream. Without a W&B logger it logs nothing.
- **Four details of the W&B panel are not in the SDK's surface and would break silently if renamed upstream**: the query is rewritten to `historyTable` with its `extraKeys` and `index` arguments (without `index` the panel comes up empty), the panel sets `showStepSelector` (without it there is no slider), `index` is fixed at the *first* logged step so the configuration stops moving (W&B rewrites it on every log, and pointing at a still-uploading table made panels blink), and a colour bar is suppressed with `"legend": false`, not the `null` the schema specifies. The specifications live in the run's own entity, and `charts.register` must use `updateCustomChart` as well as `create`, since `create_custom_chart` cannot update.

## Naming

**Code names follow the paper's, and a new name is chosen by looking the concept up there first.** A name disagreeing with this table is a bug in the name.

| Paper | Code |
| --- | --- |
| relations / predicates `p`, arity `\|p\|` | `relation_names`, `relation_arities` |
| action schema `a`, arity `\|a\|` | `action_names`, `action_arities` |
| keys `K` | the encoder's `node_embeddings` |
| queries `Q^a` | `SlotSelector.slot_queries` |
| correspondence matrix `C^a` | `correspondence` |
| soft assignment `S^a`, selection `S̃^a` | `selection` |
| slot probabilities `w^a` | `slot_weight_logits`, `slot_weights` |
| number of slots `M`, objects `O`, embedding width `d` | `num_slots`, `num_objects`, `object_dim` |
| lifted atom `p(x1, ..., x\|p\|)` | `schema.action.LiftedAtom`, over `slots` |
| `P_add` / `P_del` / `P_pre+` / `P_pre-` | `Channels.positive` / `.negative` on `effects` / `preconditions` |
| `p_pre` | `precondition_satisfaction` |
| τ, the root degree of the relaxed product | `precondition_tau` (the schedule), `tau` (the value at a step) |
| `L_adj`, `L_aux`, weight `α` | `Losses.adjacency`, `Losses.auxiliary`, `aux_loss_alpha` (`alpha` where the quantity stands alone) |
| PCGrad's norm cap (App. A.2.2) | `pcgrad_norm_cap` |
| `N_min^tr` .. `N_max^val`, per action name | `min_per_action_train` .. `max_per_action_valid` |
| derivation of action schemas (Sec. 4.5) | `validation.decoding.derive_schema` |
| soundness / completeness | `precision` / `recall`, `tp` / `fp` / `fn` |

Three deliberate departures. **A relation is a predicate that became a channel of the adjacency tensor**, which `schema.domain.tensor_relations` decides, so `pddl/` and `transpiler.py` say `predicate_*` for the PDDL declaration and everything tensor-side says `relation_*`. **An argument named for a tensor axis keeps the axis's name** (`y_action`, `edge_index`), which is PyG's convention. And **the MLP-effects model is not a baseline**: it is one configuration of the same model, so it is "MLP effects" throughout, and the word baseline is left to L1, which is another system's output.

**An attribute of a PyG `Data` object must not have `index` or `batch` anywhere in its name unless it means what PyG takes those to mean.** `Data.__cat_dim__` and `Data.__inc__` key on the substring: a key containing `index` is concatenated along the last axis and offset by `num_nodes` per graph. That is why the action label on a graph is `y_action`; named `action_index`, three transitions labelled 0, 1, 2 over a 4-object instance batch to `[0, 5, 10]`, silently. The restriction is on `Data` attributes alone, so `PddlTransitionDataset.action_indices` keeps the clearer name.

## Checkpoint Compatibility

- `models/checkpoints.py` migrates old checkpoints. **There are exactly two formats and one rule between them**: `CHECKPOINT_VERSION = 1`, and an absent stamp (`UNVERSIONED`), the layout the paper's runs were written with. `_from_unversioned` does all of it; a stamp higher than the current one is refused. **A further change to what the model stores needs version 2 and a rule of its own**, because editing version 1 would change what an already-published checkpoint means.
- **Migration is driven by the stamp, not by a failed load**, because Lightning silently drops a hyper-parameter it does not recognise instead of raising.
- For the same reason, **renaming a constructor argument of `DIAS` needs an entry in `_RENAMED_HPARAMS` and a line in `tests/test_checkpoint_layout.UNVERSIONED_HPARAM_NAMES`**. A datamodule argument does not, since nothing feeds stored datamodule hyper-parameters back to a constructor.
- If checkpoint naming, run directory layout or logger config changes, update the validation loaders and sweep scripts in the same change.

## Editing Rules

- Prefer minimal, behavior-preserving changes, and preserve the distinction between single-domain training, per-checkpoint validation and sweep orchestration.
- Do not silently change experiment semantics stored in YAML or config defaults without updating the runners and the docs in the same task; the same goes for path conventions.
- Treat the `pddl/<domain>/` folder as the source of truth for per-domain paths, reached through `experiments/pipeline.py`, and `pddl/<domain>/config.yaml` for per-domain overrides.
- The test suite must run on the oldest supported Python (3.10), so `unittest.TestCase.enterContext` is out; use `tests/support.temp_dir(self)`. CI runs 3.10 to 3.13 across three jobs: `format`, `test`, and `install`, which does a non-editable install into an empty venv and imports the package from another directory, so a missing module or an unavailable wheel fails there rather than on a user's first clone. Dropping a Python version means dropping it from the matrix, `requires-python` and the README together.
- isort's `src_paths` lists `experiments` **and** `experiments/scaling`, because the scaling scripts import each other as siblings; without the nested entry the `format` job fails.

## Documentation Rules

- The public docs are `README.md` (what DIAS is, install, quickstart, domains, a short reproduction section), `experiments/README.md` (the full recipes) and `experiments/scaling/README.md`. They must not point at `local/`.
- **A config does not describe its own parameters.** Every argument a Lightning config sets is documented in its class signature and printed by `--model.help` / `--data.help`, so a comment restating what a value does is a second place to keep in step. What a comment there is for is what the signature cannot say: why *this* file sets *this* value, and how it composes with the others. `experiments/configs/*.yaml` are not Lightning configs, so they document their own schema, but by pointing at the code that reads it.
- Keep terminology precise and aligned with the paper: predicates, lifted action schemas, preconditions, add and delete effects, successor states, STRIPS, STRIPS+.
- **Prose is not hard-wrapped, and it has no dashes.** One paragraph is one line, in markdown, docstrings and `#` comments alike; an example command is one line however long, and so is a long string literal. Line structure is reserved for things that have one anyway: list items, tables, code blocks, and the real multi-line commands in `experiments/slurm/*.sh`. Where a dash would have gone, use a colon, a semicolon, a comma or parentheses. This is about the punctuation, not the character: `--` is the same device and no more welcome.

## Instance-Size Scaling Study

`experiments/scaling/` is a self-contained one-off analysis of iterations per second and peak GPU memory as the *instance* grows. It produces timings, not schemas, and is intentionally not wired into `launch.py`. It measures the ordinary training path by instantiating `dias.main.CLI(run=False)` on the real config chain and attaching a timing callback, so nothing is stubbed. Its instances are committed, so running it needs neither the upstream generators nor their toolchain. Details are in `experiments/scaling/README.md`.

## Maintenance Policy

Update this file whenever any of these change: entry points, config filenames or locations, logging or checkpoint layout, the sweep or validation procedure, the PDDL dataset layout, file or folder or parameter names, or a modeling assumption future changes should respect.
