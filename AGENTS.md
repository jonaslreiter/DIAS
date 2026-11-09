# DIAS: notes for coding agents

DIAS learns lifted STRIPS action schemas from `<s, a, s'>` transitions of one PDDL problem per domain, and is the code of the paper [Differentiable Learning of Lifted Action Schemas for Classical Planning](https://arxiv.org/abs/2605.13282). Read the paper for the method; this file covers what the code does not say on its face. Keep it current when the layout, entry points, conventions or naming change.

Setting: fully observed states over unary and binary predicates; labels are action names, STRIPS+ actions or full STRIPS actions; bindings are injective, so every learned schema requires distinct parameters; evaluation is behavioural, by the successor states a learned domain generates on the held-out problems `test-1` to `test-3`.

## Layout

- `src/dias/`: the package (`pip install -e .`, console script `dias`). `models/` (the LightningModule `dias.py` and its parts), `data/` (sampling, splitting, noise, the datamodule), `schema/` (domain metadata, `derivation.py` from parameters to schemas, `export.py` to PDDL), `pddl/` (text handling, untyping, transpiler, pymimir states), `validation/` (validators and the worker processes they score states in), `reporting/` (W&B schema heatmaps).
- `configs/`: `main.yaml`, the training setup, and overlays: `experiment-{mlp,noise}.yaml`, `labels-{strips,strips-plus}.yaml`, `ablation-*.yaml`.
- `pddl/<domain>/`: `domain.pddl`, `train.pddl`, `test-{1,2,3}.pddl`, `strips+.yaml`, `config.yaml`, `README.md`. The folder is the domain; nothing else lists them.
- `experiments/`: `launch.py` (training and validation arrays, sweeps), `validate_runs.py` (one validation job), `train.py` (one training job), `collect_results.py` (tables), `validate_checkpoint.py` (one model on one problem), the analyses `schema_sizes.py`, `precondition_categories.py`, `noop_share.py`, the shared `pipeline.py`, `configs/` (`validation.yaml`, `slurm.yaml`), `slurm/`, `l1_pddl/`, `scaling/`, and `README.md`, the commands of every paper table.
- `logs/`, `local/` and `experiments/results/` are gitignored.

## Commands

```bash
pytest                                   # a few minutes on CPU
black --check . && isort --check-only .
dias fit --config configs/main.yaml --config pddl/<domain>/config.yaml
python experiments/launch.py --experiment <name> [--backend local] [--dry-run]
```

## Conventions

- **Configs.** A run is `main.yaml`, the domain's `config.yaml`, the overlays of `--experiment-type`, `--labels` and `--ablation`, then `--set` and sweep overrides. Each layer states only what it changes; the class defaults of `DIAS` and `PddlGraphDataModule` are the paper's model, and `domain_path` / `problem_path` have none. Every `__init__` argument of both has a line in its `Args:` docstring, which `--model.help` / `--data.help` print. A dataclass-typed argument (`precondition_tau`, `noise`) defaults to None, since jsonargparse merges a configured value onto a default, and is stored in hyper-parameters as a `class_path` / `init_args` mapping (`dias.hparams`).
- **Seeds.** `data.seed` defaults to the run's `seed_everything` (`main.CLI`), and training is deterministic given `(domain, seed)`. Validation draws its states with a seed it records.
- **Reading a model.** The learned schema is read through `DIAS.schema_logits()` only, and becomes PDDL through `schema.export` (`read_learned_schemas`, `learned_domain`). Every checkpoint is saved with its learned domain beside it as `<checkpoint>.pddl` (`main.LearnedDomainCheckpoints`), which validation scores unless a threshold is given.
- **Checkpoints.** `models/checkpoints.py`: the current format is 3; format 2, that of the paper's runs in `logs/`, loads with its renamed hyper-parameters. Renaming or retiring a `DIAS` argument needs a new format and a rule, never an edit of an old one. `experiments/noop_share.py` maps the retired datamodule arguments of saved configs the same way.
- **Equality.** By default `=` has no channel and the learned domain requires distinct parameters; `include_equality` gives it one. Validation writes equality as the declared predicate `equal` in both domains; the recorded sample seeds draw the states they drew only with this rewrite.
- **Validation** (`validation/`): states are scored in subprocesses under an address-space limit (`workers.py`), so modules they import must not import torch. Results go to `logs/validation/<experiment>/<domain>/seed<seed>.json` with a `status` (`ok`, `partial`, `out_of_memory`, `validation_failed`); each state's score is appended to a `.states.jsonl` beside it, which `--resume` continues from.
- **Experiments.** `pipeline.py` is the only place that builds paths and imports neither torch nor pymimir. Run ids are `<experiment>-<domain>-s<seed>`, the Slurm array task id is the seed, and a sweep value is the experiment `<experiment>-<tag><value>`. Slurm jobs import `safe_slurm` in their entry points (`train.py`, `validate_runs.py`, `scaling/run_scaling.py`), never in the package; everything cluster-specific is in the scripts' `site configuration` block and `configs/slurm.yaml`.
- **PyG.** A `Data` attribute with `index` or `batch` in its name is offset or concatenated by PyG, which is why the action label is `y_action`.
- **Goldens.** `tests/fixtures/golden.pt` (model), `recipes.json` (the arguments every README recipe resolves to) and `domains.json` (sampling, first training steps and scoring on the real domains) pin behaviour. Regenerate them only for an intended change, with `DIAS_REGENERATE_GOLDEN=1 pytest <test>`, and review the diff.
- **Tests** run on Python 3.10 to 3.13 (CI: `format`, `test`, `install`); use `tests/support.temp_dir`, not `enterContext`. Test modules are named after the module they cover.
- **Logging.** The package logs through `logging.getLogger(__name__)` and never prints; only entry points configure logging. INFO is for events a user waits on, DEBUG for anything per step; values go in `%s` arguments.

## Style

- Names follow the paper; look a concept up there before naming it. Code says `relation` for a predicate that is a channel of the adjacency tensor, `predicate` for the PDDL declaration. "MLP effects" is a variant of DIAS; L1 is the symbolic baseline.
- Comments and docstrings say what the code does, briefly, and add a constraint only where the code cannot show it. No history, no justification, no restating the paper.
- Prose is never hard-wrapped, one paragraph per line, and uses no dashes as punctuation (`--` included); use a colon, semicolon, comma or parentheses.
- Result markdown holds tables only.

| Paper | Code |
| --- | --- |
| predicates `p`, arity `\|p\|` | `relation_names`, `relation_arities` |
| action schema `a`, arity `\|a\|` | `action_names`, `action_arities` |
| keys `K`, queries `Q^a` | the encoder's `node_embeddings`, `SlotSelector.slot_queries` |
| correspondence `C^a`, selection `S̃^a` | `correspondence`, `selection` |
| slot weights `w^a` | `slot_weight_logits` |
| slots `M`, objects `O`, embedding width `d` | `num_slots`, `num_objects`, `object_dim` |
| `P_eff`, `P_pre` | `StaticEffects`, `Preconditions`; `Channels.positive` / `.negative` |
| `p_pre`, τ | `precondition_satisfaction`, `precondition_tau` |
| `L_main`, `L_aux`, α | `Losses.main`, `Losses.aux`, `aux_loss_alpha` |
| `g_main`, `g_aux` | `PCGrad`'s `main`, `aux` |
| derivation of action schemas, its threshold | `schema.derivation.derive_schema`, `derivation_threshold` |
| random node features | `random_node_features` |
| MLP-effects variant | `mlp_effects`, `MlpEffects` |
| `N_min^tr` .. `N_max^val` | `min_per_action_train` .. `max_per_action_valid` |
| soundness, completeness | precision, recall |
