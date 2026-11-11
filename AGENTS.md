# DIAS: notes for coding agents

DIAS learns lifted STRIPS action schemas from `<s, a, s'>` transitions of one PDDL problem per domain. It is the code of the paper [Differentiable Learning of Lifted Action Schemas for Classical Planning](https://arxiv.org/abs/2605.13282), which describes the method. This file covers what the code does not say on its face. Keep it current when the layout, entry points, conventions or naming change.

States are fully observed and use unary and binary predicates only. Transitions are labelled with action names, STRIPS+ actions or full STRIPS actions. Bindings are injective, so every learned schema requires distinct parameters. Evaluation is behavioural: a learned domain is scored by the successor states it generates on the held-out problems `test-1` to `test-3`.

## Layout

- `src/dias/` is the package, installed with `pip install -e .`, with the console script `dias`. It holds `models/` (the LightningModule in `dias.py` and its parts), `data/` (sampling, splitting, noise and the datamodule), `schema/` (domain metadata, the derivation of schemas from parameters in `derivation.py`, and their export to PDDL in `export.py`), `pddl/` (text handling, untyping, the transpiler and pymimir states), `validation/` (the validators and the worker processes that score states) and `reporting/` (the W&B heatmaps of the learned schema).
- `configs/` holds `main.yaml`, the training setup, and the overlays `experiment-{mlp,noise}.yaml`, `labels-{strips,strips-plus}.yaml` and `ablation-*.yaml`.
- `pddl/<domain>/` holds `domain.pddl`, `train.pddl`, `test-{1,2,3}.pddl`, `strips+.yaml`, `config.yaml` and `README.md`. The folder is the domain, and nothing else lists the domains.
- `experiments/` holds the scripts: `launch.py` submits training and validation arrays and sweeps, `train.py` and `validate_runs.py` are the jobs, `collect_results.py` writes the tables, `validate_checkpoint.py` scores one model on one problem, and `schema_sizes.py`, `precondition_categories.py` and `noop_share.py` analyse the learned schemas. The shared code is in `pipeline.py`, the configs in `configs/` (`validation.yaml` and `slurm.yaml`), and the Slurm scripts in `slurm/`. The folder also holds `l1_pddl/`, the scaling study in `scaling/`, and `README.md`, which gives the commands of every table of the paper.
- `logs/`, `local/` and `experiments/results/` are gitignored.

## Commands

```bash
pytest
black --check . && isort --check-only .
dias fit --config configs/main.yaml --config pddl/<domain>/config.yaml
python experiments/launch.py --experiment <name> --backend local --dry-run
```

## Conventions

- **Configs.** A run is composed from `main.yaml`, the domain's `config.yaml`, the overlays of `--experiment-type`, `--labels` and `--ablation`, and finally the `--set` and sweep overrides. Each layer states only what it changes. The class defaults of `DIAS` and `PddlGraphDataModule` are the model of the paper, and `domain_path` and `problem_path` have no default. Every `__init__` argument of the two classes has a line in its `Args:` docstring, which `--model.help` and `--data.help` print. A dataclass-typed argument (`precondition_tau`, `noise`) defaults to None, because jsonargparse merges a configured value onto a default. It is stored in hyper-parameters as a `class_path` and `init_args` mapping (`dias.hparams`).
- **Seeds.** The data seed defaults to the run's `seed_everything` (`main.CLI`), and training is deterministic given the domain and the seed. Validation draws its states with a seed that it records.
- **Reading a model.** The learned schema is read through `DIAS.schema_logits()` only, and it becomes PDDL through `schema.export` (`read_learned_schemas` and `learned_domain`). Every checkpoint is saved with its learned domain beside it as `<checkpoint>.pddl` (`main.LearnedDomainCheckpoints`), and validation scores that file unless a threshold is given.
- **Checkpoints.** The current checkpoint format is 3 (`models/checkpoints.py`). Format 2, in which the runs of the paper were written, loads with its renamed hyper-parameters. Renaming or retiring an argument of `DIAS` needs a new format and a new rule, and an old rule is never edited. `experiments/noop_share.py` maps the retired datamodule arguments of saved configs in the same way.
- **Equality.** By default, `=` has no channel and the learned domain requires distinct parameters. With `include_equality`, it has a channel. Validation writes equality as the declared predicate `equal` in both domains, and the recorded sample seeds reproduce their states only with this rewrite.
- **Validation.** States are scored in subprocesses under an address-space limit (`validation/workers.py`), so the modules they import must not import torch. Results go to `logs/validation/<experiment>/<domain>/seed<seed>.json` with a `status` of `ok`, `partial`, `out_of_memory` or `validation_failed`. The score of each state is appended to a `.states.jsonl` file beside it, from which `--resume` continues.
- **Experiments.** `pipeline.py` is the only place that builds paths, and it imports neither torch nor pymimir. Run ids are `<experiment>-<domain>-s<seed>`, the Slurm array task id is the seed, and each value of a sweep is the experiment `<experiment>-<tag><value>`. Everything specific to a cluster is in the `site configuration` block of the Slurm scripts and in `configs/slurm.yaml`.
- **PyG.** PyG offsets or concatenates a `Data` attribute with `index` or `batch` in its name, which is why the action label is called `y_action`.
- **Goldens.** The fixtures `tests/fixtures/golden.pt` (the model), `recipes.json` (the arguments every README recipe resolves to) and `domains.json` (sampling, the first training steps and scoring on the real domains) pin behaviour. Regenerate them only for an intended change, with `DIAS_REGENERATE_GOLDEN=1 pytest <test>`, and review the diff.
- **Tests.** The tests run on Python 3.10 to 3.13 in CI, in the jobs `format`, `test` and `install`. Use `tests/support.temp_dir`, because `enterContext` needs Python 3.11. Each test module is named after the module it covers.
- **Logging.** The package logs through `logging.getLogger(__name__)` and never prints, and only entry points configure logging. INFO is for events a user waits for, and DEBUG for anything per step. Values go into `%s` arguments.

## Style

- Names follow the paper, so look a concept up there before naming it. The code says `relation` for a predicate that is a channel of the adjacency tensor, and `predicate` for the PDDL declaration. "MLP effects" is a variant of DIAS, and L1 is the symbolic baseline.
- Comments and docstrings say briefly what the code does. They add a constraint only where the code cannot show it, and they contain no history, no justification and nothing restated from the paper.
- Documentation is written in full sentences. Prose is never hard-wrapped, so a paragraph is one line, and it uses neither semicolons nor dashes as punctuation (`--` included).
- Result markdown holds tables only.

| Paper | Code |
| --- | --- |
| predicates `p`, arity `\|p\|` | `relation_names`, `relation_arities` |
| action schema `a`, arity `\|a\|` | `action_names`, `action_arities` |
| keys `K`, queries `Q^a` | the encoder's `node_embeddings`, `SlotSelector.slot_queries` |
| correspondence `C^a`, selection `S̃^a` | `correspondence`, `selection` |
| slot weights `w^a` | `slot_weight_logits` |
| slots `M`, objects `O`, embedding width `d` | `num_slots`, `num_objects`, `object_dim` |
| `P_eff`, `P_pre` | `StaticEffects`, `Preconditions`, with `Channels.positive` and `.negative` |
| `p_pre`, τ | `precondition_satisfaction`, `precondition_tau` |
| `L_main`, `L_aux`, α | `Losses.main`, `Losses.aux`, `aux_loss_alpha` |
| `g_main`, `g_aux` | `main` and `aux` in `PCGrad` |
| derivation of action schemas, its threshold | `schema.derivation.derive_schema`, `derivation_threshold` |
| random node features | `random_node_features` |
| MLP-effects variant | `mlp_effects`, `MlpEffects` |
| `N_min^tr` .. `N_max^val` | `min_per_action_train` .. `max_per_action_valid` |
| soundness, completeness | precision, recall |
