"""Count, per domain and action, the sampled transitions of an experiment's runs whose successor equals the state, sampling each run's transitions again from its saved config.

Example:
    python experiments/noop_share.py --experiment dias-names --markdown experiments/results/noop-share.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (  # noqa: E402
    LOGS_DIR,
    VALIDATION_CONFIG,
    add_domain_arguments,
    add_logging_arguments,
    configure_logging,
    load_yaml,
    resolve_seeds,
    select_domains,
)
from validate_runs import run_id_for  # noqa: E402

logger = logging.getLogger(__name__)

# Data arguments of earlier runs that no longer exist, with the value now fixed, and renamed ones.
RETIRED_DATA_ARGUMENTS = {
    "include_statics": True,
    "default_statics": ["number", "object"],
    "pt_file": None,
    "min_per_action_train_floor": None,
    "min_per_action_valid_floor": None,
    "batch_sampling_strategy": "stratified",
    "shuffle_train": True,
}
RENAMED_DATA_ARGUMENTS = {"max_steps": "max_walk_steps"}


def sampled_transitions(config_path: Path):
    """The transitions a run sampled, sampled again from its saved config."""
    from dias.data.datamodule import PddlGraphDataModule

    arguments = dict(load_yaml(config_path)["data"]["init_args"])
    for name, value in RETIRED_DATA_ARGUMENTS.items():
        if arguments.pop(name, value) != value:
            raise ValueError(f"{config_path}: {name} other than {value!r}")
    for old, new in RENAMED_DATA_ARGUMENTS.items():
        if old in arguments:
            arguments[new] = arguments.pop(old)
    # The noise corrupts copies of the training states only.
    arguments["noise"] = None
    datamodule = PddlGraphDataModule(**arguments)
    datamodule.setup("fit")
    return datamodule.base


def count_noops(base) -> dict[str, tuple[int, int]]:
    """Per action, (no-op transitions, transitions)."""
    flat = lambda t: t.reshape(t.shape[0], -1)  # noqa: E731
    unchanged = (flat(base.states) == flat(base.next_states)).all(dim=1)
    noops, totals = Counter(), Counter()
    for action, same in zip(base.action_indices.tolist(), unchanged.tolist()):
        name = base.action_names[action]
        totals[name] += 1
        noops[name] += int(same)
    return {name: (noops[name], totals[name]) for name in base.action_names}


def write_markdown(path: Path, experiment: str, results: dict) -> None:
    lines = [
        f"# No-op transitions, {experiment}",
        "",
        "| domain | n | action | no-ops | transitions | share |",
        "|---|---|---|---|---|---|",
    ]
    for domain, (seeds, counts) in results.items():
        for i, (action, (noops, total)) in enumerate(counts.items()):
            first = [domain, str(len(seeds))] if i == 0 else ["", ""]
            share = noops / total if total else float("nan")
            lines.append(
                "| "
                + " | ".join([*first, action, str(noops), str(total), f"{share:.1%}"])
                + " |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--experiment",
        required=True,
        help="Experiment whose runs to sample again.",
    )
    add_domain_arguments(p)
    p.add_argument(
        "--validation-config",
        type=Path,
        default=VALIDATION_CONFIG,
        help="Its seeds are the default.",
    )
    p.add_argument("--log-dir", type=Path, default=LOGS_DIR)
    p.add_argument(
        "--markdown", type=Path, default=None, help="Also write the table here."
    )
    add_logging_arguments(p)
    args = p.parse_args()
    configure_logging(args)

    seeds = resolve_seeds(args)
    results = {}
    for domain in select_domains(args.domains):
        read, pooled = [], {}
        for seed in seeds:
            config = (
                args.log_dir
                / "dias"
                / run_id_for(args.experiment, domain, seed)
                / "config.yaml"
            )
            if not config.is_file():
                logger.warning("%s: no saved config", config.parent.name)
                continue
            for action, (noops, total) in count_noops(
                sampled_transitions(config)
            ).items():
                before = pooled.get(action, (0, 0))
                pooled[action] = (before[0] + noops, before[1] + total)
            read.append(seed)
        if read:
            results[domain] = (read, pooled)
    if not results:
        raise SystemExit(f"no runs of {args.experiment} found under {args.log_dir}")

    for domain, (read, pooled) in results.items():
        for action, (noops, total) in pooled.items():
            print(
                f"{domain:<16} {action:<20} {noops:>7} / {total:<7} {noops / total:6.1%}"
            )
    if args.markdown:
        write_markdown(args.markdown, args.experiment, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
