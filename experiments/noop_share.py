"""Share of an experiment's sampled transitions in which the action changed nothing, per domain and action.

Each run's transitions are sampled again as its datamodule sampled them for training, from the run's saved ``config.yaml`` (domain, problem, quotas and seed), and a transition counts as a no-op when its successor equals its state. Train and valid splits are counted together.

Usage:
    python experiments/noop_share.py --experiment dias-names --domains satellite --markdown experiments/results/noop-share.md
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import pipeline`
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

# The sections of a saved run config that LightningCLI(run=False) accepts; the rest belong to the fit subcommand.
CONFIG_SECTIONS = ("seed_everything", "trainer", "model", "data")


def sampled_transitions(config_path: Path):
    """The run's datamodule, set up: the transitions it trained on, sampled again from its saved config."""
    import yaml

    from dias.main import CLI

    config = {k: v for k, v in load_yaml(config_path).items() if k in CONFIG_SECTIONS}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        cli = CLI(
            args=[
                "--config",
                handle.name,
                "--trainer.logger=false",
                "--trainer.accelerator=cpu",
                "--trainer.callbacks=[]",
                "--trainer.enable_checkpointing=false",
            ],
            run=False,
            save_config_callback=None,
        )
    finally:
        sys.argv = argv
        Path(handle.name).unlink()
    datamodule = cli.datamodule
    datamodule.setup("fit")
    return datamodule.base


def count_noops(base) -> dict[str, tuple[int, int]]:
    """Per action name, (no-op transitions, all transitions)."""
    flat = lambda t: t.reshape(t.shape[0], -1)  # noqa: E731
    unchanged = (flat(base.states) == flat(base.next_states)).all(dim=1)
    noops, totals = Counter(), Counter()
    for action, same in zip(base.action_indices.tolist(), unchanged.tolist()):
        name = base.action_names[action]
        totals[name] += 1
        noops[name] += int(same)
    return {name: (noops[name], totals[name]) for name in base.action_names}


def write_markdown(path: Path, results: dict, args: argparse.Namespace) -> None:
    lines = [
        f"# No-op share of sampled transitions (`{args.experiment}`)",
        "",
        "Per action, the share of sampled transitions (train and valid together, summed over the seeds read) whose successor equals the state.",
        "",
        f"`python experiments/noop_share.py --experiment {args.experiment} --markdown {path}`: each run's transitions are sampled again as its datamodule sampled them, from its saved `config.yaml`.",
        "",
        "| domain | action | no-ops | transitions | share |",
        "|---|---|--:|--:|--:|",
    ]
    for domain, (seeds, counts) in results.items():
        for i, (action, (noops, total)) in enumerate(counts.items()):
            share = noops / total if total else float("nan")
            lines.append(
                f"| {domain if i == 0 else ''} | {action} | {noops} | {total} | {share:.1%} |"
            )
    lines.append("")
    lines.append(
        "Seeds read: "
        + "; ".join(f"{d} {len(s)}" for d, (s, _) in results.items())
        + "."
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
        help="Experiment whose runs to sample again, the prefix of the run ids.",
    )
    add_domain_arguments(p)
    p.add_argument(
        "--validation-config",
        type=Path,
        default=VALIDATION_CONFIG,
        help="The validation.yaml whose seeds are read without --seeds or --num-runs.",
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
        write_markdown(args.markdown, results, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
