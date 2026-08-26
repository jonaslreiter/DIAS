#!/usr/bin/env python
"""Aggregate the scaling study into domain x instance-size tables.

Reads ``logs/scaling/<experiment>/<domain>/<instance>-s<seed>.json`` and prints
one matrix per metric (rows = domains, columns = the object counts of the size
ladder), with mean +/- population stdev over the seeds. OOM points show as
``oom`` and missing points as ``-``.

Metrics: iterations/second, peak GPU memory (allocated and reserved, MiB),
step time (ms), sampling time (s), and the seed count behind each cell.

    python experiments/scaling/collect_scaling.py --experiment scaling-a10
    python experiments/scaling/collect_scaling.py --experiment scaling-a10 \
        --markdown experiments/results/scaling.md
    python experiments/scaling/collect_scaling.py --experiment scaling-a10 \
        --csv experiments/results/scaling.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import yaml

from benchmark import DEFAULT_RESULTS_DIR, SIZES_CONFIG
from run_scaling import ladder_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# (json key, column header, format spec). Order defines the printed order.
METRICS = [
    ("iterations_per_second", "iterations / second", "{:.2f}"),
    ("gpu_peak_allocated_mib", "peak GPU allocated (MiB)", "{:.0f}"),
    ("gpu_peak_reserved_mib", "peak GPU reserved (MiB)", "{:.0f}"),
    ("step_time_ms_median", "step time (ms)", "{:.1f}"),
    ("sampling_seconds", "transition sampling (s)", "{:.1f}"),
    ("host_peak_rss_mib", "peak host RSS (MiB)", "{:.0f}"),
]


def load_results(
    results_dir: Path, experiment: str, domain: str
) -> dict[str, list[dict]]:
    """{instance id: [record per seed]} for one domain."""
    out: dict[str, list[dict]] = {}
    directory = results_dir / experiment / domain
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        record = json.loads(path.read_text())
        out.setdefault(record["instance"], []).append(record)
    return out


def _cell(records: list[dict], key: str, fmt: str) -> str:
    if not records:
        return "-"
    values = [
        r[key] for r in records if r.get("status") == "ok" and r.get(key) is not None
    ]
    if not values:
        statuses = {r.get("status", "?") for r in records}
        return "oom" if statuses == {"oom"} else "/".join(sorted(statuses))
    cell = fmt.format(statistics.fmean(values))
    if len(values) > 1:
        cell += " ± " + fmt.format(statistics.pstdev(values))
    return cell


def _columns(
    domains: list[str], ladders: dict[str, list[tuple[int, str]]]
) -> list[int]:
    """The shared ladder rungs, ascending."""
    return sorted({target for d in domains for target, _ in ladders[d]})


def _table(
    domains: list[str],
    ladders: dict[str, list[tuple[int, str]]],
    results: dict[str, dict[str, list[dict]]],
    key: str,
    fmt: str,
) -> tuple[list[str], list[list[str]]]:
    columns = _columns(domains, ladders)
    header = ["domain"] + [str(c) for c in columns]
    rows = []
    for domain in domains:
        by_rung = {t: results[domain].get(i, []) for t, i in ladders[domain]}
        rows.append([domain] + [_cell(by_rung.get(c, []), key, fmt) for c in columns])
    return header, rows


def _print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))
    ]
    print(f"\n{title}")
    print("  ".join(h.rjust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", default="scaling")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--sizes-config", type=Path, default=SIZES_CONFIG)
    p.add_argument("--domains", nargs="+", default=None)
    p.add_argument(
        "--markdown", type=Path, default=None, help="Write tables as markdown."
    )
    p.add_argument("--csv", type=Path, default=None, help="Write the raw per-run rows.")
    args = p.parse_args()

    cfg = yaml.safe_load(args.sizes_config.read_text())
    domains = args.domains or list(cfg["domains"])
    ladders = {d: ladder_for(d, cfg) for d in domains}
    results = {d: load_results(args.results_dir, args.experiment, d) for d in domains}

    found = sum(len(v) for r in results.values() for v in r.values())
    if not found:
        raise SystemExit(
            f"No results under {args.results_dir / args.experiment}. "
            "Run experiments/scaling/launch_scaling.py first."
        )
    print(f"experiment={args.experiment}  runs={found}")
    gpus = {
        r.get("gpu_name", "?") for d in results.values() for v in d.values() for r in v
    }
    print(f"GPU(s): {', '.join(sorted(gpus))}")
    print("columns = ladder rung; cells = mean ± stdev over seeds")

    columns = _columns(domains, ladders)
    sections: list[tuple[str, list[str], list[list[str]]]] = [
        # The rungs are nominal, so state the object count each domain actually
        # reaches before reporting anything against it.
        (
            "objects in the instance",
            ["domain"] + [str(c) for c in columns],
            [
                [d]
                + [
                    # Instance ids encode the exact count: n0068 -> 68.
                    str(int(dict(ladders[d])[c][1:])) if c in dict(ladders[d]) else "-"
                    for c in columns
                ]
                for d in domains
            ],
        )
    ]
    for key, title, fmt in METRICS:
        header, rows = _table(domains, ladders, results, key, fmt)
        sections.append((title, header, rows))
    # How many seeds each cell rests on (counts, not an average). Written as
    # `ok/total` whenever they differ, so a cell whose average hides an OOM or a
    # failed seed says so.

    def _count(domain: str, rung: int) -> str:
        records = results[domain].get(dict(ladders[domain]).get(rung, ""), [])
        ok = sum(1 for r in records if r.get("status") == "ok")
        return str(ok) if ok == len(records) else f"{ok}/{len(records)}"

    sections.append(
        (
            "seeds per cell (ok/total)",
            ["domain"] + [str(c) for c in columns],
            [[d] + [_count(d, c) for c in columns] for d in domains],
        )
    )

    for title, hdr, rws in sections:
        _print_table(title, hdr, rws)

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        blocks = [f"# Scaling study: {args.experiment}", ""]
        blocks.append(f"GPU(s): {', '.join(sorted(gpus))}. ")
        blocks.append("Columns are the number of objects in the instance; cells are")
        blocks.append(f"mean ± stdev over {len(cfg['seeds'])} seeds.\n")
        for title, hdr, rws in sections:
            blocks += [f"## {title}", "", _markdown_table(hdr, rws), ""]
        args.markdown.write_text("\n".join(blocks) + "\n")
        print(f"\nWrote {args.markdown}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "domain", "target", "instance", "num_objects", "seed", "status",
            "iterations_per_second", "step_time_ms_median",
            "gpu_peak_allocated_mib", "gpu_peak_reserved_mib",
            "host_peak_rss_mib", "sampling_seconds", "num_relations",
            "num_actions", "batch_size", "gpu_name",
        ]  # fmt: skip
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for domain in domains:
                for target, instance in ladders[domain]:
                    for record in sorted(
                        results[domain].get(instance, []), key=lambda r: r["seed"]
                    ):
                        writer.writerow({**record, "target": target})
        print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
