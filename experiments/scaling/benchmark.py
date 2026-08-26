#!/usr/bin/env python
"""Benchmark one (domain, instance, seed) of the scaling study.

Measures how the DIAS *training step* behaves as the instance grows:

  * throughput      — iterations (training steps) per second, steady state
  * GPU memory      — peak allocated / reserved bytes over the measured window
  * host memory     — peak RSS, and the size of the pre-computed sample pool
  * setup time      — transition sampling + tensor construction

What runs is the ordinary training path: the Lightning config chain of a real
run (``configs/main.yaml`` -> ``pddl/<domain>/config.yaml`` ->
``experiments/scaling/configs/scaling.yaml``) instantiated through
``dias.main.CLI``, with a
callback that times the steps. Nothing about the model or the training step is
stubbed; the overlay only pins the data pipeline and turns off validation,
logging and checkpointing (see the comments in that overlay).

The first ``--warmup`` steps are discarded and the CUDA peak-memory counters are
reset right after them, so neither the CUDA context nor the first-step allocator
growth is counted. The remaining ``--measure`` steps give the throughput; each
step is bracketed by ``torch.cuda.synchronize()`` so the times are real.

Out-of-memory is a result, not a crash: it is caught and written out as
``status: "oom"`` so the ladder can be walked to its end.

Usage
-----
    python experiments/scaling/benchmark.py --domain blocks-3 --instance n0068 --seed 0
    python experiments/scaling/benchmark.py --domain delivery --instance n0150 \
        --seed 0 \
        --experiment scaling-l40s --warmup 20 --measure 100
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
import traceback
from pathlib import Path

import yaml

SCALING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCALING_DIR.parents[1]

# Where a domain's files live is the main pipeline's business, not this
# study's; imported rather than re-derived, as the sbatch helpers are.
sys.path.append(str(SCALING_DIR.parent))
from pipeline import domain_config, domain_pddl  # noqa: E402

SIZES_CONFIG = SCALING_DIR / "configs" / "sizes.yaml"
SCALING_OVERLAY = SCALING_DIR / "configs" / "scaling.yaml"
INSTANCES_DIR = SCALING_DIR / "instances"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "logs" / "scaling"


def result_path(
    results_dir: Path, experiment: str, domain: str, instance: str, seed: int
) -> Path:
    """Where one benchmark point is written (mirrors logs/validation/ layout)."""
    return results_dir / experiment / domain / f"{instance}-s{seed}.json"


def instance_path(domain: str, instance: str) -> Path:
    return INSTANCES_DIR / domain / f"{instance}.pddl"


def _peak_rss_mib() -> float:
    # ru_maxrss is in KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _build_callback():
    """Defined lazily so ``--help`` and the launcher do not need torch."""
    import torch
    from lightning.pytorch.callbacks import Callback

    class ThroughputBenchmark(Callback):
        """Times the training steps and records peak memory over the window.

        Batch counting is done here rather than off ``trainer.global_step``:
        DIAS uses manual optimization, so the global step is whatever the module
        chooses to make it. One callback tick == one training step.
        """

        def __init__(self, warmup: int, measure: int):
            self.warmup = warmup
            self.measure = measure
            self.batches = 0
            self.step_times: list[float] = []
            self.fit_start = time.perf_counter()
            self.pre_train_seconds: float | None = None
            self.window_seconds: float | None = None
            self._cuda = torch.cuda.is_available()
            self._t0 = 0.0
            self._window_start = 0.0

        def _sync(self):
            if self._cuda:
                torch.cuda.synchronize()

        def on_train_start(self, trainer, pl_module):
            # Everything Trainer.fit does before the first training step:
            # datamodule setup, dataloader construction, model transfer.
            self.pre_train_seconds = time.perf_counter() - self.fit_start

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if self.batches == self.warmup:
                # Steady state reached: forget the warm-up peaks and start the
                # clock. reset_peak_memory_stats() sets the peak to the current
                # usage, so the warmed-up allocator pool stays in the number.
                self._sync()
                if self._cuda:
                    torch.cuda.reset_peak_memory_stats()
                self._window_start = time.perf_counter()
            self._sync()
            self._t0 = time.perf_counter()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self._sync()
            now = time.perf_counter()
            if self.batches >= self.warmup:
                self.step_times.append(now - self._t0)
            self.batches += 1
            if self.batches >= self.warmup + self.measure:
                self.window_seconds = now - self._window_start
                trainer.should_stop = True

        def metrics(self) -> dict:
            times = self.step_times
            out = {
                "measured_steps": len(times),
                "pre_train_seconds": self.pre_train_seconds,
                "window_seconds": self.window_seconds,
            }
            if times:
                out.update(
                    {
                        # Median first: robust to a stray descheduled step.
                        "iterations_per_second": 1.0 / statistics.median(times),
                        "iterations_per_second_mean": len(times) / sum(times),
                        "step_time_ms_median": 1e3 * statistics.median(times),
                        "step_time_ms_mean": 1e3 * statistics.fmean(times),
                        "step_time_ms_min": 1e3 * min(times),
                        "step_time_ms_max": 1e3 * max(times),
                        "step_time_ms_stdev": (
                            1e3 * statistics.stdev(times) if len(times) > 1 else 0.0
                        ),
                    }
                )
            if self._cuda:
                out.update(
                    {
                        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated()
                        / 2**20,
                        "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved()
                        / 2**20,
                    }
                )
            return out

    return ThroughputBenchmark


def _lightning_args(
    domain: str,
    domain_file: Path,
    problem_pddl: Path,
    seed: int,
    max_steps: int,
    accelerator: str | None,
) -> list[str]:
    """The config chain of a real run, with the instance swapped in."""
    return [
        "--config",
        str(PROJECT_ROOT / "configs" / "main.yaml"),
        "--config",
        str(domain_config(domain)),
        "--config",
        str(SCALING_OVERLAY),
        "--data.init_args.domain_path",
        str(domain_file),
        "--data.init_args.problem_path",
        str(problem_pddl),
        "--data.init_args.seed",
        str(seed),
        "--seed_everything",
        str(seed),
        "--trainer.max_steps",
        str(max_steps),
        *(["--trainer.accelerator", accelerator] if accelerator else []),
    ]


def run_benchmark(
    domain: str,
    instance: str,
    seed: int,
    warmup: int,
    measure: int,
    accelerator: str | None = None,
) -> dict:
    """Run one benchmark point and return its record (never raises on OOM)."""
    import torch

    from dias.main import CLI

    torch.set_float32_matmul_precision("medium")

    domain_file = domain_pddl(domain)
    problem_pddl = instance_path(domain, instance)
    if not problem_pddl.exists():
        raise SystemExit(
            f"Missing instance {problem_pddl} — run "
            "experiments/scaling/generate_instances.py first."
        )

    record: dict = {
        "domain": domain,
        "instance": instance,
        "seed": seed,
        "warmup_steps": warmup,
        "measure_steps": measure,
        "problem_path": str(problem_pddl.relative_to(PROJECT_ROOT)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "torch_version": torch.__version__,
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        record["gpu_name"] = props.name
        record["gpu_total_mib"] = props.total_memory / 2**20

    callback = _build_callback()(warmup=warmup, measure=measure)
    t_start = time.perf_counter()
    # Filled by the setup wrapper below; declared out here so the `finally`
    # clause can read it even when the CLI itself fails.
    sampling: dict = {}
    try:
        # LightningCLI parses `args`, but warns (and would be ambiguous) while
        # this script's own argv is still visible, so hide it for the call.
        argv, sys.argv = sys.argv, sys.argv[:1]
        try:
            cli = CLI(
                args=_lightning_args(
                    domain,
                    domain_file,
                    problem_pddl,
                    seed,
                    warmup + measure,
                    accelerator,
                ),
                run=False,
                save_config_callback=None,
                # No W&B logger, no checkpoint callbacks: unlike dias.main.main()
                # this run is not supposed to produce anything but timings.
                trainer_defaults={"callbacks": [callback]},
            )
        finally:
            sys.argv = argv
        datamodule, model, trainer = cli.datamodule, cli.model, cli.trainer
        # Time transition sampling by wrapping the hook itself: Lightning runs
        # the *callback* setup hook after the datamodule's, so a callback cannot
        # bracket it, and calling setup() here as well would sample twice.
        inner_setup = datamodule.setup

        def timed_setup(stage=None):
            t0 = time.perf_counter()
            inner_setup(stage)
            sampling["sampling_seconds"] = time.perf_counter() - t0

        datamodule.setup = timed_setup
        record.update(
            {
                "num_objects": datamodule.num_objects,
                "num_relations": datamodule.num_relations,
                "num_actions": datamodule.num_actions,
                "batch_size": datamodule.batch_size,
                "num_slots": model.num_slots,
                "num_parameters": sum(p.numel() for p in model.parameters()),
            }
        )
        callback.fit_start = time.perf_counter()
        trainer.fit(model, datamodule=datamodule)
        record["status"] = "ok"
        # Size of the pre-computed adjacency pool held in host memory — the
        # other quadratic term (num_samples x R x N x N), and in practice the
        # thing that limits the ladder before the GPU does.
        base = datamodule.base
        record["num_samples"] = len(base)
        record["sample_pool_mib"] = (
            base.states.element_size() * base.states.numel() * 2 / 2**20
        )
    except torch.cuda.OutOfMemoryError as exc:
        record["status"] = "oom"
        record["error"] = str(exc).splitlines()[0]
        print(
            f"[oom] {domain}/{instance} seed={seed}: {record['error']}", file=sys.stderr
        )
    except Exception as exc:  # noqa: BLE001 - recorded, the ladder continues
        # CPU OOM and a few CUDA paths surface as plain RuntimeError/MemoryError.
        oom = isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()
        record["status"] = "oom" if oom else "error"
        record["error"] = str(exc).splitlines()[0] if str(exc) else repr(exc)
        traceback.print_exc()
    finally:
        record.update(callback.metrics())
        record.update(sampling)
        record["total_seconds"] = time.perf_counter() - t_start
        record["host_peak_rss_mib"] = _peak_rss_mib()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return record


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--domain", required=True)
    p.add_argument("--instance", required=True, help="Instance id, e.g. n0068.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--experiment", default="scaling", help="Results group name.")
    p.add_argument("--warmup", type=int, default=None, help="Default: sizes.yaml.")
    p.add_argument("--measure", type=int, default=None, help="Default: sizes.yaml.")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--sizes-config", type=Path, default=SIZES_CONFIG)
    p.add_argument(
        "--accelerator", default=None, help="Override trainer.accelerator (e.g. cpu)."
    )
    p.add_argument("--out", type=Path, default=None, help="Override the output path.")
    args = p.parse_args()

    cfg = yaml.safe_load(args.sizes_config.read_text())
    warmup = args.warmup if args.warmup is not None else int(cfg["warmup_steps"])
    measure = args.measure if args.measure is not None else int(cfg["measure_steps"])

    record = run_benchmark(
        domain=args.domain,
        instance=args.instance,
        seed=args.seed,
        warmup=warmup,
        measure=measure,
        accelerator=args.accelerator,
    )
    record["experiment"] = args.experiment

    out = args.out or result_path(
        args.results_dir, args.experiment, args.domain, args.instance, args.seed
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")

    its = record.get("iterations_per_second")
    print(
        f"[{record['status']}] {args.domain}/{args.instance} seed={args.seed}: "
        f"{'%.2f it/s' % its if its else 'no timing'}, "
        f"peak {record.get('gpu_peak_allocated_mib', float('nan')):.0f} MiB allocated "
        f"/ {record.get('gpu_peak_reserved_mib', float('nan')):.0f} MiB reserved "
        f"-> {out}",
        flush=True,
    )
    # OOM is a recorded outcome; only genuine failures are errors.
    return 1 if record["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
