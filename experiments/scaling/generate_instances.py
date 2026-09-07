#!/usr/bin/env python
"""Regenerate the scaling-study instances from the upstream PDDL generators.

One instance per (domain, size) entry of ``sizes.yaml``, written to ``instances/<domain>/n<num_objects>.pddl`` plus an ``index.json``. Both are committed, so running the study needs neither this script nor the generators, which is why they are not vendored. To change the ladder, build them yourself and point ``--generators-dir`` at the checkout:

    git clone https://github.com/AI-Planning/pddl-generators.git

Each generator emits its own dialect, rewritten here into the repository's ``pddl/<domain>/domain.pddl``:

blocks-3
    ``blocksworld-3ops`` to ``blocksworld``: objects get the ``- object`` type, and one ``(eq b b)`` fact per block, without which the ``(not (eq ?x ?y))`` preconditions are vacuously true.

logistics
    ``logistics-strips`` to ``logistics``: ``OBJ`` -> ``object``, ``in-city`` -> ``loc``, plus a ``(vehicle ?v)`` fact per truck and airplane.

delivery
    Already the repo's domain, so only the problem name changes. It needs `tarski`, so it runs from a throwaway venv under ``experiments/scaling/.venv-delivery``; its CLI seeds nothing and emits a whole tree at once, so a wrapper seeds ``random`` first and the needed instances are picked out.

Usage
-----
    python experiments/scaling/generate_instances.py --generators-dir ../pddl-generators [--domains blocks-3] [--force]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

import yaml

SCALING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCALING_DIR.parents[1]
SIZES_CONFIG = SCALING_DIR / "configs" / "sizes.yaml"
INSTANCES_DIR = SCALING_DIR / "instances"
DEFAULT_DELIVERY_VENV = SCALING_DIR / ".venv-delivery"

# The repo's own domain file, which instances are rewritten into and checked against.
sys.path.append(str(SCALING_DIR.parent))
from pipeline import domain_pddl  # noqa: E402

# Objects implied by a size entry. Checked against the instance; the id is built from it.
OBJECT_COUNT = {
    "blocksworld": lambda s: s["blocks"],
    # airplanes + cities + trucks (= cities, the generator default) + locations
    # + packages.
    "logistics": lambda s: (
        s["airplanes"] + 2 * s["cities"] + s["cities"] * s["city_size"] + s["packages"]
    ),
    # grid^2 cells + packages + the single truck.
    "delivery": lambda s: s["grid"] ** 2 + s["packages"] + 1,
}


def instance_id(num_objects: int) -> str:
    """Instance id: zero-padded object count, so ids sort like the ladder."""
    return f"n{num_objects:04d}"


# ── generator drivers ──────────────────────────────────────────────────────────


UPSTREAM_GENERATORS = "https://github.com/AI-Planning/pddl-generators.git"


def require_generators(path: Path | None) -> Path:
    """The generator checkout, or an error naming how to obtain one."""
    if path is None:
        raise SystemExit(
            f"--generators-dir is required to regenerate instances. Clone the upstream generators first:\n    git clone {UPSTREAM_GENERATORS}"
        )
    if not path.is_dir():
        raise SystemExit(
            f"--generators-dir {path} does not exist. Clone the upstream generators:\n    git clone {UPSTREAM_GENERATORS}"
        )
    return path


def _require_binary(path: Path, build_hint: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"Missing generator binary {path}.\n" f"Build it first:  {build_hint}"
        )


def _gen_blocksworld(generators: Path, size: dict, seed: int) -> str:
    script = generators / "blocksworld" / "blocksworld"
    _require_binary(
        generators / "blocksworld" / "bwstates.1" / "bwstates",
        f"make -C {generators}/blocksworld/bwstates.1",
    )
    _require_binary(
        generators / "blocksworld" / "3ops" / "2pddl" / "2pddl",
        f"make -C {generators}/blocksworld/3ops/2pddl",
    )
    # The script drops a STATES scratch file into the working directory.
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run(
            [str(script), "3", str(size["blocks"]), str(seed)],
            cwd=tmp,
            check=True,
            capture_output=True,
            text=True,
        )
    return out.stdout


def _gen_logistics(generators: Path, size: dict, seed: int) -> str:
    binary = generators / "logistics" / "logistics"
    _require_binary(binary, f"make -C {generators}/logistics")
    out = subprocess.run(
        [
            str(binary),
            "-c",
            str(size["cities"]),
            "-s",
            str(size["city_size"]),
            "-p",
            str(size["packages"]),
            "-a",
            str(size["airplanes"]),
            # The generator requires a seed >= 1.
            "-r",
            str(max(1, seed)),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout


def _ensure_delivery_venv(venv_dir: Path) -> Path:
    """Return the python of a venv that has `tarski` (creating it if needed)."""
    python = venv_dir / "bin" / "python"
    if python.exists():
        probe = subprocess.run(
            [str(python), "-c", "import tarski"], capture_output=True, text=True
        )
        if probe.returncode == 0:
            return python
    print(f"[delivery] creating venv with tarski at {venv_dir}", flush=True)
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "tarski"], check=True
    )
    return python


def _gen_delivery_tree(
    generators: Path, sizes: list[dict], seed: int, venv_dir: Path
) -> dict:
    """Run the delivery generator once; return {(grid, packages): pddl text}.

    Its CLI has no seed and no single-instance mode, always writing the full cross product into ``./output``, so it runs once per domain in a scratch directory behind a seeding wrapper.
    """
    generator = generators / "delivery" / "generate.py"
    domain = generators / "delivery" / "domain.pddl"
    if not generator.exists():
        raise SystemExit(f"Missing {generator} in the generator checkout.")
    python = _ensure_delivery_venv(venv_dir)
    max_grid = max(s["grid"] for s in sizes)
    max_packages = max(s["packages"] for s in sizes)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # generate.py reads "domain.pddl" and writes "output/" relative to cwd.
        shutil.copy(domain, tmp_path / "domain.pddl")
        wrapper = f"import random, sys\nrandom.seed({seed})\nsys.argv = ['generate.py', '--grid_size_splits={max_grid}', '--max_nr_packages={max_packages}', '--nr_instances_per_setup=1']\nexec(open({str(generator)!r}).read())\n"
        subprocess.run(
            [str(python), "-c", wrapper],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        out: dict[tuple[int, int], str] = {}
        for size in sizes:
            path = (
                tmp_path
                / "output"
                / "train"
                / f"instance_s-{size['grid']}_p-{size['packages']}_v-0.pddl"
            )
            if not path.exists():
                raise SystemExit(f"delivery generator did not produce {path.name}")
            out[(size["grid"], size["packages"])] = path.read_text()
    return out


# ── rewriting to the repository's domain dialect ───────────────────────────────

_OBJECTS_RE = re.compile(r"\(:objects(.*?)\)", re.DOTALL | re.IGNORECASE)
_INIT_RE = re.compile(r"\(:init(.*)\(:goal", re.DOTALL | re.IGNORECASE)


def _parse_objects(text: str) -> list[str]:
    match = _OBJECTS_RE.search(text)
    if match is None:
        raise ValueError("no (:objects ...) block found")
    return match.group(1).split()


def _parse_init_atoms(text: str) -> list[str]:
    match = _INIT_RE.search(text)
    if match is None:
        raise ValueError("no (:init ...) block found")
    # Everything up to the last ')' that closes (:init ...).
    body = match.group(1).rsplit(")", 1)[0]
    return [f"({atom.strip()})" for atom in re.findall(r"\(([^()]*)\)", body)]


def _rewrite_blocksworld(text: str, num_objects: int) -> str:
    objects = _parse_objects(text)
    atoms = _parse_init_atoms(text)
    # The repo domain guards its parameters with (not (eq ?x ?y)), which needs these.
    atoms += [f"(eq {o} {o})" for o in objects]
    return _emit_problem(
        name=f"blocksworld-n{num_objects}",
        domain="blocksworld",
        objects=" ".join(objects) + " - object",
        atoms=atoms,
    )


def _rewrite_logistics(text: str, num_objects: int) -> str:
    objects = _parse_objects(text)
    atoms = []
    vehicles: list[str] = []
    for atom in _parse_init_atoms(text):
        parts = atom.strip("()").split()
        name, args = parts[0], parts[1:]
        lowered = name.lower()
        if lowered == "obj":
            name = "object"  # the repo domain calls the package type `object`
        elif lowered == "in-city":
            name = "loc"  # ... and the city topology `loc`
        elif lowered in ("truck", "airplane"):
            vehicles.append(args[0])
        atoms.append(f"({' '.join([name, *args])})")
    # The repo domain shares load/unload via a `vehicle` predicate the generator omits.
    atoms = [f"(vehicle {v})" for v in vehicles] + atoms
    return _emit_problem(
        name=f"logistics-n{num_objects}",
        domain="logistics",
        objects=" ".join(objects),
        atoms=atoms,
    )


def _rewrite_delivery(text: str, num_objects: int) -> str:
    # Already the repo's domain, so only the problem name changes.
    return re.sub(
        r"\(problem\s+[^)]*\)", f"(problem delivery-n{num_objects})", text, count=1
    ).lstrip()


def _emit_problem(name: str, domain: str, objects: str, atoms: list[str]) -> str:
    body = "\n".join(f"    {atom}" for atom in atoms)
    # Training walks from the initial state, so a rebuilt problem gets the empty goal.
    return f"(define (problem {name})\n (:domain {domain})\n (:objects {objects})\n (:init\n{body})\n (:goal  (and)))\n"


REWRITERS = {
    "blocksworld": _rewrite_blocksworld,
    "logistics": _rewrite_logistics,
    "delivery": _rewrite_delivery,
}


# ── verification ───────────────────────────────────────────────────────────────


def _verify(domain_file: Path, problem_pddl: Path, expected_objects: int) -> dict:
    """Load the instance with pymimir and confirm the derived object count, returning the metadata for ``index.json``, or ``{}`` when pymimir is unavailable."""
    try:
        import pymimir
    except ImportError:
        print(
            "[warn] pymimir not importable, skipping the instance check. Run this in the training environment to validate the instances.",
            file=sys.stderr,
        )
        return {}

    domain = pymimir.Domain(domain_file)
    problem = pymimir.Problem(domain, problem_pddl)
    num_objects = len(problem.get_objects())
    if num_objects != expected_objects:
        raise SystemExit(
            f"{problem_pddl}: generated {num_objects} objects but the size entry implies {expected_objects}. Fix OBJECT_COUNT in experiments/scaling/generate_instances.py or the size entry."
        )
    return {
        "num_objects": num_objects,
        "num_actions": len(domain.get_actions()),
        "num_predicates": len(domain.get_predicates()),
        "num_initial_atoms": len(problem.get_initial_state().get_atoms()),
    }


# ── driver ─────────────────────────────────────────────────────────────────────


def generate_domain(
    domain: str,
    spec: dict,
    force: bool,
    generators: Path,
    delivery_venv: Path,
) -> list[dict]:
    generator = spec["generator"]
    seed = int(spec.get("generator_seed", 1))
    sizes = spec["sizes"]
    out_dir = INSTANCES_DIR / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    domain_file = domain_pddl(domain)

    delivery_tree = (
        _gen_delivery_tree(generators, sizes, seed, delivery_venv)
        if generator == "delivery"
        else None
    )

    entries: list[dict] = []
    for size in sizes:
        num_objects = OBJECT_COUNT[generator](size)
        iid = instance_id(num_objects)
        path = out_dir / f"{iid}.pddl"
        if path.exists() and not force:
            print(f"[{domain}] {iid}: exists, skipping (--force to regenerate)")
        else:
            if generator == "blocksworld":
                raw = _gen_blocksworld(generators, size, seed)
            elif generator == "logistics":
                raw = _gen_logistics(generators, size, seed)
            else:
                raw = delivery_tree[(size["grid"], size["packages"])]
            path.write_text(REWRITERS[generator](raw, num_objects))
            print(f"[{domain}] {iid}: wrote {path.relative_to(PROJECT_ROOT)}")

        meta = _verify(domain_file, path, num_objects)
        entries.append(
            {
                "id": iid,
                "domain": domain,
                "generator": generator,
                "generator_seed": seed,
                "params": size,
                "num_objects": num_objects,
                "path": str(path.relative_to(PROJECT_ROOT)),
                **meta,
            }
        )
    return entries


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--generators-dir",
        type=Path,
        default=None,
        help=f"Checkout of the upstream PDDL generators ({UPSTREAM_GENERATORS}), built. Required: they are not vendored, because the instances they produce are committed.",
    )
    p.add_argument("--sizes-config", type=Path, default=SIZES_CONFIG)
    p.add_argument("--domains", nargs="+", default=None, help="Subset of domains.")
    p.add_argument(
        "--force", action="store_true", help="Regenerate instances that already exist."
    )
    p.add_argument(
        "--delivery-venv",
        type=Path,
        default=DEFAULT_DELIVERY_VENV,
        help="Venv with `tarski` for the delivery generator (created if missing).",
    )
    args = p.parse_args()

    generators = require_generators(args.generators_dir)
    cfg = yaml.safe_load(args.sizes_config.read_text())
    domains = args.domains or list(cfg["domains"])
    unknown = [d for d in domains if d not in cfg["domains"]]
    if unknown:
        raise SystemExit(f"Unknown domain(s): {unknown}. Known: {list(cfg['domains'])}")

    index: dict[str, list[dict]] = {}
    for domain in domains:
        index[domain] = generate_domain(
            domain,
            cfg["domains"][domain],
            args.force,
            generators,
            args.delivery_venv,
        )

    # Merged, so a per-domain run does not drop the others.
    index_path = INSTANCES_DIR / "index.json"
    merged = json.loads(index_path.read_text()) if index_path.exists() else {}
    merged.update(index)
    index_path.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"\nWrote {index_path.relative_to(PROJECT_ROOT)}")
    for domain, entries in merged.items():
        ladder = ", ".join(str(e["num_objects"]) for e in entries)
        print(f"  {domain:12s} objects: {ladder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
