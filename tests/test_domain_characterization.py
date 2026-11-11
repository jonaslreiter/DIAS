"""Golden on the real domains: the transitions a blocks-3 run samples and its first training steps in each setting, the states validation draws on every domain, and how L1's domains score on them.

Regenerate after an intended change with ``DIAS_REGENERATE_GOLDEN=1 pytest tests/test_domain_characterization.py`` and review the fixture diff.
"""

import hashlib
import json
import os
import unittest
from pathlib import Path

import pymimir
import torch

from dias.main import CLI
from dias.validation.pddl import PddlValidator
from dias.validation.successors import WalkSampling
from test_configs import MAIN, unseeded

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "domains.json"
REGENERATE = os.environ.get("DIAS_REGENERATE_GOLDEN") == "1"
TOLERANCE = 1e-4
DOMAINS = sorted(path.parent.name for path in (ROOT / "pddl").glob("*/domain.pddl"))

SETTINGS = {
    "names": [],
    "strips": ["configs/labels-strips.yaml"],
    "strips-plus": ["configs/labels-strips-plus.yaml"],
    "mlp": ["configs/experiment-mlp.yaml"],
    "noise": ["configs/experiment-noise.yaml"],
}
STEPS = 2


def digest(*tensors) -> str:
    sha = hashlib.sha256()
    for tensor in tensors:
        sha.update(tensor.detach().cpu().to(torch.float64).numpy().tobytes())
    return sha.hexdigest()[:16]


def run(overlays: list[str]) -> dict:
    """What a blocks-3 run samples, and its losses on a validation batch after a few steps, on CPU."""
    args = ["--config", str(MAIN), "--config", str(ROOT / "pddl/blocks-3/config.yaml")]
    for overlay in overlays:
        args += ["--config", str(ROOT / overlay)]
    args += [
        "--seed_everything=0",
        "--trainer.accelerator=cpu",
        "--trainer.logger=false",
        "--trainer.callbacks=[]",
        "--trainer.enable_checkpointing=false",
        "--trainer.enable_progress_bar=false",
        "--trainer.enable_model_summary=false",
        f"--trainer.max_steps={STEPS}",
        "--trainer.limit_val_batches=0",
        "--trainer.num_sanity_val_steps=0",
    ]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with unseeded():
            cli = CLI(args=args, run=False, save_config_callback=None)
            data, model = cli.datamodule, cli.model
            cli.trainer.fit(model, datamodule=data)
            base = data.base
            train = data._train_ds
            noisy = [
                t
                for t in (train.noisy_states, train.noisy_next_states)
                if t is not None
            ]
            batch = next(iter(data.val_dataloader()))
            with torch.no_grad(), torch.random.fork_rng():
                torch.manual_seed(0)
                losses = model.step(batch)
    finally:
        torch.set_num_threads(threads)
    return {
        "transitions": digest(
            base.states, base.next_states, base.action_indices, base.action_args
        ),
        "split": digest(torch.tensor(data._train_idx), torch.tensor(data._valid_idx)),
        "noise": digest(*noisy) if noisy else None,
        "loss_main": float(losses.main),
        "loss_aux": float(losses.aux),
        "parameters": float(
            sum(p.detach().double().abs().sum() for p in model.parameters())
        ),
    }


def problem(domain: str):
    return pymimir.Problem(
        pymimir.Domain(ROOT / "pddl" / domain / "domain.pddl"),
        ROOT / "pddl" / domain / "test-1.pddl",
    )


def drawn_states(domain: str) -> str:
    states = WalkSampling(num_samples=10).draw(problem(domain), 0)
    atoms = [
        sorted(
            str(atom)
            for atom in state.get_atoms(ignore_static=False, ignore_derived=True)
        )
        for state in states
    ]
    return hashlib.sha256(json.dumps(atoms).encode()).hexdigest()[:16]


def l1_scores(domain: str) -> dict:
    validator = PddlValidator(
        ROOT / "pddl" / domain / "domain.pddl",
        ROOT / "experiments" / "l1_pddl" / f"{domain}.pddl",
        ROOT / "pddl" / domain / "test-1.pddl",
        num_workers=4,
    )
    validator.setup(WalkSampling(num_samples=4), seed=0)
    stats = validator.validate()["action_stats"]
    return {action: [s["tp"], s["fp"], s["fn"]] for action, s in sorted(stats.items())}


class DomainCharacterizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = (
            json.loads(FIXTURE.read_text())
            if FIXTURE.exists() and not REGENERATE
            else {}
        )
        cls.results = {}

    @classmethod
    def tearDownClass(cls):
        if REGENERATE:
            FIXTURE.write_text(json.dumps(cls.results, indent=1, sort_keys=True) + "\n")

    def check(self, key, value, close=()):
        self.results[key] = value
        if REGENERATE:
            return
        expected = self.fixture[key]
        for field in close:
            self.assertAlmostEqual(
                value.pop(field),
                expected[field],
                delta=TOLERANCE * max(1.0, abs(expected[field])),
                msg=f"{key} {field}",
            )
        self.assertEqual(
            value, {k: v for k, v in expected.items() if k not in close}, key
        )

    def test_a_blocks_3_run_samples_and_trains_as_recorded(self):
        for setting, overlays in SETTINGS.items():
            with self.subTest(setting=setting):
                self.check(
                    f"run/{setting}",
                    run(overlays),
                    close=("loss_main", "loss_aux", "parameters"),
                )

    def test_validation_draws_the_recorded_states(self):
        for domain in DOMAINS:
            with self.subTest(domain=domain):
                self.check(f"states/{domain}", {"digest": drawn_states(domain)})

    def test_l1_domains_score_as_recorded(self):
        for domain in DOMAINS:
            if not (ROOT / "experiments" / "l1_pddl" / f"{domain}.pddl").exists():
                continue
            with self.subTest(domain=domain):
                self.check(f"l1/{domain}", l1_scores(domain))


if __name__ == "__main__":
    unittest.main()
