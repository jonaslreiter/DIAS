"""Golden tests pinning the DIAS training path.

A full forward pass and a full three-step training trajectory against a stored fixture, which is the guard for "this change computes the same thing". The training-step golden sees what a forward-only one cannot: RNG order during construction, loss-term order, drift in the PCGrad projection.

To regenerate after an intended change:

    DIAS_REGENERATE_GOLDEN=1 pytest tests/test_training_characterization.py

and review the resulting fixture diff.
"""

import os
import unittest
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from dias.models.dias import DIAS

FIXTURE = Path(__file__).parent / "fixtures" / "golden.pt"
REGENERATE = os.environ.get("DIAS_REGENERATE_GOLDEN") == "1"

TOLERANCE = 1e-6

# Wider: Adam scales a CPU's round-off by the learning rate per step.
TRAJECTORY_TOLERANCE = 1e-4

# The thread count decides how reductions are split.
torch.set_num_threads(1)


NUM_OBJECTS = 4
ARITY_RELATIONS = [1, 2, 2]
NUM_ACTIONS = 2
NUM_SLOTS = 3

MODEL_DEFAULTS = dict(
    num_objects=NUM_OBJECTS,
    relation_arities=ARITY_RELATIONS,
    num_actions=NUM_ACTIONS,
    object_dim=8,
    aux_loss_alpha=1.0,
    use_applicability_loss=False,
    learning_rate=0.01,
    num_slots=NUM_SLOTS,
    action_arities=[2, 2],
    action_arg_positions=[[0, 1], [0]],
    encoder_hidden_dims=[4, 8, 8],
)

FLAG_CASES = {
    "full": {},
    "mlp_effects": {"mlp_effects": True},
    "no_slot_weights": {"use_slot_weights": False},
    "no_random_features": {"random_node_features": False},
}


def make_transition(action: int, shift: int) -> tuple[Data, Data]:
    """One <s, a, s'> pair: s' differs from s by one relabelled binary edge."""
    unary = torch.tensor([[0, 1], [0, 1]])
    binary = torch.tensor([[0, 1, 2], [1, 2, 3]])
    state = Data(
        edge_index=torch.cat([unary, binary], dim=1),
        edge_attr=torch.tensor([0, 0, 1, 1, 2]),
        num_nodes=NUM_OBJECTS,
    )
    state.y_action = torch.tensor([action])
    first = shift % NUM_OBJECTS
    state.y_action_args = torch.tensor([[first, (first + 1) % NUM_OBJECTS]])

    successor = Data(
        edge_index=torch.cat([unary, binary.roll(shifts=1, dims=1)], dim=1),
        edge_attr=torch.tensor([0, 0, 1, 2, 2]),
        num_nodes=NUM_OBJECTS,
    )
    successor.y_action = state.y_action
    successor.y_action_args = state.y_action_args
    return state, successor


def make_batch(actions=(0, 1), shift: int = 0) -> tuple[Batch, Batch]:
    pairs = [make_transition(a, shift + i) for i, a in enumerate(actions)]
    return (
        Batch.from_data_list([s for s, _ in pairs]),
        Batch.from_data_list([n for _, n in pairs]),
    )


class Golden:
    """Compare against the stored fixture, or rewrite it under DIAS_REGENERATE_GOLDEN."""

    def __init__(self):
        self.stored = {} if REGENERATE else torch.load(FIXTURE, weights_only=True)
        self.produced = {}

    def check(
        self,
        case: unittest.TestCase,
        key: str,
        value: torch.Tensor,
        atol: float = TOLERANCE,
    ) -> None:
        value = value.detach().clone()
        self.produced[key] = value
        if REGENERATE:
            return
        case.assertIn(key, self.stored, f"{key} missing from {FIXTURE.name}")
        torch.testing.assert_close(
            value, self.stored[key], rtol=0, atol=atol, msg=lambda m: f"{key}: {m}"
        )

    def save(self) -> None:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.produced | self.stored, FIXTURE)


GOLDEN = Golden()


def tearDownModule():
    if REGENERATE:
        GOLDEN.save()


class ForwardCharacterizationTest(unittest.TestCase):
    def test_forward_outputs_match_golden(self):
        for name, flags in FLAG_CASES.items():
            with self.subTest(case=name):
                torch.manual_seed(11)
                model = DIAS(**(MODEL_DEFAULTS | flags))
                self.addCleanup(model.teardown, "test")
                model.eval()

                torch.manual_seed(29)
                prediction = model(*make_batch())

                for label, value in (
                    ("adjacency_out", prediction.adjacency),
                    ("ignore_effects", prediction.effects.ignore),
                    ("ignore_precond", prediction.preconditions.ignore),
                    ("delta", prediction.delta),
                ):
                    GOLDEN.check(self, f"forward/{name}/{label}", value)

    def test_step_losses_match_golden(self):
        for name, flags in FLAG_CASES.items():
            with self.subTest(case=name):
                torch.manual_seed(11)
                model = DIAS(**(MODEL_DEFAULTS | flags))
                self.addCleanup(model.teardown, "test")
                model.eval()

                torch.manual_seed(29)
                losses = model.step(make_batch())
                for key, value in (
                    ("loss_main", losses.main),
                    ("loss_aux", losses.aux),
                    ("hamming_distance", losses.hamming_distance),
                ):
                    GOLDEN.check(self, f"step/{name}/{key}", value)


def trained_values(model: DIAS) -> dict[str, torch.Tensor]:
    """Every parameter, the schema tensors arity-masked."""
    schema = model.schema_logits()
    masked = {
        "predictor.preconditions.logits": schema.preconditions,
        "predictor.effects.logits": schema.effects,
    }
    return {
        name: masked.get(name, parameter)
        for name, parameter in model.named_parameters()
    }


class TrainingTrajectoryTest(unittest.TestCase):
    """Three real training_steps, then every parameter compared to the fixture."""

    def run_steps(self, **flags) -> DIAS:
        torch.manual_seed(11)
        model = DIAS(**(MODEL_DEFAULTS | flags))
        self.addCleanup(model.teardown, "test")
        self.addCleanup(torch.use_deterministic_algorithms, False)

        batches = DataLoader([make_batch(shift=i) for i in range(3)], batch_size=None)
        trainer = L.Trainer(
            deterministic=True,
            max_steps=3,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            limit_val_batches=0,
            num_sanity_val_steps=0,
        )
        torch.manual_seed(29)
        trainer.fit(model, train_dataloaders=batches)
        return model

    def test_parameters_after_three_steps_match_golden(self):
        for name in ("full", "no_pcgrad"):
            with self.subTest(case=name):
                flags = {} if name == "full" else {"use_pcgrad": False}
                model = self.run_steps(**flags)
                for param_name, value in sorted(trained_values(model).items()):
                    GOLDEN.check(
                        self,
                        f"train/{name}/{param_name}",
                        value,
                        atol=TRAJECTORY_TOLERANCE,
                    )


if __name__ == "__main__":
    unittest.main()
