"""Building checkpoints for the validation tests, without training anything."""

from pathlib import Path

import torch

from dias.schema import DomainSchema

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAIN = PROJECT_ROOT / "pddl" / "blocks-3" / "domain.pddl"
TRAIN_PROBLEM = PROJECT_ROOT / "pddl" / "blocks-3" / "train.pddl"
TEST_PROBLEM = PROJECT_ROOT / "pddl" / "blocks-3" / "test-1.pddl"

MODEL_SEED = 5


def datamodule_hparams(equality: bool = False) -> dict:
    """What a run stores of its datamodule, where `equality` gives `=` a channel."""
    return {
        "domain_path": str(DOMAIN),
        "problem_path": str(TRAIN_PROBLEM),
        "include_equality": equality,
        "action_arguments": False,
        "strips_plus_actions": False,
    }


def domain_schema(equality: bool = False) -> DomainSchema:
    return DomainSchema.from_pddl(DOMAIN, TRAIN_PROBLEM, include_equality=equality)


def build_model(equality: bool = False, **overrides):
    """A small untrained DIAS for blocks-3, and the hparams it was built from."""
    from dias.models.dias import DIAS

    schema = domain_schema(equality)
    hparams = {
        "num_objects": schema.num_objects,
        "relation_arities": list(schema.relation_arities),
        "num_actions": schema.num_actions,
        "object_dim": 8,
        "aux_loss_alpha": 1.0,
        "num_slots": max(schema.action_arities),
        "action_arities": list(schema.action_arities),
        "action_arg_positions": [list(p) for p in schema.action_arg_positions],
        "encoder_hidden_dims": [4, 8, 8],
        **overrides,
    }
    torch.manual_seed(MODEL_SEED)
    return DIAS(**hparams), hparams


def save(
    directory: Path,
    model,
    model_hparams: dict,
    model_id: str = "run0",
    equality: bool = False,
) -> str:
    """Write `model` where the validators expect to find a finished run."""
    checkpoints = directory / "dias" / model_id / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "datamodule_hyper_parameters": datamodule_hparams(equality),
        "pytorch-lightning_version": "2.6.1",
        "epoch": 1,
        "global_step": 10,
    }
    model.on_save_checkpoint(checkpoint)
    torch.save(checkpoint, checkpoints / "final-epoch=1-step=10.ckpt")
    model.teardown("build")
    return model_id
