"""Building checkpoints for the validation tests, without training anything."""

from pathlib import Path

import torch

from dias.schema import DomainSchema

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAIN = PROJECT_ROOT / "pddl" / "blocks-3" / "domain.pddl"
TRAIN_PROBLEM = PROJECT_ROOT / "pddl" / "blocks-3" / "train.pddl"
TEST_PROBLEM = PROJECT_ROOT / "pddl" / "blocks-3" / "test-1.pddl"

MODEL_SEED = 5


def datamodule_hparams() -> dict:
    return {
        "domain_path": str(DOMAIN),
        "problem_path": str(TRAIN_PROBLEM),
        "include_statics": True,
        "action_arguments": False,
        "strips_plus_actions": False,
    }


def domain_schema() -> DomainSchema:
    hparams = datamodule_hparams()
    return DomainSchema.from_pddl(
        hparams["domain_path"], hparams["problem_path"], include_statics=True
    )


def build_model(**overrides):
    """A small untrained DIAS for blocks-3, plus the hparams it was built from.

    The narrow encoder keeps the golden fixtures small; nothing here depends on
    the widths.
    """
    from dias.models.dias import DIAS

    schema = domain_schema()
    hparams = {
        "num_objects": schema.num_objects,
        "arity_relations": list(schema.relation_arities),
        "num_actions": schema.num_actions,
        "object_dim": 8,
        "num_slots": max(schema.action_arities),
        "action_arities": list(schema.action_arities),
        "action_arg_positions": [list(p) for p in schema.action_arg_positions],
        "successor_relations": True,
        "parameter_plotting": None,
        "batch_size": 2,
        "encoder_hidden_dims": [4, 8, 8],
        **overrides,
    }
    torch.manual_seed(MODEL_SEED)
    return DIAS(**hparams), hparams


def save(directory: Path, model, model_hparams: dict, model_id: str = "run0") -> str:
    """Write `model` where the validators expect to find a finished run."""
    checkpoints = directory / "dias" / model_id / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "datamodule_hyper_parameters": datamodule_hparams(),
        "pytorch-lightning_version": "2.6.1",
        "epoch": 1,
        "global_step": 10,
    }
    # Stamped the way Lightning would, so it is not mistaken for a v1 file.
    model.on_save_checkpoint(checkpoint)
    torch.save(checkpoint, checkpoints / "best-epoch=1-step=10.ckpt")
    model.teardown("build")
    return model_id
