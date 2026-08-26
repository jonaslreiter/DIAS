"""Building checkpoints for the validation tests, without training anything."""

import re
from pathlib import Path

import torch

from dias.schema import DomainSchema

# What today's parameters were called before the models/ split, stated here
# rather than read out of dias.models.checkpoints -- a rename that forgets its
# migration rule has to fail a test, not an archived run.
_PRE_SPLIT_NAMES = {
    "graph_edit.selector.slot_encodings": "graph_edit.slot_encodings",
    "graph_edit.selector.slot_weight_logits": "graph_edit.slot_weight_logits",
    "graph_edit.preconditions.logits": "graph_edit.preconditions",
    "graph_edit.effects.logits": "graph_edit.effects",
}
_ENCODER_LAYER = re.compile(r"^state_encoder\.(convs|norms)\.(\d+)\.")
_PRE_SPLIT_ENCODER = {"convs": "conv", "norms": "ln"}


def pre_split_name(name: str) -> str:
    """The name a parameter of today's model used to have."""
    layer = _ENCODER_LAYER.match(name)
    if layer is None:
        return _PRE_SPLIT_NAMES.get(name, name)
    kind = _PRE_SPLIT_ENCODER[layer.group(1)]
    return f"dynamics_model.{kind}{int(layer.group(2)) + 1}.{name[layer.end():]}"


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
