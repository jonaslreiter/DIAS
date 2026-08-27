"""Building checkpoints for the validation tests, without training anything."""

import re
from pathlib import Path

import torch

from dias.schema import DomainSchema

# What today's parameters were called before the models/ split. Stated independently of dias.models.checkpoints, so a rename that forgets its migration rule fails here.
_PRE_SPLIT_NAMES = {
    "predictor.selector.slot_queries": "graph_edit.slot_encodings",
    "predictor.selector.slot_weight_logits": "graph_edit.slot_weight_logits",
    "predictor.preconditions.logits": "graph_edit.preconditions",
    "predictor.effects.logits": "graph_edit.effects",
}
_ENCODER_LAYER = re.compile(r"^encoder\.(convs|norms)\.(\d+)\.")
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
    """A small untrained DIAS for blocks-3, plus the hparams it was built from."""
    from dias.models.dias import DIAS

    schema = domain_schema()
    hparams = {
        "num_objects": schema.num_objects,
        "relation_arities": list(schema.relation_arities),
        "num_actions": schema.num_actions,
        "object_dim": 8,
        "aux_loss_weight": 1.0,
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
    # Stamped as Lightning would, so it is not mistaken for an older layout.
    model.on_save_checkpoint(checkpoint)
    torch.save(checkpoint, checkpoints / "best-epoch=1-step=10.ckpt")
    model.teardown("build")
    return model_id
