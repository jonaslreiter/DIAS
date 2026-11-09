"""The derivation of action schemas from a model's parameters (paper, Sec. 4.5)."""

import numpy as np
import torch
from torch.nn import functional as F

from dias.models.outputs import Channels


def derive_schema(
    effect_logits: torch.Tensor | None,
    precondition_logits: torch.Tensor,
    slot_weight_logits: torch.Tensor,
    force_action_args: bool,
    action_arguments: bool,
    strips_plus_actions: bool,
    strips_plus_positions: dict,
    action_names,
    action_arities,
    num_slots,
    random_flip: bool = False,
    threshold: float = 0.5,
):
    """Each action's active slots and its add effects, delete effects, positive and negative preconditions over them, as 0/1 arrays ``(A, R, M, M)`` thresholded at `threshold`; add and delete are None without `effect_logits`. `random_flip` flips one random cell of each array."""
    effects = None if effect_logits is None else Channels.from_logits(effect_logits)
    add_probabilities = None if effects is None else effects.positive
    delete_probabilities = None if effects is None else effects.negative

    preconditions = Channels.from_logits(precondition_logits)
    positive_probabilities = preconditions.positive
    negative_probabilities = preconditions.negative

    slot_activations = F.sigmoid(slot_weight_logits)

    # A slot whose object the label names is active.
    if action_arguments and force_action_args:
        if strips_plus_actions:
            forced_slots = torch.zeros(len(action_names), num_slots, dtype=torch.bool)
            for action, name in enumerate(action_names):
                for position in strips_plus_positions.get(name, []):
                    if position < num_slots:
                        forced_slots[action, position] = True
        else:
            slot_indices = torch.arange(num_slots).unsqueeze(0)
            arities = torch.tensor(action_arities, dtype=torch.long).unsqueeze(1)
            forced_slots = slot_indices < arities  # (num_actions, num_slots)
        slot_activations[forced_slots] = 1.0

    active_slots = slot_activations.detach().cpu().numpy() > threshold

    schema = [positive_probabilities, negative_probabilities]
    if effects is not None:
        schema += [add_probabilities, delete_probabilities]
    for action, active in enumerate(active_slots):
        inactive = ~active  # (num_slots,)
        for probabilities in schema:
            probabilities[action, :, inactive, :] = 0
            probabilities[action, :, :, inactive] = 0

    add_effects = _binarize(add_probabilities, threshold)
    delete_effects = _binarize(delete_probabilities, threshold)
    positive_preconditions = _binarize(positive_probabilities, threshold)
    negative_preconditions = _binarize(negative_probabilities, threshold)

    if random_flip:
        for array in (
            add_effects,
            delete_effects,
            positive_preconditions,
            negative_preconditions,
        ):
            if array is None:
                continue
            cell = tuple(np.random.randint(0, size) for size in array.shape)
            array[cell] = 1 - array[cell]

    return (
        add_effects,
        delete_effects,
        positive_preconditions,
        negative_preconditions,
        active_slots,
    )


def _binarize(probabilities, threshold: float):
    """`probabilities` thresholded to 0/1, or None."""
    if probabilities is None:
        return None
    return np.where(probabilities > threshold, 1, 0)
