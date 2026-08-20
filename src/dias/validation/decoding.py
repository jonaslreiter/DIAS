"""Reading a lifted schema off a trained model's logits."""

import numpy as np
import torch
from torch.nn import functional as F

from dias.models.outputs import Channels


def process_model_logits(
    effect_logits: torch.Tensor | None,
    precondition_logits: torch.Tensor,
    slot_weight_logits: torch.Tensor,
    force_action_args: bool,
    action_arguments: bool,
    strips_plus_actions: bool,
    free_pos_map: dict,
    action_names,
    action_arities,
    num_slots,
    random_flip: bool = False,
    threshold: float = 0.5,
):
    """Binarize the learned schema tensors at *threshold*.

    ``threshold`` is applied uniformly to every probability the schema is read
    off from: slot activation and the four add/delete/positive/negative
    tensors. Lowering it yields more active slots (higher learned arity) and
    both more effects and more preconditions; raising it yields fewer of each.
    0.5 is the default the paper's numbers use.

    `effect_logits` is None for a model whose effects are predicted from the
    state: there is no lifted effect tensor to read off, and the add and delete
    arrays come back as None with it.
    """
    effects = None if effect_logits is None else Channels.from_logits(effect_logits)
    add_effects_probs = None if effects is None else effects.positive
    del_effects_probs = None if effects is None else effects.negative

    preconditions = Channels.from_logits(precondition_logits)
    pos_precond_probs = preconditions.positive
    neg_precond_probs = preconditions.negative

    slot_probs = F.sigmoid(slot_weight_logits)

    # When force_action_args is enabled, compute a mask of known argument
    # slots per action and force those slots to be active in slot_probs.
    # - strips_plus: only the positions listed in strips+.yaml are known
    # - otherwise:   all argument positions (up to action arity) are known
    if action_arguments and force_action_args:
        if strips_plus_actions:
            forced_slot_mask = torch.zeros(
                len(action_names), num_slots, dtype=torch.bool
            )
            for a_idx, name in enumerate(action_names):
                for pos in free_pos_map.get(name, []):
                    if pos < num_slots:
                        forced_slot_mask[a_idx, pos] = True
        else:
            # All positions up to the action arity are known
            slot_indices = torch.arange(num_slots).unsqueeze(0)
            arity_t = torch.tensor(action_arities, dtype=torch.long).unsqueeze(1)
            forced_slot_mask = slot_indices < arity_t  # (num_actions, num_slots)
        slot_probs[forced_slot_mask] = 1.0

    # (num_actions, num_slots)
    active_slots = slot_probs.detach().cpu().numpy() > threshold

    # An inactive slot is not a parameter of the learned action, so nothing may
    # be asserted over it. Shapes: (actions, relations, slots, slots).
    schema = [pos_precond_probs, neg_precond_probs]
    if effects is not None:
        schema += [add_effects_probs, del_effects_probs]
    for action, active in enumerate(active_slots):
        inactive = ~active  # (num_slots,)
        for probabilities in schema:
            probabilities[action, :, inactive, :] = 0
            probabilities[action, :, :, inactive] = 0

    # Threshold after masking
    add_eff = _binarize(add_effects_probs, threshold)
    del_eff = _binarize(del_effects_probs, threshold)
    pos_prec = _binarize(pos_precond_probs, threshold)
    neg_prec = _binarize(neg_precond_probs, threshold)

    if random_flip:
        for arr in (add_eff, del_eff, pos_prec, neg_prec):
            if arr is None:
                continue
            idx = tuple(np.random.randint(0, s) for s in arr.shape)
            arr[idx] = 1 - arr[idx]

    return add_eff, del_eff, pos_prec, neg_prec, active_slots


def _binarize(probabilities, threshold: float):
    """0/1 array at `threshold`, or None for a schema part that does not exist."""
    if probabilities is None:
        return None
    return np.where(probabilities > threshold, 1, 0)
