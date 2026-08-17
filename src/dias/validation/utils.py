from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def resolve_best_checkpoint(ckpt_dir: Path) -> Path:
    """Return the ``best-epoch*.ckpt`` most recently written in ``ckpt_dir``.

    Re-running a seed (``launch.py --seeds``) reuses the run id and therefore the
    checkpoint directory, and Lightning does not delete the previous attempt's
    file: it side-steps the name clash with a ``-v1`` suffix. Selecting by mtime
    picks the current attempt's checkpoint; plain glob order is arbitrary and
    could silently validate the stale one.
    """
    matches = list(ckpt_dir.glob("best-epoch*.ckpt"))
    if not matches:
        raise FileNotFoundError(
            f"No best-epoch*.ckpt found in {ckpt_dir} (run did not finish?)"
        )
    return max(matches, key=lambda p: p.stat().st_mtime)


def process_model_logits(
    preconditions_logits: torch.tensor,
    effect_logits: torch.tensor,
    slot_weight_logits: torch.tensor,
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
    """
    effect_probs = F.softmax(preconditions_logits, dim=-1)
    add_effects_probs = effect_probs[..., 1]
    del_effects_probs = effect_probs[..., -1]

    precond_probs = F.softmax(effect_logits, dim=-1)
    pos_precond_probs = precond_probs[..., 1]
    neg_precond_probs = precond_probs[..., -1]

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

    active_slots = np.array(slot_probs) > threshold  # shape: (num_actions, num_slots)

    # Zero out rows and columns for inactive slots in effects/preconditions
    # Shapes: (num_actions, num_relations, num_slots, num_slots)
    for a in range(active_slots.shape[0]):
        inactive = ~active_slots[a]  # (num_slots,)
        add_effects_probs[a, :, inactive, :] = 0
        add_effects_probs[a, :, :, inactive] = 0
        del_effects_probs[a, :, inactive, :] = 0
        del_effects_probs[a, :, :, inactive] = 0
        pos_precond_probs[a, :, inactive, :] = 0
        pos_precond_probs[a, :, :, inactive] = 0
        neg_precond_probs[a, :, inactive, :] = 0
        neg_precond_probs[a, :, :, inactive] = 0

    # Threshold after masking
    add_eff = np.where(add_effects_probs > threshold, 1, 0)
    del_eff = np.where(del_effects_probs > threshold, 1, 0)
    pos_prec = np.where(pos_precond_probs > threshold, 1, 0)
    neg_prec = np.where(neg_precond_probs > threshold, 1, 0)

    if random_flip:
        for arr in (add_eff, del_eff, pos_prec, neg_prec):
            idx = tuple(np.random.randint(0, s) for s in arr.shape)
            arr[idx] = 1 - arr[idx]

    return add_eff, del_eff, pos_prec, neg_prec, active_slots
