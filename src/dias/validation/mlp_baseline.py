from __future__ import annotations

import multiprocessing as _mp
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pymimir
import torch
import torch.nn.functional as F
from pymimir.advanced.formalism import FluentGroundAtomList, ObjectList
from pymimir.wrapper_formalism import State
from tqdm import tqdm

from dias.data.datamodule import PddlGraphDataModule
from dias.data.generate_samples import generate_bfs
from dias.models.dias import DIAS
from dias.pddl.states import state_fluents, transfer_state
from dias.pddl.transpiler import PDDLTranspiler
from dias.pddl.untype import extract_domain_header, strip_problem_types
from dias.schema import DomainSchema

from .artifacts import PddlWorkspace
from .utils import process_model_logits, resolve_best_checkpoint


# ──────────────────────────────────────────────────────────────────────────
#  Phase-1 worker (top-level so it can be pickled by spawn-mode pool).
#  pymimir state objects can't cross processes, so each worker reloads
#  Domain/Problem from PDDL files and reconstructs each state from a
#  serializable payload of (predicate_name, term_names) tuples.
# ──────────────────────────────────────────────────────────────────────────
def _phase1_worker(
    payloads: list[list[tuple[str, tuple[str, ...]]]],
    config: dict,
):
    original_domain = pymimir.Domain(Path(config["original_domain_path"]))
    original_problem = pymimir.Problem(
        original_domain, Path(config["original_problem_path"])
    )
    learned_domain = pymimir.Domain(Path(config["learned_domain_path"]))
    learned_problem = pymimir.Problem(
        learned_domain, Path(config["learned_problem_path"])
    )

    relation_arities = config["relation_arities"]
    relation_name_to_idx = config["relation_name_to_idx"]
    object_name_to_idx = config["object_name_to_idx"]
    action_name_to_idx = config["action_name_to_idx"]
    num_objects = config["num_objects"]
    R = config["num_relations"]
    eq_channel_idx = config["eq_channel_idx"]
    prefix = config["learned_action_prefix"]
    prefix_len = len(prefix)
    max_arity = config["max_arity"]
    keep_channels = (
        np.array([i for i in range(R) if i != eq_channel_idx], dtype=np.int64)
        if eq_channel_idx is not None
        else None
    )

    def state_to_adj(state) -> np.ndarray:
        adj = np.zeros((R, num_objects, num_objects), dtype=np.uint8)
        for atom in state.get_atoms(ignore_static=False, ignore_derived=True):
            r = relation_name_to_idx.get(atom.get_predicate().get_name())
            if r is None:
                continue
            arity = relation_arities[r]
            terms = atom.get_terms()
            if arity == 2:
                i = object_name_to_idx[terms[0].get_name()]
                j = object_name_to_idx[terms[1].get_name()]
                adj[r, i, j] = 1
            elif arity == 1:
                i = object_name_to_idx[terms[0].get_name()]
                adj[r, i, i] = 1
            elif arity == 0:
                adj[r, 0, 0] = 1
        return adj

    def adj_signature(adj: np.ndarray) -> bytes:
        if keep_channels is not None:
            adj = adj[keep_channels]
        return adj.tobytes()

    def reconstruct_state(payload, target_problem):
        adv_problem = target_problem._advanced_problem
        fluent_pred_by_name = {
            p.get_name(): p for p in adv_problem.get_domain().get_fluent_predicates()
        }
        obj_by_name = {o.get_name(): o for o in adv_problem.get_objects()}
        target_adv_atoms = []
        for pred_name, term_names in payload:
            fluent_pred = fluent_pred_by_name.get(pred_name)
            if fluent_pred is None:
                continue
            objs = ObjectList([obj_by_name[n] for n in term_names])
            adv_atom = adv_problem.get_or_create_ground_atom(fluent_pred, objs)
            target_adv_atoms.append(adv_atom)
        fluent_list = FluentGroundAtomList(target_adv_atoms)
        (
            adv_state,
            _,
        ) = target_problem._search_context.get_state_repository().get_or_create_state(
            fluent_list, []
        )
        return State(adv_state, target_problem)

    chunk_S = len(payloads)
    input_adjs_arr = np.zeros((chunk_S, R, num_objects, num_objects), dtype=np.uint8)
    orig_sigs_list: list[set[bytes]] = []
    state_idx_local: list[int] = []
    action_idx_local: list[int] = []
    binding_rows: list[np.ndarray] = []

    for i, payload in enumerate(payloads):
        orig_state = reconstruct_state(payload, original_problem)
        learned_state = reconstruct_state(payload, learned_problem)

        input_adjs_arr[i] = state_to_adj(orig_state)

        sigs: set[bytes] = set()
        for ga in orig_state.generate_applicable_actions():
            sigs.add(adj_signature(state_to_adj(ga.apply(orig_state))))
        orig_sigs_list.append(sigs)

        for ga in learned_state.generate_applicable_actions():
            name = ga.get_action().get_name()
            if not name.startswith(prefix):
                continue
            a_idx = action_name_to_idx[name[prefix_len:]]
            objs = ga.get_objects()
            row = np.full(max_arity, -1, dtype=np.int64)
            for p, o in enumerate(objs):
                if p >= max_arity:
                    break
                row[p] = object_name_to_idx[o.get_name()]
            binding_rows.append(row)
            action_idx_local.append(a_idx)
            state_idx_local.append(i)

    if binding_rows:
        bindings_padded = np.stack(binding_rows, axis=0)
    else:
        bindings_padded = np.zeros((0, max_arity), dtype=np.int64)
    state_idx_arr = np.asarray(state_idx_local, dtype=np.int64)
    action_idx_arr = np.asarray(action_idx_local, dtype=np.int64)

    return (
        input_adjs_arr,
        orig_sigs_list,
        state_idx_arr,
        action_idx_arr,
        bindings_padded,
    )


class ModelValidation:
    LEARNED_ACTION_PREFIX = "learned_"
    EQUALITY_PREDICATE_NAME = "eq"

    def __init__(self, model_id, logs_path_abs, problem_path=None):
        self.model_id = model_id
        self.logs_path_abs = logs_path_abs

        self.sub_task_args = {
            "model_id": model_id,
            "logs_path_abs": logs_path_abs,
            "problem_path": problem_path,
        }

        ckpt_dir = Path(f"{logs_path_abs}/dias/{model_id}/checkpoints")
        self.ckpt_file = str(resolve_best_checkpoint(ckpt_dir))

        ckpt = torch.load(self.ckpt_file, map_location="cpu")
        _legacy_keys = {
            "method",
            "num_val",
            "num_test",
            "balanced_min_per_label",
            "balanced_max_steps",
            "inapplicable_actions_ratio",
        }
        datamodule_hparams = {
            k: v
            for k, v in ckpt["datamodule_hyper_parameters"].items()
            if not k.startswith("_") and k not in _legacy_keys
        }
        # Relative paths in a checkpoint were relative to the working directory
        # the run was launched from, which is the repository root.
        for key in ("domain_path", "problem_path", "pt_file"):
            if datamodule_hparams.get(key) is not None:
                datamodule_hparams[key] = str(Path(datamodule_hparams[key]).resolve())

        self.schema = DomainSchema.from_pddl(
            datamodule_hparams["domain_path"],
            datamodule_hparams["problem_path"],
            include_statics=datamodule_hparams.get("include_statics", False),
            default_statics=datamodule_hparams.get("default_statics"),
            action_arguments=datamodule_hparams.get("action_arguments", False),
            strips_plus_actions=datamodule_hparams.get("strips_plus_actions", False),
        )

        self.model = DIAS.load_from_checkpoint(self.ckpt_file, map_location="cpu")
        self.model.to("cpu")
        self.model.eval()

        if not getattr(self.model.graph_edit, "state_dependent_effects", False):
            raise ValueError(
                "validate_mlp_baseline expects a model with "
                "state_dependent_effects=True. Use validate.py for the "
                "static-effects baseline."
            )

        free_pos_map = PddlGraphDataModule._load_strips_plus_yaml(
            datamodule_hparams["domain_path"],
        )
        action_arguments = datamodule_hparams.get("action_arguments", False)
        strips_plus_actions = datamodule_hparams.get("strips_plus_actions", False)

        # Note: process_model_logits has the precondition/effect arguments
        # swapped (see validate.py for the same call).  We follow that
        # convention here so the per-relation/per-slot tensors land in the
        # right buckets.  Effects come from the MLP at predict time, so the
        # add/del arrays produced here are only used to drive the
        # active-slots mask -- we discard their content for action effects.
        precondition_logits = self.model.graph_edit.preconditions.detach().cpu()
        slot_weights = self.model.graph_edit.slot_weight_logits.detach().cpu()

        # process_model_logits expects two logit tensors of identical
        # shape; the effects MLP doesn't have a static effects tensor any
        # more, so we feed a zeros tensor of the same shape.  Its output
        # add/del arrays will all be zero, which is fine (we use only the
        # precondition arrays + active_slots).
        zero_effect_logits = torch.zeros_like(precondition_logits)

        _, _, pos_prec, neg_prec, active_slots = process_model_logits(
            preconditions_logits=zero_effect_logits,
            effect_logits=precondition_logits,
            slot_weight_logits=slot_weights,
            force_action_args=self.model.force_action_args,
            action_arguments=action_arguments,
            strips_plus_actions=strips_plus_actions,
            free_pos_map=free_pos_map,
            action_names=list(self.schema.action_names),
            action_arities=list(self.schema.action_arities),
            num_slots=self.model.num_slots,
            random_flip=False,
        )

        original_domain_path = datamodule_hparams["domain_path"]
        original_domain_text = Path(original_domain_path).read_text()
        has_equality = self._domain_uses_equality(original_domain_text)

        relation_names = list(self.schema.relation_names)
        relation_arities = list(self.schema.relation_arities)
        if not has_equality:
            keep = [i for i, n in enumerate(relation_names) if n != "="]
            pos_prec = pos_prec[:, keep, :, :]
            neg_prec = neg_prec[:, keep, :, :]
            relation_names = [relation_names[i] for i in keep]
            relation_arities = [relation_arities[i] for i in keep]

        transpiler_predicate_names = [
            self.EQUALITY_PREDICATE_NAME if has_equality and n == "=" else n
            for n in relation_names
        ]

        self.relation_names = relation_names
        self.relation_arities = relation_arities
        self.has_equality = has_equality
        self.eq_channel_idx = (
            relation_names.index("=") if "=" in relation_names else None
        )
        self.action_names = list(self.schema.action_names)
        self.action_arities = list(self.schema.action_arities)
        self.num_slots = int(self.model.num_slots)
        self.num_objects = int(self.schema.num_objects)
        self.active_slots = active_slots  # (num_actions, num_slots), bool

        # Per-action ordered list of active slot indices.  These become the
        # parameter positions in the learned PDDL action: param k is named
        # ?x{active_slot_indices[a][k]}, so the precondition GroundAtoms
        # produced by the transpiler (which reference slot indices) line
        # up with the action's parameters even when active slots are not
        # contiguous (per_action_slots=False).  When pymimir grounds the
        # action, we recover the per-slot binding by mapping parameter
        # position back to slot index using this list.
        self.active_slot_indices = [
            [int(i) for i in range(self.num_slots) if bool(active_slots[a][i])]
            for a in range(len(self.action_names))
        ]

        # Use the transpiler only to derive precondition GroundAtoms; we
        # write the action blocks ourselves so we can force empty effects
        # and a fixed parameter list per action.
        zero_eff = np.zeros_like(pos_prec)
        pddl_transpiler = PDDLTranspiler(
            add_effects=zero_eff,
            delete_effects=zero_eff,
            positive_preconditions=pos_prec,
            negative_preconditions=neg_prec,
            predicate_names=transpiler_predicate_names,
            predicate_arities=relation_arities,
            action_names=list(self.schema.action_names),
        )
        pddl_transpiler.generate_action_schemas()

        num_learned_actions = len(pddl_transpiler.action_schemas)
        for schema in pddl_transpiler.action_schemas:
            schema.name = f"{self.LEARNED_ACTION_PREFIX}{schema.name}"

        # All learned actions have :effect (), which would make pymimir
        # classify every predicate as static -- breaking transfer_state
        # because fluent atoms from the source state would be silently
        # dropped (and the transferred state would inherit the learned
        # problem's :init atoms instead).  add_unreachable_action keeps
        # every predicate fluent by giving a single never-applicable
        # action effects on all of them.  The guard predicate it
        # introduces is never present in :init so the action is
        # unreachable and can't pollute applicable-action enumeration.
        pddl_transpiler.add_unreachable_action()

        # Build the learned-only domain.
        header = extract_domain_header(
            original_domain_text,
            rel_names=pddl_transpiler.predicate_names,
            predicate_arities=pddl_transpiler.predicate_arities,
            keep_equality_requirement=False,
        )
        action_blocks = [
            self._format_learned_action(
                schema=schema, slot_indices=self.active_slot_indices[a]
            )
            for a, schema in enumerate(
                pddl_transpiler.action_schemas[:num_learned_actions]
            )
        ]
        # The dummy unreachable action keeps the transpiler's own serializer,
        # whose non-empty effects on every predicate are the point of it.
        action_blocks.append(
            str(pddl_transpiler.action_schemas[num_learned_actions]).rstrip()
        )

        self.workspace = PddlWorkspace()
        self.learned_domain_filename = self.workspace.write(
            "learned-domain.pddl",
            header + "\n\n" + "\n\n".join(action_blocks) + "\n)" + "\n",
        )
        self.learned_problem_filename = self.workspace.write(
            "learned-problem.pddl",
            strip_problem_types(
                Path(problem_path).read_text(),
                original_domain_text,
                equality_predicate_name=(
                    self.EQUALITY_PREDICATE_NAME if has_equality else None
                ),
            ),
        )

        self.learned_domain = pymimir.Domain(Path(self.learned_domain_filename))
        self.learned_problem = pymimir.Problem(
            self.learned_domain, Path(self.learned_problem_filename)
        )

        self.original_domain_path = str(original_domain_path)
        self.original_problem_path = str(problem_path)
        self.original_domain = pymimir.Domain(Path(original_domain_path))
        self.original_problem = pymimir.Problem(
            self.original_domain, Path(problem_path)
        )

        self.num_objects = len(self.original_problem.get_objects())
        self.model.num_objects = len(self.original_problem.get_objects())
        self.model.graph_edit.num_objects = len(self.original_problem.get_objects())
        self.model.eval()

        # Object name → adjacency-tensor index.  Built from the original
        # problem so input/output adjacencies share the same indexing as
        # the training tensors.
        self.object_name_to_idx = {
            o.get_name(): o.get_index() for o in self.original_problem.get_objects()
        }
        # Sanity: indices should be a contiguous permutation of [0, N).
        assert sorted(self.object_name_to_idx.values()) == list(range(self.num_objects))

        self.relation_name_to_idx = {n: i for i, n in enumerate(relation_names)}
        self.action_name_to_idx = {n: i for i, n in enumerate(self.action_names)}

        self.state_pairs: list[tuple] | None = None

    # ──────────────────────────────────────────────────────────────────
    #  setup
    # ──────────────────────────────────────────────────────────────────
    def setup(self, num_samples: int = 250, sample_population: int = 500):
        if num_samples > sample_population:
            num_samples = sample_population
            print(
                f"\033[93m[WARNING] num_samples ({num_samples}) exceeds "
                f"population, clamping to {sample_population}.\033[0m"
            )

        bfs_samples = generate_bfs(self.original_problem, sample_population)
        seen = set()
        population_set = []
        for s in bfs_samples:
            st = s["_pymimir_state"]
            sid = st.get_index()
            if sid not in seen:
                seen.add(sid)
                population_set.append(st)
            if len(population_set) >= sample_population:
                break

        if num_samples > len(population_set):
            print(
                f"\033[93m[WARNING] num_samples ({num_samples}) exceeds reachable "
                f"states ({len(population_set)}), clamping.\033[0m"
            )
            num_samples = len(population_set)

        sampled_orig_states = random.sample(population_set, num_samples)

        # Sanity check: the shared-fluents view should agree across
        # original / learned problems for every sampled state.
        shared_preds = {p.get_name() for p in self.original_domain.get_predicates()} & {
            p.get_name() for p in self.learned_domain.get_predicates()
        }

        self.state_pairs = []
        for orig_state in sampled_orig_states:
            learned_state = transfer_state(orig_state, self.learned_problem)
            orig_shared = state_fluents(orig_state, shared_preds)
            learned_shared = state_fluents(learned_state, shared_preds)
            assert orig_shared == learned_shared, (
                f"State transfer mismatch (shared predicates):\n"
                f"  shared: {sorted(shared_preds)}\n"
                f"  only in original: {orig_shared - learned_shared}\n"
                f"  only in learned:  {learned_shared - orig_shared}"
            )
            self.state_pairs.append((orig_state, learned_state))

    def validate(self, batch_size: int = 4096, num_workers: int = 1):
        """
        Validates states by batching all MLP forward passes on GPU.

        Phase 1 (pymimir, CPU): for every state pair we enumerate the
        ground-truth applicable actions / successors and the learned-domain
        applicable groundings.  This is the slow part for domains with
        large grounding spaces; pass ``num_workers > 1`` to spread the
        state pairs across worker processes (pymimir state instances can
        not be shared across processes, so each worker reloads the domain
        / problem from PDDL files and reconstructs states from atom
        payloads).

        Phase 2 (GPU): all groundings across all state pairs are flattened
        into one batch and run through the effects MLP in ``batch_size``
        chunks, so the MLP runs once per chunk instead of once per
        (state, action).
        """
        if self.state_pairs is None:
            raise RuntimeError("\033[91mCall setup() before validate().\033[0m")

        R = len(self.relation_names)
        N = self.num_objects
        K = self.num_slots
        S = len(self.state_pairs)

        # ── Phase 1: pymimir / CPU work (optionally multi-process) ────────
        # Build a serializable atom payload per state pair (all fluent atoms
        # are reconstructed in the worker; static atoms come from :init).
        payloads = [
            [
                (
                    atom.get_predicate().get_name(),
                    tuple(t.get_name() for t in atom.get_terms()),
                )
                for atom in orig_state.get_atoms(
                    ignore_static=False, ignore_derived=True
                )
            ]
            for orig_state, _ in self.state_pairs
        ]

        # max_arity is determined by the action schemas (one parameter per
        # active slot), so we can compute it upfront and tell the workers
        # to return already-padded binding rows -- no per-grounding Python
        # loop in the main process.
        max_arity = max((len(s) for s in self.active_slot_indices), default=0)

        worker_config = {
            "original_domain_path": self.original_domain_path,
            "original_problem_path": self.original_problem_path,
            "learned_domain_path": self.learned_domain_filename,
            "learned_problem_path": self.learned_problem_filename,
            "relation_arities": list(self.relation_arities),
            "relation_name_to_idx": dict(self.relation_name_to_idx),
            "object_name_to_idx": dict(self.object_name_to_idx),
            "action_name_to_idx": dict(self.action_name_to_idx),
            "num_relations": R,
            "num_objects": N,
            "eq_channel_idx": self.eq_channel_idx,
            "learned_action_prefix": self.LEARNED_ACTION_PREFIX,
            "max_arity": max_arity,
        }

        # Run Phase 1 (collect ordered list of (start_idx, worker_result)).
        chunk_results: list[tuple[int, tuple]] = []

        if num_workers <= 1 or S <= 1:
            chunk_results.append((0, _phase1_worker(payloads, worker_config)))
        else:
            num_workers = min(num_workers, S)
            chunk_size = (S + num_workers - 1) // num_workers
            chunks = [
                (start, payloads[start : start + chunk_size])
                for start in range(0, S, chunk_size)
            ]
            mp_ctx = _mp.get_context("spawn")
            ordered: list[tuple] = [None] * len(chunks)  # type: ignore
            fut_to_idx = {}
            with ProcessPoolExecutor(
                max_workers=num_workers, mp_context=mp_ctx
            ) as pool:
                for chunk_idx, (start, chunk_payloads) in enumerate(chunks):
                    fut = pool.submit(_phase1_worker, chunk_payloads, worker_config)
                    fut_to_idx[fut] = chunk_idx
                for fut in tqdm(
                    as_completed(fut_to_idx),
                    total=len(fut_to_idx),
                    desc=f"phase1 ({num_workers}p)",
                ):
                    ordered[fut_to_idx[fut]] = fut.result()
            chunk_results = list(zip([c[0] for c in chunks], ordered))

        # Concatenate worker outputs in original state order.
        per_state_orig_sigs: list[set[bytes]] = []
        input_adj_parts: list[np.ndarray] = []
        state_idx_parts: list[np.ndarray] = []
        action_idx_parts: list[np.ndarray] = []
        binding_parts: list[np.ndarray] = []
        for start, res in chunk_results:
            input_adjs_arr, orig_sigs_list, s_local, a_local, bind_local = res
            per_state_orig_sigs.extend(orig_sigs_list)
            input_adj_parts.append(input_adjs_arr)
            # Offset local state indices to global.
            state_idx_parts.append(s_local + start if s_local.size else s_local)
            action_idx_parts.append(a_local)
            binding_parts.append(bind_local)

        if input_adj_parts:
            input_adj_np = np.concatenate(input_adj_parts, axis=0)
        else:
            input_adj_np = np.zeros((0, R, N, N), dtype=np.uint8)
        flat_state_idx_np = (
            np.concatenate(state_idx_parts)
            if state_idx_parts
            else np.zeros(0, dtype=np.int64)
        )
        flat_action_idx_np = (
            np.concatenate(action_idx_parts)
            if action_idx_parts
            else np.zeros(0, dtype=np.int64)
        )
        flat_bindings_np = (
            np.concatenate(binding_parts, axis=0)
            if binding_parts
            else np.zeros((0, max_arity), dtype=np.int64)
        )

        # Move model to GPU only after Phase 1 finishes — keeps spawned
        # workers from inheriting an initialized CUDA context.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        # ── Phase 2: GPU batched prediction ───────────────────────────────
        per_state_pred_sigs: list[set[bytes]] = [set() for _ in range(S)]
        total = int(flat_state_idx_np.shape[0])

        # Cache python list once — we'll use it to assign signatures back per
        # state in the predict loop without round-tripping through tensors.
        flat_state_idx_list = flat_state_idx_np.tolist()

        if total > 0:
            with torch.no_grad():
                # One bulk H2D copy per array.
                all_input_adj = torch.from_numpy(input_adj_np).to(
                    device, dtype=torch.float32
                )  # (S, R, N, N)
                state_idx_t = torch.from_numpy(flat_state_idx_np).to(device)
                action_idx_t = torch.from_numpy(flat_action_idx_np).to(device)
                padded_bind = torch.from_numpy(flat_bindings_np).to(device)

                # padded_slots: small (num_actions × max_arity), build in numpy.
                padded_slots_np = np.full(
                    (len(self.action_names), max_arity), -1, dtype=np.int64
                )
                for a, s in enumerate(self.active_slot_indices):
                    if s:
                        padded_slots_np[a, : len(s)] = s
                padded_slots = torch.from_numpy(padded_slots_np).to(device)

                if self.eq_channel_idx is not None:
                    keep_channels = torch.tensor(
                        [i for i in range(R) if i != self.eq_channel_idx],
                        dtype=torch.long,
                        device=device,
                    )
                else:
                    keep_channels = None

                for chunk_start in tqdm(
                    range(0, total, batch_size), desc=f"predict ({device})"
                ):
                    chunk_end = min(chunk_start + batch_size, total)
                    B = chunk_end - chunk_start

                    s_chunk = state_idx_t[chunk_start:chunk_end]
                    a_chunk = action_idx_t[chunk_start:chunk_end]

                    adj_batch = all_input_adj.index_select(0, s_chunk)  # (B,R,N,N)

                    slots_chunk = padded_slots.index_select(0, a_chunk)  # (B, P)
                    bind_chunk = padded_bind[chunk_start:chunk_end]  # (B, P)
                    valid = (slots_chunk >= 0) & (bind_chunk >= 0)
                    b_idx = (
                        torch.arange(B, device=device)
                        .unsqueeze(1)
                        .expand(-1, max_arity)
                    )
                    selection = torch.zeros(B, K, N, dtype=torch.float32, device=device)
                    selection[b_idx[valid], slots_chunk[valid], bind_chunk[valid]] = 1.0

                    adj_selection = torch.einsum(
                        "bai,brij,bcj->brac", selection, adj_batch, selection
                    )
                    effects_logits = self.model.graph_edit.effects_mlp(adj_selection)
                    effects = F.softmax(effects_logits, dim=-1)
                    effects = effects[torch.arange(B, device=device), a_chunk]

                    add_effects = effects[..., 1]
                    del_effects = effects[..., -1]

                    add_proj = torch.einsum(
                        "bai,brac,bcj->brij", selection, add_effects, selection
                    )
                    del_proj = torch.einsum(
                        "bai,brac,bcj->brij", selection, del_effects, selection
                    )
                    delta = (1.0 - adj_batch) * add_proj - adj_batch * del_proj
                    adj_out = (adj_batch + delta).clamp(0.0, 1.0)
                    adj_bin = (adj_out > 0.5).to(torch.uint8)
                    adj_bin = self._canonicalize_adj(adj_bin)  # (B,R,N,N)

                    if keep_channels is not None:
                        adj_bin = adj_bin.index_select(1, keep_channels)

                    sigs_np = adj_bin.contiguous().cpu().numpy()  # (B,R',N,N)
                    sigs_flat = sigs_np.reshape(B, -1)
                    s_chunk_cpu = flat_state_idx_list[chunk_start:chunk_end]
                    for b in range(B):
                        per_state_pred_sigs[s_chunk_cpu[b]].add(sigs_flat[b].tobytes())
                print("\n")
                print("GPU task done.")

        # ── Phase 3: metrics ──────────────────────────────────────────────
        matches = tp_total = fp_total = fn_total = 0
        for orig_sigs, pred_sigs in zip(per_state_orig_sigs, per_state_pred_sigs):
            tp = len(orig_sigs & pred_sigs)
            tp_total += tp
            fp_total += len(pred_sigs) - tp
            fn_total += len(orig_sigs) - tp
            if orig_sigs == pred_sigs:
                matches += 1

        denom_p = tp_total + fp_total
        denom_r = tp_total + fn_total
        return {
            "matches": matches,
            "total": S,
            "accuracy": matches / S if S else 0.0,
            "precision": tp_total / denom_p if denom_p else 0.0,
            "recall": tp_total / denom_r if denom_r else 0.0,
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
        }

    # ──────────────────────────────────────────────────────────────────
    #  prediction
    # ──────────────────────────────────────────────────────────────────
    def _predict_successors(
        self,
        input_adj: torch.Tensor,
        action_name: str,
        bindings: list[list[int]],
    ) -> torch.Tensor:
        """Return predicted next-state adjacencies for each grounding.

        Mirrors the state-dependent branch in
        :class:`PredictionModule.forward`:

            adj_selection = einsum("bai,brij,bcj->brac", sel, adj, sel)
            effects = softmax(effects_mlp(adj_selection))[arange(B), y_action]

        Then projects the soft per-slot add/del effects back to the dense
        adjacency and applies the XOR-style update
        ``adj' = adj + (1 - adj) * add - adj * del``, before thresholding.
        """
        B = len(bindings)
        K = self.num_slots
        N = self.num_objects
        R = len(self.relation_names)
        a_idx = self.action_names.index(action_name)
        slot_indices = self.active_slot_indices[a_idx]

        device = input_adj.device
        adj_float = input_adj.to(torch.float32)

        # Map each parameter position to its real slot index.  Unused
        # slots (those not in active_slot_indices) stay zero, which is
        # what the model's active-slot mask requires.
        selection = torch.zeros(B, K, N, dtype=torch.float32, device=device)
        for b, binding in enumerate(bindings):
            for param_pos, obj_idx in enumerate(binding):
                if param_pos >= len(slot_indices):
                    break
                slot = slot_indices[param_pos]
                selection[b, slot, obj_idx] = 1.0

        adj_batch = adj_float.unsqueeze(0).expand(B, R, N, N).contiguous()
        y_action = torch.full((B,), a_idx, dtype=torch.long, device=device)

        adj_selection = torch.einsum(
            "bai,brij,bcj->brac", selection, adj_batch, selection
        )  # (B, R, K, K)
        effects_logits = self.model.graph_edit.effects_mlp(adj_selection)
        effects = F.softmax(effects_logits, dim=-1)
        effects = effects[torch.arange(B, device=device), y_action]  # (B, R, K, K, 3)

        add_effects = effects[..., 1]
        del_effects = effects[..., -1]

        add_proj = torch.einsum("bai,brac,bcj->brij", selection, add_effects, selection)
        del_proj = torch.einsum("bai,brac,bcj->brij", selection, del_effects, selection)

        delta = (1.0 - adj_batch) * add_proj - adj_batch * del_proj
        adj_out = (adj_batch + delta).clamp(0.0, 1.0)
        adj_bin = (adj_out > 0.5).to(torch.uint8)
        return self._canonicalize_adj(adj_bin)

    # ──────────────────────────────────────────────────────────────────
    #  state ↔ adjacency
    # ──────────────────────────────────────────────────────────────────
    def _state_to_adj(self, state) -> torch.Tensor:
        adj = torch.zeros(
            len(self.relation_names),
            self.num_objects,
            self.num_objects,
            dtype=torch.uint8,
        )
        for atom in state.get_atoms(ignore_static=False, ignore_derived=True):
            pred_name = atom.get_predicate().get_name()
            r = self.relation_name_to_idx.get(pred_name)
            if r is None:
                continue
            arity = self.relation_arities[r]
            terms = atom.get_terms()
            if arity == 2:
                i = self.object_name_to_idx[terms[0].get_name()]
                j = self.object_name_to_idx[terms[1].get_name()]
                adj[r, i, j] = 1
            elif arity == 1:
                i = self.object_name_to_idx[terms[0].get_name()]
                adj[r, i, i] = 1
            elif arity == 0:
                adj[r, 0, 0] = 1
        return adj

    def _canonicalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Zero out off-diagonal cells for arity-1 channels and any
        non-(0,0) cells for arity-0, so predicted and ground-truth adj
        share the same encoding before signature.
        """
        if adj.dim() == 3:
            out = adj.clone()
            N = out.shape[-1]
            eye_idx = torch.arange(N, device=out.device)
            for r, ar in enumerate(self.relation_arities[0 : adj.shape[1]]):
                if ar == 1:
                    diag = out[r, eye_idx, eye_idx].clone()
                    out[r] = 0
                    out[r, eye_idx, eye_idx] = diag
                elif ar == 0:
                    v = out[r, 0, 0].clone()
                    out[r] = 0
                    out[r, 0, 0] = v
            return out
        elif adj.dim() == 4:  # (B, R, N, N)
            out = adj.clone()
            N = out.shape[-1]
            eye_idx = torch.arange(N, device=out.device)
            for r, ar in enumerate(self.relation_arities[0 : adj.shape[1]]):
                if ar == 1:
                    diag = out[:, r, eye_idx, eye_idx].clone()  # (B, N)
                    out[:, r] = 0
                    out[:, r, eye_idx, eye_idx] = diag
                elif ar == 0:
                    v = out[:, r, 0, 0].clone()  # (B,)
                    out[:, r] = 0
                    out[:, r, 0, 0] = v
            return out
        else:
            raise ValueError(f"Unexpected adj rank: {adj.shape}")

    def _adj_signature(self, adj: torch.Tensor) -> bytes:
        """Canonical bytes for an adjacency tensor; eq channel removed."""
        if self.eq_channel_idx is not None:
            keep = [i for i in range(adj.shape[0]) if i != self.eq_channel_idx]
            adj = adj[keep]
        return adj.to(torch.uint8).contiguous().cpu().numpy().tobytes()

    # ──────────────────────────────────────────────────────────────────
    #  PDDL helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _format_learned_action(schema, slot_indices: list[int]) -> str:
        """Serialize a learned action with empty effects and one parameter
        per active slot, named ``?x{slot_index}``.  This keeps the
        parameter names aligned with the slot indices used by the
        transpiler's precondition atoms (so e.g. an atom on slot 2 reads
        ``?x2``), regardless of whether the active slots are contiguous.
        Slots without any precondition stay declared as parameters --
        pymimir will ground them over all objects, which matches the
        model's view that those slots have no learned constraint.
        """
        params = " ".join(f"?x{i}" for i in slot_indices)
        precs = [str(p) for p in schema.preconditions]
        prec_str = "(and " + " ".join(precs) + ")" if precs else "()"
        return (
            f"(:action {schema.name}\n"
            f"  :parameters ({params})\n"
            f"  :precondition {prec_str}\n"
            f"  :effect ()\n"
            f")"
        )

    @staticmethod
    def _domain_uses_equality(domain_text: str) -> bool:
        req_match = re.search(r"\(:requirements([^)]*)\)", domain_text, re.IGNORECASE)
        return bool(req_match and ":equality" in req_match.group(1).lower())

    # ──────────────────────────────────────────────────────────────────
    #  state utilities (kept from the previous validator)
    # ──────────────────────────────────────────────────────────────────
