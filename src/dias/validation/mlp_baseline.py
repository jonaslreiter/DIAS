from __future__ import annotations

import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pymimir
import torch
from pymimir.advanced.formalism import FluentGroundAtomList, ObjectList
from pymimir.wrapper_formalism import State
from tqdm import tqdm

from dias.pddl import text as pddl_text
from dias.pddl.transpiler import PDDLTranspiler
from dias.pddl.untype import (
    EQUALITY_PREDICATE,
    extract_domain_header,
    strip_problem_types,
)

from .artifacts import PddlWorkspace
from .checkpoint import load_run
from .decoding import process_model_logits
from .successors import (
    LEARNED_ACTION_PREFIX,
    resolve_sample_seed,
    sample_reachable_states,
    transfer_states,
)


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
    def __init__(
        self,
        model_id,
        logs_path_abs,
        problem_path=None,
        batch_size: int = 4096,
        num_workers: int = 1,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_id = model_id

        run = load_run(model_id, logs_path_abs)
        self.schema = run.schema
        self.model = run.model
        datamodule_hparams = run.datamodule_hparams

        schema = self.model.schema_logits()
        if schema.effects is not None:
            raise ValueError(
                "validate_mlp_baseline expects a model with "
                "state_dependent_effects=True. Use validate.py for the "
                "static-effects baseline."
            )

        free_pos_map = (
            strips_plus_arguments(datamodule_hparams["domain_path"])
            if datamodule_hparams.get("strips_plus_actions", False)
            else {}
        )
        action_arguments = datamodule_hparams.get("action_arguments", False)
        strips_plus_actions = datamodule_hparams.get("strips_plus_actions", False)

        # This baseline predicts its effects from the state at validation time,
        # so `schema.effects` is None and only the preconditions and the slot
        # activations are read off here.
        _, _, pos_prec, neg_prec, active_slots = process_model_logits(
            effect_logits=schema.effects,
            precondition_logits=schema.preconditions,
            slot_weight_logits=schema.slot_weight_logits,
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
        has_equality = pddl_text.domain_uses_equality(original_domain_text)

        relation_names = list(self.schema.relation_names)
        relation_arities = list(self.schema.relation_arities)
        if not has_equality:
            keep = [i for i, n in enumerate(relation_names) if n != "="]
            pos_prec = pos_prec[:, keep, :, :]
            neg_prec = neg_prec[:, keep, :, :]
            relation_names = [relation_names[i] for i in keep]
            relation_arities = [relation_arities[i] for i in keep]

        transpiler_predicate_names = [
            EQUALITY_PREDICATE if has_equality and n == "=" else n
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
            schema.name = f"{LEARNED_ACTION_PREFIX}{schema.name}"

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
                equality_predicate_name=(EQUALITY_PREDICATE if has_equality else None),
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
        self.model.retarget(self.num_objects)
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

    def setup(
        self,
        num_samples: int = 250,
        sample_population: int = 500,
        seed: int | None = None,
    ):
        # Recorded, so a result drawn without a fixed seed can still be redrawn.
        self.sample_seed = resolve_sample_seed(seed)
        original_states = sample_reachable_states(
            self.original_problem, num_samples, sample_population, self.sample_seed
        )
        learned_states = transfer_states(
            original_states,
            self.learned_problem,
            self.original_domain,
            self.learned_domain,
        )
        self.state_pairs = list(zip(original_states, learned_states))

    def validate(self):
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
        batch_size = self.batch_size
        num_workers = self.num_workers
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
            "learned_action_prefix": LEARNED_ACTION_PREFIX,
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

                    # The binding is already known -- pymimir grounded the
                    # learned action to get it -- so the model is asked only
                    # what the action changes, not whether it applies.
                    adj_out = self.model.apply_schema(adj_batch, selection, a_chunk)
                    adj_bin = (adj_out > 0.5).to(torch.uint8)  # (B,R,N,N)

                    if keep_channels is not None:
                        adj_bin = adj_bin.index_select(1, keep_channels)

                    sigs_np = adj_bin.contiguous().cpu().numpy()  # (B,R',N,N)
                    sigs_flat = sigs_np.reshape(B, -1)
                    s_chunk_cpu = flat_state_idx_list[chunk_start:chunk_end]
                    for b in range(B):
                        per_state_pred_sigs[s_chunk_cpu[b]].add(sigs_flat[b].tobytes())

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
