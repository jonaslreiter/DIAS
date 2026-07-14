"""Validation heatmaps and the background worker that renders them."""

import queue
import sys
import threading
import traceback

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb


class PlotWorker:
    """Background plotting worker; keeps at most one pending job (newest wins)."""

    def __init__(self):
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="PlotWorker", daemon=True
        )
        self._thread.start()

    def submit(self, fn, *args, **kwargs) -> None:
        if self._stopping:
            return
        while True:
            try:
                old = self._q.get_nowait()
            except queue.Empty:
                break
            else:
                self._q.task_done()
                if old is None:
                    return
        try:
            self._q.put_nowait((fn, args, kwargs))
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            fn, args, kwargs = item
            try:
                fn(*args, **kwargs)
            except Exception as e:
                print(f"[PlotWorker] error: {e}", file=sys.stderr)
                traceback.print_exc()
            finally:
                self._q.task_done()

    def shutdown(self, timeout: float = 60.0) -> None:
        self._stopping = True
        self._q.put(None)
        self._thread.join(timeout=timeout)


def _annotate(ax, mat, k_rows, k_cols, threshold: float = 0.5) -> None:
    for i in range(k_rows):
        for j in range(k_cols):
            val = mat[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(float(val) - threshold) > 0.3 else "black",
            )


def render_action_relation_heatmap(
    values: np.ndarray,
    action_names: list,
    relation_names: list,
    title: str,
    epoch: int,
) -> wandb.Image:
    num_actions, num_relations, k, _ = values.shape
    slot_labels = [f"Slot {s}" for s in range(k)]
    fig, axes = plt.subplots(
        num_actions,
        num_relations,
        figsize=(3 * num_relations + 1, 3 * num_actions),
        squeeze=False,
        constrained_layout=True,
    )

    im = None
    for a in range(num_actions):
        for r in range(num_relations):
            ax = axes[a][r]
            im = ax.imshow(values[a, r], vmin=0, vmax=1, cmap="RdYlBu_r")
            ax.set_xticks(range(k))
            ax.set_xticklabels(slot_labels, fontsize=7)
            ax.set_yticks(range(k))
            ax.set_yticklabels(slot_labels, fontsize=7)
            if a == 0:
                ax.set_title(
                    relation_names[r] if r < len(relation_names) else f"Rel {r}",
                    fontsize=9,
                )
            if r == 0:
                ax.set_ylabel(
                    action_names[a] if a < len(action_names) else f"Action {a}",
                    fontsize=9,
                )
            _annotate(ax, values[a, r], k, k)

    fig.suptitle(f"{title} (epoch {epoch})", fontsize=12)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label="probability")
    image = wandb.Image(fig)
    plt.close(fig)
    return image


def render_selection_heatmap(
    samples: dict,
    slot_weights: dict,
    action_names: list,
    object_names: list,
    epoch: int,
) -> wandb.Image:
    sorted_actions = sorted(samples.keys())
    num_actions = len(sorted_actions)
    k, n_nodes = next(iter(samples.values())).shape
    slot_labels = [f"Slot {s}" for s in range(k)]
    node_labels = object_names[:n_nodes]

    fig, axes = plt.subplots(
        num_actions,
        2,
        figsize=(max(4, 0.5 * n_nodes) + 2, 2 * num_actions),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [n_nodes, 1]},
    )

    im = None
    for row, a in enumerate(sorted_actions):
        ax = axes[row][0]
        mat = samples[a]
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(n_nodes))
        ax.set_xticklabels(node_labels, fontsize=6, rotation=45)
        ax.set_yticks(range(k))
        ax.set_yticklabels(slot_labels, fontsize=7)
        ax.set_ylabel(action_names[a], fontsize=9)
        if row == 0:
            ax.set_title("Slot → Node Selection", fontsize=9)
        _annotate(ax, mat, k, n_nodes)

        ax_sw = axes[row][1]
        sw = slot_weights.get(a, np.zeros(k))
        ax_sw.imshow(sw.reshape(k, 1), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax_sw.set_xticks([])
        ax_sw.set_yticks(range(k))
        ax_sw.set_yticklabels(slot_labels, fontsize=7)
        if row == 0:
            ax_sw.set_title("Slot\nWeight", fontsize=9)
        _annotate(ax_sw, sw.reshape(k, 1), k, 1)

    fig.suptitle(f"Selection Matrices (epoch {epoch})", fontsize=12)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label="weight")
    image = wandb.Image(fig)
    plt.close(fig)
    return image


def _render_and_log_validation_heatmaps(payload: dict, wandb_run) -> None:
    media: dict = {}
    for values, title, log_key in payload.get("components", []):
        media[log_key] = render_action_relation_heatmap(
            values,
            payload["action_names"],
            payload["relation_names"],
            title,
            payload["epoch"],
        )

    samples = payload.get("selection_samples")
    slot_weights = payload.get("selection_slot_weights")
    if samples and slot_weights:
        media["selection_heatmap"] = render_selection_heatmap(
            samples,
            slot_weights,
            payload["action_names"],
            payload["object_names"],
            payload["epoch"],
        )

    if media:
        wandb_run.log(media, step=payload["step"])


def queue_validation_heatmaps(model) -> None:
    """Snapshot the model's validation visuals and enqueue one media-log job."""
    if not isinstance(model.logger, L.pytorch.loggers.WandbLogger):
        return
    step = int(model.global_step)
    if model._last_plot_step_submitted == step:
        return

    action_names, relation_names, object_names = model.label_names
    plot_effects = bool(model.parameter_plotting.get("effect_and_precondition", False))
    plot_selection = bool(model.parameter_plotting.get("selection", False))
    if not plot_effects and not plot_selection:
        return

    payload: dict = {
        "epoch": int(model.current_epoch),
        "step": step,
        "action_names": action_names,
        "relation_names": relation_names,
        "object_names": object_names,
    }

    if plot_effects:
        with torch.no_grad():
            precond_probs = F.softmax(model.graph_edit.preconditions.detach(), dim=-1)
            effect_probs = F.softmax(model.graph_edit.effects.detach(), dim=-1)
            payload["components"] = [
                (
                    effect_probs[..., 1].cpu().numpy(),
                    "Add Effects",
                    "add_effects_heatmap",
                ),
                (
                    effect_probs[..., -1].cpu().numpy(),
                    "Delete Effects",
                    "delete_effects_heatmap",
                ),
                (
                    precond_probs[..., 1].cpu().numpy(),
                    "Positive Preconditions",
                    "positive_preconditions_heatmap",
                ),
                (
                    precond_probs[..., -1].cpu().numpy(),
                    "Negative Preconditions",
                    "negative_preconditions_heatmap",
                ),
            ]

    if (
        plot_selection
        and model._last_selection is not None
        and model._last_y_action is not None
    ):
        selection = model._last_selection.numpy()
        y_action = model._last_y_action.numpy()
        samples: dict = {}
        for idx in range(len(y_action)):
            a = int(y_action[idx])
            samples.setdefault(a, selection[idx].copy())
        if samples:
            with torch.no_grad():
                all_sw = (
                    torch.sigmoid(model.graph_edit.slot_weight_logits.detach())
                    .cpu()
                    .numpy()
                )
            payload["selection_samples"] = samples
            payload["selection_slot_weights"] = {a: all_sw[a] for a in samples}

    if len(payload) == 5:
        return
    model._last_plot_step_submitted = step
    model._plot_worker.submit(
        _render_and_log_validation_heatmaps, payload, model.logger.experiment
    )
