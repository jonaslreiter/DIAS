import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch_geometric.data import Batch, Data

from dias.models.dias import DIAS


def make_batch(action_args=(0, 1), changed=False):
    state = Data(
        edge_index=torch.tensor([[0, 1], [0, 1]]),
        edge_attr=torch.tensor([0, 1]),
        num_nodes=2,
    )
    state.y_action = torch.tensor([0])
    state.y_action_args = torch.tensor([list(action_args)])
    if changed:
        next_state = Data(
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.tensor([0, 1]),
            num_nodes=2,
        )
    else:
        next_state = Data(
            edge_index=torch.tensor([[0, 1], [0, 1]]),
            edge_attr=torch.tensor([0, 1]),
            num_nodes=2,
        )
    return Batch.from_data_list([state]), Batch.from_data_list([next_state])


class DIASModelTest(unittest.TestCase):
    def make_model(self, **kwargs):
        defaults = {
            "num_objects": 2,
            "arity_relations": [1, 2],
            "num_actions": 1,
            "object_dim": 4,
            "num_slots": 2,
            "action_arities": [2],
            "action_arg_positions": [[0, 1]],
            "successor_relations": True,
            "parameter_plotting": None,
        }
        defaults.update(kwargs)
        model = DIAS(**defaults)
        self.addCleanup(model.teardown, "test")
        return model

    def test_constructor_validates_slot_configuration_and_clamps_slots(self):
        with self.assertRaisesRegex(AssertionError, "num_slots must be set"):
            DIAS(2, [1], 1, 2, parameter_plotting=None)
        with self.assertRaisesRegex(AssertionError, "action_arities"):
            DIAS(
                2,
                [1],
                1,
                2,
                per_action_slots=True,
                parameter_plotting=None,
            )
        with self.assertRaisesRegex(AssertionError, "must be >= max action arity"):
            DIAS(
                2,
                [1],
                1,
                2,
                num_slots=1,
                action_arities=[2],
                parameter_plotting=None,
            )
        clamped = self.make_model(num_objects=2, num_slots=5, action_arities=[2])
        self.assertEqual(clamped.num_slots, 2)
        per_action = self.make_model(
            num_slots=None,
            per_action_slots=True,
            action_arities=[1],
            action_arg_positions=[[0]],
        )
        self.assertEqual(per_action.num_slots, 1)

    def test_constructor_uses_default_argument_positions_and_custom_schedule(self):
        model = self.make_model(
            action_arg_positions=None,
            loss_weight_schedule={
                "learning_rate": {"type": "linear", "keypoints": {0: 0.2}}
            },
        )
        self.assertEqual(model.num_action_relations, 0)
        self.assertEqual(model.loss_weight_schedule_collection["learning_rate"], 0.2)

    def test_forward_handles_action_argument_padding_truncation_and_changes(self):
        model = self.make_model(force_action_args=True)
        model.eval()
        batch_s1, batch_s2 = make_batch(changed=True)
        short_s1 = batch_s1.clone()
        short_s1.y_action_args = torch.tensor([[-1]])
        short_output = model(short_s1, batch_s2)
        self.assertEqual(short_output[0].shape, (1, 2, 2, 2))

        long_model = self.make_model(force_action_args=False)
        long_model.eval()
        long_s1 = batch_s1.clone()
        long_s1.y_action_args = torch.tensor([[0, 1, 0]])
        long_output = long_model(long_s1, batch_s2)
        self.assertEqual(long_output[0].shape, (1, 2, 2, 2))
        self.assertTrue(torch.isfinite(long_output[0]).all())

    def test_forward_without_successor_batch_uses_reverse_edges_only(self):
        model = self.make_model(successor_relations=False, random_node_embeddings=False)
        model.eval()
        batch_s1, _ = make_batch(action_args=(-1, -1))
        output = model(batch_s1)
        self.assertEqual(output[0].shape, (1, 2, 2, 2))
        self.assertTrue(torch.isfinite(output[0]).all())

    def test_step_and_eval_steps_return_losses(self):
        model = self.make_model(successor_relations=False)
        model.eval()
        batch = (*make_batch(changed=True), torch.tensor([0]))
        results = model.step(batch)
        self.assertEqual(
            set(results),
            {"loss_adjacency", "loss_aux", "loss_mse", "pre_mean", "pre_prod_mean"},
        )
        for value in results.values():
            self.assertTrue(torch.isfinite(value.float()).all())

        model.log_results = Mock()
        valid_loss = model.validation_step(batch, 0)
        test_loss = model.test_step(batch, 0)
        self.assertTrue(torch.isfinite(valid_loss))
        self.assertTrue(torch.isfinite(test_loss))
        self.assertEqual(model.log_results.call_count, 2)

    def test_optimizer_and_schedule_hooks_update_and_log(self):
        model = self.make_model()
        optimizer = model.configure_optimizers()
        self.assertEqual(optimizer.param_groups[0]["lr"], model.learning_rate)
        model.optimizers = Mock(return_value=optimizer)
        model.log_dict = Mock()
        with patch.object(
            type(model), "global_step", new_callable=lambda: property(lambda self: 0)
        ):
            model.on_train_batch_start(None, 0)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)
        model.log_dict.assert_called_once()
        model.on_validation_epoch_start()

    def test_label_names_are_cached_and_noise_fields_are_logged(self):
        model = self.make_model()
        model._trainer = SimpleNamespace(
            datamodule=SimpleNamespace(
                action_names=["go"],
                relation_names=["clear", "on"],
                object_names={"0": "a", "1": "b"},
            )
        )
        labels = model.label_names
        self.assertEqual(labels, (["go"], ["clear", "on"], ["a", "b"]))
        self.assertIs(model.label_names, labels)

        sample = Data(
            num_flips_added=torch.tensor([1]),
            num_flips_removed=torch.tensor([2]),
            num_flips_added_per_rel=torch.tensor([[1, 0]]),
            num_flips_removed_per_rel=torch.tensor([[0, 1]]),
            num_true_predicates=torch.tensor([3]),
        )
        model.log = Mock()
        model.log_dict = Mock()
        model._log_noise(sample)
        self.assertEqual(model.log.call_count, 3)
        model.log_dict.assert_called_once()

    def test_log_results_skips_nan_and_handles_train_prefix(self):
        model = self.make_model()
        model.log = Mock()
        model.log_results(
            "train",
            {"loss": torch.tensor(1.0), "nan": torch.tensor(float("nan")), "step": 2},
        )
        self.assertEqual(model.log.call_count, 2)
        first_call = model.log.call_args_list[0]
        self.assertEqual(first_call.args[0], "train/loss")
        self.assertTrue(first_call.kwargs["on_step"])
        self.assertFalse(first_call.kwargs["on_epoch"])

    def test_training_step_runs_pcgrad_projection_and_optimizer_hooks(self):
        shared = torch.nn.Parameter(torch.tensor(1.0))
        aligned = torch.nn.Parameter(torch.tensor(1.0))
        adjacency_only = torch.nn.Parameter(torch.tensor(2.0))
        auxiliary_only = torch.nn.Parameter(torch.tensor(3.0))
        unused = torch.nn.Parameter(torch.tensor(4.0))

        class Optimizer:
            def __init__(self):
                self.zeroed = False
                self.stepped = False

            def zero_grad(self):
                self.zeroed = True

            def step(self):
                self.stepped = True

        optimizer = Optimizer()

        class FakeModel:
            pcgrad_dyn_clip = True
            loss_weight_schedule_collection = {"aux_loss_weight": 1.0}

            def parameters(self):
                return iter([shared, aligned, adjacency_only, auxiliary_only, unused])

            def optimizers(self):
                return optimizer

            def _log_noise(self, batch_s1):
                self.noise_batch = batch_s1

            def step(self, batch):
                return {
                    "loss_adjacency": shared + aligned + adjacency_only,
                    "loss_aux": -2 * shared + 2 * aligned + auxiliary_only,
                    "loss_mse": torch.tensor(0),
                    "pre_mean": torch.tensor(1.0),
                    "pre_prod_mean": torch.tensor(1.0),
                }

            def log_results(self, prefix, results):
                self.logged = (prefix, results)

        fake = FakeModel()
        result = DIAS.training_step(fake, ("batch",), 0)
        self.assertIsNone(result)
        self.assertTrue(optimizer.zeroed)
        self.assertTrue(optimizer.stepped)
        self.assertIs(fake.noise_batch, "batch")
        self.assertEqual(fake.logged[0], "train")
        self.assertTrue(hasattr(fake, "manual_backward_done"))

    def test_validation_epoch_end_queues_only_when_plotting_enabled(self):
        model = self.make_model(parameter_plotting=None)
        with patch("dias.models.dias.queue_validation_heatmaps") as queue_heatmaps:
            model.on_validation_epoch_end()
            queue_heatmaps.assert_not_called()

        plotting_model = self.make_model(parameter_plotting={"selection": True})
        with patch("dias.models.dias.queue_validation_heatmaps") as queue_heatmaps:
            plotting_model.on_validation_epoch_end()
            queue_heatmaps.assert_called_once_with(plotting_model)


if __name__ == "__main__":
    unittest.main()
