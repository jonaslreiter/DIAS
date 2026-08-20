import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch_geometric.data import Batch, Data

from dias.models.dias import DIAS
from dias.schedules import Exponential


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

    def test_constructor_defaults_the_argument_positions_to_none(self):
        model = self.make_model(action_arg_positions=None)
        self.assertEqual(model.edges.num_action_relations, 0)

    def test_the_tau_schedule_is_kept_as_given(self):
        model = self.make_model(
            gmean_temperature=Exponential(start=1.0, factor=0.1, every=10)
        )
        self.assertEqual(
            model.gmean_temperature,
            Exponential(start=1.0, floor=0.0, factor=0.1, every=10),
        )
        self.assertAlmostEqual(model.gmean_temperature(10), 0.1)

    def test_the_tau_schedule_is_stored_as_primitives(self):
        """A dataclass here would make the checkpoint unloadable."""
        model = self.make_model(
            gmean_temperature=Exponential(start=1.0, factor=0.1, every=10)
        )
        self.assertEqual(
            model.hparams["gmean_temperature"],
            {
                "class_path": "dias.schedules.Exponential",
                "init_args": {
                    "start": 1.0,
                    "floor": 0.0,
                    "factor": 0.1,
                    "every": 10,
                },
            },
        )

    def test_forward_handles_action_argument_padding_truncation_and_changes(self):
        model = self.make_model(force_action_args=True)
        model.eval()
        batch_s1, batch_s2 = make_batch(changed=True)
        short_s1 = batch_s1.clone()
        short_s1.y_action_args = torch.tensor([[-1]])
        short_output = model(short_s1, batch_s2)
        self.assertEqual(short_output.adjacency.shape, (1, 2, 2, 2))

        long_model = self.make_model(force_action_args=False)
        long_model.eval()
        long_s1 = batch_s1.clone()
        long_s1.y_action_args = torch.tensor([[0, 1, 0]])
        long_output = long_model(long_s1, batch_s2)
        self.assertEqual(long_output.adjacency.shape, (1, 2, 2, 2))
        self.assertTrue(torch.isfinite(long_output.adjacency).all())

    def test_forward_without_successor_batch_uses_reverse_edges_only(self):
        model = self.make_model(successor_relations=False, random_node_embeddings=False)
        model.eval()
        batch_s1, _ = make_batch(action_args=(-1, -1))
        output = model(batch_s1)
        self.assertEqual(output.adjacency.shape, (1, 2, 2, 2))
        self.assertTrue(torch.isfinite(output.adjacency).all())

    def test_step_and_eval_steps_return_losses(self):
        model = self.make_model(successor_relations=False)
        model.eval()
        batch = (*make_batch(changed=True), torch.tensor([0]))
        losses = model.step(batch)
        self.assertEqual(
            set(losses.logged(model.aux_loss_weight)),
            {
                "loss",
                "loss_adjacency",
                "loss_aux",
                "hamming_distance",
                "pre_mean",
                "pre_prod_mean",
            },
        )
        for value in losses.logged(model.aux_loss_weight).values():
            self.assertTrue(torch.isfinite(value.float()).all())

        model.log_results = Mock()
        valid_loss = model.validation_step(batch, 0)
        test_loss = model.test_step(batch, 0)
        self.assertTrue(torch.isfinite(valid_loss))
        self.assertTrue(torch.isfinite(test_loss))
        self.assertEqual(model.log_results.call_count, 2)

    def test_retarget_accepts_a_bigger_instance_and_refuses_a_smaller_one(self):
        """A lifted schema transfers; the recorded instance size does not."""
        model = self.make_model(num_objects=4, num_slots=2, action_arities=[2])
        model.retarget(9)
        self.assertEqual(model.num_objects, 9)
        with self.assertRaisesRegex(ValueError, "fewer than the model's 2 slots"):
            model.retarget(1)
        self.assertEqual(model.num_objects, 9)

    def test_apply_schema_changes_only_what_the_binding_reaches(self):
        """Slots bind to objects 0 and 1, so object 2 cannot be touched."""
        model = self.make_model(num_objects=3, num_slots=2, action_arities=[2])
        model.eval()
        adjacency = torch.zeros(1, 2, 3, 3)
        selection = torch.zeros(1, 2, 3)
        selection[0, 0, 0] = 1.0
        selection[0, 1, 1] = 1.0
        with torch.no_grad():
            successor = model.apply_schema(adjacency, selection, torch.tensor([0]))
        self.assertEqual(successor.shape, (1, 2, 3, 3))
        self.assertTrue(torch.all(successor[0, :, 2, :] == 0))
        self.assertTrue(torch.all(successor[0, :, :, 2] == 0))
        self.assertTrue((successor[0, :, :2, :2] != 0).any())

    def test_optimizer_uses_the_configured_learning_rate(self):
        model = self.make_model()
        optimizer = model.configure_optimizers()
        self.assertEqual(optimizer.param_groups[0]["lr"], model.learning_rate)

    def test_the_scheduled_tau_is_logged_for_the_current_step(self):
        model = self.make_model(
            gmean_temperature=Exponential(start=1.0, factor=0.1, every=10)
        )
        model.log = Mock()
        with patch.object(
            type(model), "global_step", new_callable=lambda: property(lambda self: 10)
        ):
            model.on_train_batch_start(None, 0)
        name, value = model.log.call_args.args
        self.assertEqual(name, "scheduled_w/gmean_temperature")
        self.assertAlmostEqual(value, 0.1)

    def with_labels(self) -> "DIAS":
        model = self.make_model()
        model._trainer = SimpleNamespace(
            datamodule=SimpleNamespace(
                action_names=["go"],
                relation_names=["clear", "on"],
                object_names={"0": "a", "1": "b"},
            )
        )
        return model

    def test_label_names_are_read_from_the_datamodule_once(self):
        model = self.with_labels()
        labels = model.label_names
        self.assertEqual(labels, (["go"], ["clear", "on"], ["a", "b"]))
        self.assertIs(model.label_names, labels)

    def test_a_noisy_batch_logs_its_flips_per_relation(self):
        model = self.with_labels()
        model.log_dict = Mock()
        model._log_noise(
            Data(
                num_flips_added=torch.tensor([1]),
                num_flips_removed=torch.tensor([2]),
                num_flips_added_per_rel=torch.tensor([[1, 0]]),
                num_flips_removed_per_rel=torch.tensor([[0, 1]]),
                num_true_predicates=torch.tensor([3]),
            )
        )
        logged = model.log_dict.call_args.args[0]
        self.assertEqual(
            set(logged),
            {
                "noise/num_flips_added",
                "noise/num_flips_removed",
                "noise/num_true_predicates",
                "noise/clear/added",
                "noise/clear/removed",
                "noise/on/added",
                "noise/on/removed",
            },
        )
        self.assertEqual(logged["noise/clear/added"].item(), 1.0)
        self.assertEqual(logged["noise/on/removed"].item(), 1.0)

    def test_a_clean_batch_logs_no_noise_metrics_at_all(self):
        model = self.with_labels()
        model.log_dict = Mock()
        model._log_noise(Data(y_action=torch.tensor([0])))
        model.log_dict.assert_not_called()

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
