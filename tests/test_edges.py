import unittest

import torch
from torch_geometric.data import Batch, Data

from dias.models.edges import EdgeTypeLayout


def transition(edges, types, action=0, arguments=(-1, -1), num_nodes=2):
    state = Data(
        edge_index=torch.tensor(edges),
        edge_attr=torch.tensor(types),
        num_nodes=num_nodes,
    )
    state.y_action = torch.tensor([action])
    state.y_action_args = torch.tensor([list(arguments)])
    return Batch.from_data_list([state])


class EdgeTypeLayoutTest(unittest.TestCase):
    def test_the_four_bands_partition_the_edge_types(self):
        layout = EdgeTypeLayout(
            num_relations=3,
            num_actions=2,
            action_arg_positions=[[0, 1], [0]],
            successor_relations=True,
        )
        # 3 relations + 3 reversed + 3 action arguments + 4 * 3 changes.
        self.assertEqual(layout.num_action_relations, 3)
        self.assertEqual(layout.num_edge_types, 21)

        before = transition([[0], [1]], [0], action=0, arguments=(0, 1))
        after = transition([[1], [0]], [1], action=0, arguments=(0, 1))
        _, edge_type = layout(before, after)
        self.assertTrue(int(edge_type.min()) >= 0)
        self.assertLess(int(edge_type.max()), layout.num_edge_types)

    def test_an_action_without_arguments_contributes_no_relations(self):
        layout = EdgeTypeLayout(num_relations=2, num_actions=2)
        self.assertEqual(layout.num_action_relations, 0)
        self.assertEqual(layout.num_edge_types, 4)

    def test_changes_are_reported_in_both_directions(self):
        layout = EdgeTypeLayout(
            num_relations=2,
            num_actions=1,
            action_arg_positions=[[0]],
            successor_relations=True,
        )
        before = transition([[0], [1]], [0])
        after = transition([[1], [0]], [1])
        _, edge_type = layout(before, after)
        # 2 state + 2 reversed, then the deleted atom and the added one, each in both directions, at base = 2 * 2 + 1.
        self.assertEqual(edge_type.tolist()[-4:], [5, 7, 10, 12])

    def test_a_missing_successor_or_an_unchanged_one_adds_nothing(self):
        layout = EdgeTypeLayout(
            num_relations=1,
            num_actions=1,
            action_arg_positions=[[0]],
            successor_relations=True,
        )
        state = transition([[0], [1]], [0])
        self.assertEqual(len(layout(state, None)[1]), 2)
        self.assertEqual(len(layout(state, state)[1]), 2)

    def test_action_self_loops_land_on_the_objects_the_action_used(self):
        layout = EdgeTypeLayout(
            num_relations=1, num_actions=1, action_arg_positions=[[0, 1]]
        )
        state = transition([[0], [0]], [0], arguments=(1, 0), num_nodes=3)
        edge_index, edge_type = layout(state)
        # The first two edges are the state atom and its reverse.
        self.assertEqual(edge_index[:, 2:].tolist(), [[1, 0], [1, 0]])
        self.assertEqual(edge_type[2:].tolist(), [2, 3])

    def test_a_position_the_action_does_not_declare_is_ignored(self):
        """y_action_args is padded to the widest action, not to this one."""
        layout = EdgeTypeLayout(
            num_relations=1, num_actions=2, action_arg_positions=[[0], [0, 1]]
        )
        state = transition([[0], [0]], [0], action=0, arguments=(1, 1), num_nodes=2)
        _, edge_type = layout(state)
        self.assertEqual(edge_type[2:].tolist(), [2])


if __name__ == "__main__":
    unittest.main()
