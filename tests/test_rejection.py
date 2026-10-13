"""The rejection loss's estimate of what the preconditions admit, on states built by hand."""

import itertools
import unittest

import torch

from dias.models.outputs import Channels
from dias.models.rejection import admission, admitted_fraction, lift, sample_bindings

# Remark 4 of the recovery theorem: unary A, B and binary P, R over x1, x2, c1, c2, u, r.
A, B, P, R = 0, 1, 2, 3
X1, X2, C1, C2, U, HUB = range(6)


def remark_4_states() -> torch.Tensor:
    """The four reachable states: a(x, c) needs A(x) and P(x, c), deletes A(x) and adds B(x)."""
    initial = torch.zeros(4, 6, 6)
    for x, c in ((X1, C1), (X2, C2)):
        initial[P, x, c] = 1
        initial[R, x, HUB] = 1
    initial[R, U, HUB] = 1
    for thing in (X1, X2, U, C1):
        initial[A, thing, thing] = 1
    states = []
    for done in ((), (X1,), (X2,), (X1, X2)):
        state = initial.clone()
        for x in done:
            state[A, x, x] = 0
            state[B, x, x] = 1
        states.append(state)
    return torch.stack(states)


def hard(required: dict, forbidden: dict, num_slots=2, num_relations=4) -> Channels:
    """One action's preconditions, requiring and forbidding exactly the given ``(relation, slot, slot)`` cells."""
    positive = torch.zeros(1, num_relations, num_slots, num_slots)
    negative = torch.zeros_like(positive)
    for cell in required:
        positive[(0, *cell)] = 1
    for cell in forbidden:
        negative[(0, *cell)] = 1
    return Channels(
        ignore=1 - positive - negative, positive=positive, negative=negative
    )


class SampleBindingsTest(unittest.TestCase):
    def test_every_binding_puts_distinct_objects_in_its_slots(self):
        bindings = sample_bindings(3, 2, 50, num_objects=5, num_slots=3)
        self.assertEqual(bindings.shape, (3, 2, 50, 3))
        self.assertTrue(((bindings >= 0) & (bindings < 5)).all())
        distinct = torch.stack(
            [bindings[..., i] != bindings[..., j] for i, j in ((0, 1), (0, 2), (1, 2))]
        )
        self.assertTrue(distinct.all())


class LiftTest(unittest.TestCase):
    def test_a_binding_reads_the_state_over_its_objects(self):
        states = remark_4_states()
        lifted = lift(
            states, torch.tensor([X1, C1]).view(1, 1, 1, 2).expand(4, 1, 1, 2)
        )
        self.assertEqual(lifted.shape, (4, 1, 1, 4, 2, 2))
        initial = lifted[0, 0, 0]
        self.assertEqual(initial[A, 0, 0], 1)  # A(x1)
        self.assertEqual(initial[A, 1, 1], 1)  # A(c1)
        self.assertEqual(initial[P, 0, 1], 1)  # P(x1, c1)
        self.assertEqual(initial[P, 1, 0], 0)
        self.assertEqual(lifted[1, 0, 0, A, 0, 0], 0)  # x1 is done in the second state


class AdmittedFractionTest(unittest.TestCase):
    def estimate(self, preconditions, slot_weights=None, samples=4000, states=None):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            return admitted_fraction(
                remark_4_states() if states is None else states,
                preconditions,
                torch.ones(1, 2) if slot_weights is None else slot_weights,
                samples=samples,
                tau=0.0,
            )

    def test_the_hub_binding_of_remark_4_admits_twice_as_much(self):
        """Bound to the hub r the precondition has more literals, 12 against 11, yet admits 8 ground actions over the four states against 4."""
        shared = {
            (B, 1, 1),
            (P, 1, 0),
            (P, 0, 0),
            (P, 1, 1),
            (R, 1, 0),
            (R, 0, 0),
            (R, 1, 1),
        }
        true_binding = hard({(A, 0, 0), (P, 0, 1)}, shared | {(B, 0, 0), (R, 0, 1)})
        hub_binding = hard(
            {(A, 0, 0), (R, 0, 1)}, shared | {(B, 0, 0), (A, 1, 1), (P, 0, 1)}
        )
        # Every one of the 30 ordered pairs of distinct objects, in each of the four states.
        every_pair = torch.tensor(list(itertools.permutations(range(6), 2)))
        every_pair = every_pair.expand(4, 1, 30, 2)
        for preconditions, admitted in (
            (true_binding, [2, 1, 1, 0]),
            (hub_binding, [3, 2, 2, 1]),
        ):
            exact = admission(
                remark_4_states(), preconditions, torch.ones(1, 2), every_pair, tau=0.0
            )
            torch.testing.assert_close(
                exact.sum(dim=(1, 2)),
                torch.tensor(admitted, dtype=torch.float),
                atol=1e-5,
                rtol=0,
            )
            # Sampling estimates the same fraction.
            torch.testing.assert_close(
                self.estimate(preconditions),
                torch.tensor(admitted, dtype=torch.float).view(4, 1) / 30,
                atol=0.02,
                rtol=0,
            )

    def test_a_literal_true_at_every_ground_action_is_given_no_push(self):
        """Forbidding P(y, y) rejects nothing, since P is never reflexive, so the loss leaves it alone; requiring P(y, z) rejects almost everything."""
        positive = torch.zeros(1, 4, 2, 2, requires_grad=True)
        negative = torch.zeros(1, 4, 2, 2, requires_grad=True)
        preconditions = Channels(
            ignore=1 - positive - negative, positive=positive, negative=negative
        )
        self.estimate(preconditions, samples=500).sum().backward()
        self.assertEqual(negative.grad[0, P, 0, 0].item(), 0.0)
        self.assertLess(positive.grad[0, P, 0, 1].item(), 0.0)

    def test_an_inactive_slot_restricts_nothing(self):
        on_the_second_slot = hard({(A, 1, 1), (P, 0, 1)}, {(B, 1, 1)})
        admitted = self.estimate(
            on_the_second_slot, torch.tensor([[1.0, 0.0]]), samples=50
        )
        torch.testing.assert_close(admitted, torch.ones(4, 1))

    def test_gradients_reach_the_precondition_logits_and_the_slot_weights(self):
        logits = torch.randn(1, 4, 2, 2, 3, requires_grad=True)
        slot_logits = torch.zeros(1, 2, requires_grad=True)
        self.estimate(
            Channels.from_logits(logits), torch.sigmoid(slot_logits), samples=20
        ).mean().backward()
        self.assertTrue(logits.grad.isfinite().all())
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)
        self.assertGreater(slot_logits.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
