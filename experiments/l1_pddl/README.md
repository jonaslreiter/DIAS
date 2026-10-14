# L1's learned domains

The action schemas L1 (Balyo et al., 2024) learned for the paper's comparison, one file per domain in `pddl/`, as its authors returned them from one random walk each. Two things are respelled to match that domain's `domain.pddl`, which scores identically: equality, which L1 writes as the `eq` or `equal` predicate its input domain had, is `=`, and the types and predicate declarations are the domain's own. Gripper's input asserted its equality predicate for the two rooms only, so there L1's `(= ?r ?r)` stood for `(room ?r)`, which every schema writing it also requires.

```bash
python experiments/launch.py --experiment l1 --experiment-type l1
python experiments/collect_results.py --experiment l1
```

L1 learned from each domain's `train.pddl`, except in two, whose training instances changed after its walks were sampled:

| Domain | L1 learned from | DIAS trains on |
| --- | --- | --- |
| hanoi | 3 pegs, 7 discs (10 objects) | 3 pegs, 6 discs |
| spanner | 6 spanners, 4 nuts, 8 locations (19 objects) | 10 spanners, 7 nuts, 10 locations |

Neither the walks nor those two instances are in the repository.

TODO: check whether DIAS learns hanoi and spanner from L1's instances as well as from `train.pddl`.
