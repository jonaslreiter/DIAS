# L1's learned domains

The action schemas L1 (Balyo et al., 2024) learned for the paper's comparison, one file per domain in `pddl/`, as its authors returned them from one random walk each. Two things are respelled to match that domain's `domain.pddl`, which scores identically: equality, which L1 writes as the `eq` or `equal` predicate its input domain had, is `=`, and the types and predicate declarations are the domain's own. Gripper's input asserted its equality predicate for the two rooms only, so there L1's `(= ?r ?r)` stood for `(room ?r)`, which every schema writing it also requires.

```bash
python experiments/launch.py --experiment l1 --experiment-type l1
python experiments/collect_results.py --experiment l1
```

L1 learned from each domain's `train.pddl`, the instance DIAS trains on. Its walks are not in the repository.
