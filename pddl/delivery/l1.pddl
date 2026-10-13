; The action schemas L1 (Balyo et al., 2024) learned for the paper's comparison, as its authors returned them, with equality written as = and the types as in domain.pddl.
; Learned from one random walk on train.pddl.
(define (domain delivery)
  (:requirements :negative-preconditions :typing :equality)
  (:types cell - object locatable - object package - locatable truck - locatable)
  (:predicates
    (at ?x1 - locatable ?x2 - cell)
    (carrying ?x1 - truck ?x2 - package)
    (empty ?x1 - truck)
    (adjacent ?x1 - cell ?x2 - cell)
  )

  (:action move
    :parameters (?cell1 - cell ?cell2 - cell ?truck1 - truck)
    :precondition (and
      (adjacent ?cell1 ?cell2)
      (adjacent ?cell2 ?cell1)
      (at ?truck1 ?cell2)
      (= ?cell1 ?cell1)
      (= ?cell2 ?cell2)
      (= ?truck1 ?truck1)
      (not(at ?truck1 ?cell1))
    )
    :effect (and
      (at ?truck1 ?cell1)
      (not(at ?truck1 ?cell2))
    )
  )

  (:action pick-package
    :parameters (?cell1 - cell ?package1 - package ?truck1 - truck)
    :precondition (and
      (at ?package1 ?cell1)
      (at ?truck1 ?cell1)
      (empty ?truck1)
      (= ?cell1 ?cell1)
      (= ?package1 ?package1)
      (= ?truck1 ?truck1)
      (not(carrying ?truck1 ?package1))
    )
    :effect (and
      (carrying ?truck1 ?package1)
      (not(at ?package1 ?cell1))
      (not(empty ?truck1))
    )
  )

  (:action drop-package
    :parameters (?cell1 - cell ?package1 - package ?truck1 - truck)
    :precondition (and
      (at ?truck1 ?cell1)
      (carrying ?truck1 ?package1)
      (= ?cell1 ?cell1)
      (= ?package1 ?package1)
      (= ?truck1 ?truck1)
      (not(at ?package1 ?cell1))
      (not(empty ?truck1))
    )
    :effect (and
      (at ?package1 ?cell1)
      (empty ?truck1)
      (not(carrying ?truck1 ?package1))
    )
  )
)
