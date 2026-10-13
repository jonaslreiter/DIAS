; The action schemas L1 (Balyo et al., 2024) learned for the paper's comparison, as its authors returned them, with equality written as = and the types as in domain.pddl.
; Learned from one random walk on train.pddl.
; L1's input asserted its equality predicate for the two rooms only, so (= ?r ?r) stood for (room ?r), which every schema writing it also requires.
(define (domain gripper-strips)
  (:requirements :negative-preconditions :typing :equality)
  (:predicates
    (room ?r)
    (ball ?b)
    (gripper ?g)
    (at-robby ?r)
    (at ?b ?r)
    (free ?g)
    (carry ?o ?g)
  )

  (:action pick
    :parameters (?untyped1 ?untyped2 ?untyped3)
    :precondition (and
      (at ?untyped1 ?untyped3)
      (at-robby ?untyped3)
      (ball ?untyped1)
      (= ?untyped3 ?untyped3)
      (free ?untyped2)
      (gripper ?untyped2)
      (not(carry ?untyped1 ?untyped2))
      (room ?untyped3)
    )
    :effect (and
      (carry ?untyped1 ?untyped2)
      (not(at ?untyped1 ?untyped3))
      (not(free ?untyped2))
    )
  )

  (:action move
    :parameters (?untyped1 ?untyped2)
    :precondition (and
      (at-robby ?untyped2)
      (= ?untyped1 ?untyped1)
      (= ?untyped2 ?untyped2)
      (not(at-robby ?untyped1))
      (room ?untyped1)
      (room ?untyped2)
    )
    :effect (and
      (at-robby ?untyped1)
      (not(at-robby ?untyped2))
    )
  )

  (:action drop
    :parameters (?untyped1 ?untyped2 ?untyped3)
    :precondition (and
      (at-robby ?untyped3)
      (ball ?untyped2)
      (carry ?untyped2 ?untyped1)
      (= ?untyped3 ?untyped3)
      (gripper ?untyped1)
      (not(at ?untyped2 ?untyped3))
      (not(free ?untyped1))
      (room ?untyped3)
    )
    :effect (and
      (at ?untyped2 ?untyped3)
      (free ?untyped1)
      (not(carry ?untyped2 ?untyped1))
    )
  )
)
