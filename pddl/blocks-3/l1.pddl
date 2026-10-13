; The action schemas L1 (Balyo et al., 2024) learned for the paper's comparison, as its authors returned them, with equality written as = and the types as in domain.pddl.
; Learned from one random walk on train.pddl.
(define (domain blocksworld)
  (:requirements :negative-preconditions :typing :equality)
  (:predicates
    (clear ?x)
    (on-table ?x)
    (on ?x ?y)
  )

  (:action stack
    :parameters (?untyped1 ?untyped2)
    :precondition (and
      (clear ?untyped1)
      (clear ?untyped2)
      (= ?untyped1 ?untyped1)
      (= ?untyped2 ?untyped2)
      (not(on ?untyped2 ?untyped1))
      (on-table ?untyped2)
    )
    :effect (and
      (not(clear ?untyped1))
      (not(on-table ?untyped2))
      (on ?untyped2 ?untyped1)
    )
  )

  (:action move
    :parameters (?untyped1 ?untyped2 ?untyped3)
    :precondition (and
      (clear ?untyped2)
      (clear ?untyped3)
      (= ?untyped1 ?untyped1)
      (= ?untyped2 ?untyped2)
      (= ?untyped3 ?untyped3)
      (not(clear ?untyped1))
      (not(on ?untyped3 ?untyped2))
      (on ?untyped3 ?untyped1)
    )
    :effect (and
      (clear ?untyped1)
      (not(clear ?untyped2))
      (not(on ?untyped3 ?untyped1))
      (on ?untyped3 ?untyped2)
    )
  )

  (:action newtower
    :parameters (?untyped1 ?untyped2)
    :precondition (and
      (clear ?untyped2)
      (= ?untyped1 ?untyped1)
      (= ?untyped2 ?untyped2)
      (not(clear ?untyped1))
      (not(on-table ?untyped2))
      (on ?untyped2 ?untyped1)
    )
    :effect (and
      (clear ?untyped1)
      (not(on ?untyped2 ?untyped1))
      (on-table ?untyped2)
    )
  )
)
