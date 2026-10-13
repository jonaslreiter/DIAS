; The action schemas L1 (Balyo et al., 2024) learned for the paper's comparison, as its authors returned them, with equality written as = and the types as in domain.pddl.
; Learned from one random walk on hanoi-3-7 (3 pegs, 7 discs), not on train.pddl, whose 6 discs DIAS trains on.
(define (domain hanoi)
  (:requirements :negative-preconditions :equality :strips)
  (:predicates
    (clear ?x)
    (on ?x ?y)
    (smaller ?x ?y)
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
      (smaller ?untyped3 ?untyped1)
      (smaller ?untyped3 ?untyped2)
    )
    :effect (and
      (clear ?untyped1)
      (not(clear ?untyped2))
      (not(on ?untyped3 ?untyped1))
      (on ?untyped3 ?untyped2)
    )
  )
)
