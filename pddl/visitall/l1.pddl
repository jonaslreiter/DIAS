; The action schemas L1 (Balyo et al., 2024) learned for the paper's comparison, as its authors returned them, with equality written as = and the types as in domain.pddl.
; Learned from one random walk on train.pddl.
(define (domain grid-visit-all)
  (:requirements :typing :negative-preconditions)
  (:types place - object)
  (:predicates
    (connected ?x ?y - place)
    (at-robot ?x - place)
    (visited ?x - place)
  )

  (:action move
    :parameters (?place1 - place ?place2 - place)
    :precondition (and
      (at-robot ?place2)
      (connected ?place1 ?place2)
      (connected ?place2 ?place1)
      (not(at-robot ?place1))
      (visited ?place2)
    )
    :effect (and
      (at-robot ?place1)
      (not(at-robot ?place2))
      (visited ?place1)
    )
  )
)
