(define (domain miconic)
  (:requirements :negative-preconditions)
  (:predicates
    (person ?x)
    (floor ?x)
    (lift_pos ?x)
    (in_lift ?x)
    (in_floor ?x ?y)
    (above ?x ?y)
  )

  (:action move_down
    :parameters (?untyped1 ?untyped2)
    :precondition (and
      (above ?untyped2 ?untyped1)
      (floor ?untyped1)
      (floor ?untyped2)
      (lift_pos ?untyped2)
      (not(lift_pos ?untyped1))
    )
    :effect (and
      (lift_pos ?untyped1)
      (not(lift_pos ?untyped2))
    )
  )

  (:action unboard
    :parameters (?untyped1 ?untyped2)
    :precondition (and
      (floor ?untyped1)
      (in_lift ?untyped2)
      (lift_pos ?untyped1)
      (not(in_floor ?untyped2 ?untyped1))
      (person ?untyped2)
    )
    :effect (and
      (in_floor ?untyped2 ?untyped1)
      (not(in_lift ?untyped2))
    )
  )

  (:action board
    :parameters (?untyped1 ?untyped2)
    :precondition (and
      (floor ?untyped1)
      (in_floor ?untyped2 ?untyped1)
      (lift_pos ?untyped1)
      (not(in_lift ?untyped2))
      (person ?untyped2)
    )
    :effect (and
      (in_lift ?untyped2)
      (not(in_floor ?untyped2 ?untyped1))
    )
  )

  (:action move_up
    :parameters (?untyped1 ?untyped2)
    :precondition (and
      (above ?untyped1 ?untyped2)
      (floor ?untyped1)
      (floor ?untyped2)
      (lift_pos ?untyped2)
      (not(lift_pos ?untyped1))
    )
    :effect (and
      (lift_pos ?untyped1)
      (not(lift_pos ?untyped2))
    )
  )
)
