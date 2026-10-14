(define (domain sokoban-domain)
  (:requirements :equality :negative-preconditions)
  (:predicates
    (has_player ?x)
    (has_box ?x)
    (adjacent ?x ?y)
    (adjacent_2 ?x ?y)
  )

  (:action move-player
    :parameters (?untyped1 ?untyped2)
    :precondition (and
      (adjacent ?untyped1 ?untyped2)
      (adjacent ?untyped2 ?untyped1)
      (has_player ?untyped2)
      (not(has_player ?untyped1))
    )
    :effect (and
      (has_player ?untyped1)
      (not(has_player ?untyped2))
    )
  )

  (:action push-box
    :parameters (?untyped1 ?untyped2 ?untyped3)
    :precondition (and
      (adjacent ?untyped1 ?untyped3)
      (adjacent ?untyped2 ?untyped3)
      (adjacent ?untyped3 ?untyped1)
      (adjacent ?untyped3 ?untyped2)
      (adjacent_2 ?untyped1 ?untyped2)
      (adjacent_2 ?untyped2 ?untyped1)
      (has_box ?untyped3)
      (has_player ?untyped2)
      (not(has_box ?untyped1))
      (not(has_player ?untyped3))
    )
    :effect (and
      (has_box ?untyped1)
      (has_player ?untyped3)
      (not(has_box ?untyped3))
      (not(has_player ?untyped2))
    )
  )
)
