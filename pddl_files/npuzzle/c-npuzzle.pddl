(define (domain cpuzzle)
  (:requirements :strips :typing)
  (:types tile cell)

  (:predicates
    (at ?t - tile ?c - cell)
    (blank ?c - cell)
    (up ?from - cell ?to - cell)
    (right ?from - cell ?to - cell)
  )

  (:action move-up
    :parameters (?t - tile ?from - cell ?to - cell)
    :precondition (and
      (at ?t ?from)
      (blank ?to)
      (up ?from ?to)
    )
    :effect (and
      (not (at ?t ?from))
      (at ?t ?to)
      (not (blank ?to))
      (blank ?from)
    )
  )

  (:action move-down
    :parameters (?t - tile ?from - cell ?to - cell)
    :precondition (and
      (at ?t ?from)
      (blank ?to)
      (up ?to ?from)
    )
    :effect (and
      (not (at ?t ?from))
      (at ?t ?to)
      (not (blank ?to))
      (blank ?from)
    )
  )

  (:action move-left
    :parameters (?t - tile ?from - cell ?to - cell)
    :precondition (and
      (at ?t ?from)
      (blank ?to)
      (right ?to ?from)
    )
    :effect (and
      (not (at ?t ?from))
      (at ?t ?to)
      (not (blank ?to))
      (blank ?from)
    )
  )

  (:action move-right
    :parameters (?t - tile ?from - cell ?to - cell)
    :precondition (and
      (at ?t ?from)
      (blank ?to)
      (right ?from ?to)
    )
    :effect (and
      (not (at ?t ?from))
      (at ?t ?to)
      (not (blank ?to))
      (blank ?from)
    )
  )
)