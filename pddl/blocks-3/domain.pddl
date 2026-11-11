(define (domain blocksworld)
  (:requirements :negative-preconditions :typing :equality)
  (:predicates
    (clear ?x)
    (on-table ?x)
    (on ?x ?y))

  (:action stack
    :parameters (?bm ?bt)
    :precondition (and
      (clear ?bm)
      (clear ?bt)
      (on-table ?bm)
      (not (= ?bm ?bt)))
    :effect (and
      (not (clear ?bt))
      (not (on-table ?bm))
      (on ?bm ?bt)))

  (:action newtower
    :parameters (?bm ?bf)
    :precondition (and
      (clear ?bm)
      (on ?bm ?bf)
      (not (= ?bm ?bf)))
    :effect (and
      (not (on ?bm ?bf))
      (on-table ?bm)
      (clear ?bf)))

  (:action move
    :parameters (?bm ?bf ?bt)
    :precondition (and
      (clear ?bm)
      (clear ?bt)
      (on ?bm ?bf)
      (not (= ?bm ?bt))
      (not (= ?bm ?bf))
      (not (= ?bf ?bt)))
    :effect (and
      (not (clear ?bt))
      (not (on ?bm ?bf))
      (on ?bm ?bt)
      (clear ?bf)))
)
