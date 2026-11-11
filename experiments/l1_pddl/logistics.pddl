(define (domain logistics)
  (:requirements :strips :negative-preconditions)
  (:predicates
    (object ?o)
    (truck ?t)
    (airplane ?p)
    (vehicle ?v)
    (location ?l)
    (airport ?a)
    (city ?c)
    (loc ?l ?c)
    (at ?x ?l)
    (in ?p ?v))

  (:action fly
    :parameters (?untyped1 ?untyped2 ?untyped3)
    :precondition (and
      (airplane ?untyped3)
      (airport ?untyped1)
      (airport ?untyped2)
      (at ?untyped3 ?untyped2)
      (location ?untyped1)
      (location ?untyped2)
      (not (at ?untyped3 ?untyped1))
      (vehicle ?untyped3))
    :effect (and
      (at ?untyped3 ?untyped1)
      (not (at ?untyped3 ?untyped2))))

  (:action load
    :parameters (?untyped1 ?untyped2 ?untyped3)
    :precondition (and
      (at ?untyped1 ?untyped3)
      (at ?untyped2 ?untyped3)
      (location ?untyped3)
      (not (in ?untyped1 ?untyped2))
      (vehicle ?untyped2))
    :effect (and
      (in ?untyped1 ?untyped2)
      (not (at ?untyped1 ?untyped3))))

  (:action drive
    :parameters (?untyped1 ?untyped2 ?untyped3)
    :precondition (and
      (at ?untyped3 ?untyped2)
      (location ?untyped1)
      (location ?untyped2)
      (not (at ?untyped3 ?untyped1))
      (truck ?untyped3)
      (vehicle ?untyped3))
    :effect (and
      (at ?untyped3 ?untyped1)
      (not (at ?untyped3 ?untyped2))))

  (:action unload
    :parameters (?untyped1 ?untyped2 ?untyped3)
    :precondition (and
      (at ?untyped3 ?untyped2)
      (in ?untyped1 ?untyped3)
      (location ?untyped2)
      (not (at ?untyped1 ?untyped2))
      (vehicle ?untyped3))
    :effect (and
      (at ?untyped1 ?untyped2)
      (not (in ?untyped1 ?untyped3))))
)
