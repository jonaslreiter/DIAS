(define (domain driverlog)
  (:requirements :typing :negative-preconditions)
  (:types
    location locatable - object
    driver truck obj - locatable)
  (:predicates
    (at ?obj - locatable ?loc - location)
    (in ?obj1 - obj ?obj - truck)
    (driving ?d - driver ?v - truck)
    (link ?x ?y - location)
    (path ?x ?y - location)
    (empty ?v - truck))

  (:action load-truck
    :parameters (?location1 - location ?obj1 - obj ?truck1 - truck)
    :precondition (and
      (at ?obj1 ?location1)
      (at ?truck1 ?location1)
      (not (in ?obj1 ?truck1)))
    :effect (and
      (in ?obj1 ?truck1)
      (not (at ?obj1 ?location1))))

  (:action walk
    :parameters (?driver1 - driver ?location1 - location ?location2 - location)
    :precondition (and
      (at ?driver1 ?location2)
      (not (at ?driver1 ?location1))
      (path ?location1 ?location2)
      (path ?location2 ?location1))
    :effect (and
      (at ?driver1 ?location1)
      (not (at ?driver1 ?location2))))

  (:action unload-truck
    :parameters (?location1 - location ?obj1 - obj ?truck1 - truck)
    :precondition (and
      (at ?truck1 ?location1)
      (in ?obj1 ?truck1)
      (not (at ?obj1 ?location1)))
    :effect (and
      (at ?obj1 ?location1)
      (not (in ?obj1 ?truck1))))

  (:action board-truck
    :parameters (?driver1 - driver ?location1 - location ?truck1 - truck)
    :precondition (and
      (at ?driver1 ?location1)
      (at ?truck1 ?location1)
      (empty ?truck1)
      (not (driving ?driver1 ?truck1)))
    :effect (and
      (driving ?driver1 ?truck1)
      (not (at ?driver1 ?location1))
      (not (empty ?truck1))))

  (:action drive-truck
    :parameters (?location1 - location ?location2 - location ?truck1 - truck)
    :precondition (and
      (at ?truck1 ?location2)
      (link ?location1 ?location2)
      (link ?location2 ?location1)
      (not (at ?truck1 ?location1)))
    :effect (and
      (at ?truck1 ?location1)
      (not (at ?truck1 ?location2))))

  (:action disembark-truck
    :parameters (?driver1 - driver ?location1 - location ?truck1 - truck)
    :precondition (and
      (at ?truck1 ?location1)
      (driving ?driver1 ?truck1)
      (not (at ?driver1 ?location1))
      (not (empty ?truck1)))
    :effect (and
      (at ?driver1 ?location1)
      (empty ?truck1)
      (not (driving ?driver1 ?truck1))))
)
