; The action schemas L1 (Balyo et al., 2024) learned for the paper's comparison, as its authors returned them, with equality written as = and the types as in domain.pddl.
; Learned from one random walk on pfile2 (6 spanners, 4 nuts, 8 locations), not on train.pddl, which DIAS trains on.
(define (domain spanner)
  (:requirements :typing :strips :negative-preconditions)
  (:types location locatable - object man nut spanner - locatable)
  (:predicates
    (at ?m - locatable ?l - location)
    (carrying ?m - man ?s - spanner)
    (useable ?s - spanner)
    (link ?l1 - location ?l2 - location)
    (tightened ?n - nut)
    (loose ?n - nut)
  )

  (:action walk
    :parameters (?location1 - location ?location2 - location ?man1 - man)
    :precondition (and
      (at ?man1 ?location2)
      (link ?location2 ?location1)
      (not(at ?man1 ?location1))
    )
    :effect (and
      (at ?man1 ?location1)
      (not(at ?man1 ?location2))
    )
  )

  (:action pickup_spanner
    :parameters (?location1 - location ?man1 - man ?spanner1 - spanner)
    :precondition (and
      (at ?man1 ?location1)
      (at ?spanner1 ?location1)
      (not(carrying ?man1 ?spanner1))
      (useable ?spanner1)
    )
    :effect (and
      (carrying ?man1 ?spanner1)
      (not(at ?spanner1 ?location1))
    )
  )

  (:action tighten_nut
    :parameters (?nut1 - nut ?spanner1 - spanner)
    :precondition (and
      (loose ?nut1)
      (not(tightened ?nut1))
      (useable ?spanner1)
    )
    :effect (and
      (not(loose ?nut1))
      (not(useable ?spanner1))
      (tightened ?nut1)
    )
  )
)
