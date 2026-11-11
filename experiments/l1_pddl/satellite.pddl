(define (domain satellite)
  (:requirements :strips :equality :typing :negative-preconditions)
  (:types satellite direction instrument mode)
  (:predicates
    (on_board ?i - instrument ?s - satellite)
    (supports ?i - instrument ?m - mode)
    (pointing ?s - satellite ?d - direction)
    (power_avail ?s - satellite)
    (power_on ?i - instrument)
    (calibrated ?i - instrument)
    (have_image ?d - direction ?m - mode)
    (calibration_target ?i - instrument ?d - direction))

  (:action turn_to
    :parameters (?direction1 - direction ?direction2 - direction ?satellite1 - satellite)
    :precondition (and
      (= ?direction1 ?direction1)
      (= ?direction2 ?direction2)
      (= ?satellite1 ?satellite1)
      (not (pointing ?satellite1 ?direction1))
      (pointing ?satellite1 ?direction2))
    :effect (and
      (not (pointing ?satellite1 ?direction2))
      (pointing ?satellite1 ?direction1)))

  (:action switch_on
    :parameters (?instrument1 - instrument ?satellite1 - satellite)
    :precondition (and
      (= ?instrument1 ?instrument1)
      (= ?satellite1 ?satellite1)
      (not (power_on ?instrument1))
      (on_board ?instrument1 ?satellite1)
      (power_avail ?satellite1))
    :effect (and
      (not (calibrated ?instrument1))
      (not (power_avail ?satellite1))
      (power_on ?instrument1)))

  (:action switch_off
    :parameters (?instrument1 - instrument ?satellite1 - satellite)
    :precondition (and
      (= ?instrument1 ?instrument1)
      (= ?satellite1 ?satellite1)
      (not (power_avail ?satellite1))
      (on_board ?instrument1 ?satellite1)
      (power_on ?instrument1))
    :effect (and
      (not (power_on ?instrument1))
      (power_avail ?satellite1)))

  (:action calibrate
    :parameters (?instrument1 - instrument)
    :precondition (and
      (= ?instrument1 ?instrument1))
    :effect (and
      (calibrated ?instrument1)))

  (:action take_image
    :parameters (?direction1 - direction ?mode1 - mode)
    :precondition (and
      (= ?direction1 ?direction1)
      (= ?mode1 ?mode1))
    :effect (and
      (have_image ?direction1 ?mode1)))
)
