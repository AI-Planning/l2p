(define (domain test_domain)
    (:requirements :conditional-effects :disjunctive-preconditions :equality :negative-preconditions :strips :typing :universal-preconditions)
    (:types arm block table - physical_object)
    (:predicates (clear ?b - block)  (empty ?a - arm)  (holding ?a - arm ?b - block)  (on_table ?b - block ?t - table))
    (:action pickup
        :parameters (?a - arm ?b - block ?t - table)
        :precondition (and (empty ?a) (on_table ?b ?t) (clear ?b))
        :effect (and (not (empty ?a)) (not (on_table ?b ?t)) (not (clear ?b)) (holding ?a ?b))
    )
     (:action putdown
        :parameters (?a - arm ?b - block ?t - table)
        :precondition (and (holding ?a ?b) (empty ?a) (not (on_table ?b ?t)) (clear ?b))
        :effect (and (not (holding ?a ?b)) (on_table ?b ?t) (not (clear ?b)) (empty ?a))
    )
     (:action stack
        :parameters (?a - arm ?b1 - block ?b2 - block)
        :precondition (and (holding ?a ?b1) (clear ?b2) (on_table ?b2) (empty ?a) (not (= ?b1 ?b2)))
        :effect (and (not (holding ?a ?b1)) (not (clear ?b2)) (not (on_table ?b1)) (holding ?a ?b1))
    )
     (:action unstack
        :parameters (?a - arm ?b1 - block ?b2 - block ?t - table)
        :precondition (and (empty ?a) (clear ?b1) (on_table ?b2 ?t) (not (= ?b1 ?b2)))
        :effect (and (holding ?b1) (not (clear ?b2)) (not (on_table ?b1 ?t)))
    )
)