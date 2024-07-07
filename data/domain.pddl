(define (domain test_domain)
    (:requirements :conditional-effects :disjunctive-preconditions :equality :negative-preconditions :strips :typing :universal-preconditions)
    (:types arm block table - object)
    (:predicates (clear ?b - block)  (empty ?a - arm)  (holding ?a - arm ?b - block)  (on ?b1 - block ?b2 - block)  (on_table ?b - block ?t - table))
    (:action pickup
        :parameters (?a - arm ?b - block ?t - table)
        :precondition (and (clear ?b) (empty ?a) (on_table ?b ?t))
        :effect (and (not (on_table ?b ?t)) (not (empty ?a)) (holding ?a ?b))
    )
     (:action putdown
        :parameters (?a - arm ?b - block ?t - table)
        :precondition (and (holding ?a ?b) (clear ?t))
        :effect (and (not (holding ?a ?b)) (on_table ?b ?t) (empty ?a))
    )
     (:action stack
        :parameters (?a - arm ?b1 - block ?b2 - block)
        :precondition (and (holding ?a ?b1) (clear ?b2))
        :effect (and (not (holding ?a ?b1)) (not (clear ?b2)) (clear ?b1) (on ?b1 ?b2))
    )
     (:action unstack
        :parameters (?a - arm ?b1 - block ?b2 - block)
        :precondition (and (empty ?a) (clear ?b1) (on ?b1 ?b2))
        :effect (and (not (empty ?a)) (holding ?a ?b1) (not (on ?b1 ?b2)) (clear ?b2))
    )
)