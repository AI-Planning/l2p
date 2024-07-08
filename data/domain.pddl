(define (domain test_domain)
    (:requirements :conditional-effects :disjunctive-preconditions :equality :negative-preconditions :strips :typing :universal-preconditions)
    (:types arm block table - object)
    (:predicates (arm_empty ?a - arm)  (clear ?b - block)  (holding ?a - arm ?b - block)  (on ?top - block ?bottom - block)  (on_table ?b - block ?t - table))
    (:action pickup
        :parameters (?a - arm ?b - block ?t - table)
        :precondition (and (arm_empty ?a) (on_table ?b ?t) (clear ?b))
        :effect (and (not (arm_empty ?a)) (holding ?a ?b) (not (on_table ?b ?t)) (not (clear ?b)))
    )
     (:action putdown
        :parameters (?a - arm ?b - block ?t - table)
        :precondition (holding ?a ?b)
        :effect (and (not (holding ?a ?b)) (arm_empty ?a) (on_table ?b ?t) (clear ?b))
    )
     (:action stack
        :parameters (?a - arm ?top - block ?bottom - block)
        :precondition (and (holding ?a ?top) (clear ?bottom))
        :effect (and (arm_empty ?a) (on ?top ?bottom) (not (clear ?bottom)) (not (holding ?a ?top)))
    )
     (:action unstack
        :parameters (?a - arm ?top - block ?bottom - block)
        :precondition (and (arm_empty ?a) (clear ?top) (on ?top ?bottom))
        :effect (and (not (arm_empty ?a)) (holding ?a ?top) (not (on ?top ?bottom)) (clear ?bottom))
    )
)