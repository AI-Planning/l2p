(define (domain blocksworld)
    (:requirements :conditional-effects :disjunctive-preconditions :equality :existential-preconditions :negative-preconditions :strips :typing :universal-preconditions)
    (:types
        location - object
        block on_block table - location
    )
    (:predicates (at ?b - block ?l - location)  (clear ?b - block)  (free) (holding ?b - block)  (on ?b - block ?t - block))
    (:action pickup_block
        :parameters (?b - block ?l - location)
        :precondition (and (clear ?b) (at ?b ?l) (free))
        :effect (and (not (at ?b ?l)) (holding ?b) (not (free)))
    )
     (:action place_on_block
        :parameters (?block - block ?target - block)
        :precondition (and (holding ?block) (clear ?target))
        :effect (and (not (holding ?block)) (on ?block ?target) (not (clear ?target)) (clear ?block))
    )
     (:action place_on_table
        :parameters (?b - block ?t - table)
        :precondition (holding ?b)
        :effect (and (not (holding ?b)) (at ?b ?t))
    )
)