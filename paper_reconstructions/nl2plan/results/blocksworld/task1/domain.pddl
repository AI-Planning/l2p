(define (domain blocksworld)
    (:requirements :conditional-effects :disjunctive-preconditions :equality :existential-preconditions :negative-preconditions :strips :typing :universal-preconditions)
    (:types
        block table - object
    )
    (:predicates (clear ?b - block)  (holding ?b - block)  (on_block ?b1 - block ?b2 - block)  (on_table ?b - block))
    (:action pick_block
        :parameters (?b - block ?b2 - block)
        :precondition (and (clear ?b) (or (on_table ?b) (on_block ?b ?b2)))
        :effect (and (not (on_table ?b)) (not (on_block ?b ?b2)) (holding ?b))
    )
     (:action place_on_block
        :parameters (?b1 - block ?b2 - block)
        :precondition (and (holding ?b1) (clear ?b2))
        :effect (and (not (holding ?b1)) (on_block ?b1 ?b2) (not (clear ?b2)) (clear ?b1))
    )
     (:action place_on_table
        :parameters (?b - block)
        :precondition (holding ?b)
        :effect (and (not (holding ?b)) (on_table ?b) (clear ?b))
    )
)