(define (domain blocksworld)
    (:requirements :negative-preconditions :strips :typing)
    (:types
        location - object
        block table - location
    )
    (:predicates (clear ?b - block)  (hand-empty) (holding ?b - block)  (on ?b1 - block ?b2 - block)  (on-table ?b - block))
    (:action pick_from_block
        :parameters (?b - block ?from - block)
        :precondition (and (hand-empty) (clear ?b) (on ?b ?from))
        :effect (and (holding ?b) (not (hand-empty)) (not (on ?b ?from)) (clear ?from))
    )
     (:action pick_from_table
        :parameters (?b - block)
        :precondition (and (on-table ?b) (clear ?b) (hand-empty))
        :effect (and (not (on-table ?b)) (not (hand-empty)) (holding ?b))
    )
     (:action place_on_block
        :parameters (?b - block ?dest - block)
        :precondition (and (holding ?b) (clear ?dest))
        :effect (and (not (holding ?b)) (hand-empty) (on ?b ?dest) (not (clear ?dest)) (clear ?b))
    )
     (:action place_on_table
        :parameters (?b - block)
        :precondition (holding ?b)
        :effect (and (not (holding ?b)) (on-table ?b) (hand-empty))
    )
)