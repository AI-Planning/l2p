(define (domain blocksworld)
    (:requirements :negative-preconditions :strips :typing)
    (:types
        location - object
        block table - location
    )
    (:predicates (clear ?b - block)  (hand_empty) (holding ?b - block)  (on ?b1 - block ?b2 - block)  (on_table ?b - block ?t - table))
    (:action pick_block_from_block
        :parameters (?b - block ?above - block)
        :precondition (and (on ?b ?above) (clear ?b) (hand_empty))
        :effect (and (holding ?b) (not (hand_empty)) (not (on ?b ?above)) (clear ?above))
    )
     (:action pick_block_from_table
        :parameters (?b - block ?t - table)
        :precondition (and (hand_empty) (on_table ?b ?t) (clear ?b))
        :effect (and (holding ?b) (not (on_table ?b ?t)) (not (hand_empty)))
    )
     (:action place_block_on_block
        :parameters (?b - block ?target - block)
        :precondition (and (holding ?b) (clear ?target))
        :effect (and (not (holding ?b)) (on ?b ?target) (not (clear ?target)) (clear ?b))
    )
     (:action place_block_on_table
        :parameters (?b - block ?t - table)
        :precondition (holding ?b)
        :effect (and (not (holding ?b)) (on_table ?b ?t) (hand_empty))
    )
)