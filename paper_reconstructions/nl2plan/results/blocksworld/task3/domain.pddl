(define (domain blocksworld)
    (:requirements :disjunctive-preconditions :negative-preconditions :strips :typing)
    (:types
        location - object
        block table - location
    )
    (:predicates (clear_block ?b - block)  (clear_location ?l - location)  (hand_empty) (holding ?b - block)  (on_block ?b1 - block ?b2 - block)  (on_location ?b - block ?l - location))
    (:action pick_up_block_from_block
        :parameters (?b_to_pick - block ?b_support - block)
        :precondition (and (hand_empty) (on_block ?b_to_pick ?b_support) (clear_block ?b_to_pick))
        :effect (and (not (hand_empty)) (holding ?b_to_pick) (not (on_block ?b_to_pick ?b_support)) (clear_block ?b_support))
    )
     (:action pick_up_block_from_location
        :parameters (?b - block ?l - location)
        :precondition (and (hand_empty) (on_location ?b ?l) (clear_block ?b))
        :effect (and (holding ?b) (not (hand_empty)) (not (on_location ?b ?l)))
    )
     (:action place_block_on_block
        :parameters (?b - block ?target_b - block)
        :precondition (and (holding ?b) (clear_block ?target_b))
        :effect (and (not (holding ?b)) (on_block ?b ?target_b) (not (clear_block ?target_b)) (clear_block ?b))
    )
     (:action place_block_on_location
        :parameters (?b - block ?l - location)
        :precondition (and (holding ?b) (clear_location ?l))
        :effect (and (not (holding ?b)) (on_location ?b ?l) (not (clear_location ?l)) (clear_block ?b))
    )
)