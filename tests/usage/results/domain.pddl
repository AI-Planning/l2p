(define (domain blocksworld_domain)
    (:requirements :conditional-effects :disjunctive-preconditions :equality :negative-preconditions :strips :typing :universal-preconditions)
    (:types arm block table - object)
    (:predicates (clear ?b - block)  (holding ?a - arm ?b - block)  (on_top ?b1 - block ?b2 - block))
    (:action stack
        :parameters (?arm - arm ?top - block ?bottom - block)
        :precondition (and (holding ?arm ?top) (clear ?bottom))
        :effect (and (not (holding ?arm ?top)) (on_top ?top ?bottom) (not (clear ?bottom)))
    )
)