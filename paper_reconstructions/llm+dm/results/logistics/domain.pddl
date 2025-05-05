(define (domain logistics)
    (:requirements :conditional-effects :disjunctive-preconditions :equality :negative-preconditions :typing)
    (:types
        city location package plane truck - object
    )
    (:predicates (at ?x - location ?l - location)  (in-plane ?p - package ?a - plane)  (in-truck ?p - package ?t - truck)  (truck-empty ?t - truck))
    (:action drive_truck
        :parameters (?t - truck ?l1 - location ?l2 - location)
        :precondition (and (at ?t ?l1) (truck-empty ?t) (at ?l1 ?l1) (at ?l2 ?l2))
        :effect (and (not (at ?t ?l1)) (at ?t ?l2))
    )
     (:action fly_airplane
        :parameters (?a - plane ?l1 - location ?l2 - location ?c1 - city ?c2 - city)
        :precondition (and (at ?a ?l1) (at ?c1 ?l1) (at ?c2 ?l2))
        :effect (and (not (at ?a ?l1)) (at ?a ?l2))
    )
     (:action load_airplane
        :parameters (?p - package ?a - plane ?l - location)
        :precondition (and (at ?a ?l) (at ?p ?l))
        :effect (and (not (at ?p ?l)) (in-plane ?p ?a))
    )
     (:action load_truck
        :parameters (?p - package ?t - truck ?l - location)
        :precondition (and (at ?t ?l) (at ?p ?l) (truck-empty ?t))
        :effect (and (not (at ?p ?l)) (not (truck-empty ?t)) (in-truck ?p ?t))
    )
     (:action unload_airplane
        :parameters (?p - package ?a - plane ?l - location)
        :precondition (and (in-plane ?p ?a) (at ?a ?l))
        :effect (and (not (in-plane ?p ?a)) (at ?p ?l))
    )
     (:action unload_truck
        :parameters (?p - package ?t - truck ?l - location)
        :precondition (and (in-truck ?p ?t) (at ?t ?l) (not (truck-empty ?t)))
        :effect (and (not (in-truck ?p ?t)) (truck-empty ?t) (at ?p ?l))
    )
)