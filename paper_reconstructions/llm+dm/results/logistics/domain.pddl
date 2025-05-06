(define (domain logistics)
   (:requirements
      :typing :negative-preconditions :disjunctive-preconditions :conditional-effects :equality)

   (:types 
      object
      truck - object
      plane - object
      package - object
      city - object
      location - object
   )

   (:predicates 
      (at ?x - object ?l - location) ;  true if object ?x is located at location ?l
      (truck-empty ?t - truck) ;  true if truck ?t is empty
      (in-truck ?p - package ?t - truck) ;  true if package ?p is loaded into truck ?t
      (in-plane ?p - package ?a - plane) ;  true if package ?p is loaded into airplane ?a
   )

   (:action load_truck
      :parameters (
         ?p - package ?t - truck ?l - location
      )
      :precondition
         (and
             (at ?t ?l)
             (at ?p ?l)
             (truck-empty ?t)
         )
      :effect
         (and
             (not (at ?p ?l))
             (not (truck-empty ?t))
             (in-truck ?p ?t)
         )
   )

   (:action unload_truck
      :parameters (
         ?p - package ?t - truck ?l - location
      )
      :precondition
         (and
             (in-truck ?p ?t)
             (at ?t ?l)
         )
      :effect
         (and
             (not (in-truck ?p ?t))
             (truck-empty ?t)
             (at ?p ?l)
         )
   )

   (:action load_airplane
      :parameters (
         ?p - package ?a - plane ?l - location
      )
      :precondition
         (and
             (at ?a ?l)
             (at ?p ?l)
         )
      :effect
         (and
             (not (at ?p ?l))
             (in-plane ?p ?a)
         )
   )

   (:action unload_airplane
      :parameters (
         ?p - package ?a - plane ?l - location
      )
      :precondition
         (and
             (in-plane ?p ?a)
             (at ?a ?l)
         )
      :effect
         (and
             (not (in-plane ?p ?a))
             (at ?p ?l)
         )
   )

   (:action drive_truck
      :parameters (
         ?t - truck ?l1 ?l2 - location
      )
      :precondition
         (and
             (at ?t ?l1)
             (at ?l1 ?l2)  ; This checks if the starting location and destination are in the same city
             (at ?l2 ?l1)  ; This checks if the destination is reachable from the starting location
             (truck-empty ?t)
         )
      :effect
         (and
             (not (at ?t ?l1))
             (at ?t ?l2)
         )
   )

   (:action fly_airplane
      :parameters (
         ?a - plane ?l1 ?l2 - location
      )
      :precondition
         (and
             (at ?a ?l1)
             (at ?l1 ?l1)  ; Assuming ?l1 is a location and not a city
             (at ?l2 ?l2)  ; Assuming ?l2 is a location and not a city
             (not (= ?l1 ?l2))
         )
      :effect
         (and
             (not (at ?a ?l1))
             (at ?a ?l2)
         )
   )
)