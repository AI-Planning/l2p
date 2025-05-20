(define (domain logistics)
   (:requirements
      :typing :negative-preconditions :disjunctive-preconditions :conditional-effects :equality)

   (:types 
      city location package plane truck - object
   )

   (:predicates 
      (at ?t - truck ?l - location)
      (package-at ?p - package ?l - location)
      (truck-full ?t - truck)
      (truck-has-package ?t - truck ?p - package)
      (connected ?l1 - location ?l2 - location)
      (in-city ?l - location ?c - city)
   )

   (:action load_truck
      :parameters (
         ?p - package ?t - truck ?l - location
      )
      :precondition
         (and
             (at ?t ?l) ; the truck is at the location
             (package-at ?p ?l) ; the package is at the same location
             (not (truck-full ?t)) ; the truck is not full
         )
      :effect
         (and
             (not (package-at ?p ?l)) ; the package is no longer at the location
             (truck-has-package ?t ?p) ; the truck now has the package
         )
   )

   (:action unload_truck
      :parameters (
         ?p - package ?t - truck ?l - location
      )
      :precondition
         (and
             (at ?t ?l) ; The truck is at the location
             (truck-has-package ?t ?p) ; The truck has the package
         )
      :effect
         (and
             (not (truck-has-package ?t ?p)) ; The truck no longer has the package
             (package-at ?p ?l) ; The package is now at the location
         )
   )

   (:action load_airplane
      :parameters (
         ?p - package ?t - truck ?l - location
      )
      :precondition
         (and
             (package-at ?p ?l) ; The package is at the specified location
             (at ?t ?l) ; The truck is at the specified location
             (not (truck-full ?t)) ; The truck is not full
         )
      :effect
         (and
             (not (package-at ?p ?l)) ; The package is no longer at the location
             (truck-has-package ?t ?p) ; The truck now has the package
         )
   )

   (:action unload_airplane
      :parameters (
         ?p - package ?t - truck ?l - location
      )
      :precondition
         (and
             (at ?t ?l) ; The truck is at the location
             (truck-has-package ?t ?p) ; The truck has the package
         )
      :effect
         (and
             (not (truck-has-package ?t ?p)) ; The truck no longer has the package
             (package-at ?p ?l) ; The package is now at the location
         )
   )

   (:action drive_truck
      :parameters (
         ?t - truck ?l1 ?l2 - location ?c - city
      )
      :precondition
         (and
             (at ?t ?l1) ; The truck is at the starting location
             (connected ?l1 ?l2) ; The starting location is connected to the destination location
             (in-city ?l1 ?c) ; The starting location is in the specified city
             (in-city ?l2 ?c) ; The destination location is in the specified city
         )
      :effect
         (and
             (not (at ?t ?l1)) ; The truck is no longer at the starting location
             (at ?t ?l2) ; The truck is now at the destination location
         )
   )

   (:action fly_airplane
      :parameters (
         ?t - truck ?l1 ?l2 - location ?c1 ?c2 - city
      )
      :precondition
         (and
             (at ?t ?l1) ; The truck is at the departure airport
             (connected ?l1 ?l2) ; The departure and arrival airports are connected
             (in-city ?l1 ?c1) ; The departure airport is in city c1
             (in-city ?l2 ?c2) ; The arrival airport is in city c2
         )
      :effect
         (and
             (not (at ?t ?l1)) ; The truck is no longer at the departure airport
             (at ?t ?l2) ; The truck is now at the arrival airport
         )
   )
)