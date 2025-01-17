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
      (package-at ?p - package ?l - location) ;  true if the package ?p is located at location ?l
      (truck-at ?t - truck ?l - location) ;  true if the truck ?t is located at location ?l
      (truck-has-space ?t - truck) ;  true if the truck ?t has space to load more packages
      (truck-holding ?t - truck ?p - package) ;  true if the truck ?t is currently holding package ?p
      (airport-location ?l - location) ;  true if the location ?l is an airport
      (plane-at ?p - plane ?l - location) ;  true if the plane ?p is located at location ?l
   )

   (:action Load a package into a truck
      :parameters (
         ?p - package
         ?t - truck
         ?l - location
      )
      :precondition
         (and
             (package-at ?p ?l)
             (truck-at ?t ?l)
             (truck-has-space ?t)
         )
      :effect
         (and
             (not (package-at ?p ?l))
             (truck-holding ?t ?p)
         )
   )

   (:action Unload a package from a truck
      :parameters (
         ?p - package
         ?t - truck
         ?l - location
      )
      :precondition
         (and
             (truck-at ?t ?l)
             (truck-holding ?t ?p)
         )
      :effect
         (and
             (not (truck-holding ?t ?p))
             (package-at ?p ?l)
         )
   )

   (:action Load a package into an airplane
      :parameters (
         ?p - package
         ?t - truck
         ?l - location
      )
      :precondition
         (and
             (package-at ?p ?l)
             (truck-at ?t ?l)
             (truck-has-space ?t)
         )
      :effect
         (and
             (not (package-at ?p ?l))
             (truck-holding ?t ?p)
         )
   )

   (:action Unload a package from an airplane
      :parameters (
         ?p - package
         ?a - plane
         ?l - location
         ?t - truck
      )
      :precondition
         (and
             (truck-at ?t ?l)               ; The truck is at the unloading location
             (package-at ?p ?a)             ; The package is currently at the airplane
             (truck-has-space ?t)           ; The truck has space to load the package
         )
      :effect
         (and
             (not (package-at ?p ?a))        ; The package is no longer at the airplane
             (package-at ?p ?l)              ; The package is now at the unloading location
             (truck-holding ?t ?p)           ; The truck is now holding the package
         )
   )

   (:action Drive a truck from one location to another in a city
      :parameters (
         ?t - truck
         ?l1 - location
         ?l2 - location
         ?c - city
      )
      :precondition
         (and
             (truck-at ?t ?l1)
             (truck-has-space ?t)
         )
      :effect
         (and
             (not (truck-at ?t ?l1))
             (truck-at ?t ?l2)
         )
   )

   (:action Fly an airplane from one city to another
      :parameters (
         ?plane - plane
         ?from - location
         ?to - location
      )
      :precondition
         (and
             (plane-at ?plane ?from)
             (airport-location ?from)
             (airport-location ?to)
         )
      :effect
         (and
             (not (plane-at ?plane ?from))
             (plane-at ?plane ?to)
         )
   )
)