(define (domain tyreworld)
   (:requirements
      :typing :negative-preconditions :disjunctive-preconditions :conditional-effects :equality)

   (:types 
      object
      small_object - object
      tool - small_object
      wheel - small_object
      nut - small_object
      container - object
      hub - object
   )

   (:predicates 
      (container-open ?c - container) ;  true if the container ?c is open
      (object-in-container ?o - small_object ?c - container) ;  true if the object ?o is located in the container ?c
      (robot-holding ?o - small_object) ;  true if the robot is currently holding the object ?o
      (hub-on-ground ?h - hub) ;  true if the hub ?h is on the ground and not jacked up
      (nut-loosened ?n - nut) ;  true if the nut ?n has been loosened
      (unfastened ?h - hub) ;  true if the hub ?h has been unfastened
      (wheel-removed ?w - wheel) ;  true if the wheel ?w has been removed from its hub
      (wheel-inflated ?w - wheel) ;  true if the wheel ?w has been inflated
   )

   (:action Open a container
      :parameters (
         ?c - container
         ?o - small_object
      )
      :precondition
         (and
             (not (container-open ?c))
             (robot-holding ?o)
             (object-in-container ?o ?c)
         )
      :effect
         (and
             (container-open ?c)
             (not (robot-holding ?o))
         )
   )

   (:action Close a container
      :parameters (
         ?c - container
      )
      :precondition
         (and
             (container-open ?c)
             (not (robot-holding ?o))
         )
      :effect
         (and
             (not (container-open ?c))
         )
   )

   (:action Fetch an object from a container
      :parameters (
         ?o - small_object
         ?c - container
      )
      :precondition
         (and
             (container-open ?c)
             (object-in-container ?o ?c)
         )
      :effect
         (and
             (not (object-in-container ?o ?c))
             (robot-holding ?o)
         )
   )

   (:action Put away an object into a container
      :parameters (
         ?o - small_object
         ?c - container
      )
      :precondition
         (and
             (robot-holding ?o)
             (container-open ?c)
         )
      :effect
         (and
             (not (robot-holding ?o))
             (object-in-container ?o ?c)
         )
   )

   (:action Loosen a nut in a hub
      :parameters (
         ?n - nut
         ?h - hub
      )
      :precondition
         (and
             (robot-holding ?n)
             (hub-on-ground ?h)
         )
      :effect
         (and
             (not (robot-holding ?n))
             (nut-loosened ?n)
         )
   )

   (:action Tighten a nut in a hub
      :parameters (
         ?n - nut
         ?h - hub
      )
      :precondition
         (and
             (robot-holding ?n)
             (hub-on-ground ?h)
             (not (nut-loosened ?n))
         )
      :effect
         (and
             (not (robot-holding ?n))
             (nut-loosened ?n)
         )
   )

   (:action Jack up a hub
      :parameters (
         ?h - hub
      )
      :precondition
         (and
             (hub-on-ground ?h)
         )
      :effect
         (and
             (not (hub-on-ground ?h))
         )
   )

   (:action Jack down a hub
      :parameters (
         ?h - hub
         ?o - small_object
      )
      :precondition
         (and
             (hub-on-ground ?h)
             (robot-holding ?o)
         )
      :effect
         (and
             (not (hub-on-ground ?h))
             (not (robot-holding ?o))
         )
   )

   (:action Unfasten a hub
      :parameters (
         ?h - hub
         ?n - nut
      )
      :precondition
         (and
             (robot-holding ?n)
             (nut-loosened ?n)
             (hub-on-ground ?h)
         )
      :effect
         (and
             (not (robot-holding ?n))
             (not (nut-loosened ?n))
             (unfastened ?h)
             (robot-holding ?n)
         )
   )

   (:action Fasten a hub
      :parameters (
         ?h - hub
         ?n - nut
      )
      :precondition
         (and
             (robot-holding ?n)
             (unfastened ?h)
             (nut-loosened ?n)
             (hub-on-ground ?h)
         )
      :effect
         (and
             (not (robot-holding ?n))
             (not (unfastened ?h))
             (hub-on-ground ?h)
         )
   )

   (:action Remove wheel from hub
      :parameters (
         ?w - wheel
         ?h - hub
      )
      :precondition
         (and
             (robot-holding ?w)
             (hub-on-ground ?h)
             (unfastened ?h)
         )
      :effect
         (and
             (not (robot-holding ?w))
             (not (unfastened ?h))
             (not (hub-on-ground ?h))
             (wheel-removed ?w)
         )
   )

   (:action Put wheel on hub
      :parameters (
         ?w - wheel
         ?h - hub
      )
      :precondition
         (and
             (robot-holding ?w)
             (unfastened ?h)
             (hub-on-ground ?h)
         )
      :effect
         (and
             (not (robot-holding ?w))
             (not (wheel-removed ?w))
             (wheel-removed ?w)
         )
   )

   (:action Inflate wheel
      :parameters (
         ?w - wheel
      )
      :precondition
         (and
             (robot-holding ?w)
             (not (wheel-removed ?w))
         )
      :effect
         (and
             (not (robot-holding ?w))
             (wheel-inflated ?w)
         )
   )
)