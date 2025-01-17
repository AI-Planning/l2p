(define (domain household)
   (:requirements
      :typing :negative-preconditions :disjunctive-preconditions :conditional-effects :equality)

   (:types 
      object
      furnitureAppliance - object
      householdObject - object
      smallReceptacle - householdObject
   )

   (:predicates 
      (robot-at ?x - furnitureAppliance) ;  true if the robot is currently at the furniture or appliance ?x
      (furnitureAppliance-clear ?x - furnitureAppliance) ;  true if the furniture or appliance ?x is clear for the robot to navigate to
      (object-pickupable ?o - householdObject) ;  true if the object ?o can be picked up by the robot
      (object-stacked ?o - householdObject) ;  true if the object ?o is stacked on top of other household items
      (object-in ?o - householdObject ?f - furnitureAppliance) ;  true if the object ?o is placed in or on the furniture or appliance ?f
      (robot-holding ?o - householdObject) ;  true if the robot is currently holding the object ?o
      (dirty ?c - smallReceptacle) ;  true if the cloth ?c is dirty after cleaning
   )

   (:action Go to a Furniture Piece or an Appliance
      :parameters (
         ?from - furnitureAppliance
         ?to - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?from)
             (furnitureAppliance-clear ?to)
         )
      :effect
         (and
             (not (robot-at ?from))
             (robot-at ?to)
         )
   )

   (:action Pick up an Object on or in a Furniture Piece or an Appliance
      :parameters (
         ?o - householdObject
         ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
             (object-pickupable ?o)
             (not (object-stacked ?o))
             (not (robot-holding ?x))
         )
      :effect
         (and
             (not (furnitureAppliance-clear ?f))
             (not (object-pickupable ?o))
             (robot-holding ?o)
         )
   )

   (:action Put an Object on or in a Furniture Piece or an Appliance
      :parameters (
         ?o - householdObject
         ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
             (object-pickupable ?o)
             (not (object-stacked ?o))
         )
      :effect
         (and
             (not (object-pickupable ?o))
             (not (furnitureAppliance-clear ?f))
             (object-in ?o ?f)
         )
   )

   (:action Stack Objects
      :parameters (
         ?o1 - householdObject
         ?o2 - householdObject
         ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-holding ?o1)
             (object-pickupable ?o1)
             (object-pickupable ?o2)
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
             (not (object-stacked ?o2))
             (object-in ?o2 ?f)
         )
      :effect
         (and
             (not (robot-holding ?o1))
             (not (object-in ?o1 ?f))
             (object-stacked ?o1)
             (object-in ?o1 ?f)
         )
   )

   (:action Unstack Objects
      :parameters (
         ?o1 - householdObject
         ?o2 - householdObject
         ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
             (object-stacked ?o1)
             (object-in ?o1 ?f)
             (object-in ?o2 ?f)
             (object-pickupable ?o1)
         )
      :effect
         (and
             (not (object-stacked ?o1))
             (not (object-in ?o1 ?f))
             (not (object-in ?o2 ?f))
             (robot-holding ?o1)
             (object-in ?o2 ?f)
         )
   )

   (:action Open a Furniture Piece or an Appliance
      :parameters (
         ?f - furnitureAppliance
         ?o - householdObject
      )
      :precondition
         (and
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
             (not (robot-holding ?o))  ; Ensure the robot is not holding any household object
         )
      :effect
         (and
             (not (furnitureAppliance-clear ?f))  ; The furniture or appliance is now open
             (object-in ?o ?f)  ; The household object may now be accessible inside the furniture or appliance
         )
   )

   (:action Close a Furniture Piece or an Appliance
      :parameters (
         ?f - furnitureAppliance
         ?o - householdObject
      )
      :precondition
         (and
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
             (object-in ?o ?f)
             (not (robot-holding ?o))
         )
      :effect
         (and
             (not (object-in ?o ?f))
             (furnitureAppliance-clear ?f)
         )
   )

   (:action Toggle a Small Appliance On
      :parameters (
         ?o - householdObject
         ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
             (object-pickupable ?o)
             (not (robot-holding ?o))
             (object-in ?o ?f)
         )
      :effect
         (and
             (not (object-stacked ?o))
             (robot-holding ?o)
         )
   )

   (:action Toggle a Small Appliance Off
      :parameters (
         ?o - householdObject
         ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
             (object-in ?o ?f)
             (not (robot-holding ?o))
         )
      :effect
         (and
             (not (object-in ?o ?f))
             (object-stacked ?o)
         )
   )

   (:action Slice Objects
      :parameters (
         ?o - householdObject
         ?k - householdObject
         ?c - furnitureAppliance
         ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-holding ?k)
             (object-in ?o ?c)
             (object-pickupable ?o)
             (not (object-stacked ?o))
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
         )
      :effect
         (and
             (not (object-in ?o ?c))
             (not (robot-holding ?k))
             (robot-holding ?o)
             (object-stacked ?o)
         )
   )

   (:action Heat Food with a Microwave
      :parameters (
         ?m - furnitureAppliance
         ?r - smallReceptacle
      )
      :precondition
         (and
             (robot-at ?m)
             (furnitureAppliance-clear ?m)
             (object-in ?r ?m)
             (robot-holding ?r)
             (not (object-stacked ?r))
         )
      :effect
         (and
             (not (object-pickupable ?r))
             (not (robot-holding ?r))
             (not (object-in ?r ?m))
             (object-in ?r ?m) ; The food is now considered heated and remains in the microwave
         )
   )

   (:action Heat Food with Pan
      :parameters (
         ?f - furnitureAppliance
         ?p - furnitureAppliance
         ?h - householdObject
      )
      :precondition
         (and
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
             (object-in ?p ?f)
             (object-in ?h ?p)
             (not (robot-holding ?h))
             (object-pickupable ?h)
         )
      :effect
         (and
             (not (object-in ?h ?p))
             (not (object-pickupable ?h))
             (not (object-in ?p ?f))
             (object-in ?h ?f)
             (robot-holding ?p)
         )
   )

   (:action Transfer Food from One Small Receptacle to Another
      :parameters (
         ?food - householdObject
         ?receptacle_1 - smallReceptacle
         ?receptacle_2 - smallReceptacle
         ?furniture - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?furniture)
             (furnitureAppliance-clear ?furniture)
             (object-in ?receptacle_1 ?furniture)
             (object-in ?receptacle_2 ?furniture)
             (not (object-stacked ?receptacle_1))
             (not (object-stacked ?receptacle_2))
             (object-pickupable ?food)
             (robot-holding ?receptacle_1)
         )
      :effect
         (and
             (not (robot-holding ?receptacle_1))
             (not (object-in ?food ?receptacle_1))
             (object-in ?food ?receptacle_2)
             (robot-holding ?receptacle_2)
         )
   )

   (:action Puts an Object onto or into a Small Receptacle
      :parameters (
         ?o - householdObject
         ?r - smallReceptacle
         ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-holding ?o)
             (object-pickupable ?o)
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
             (object-in ?r ?f)
             (not (object-stacked ?r))
         )
      :effect
         (and
             (not (robot-holding ?o))
             (object-in ?o ?r)
             (not (object-pickupable ?o))
             (robot-hand-empty)
         )
   )

   (:action Pick up an Object on or in a Small Receptacle
      :parameters (
         
      )
      :precondition
         (and
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
             (object-in ?o ?r)
             (object-pickupable ?o)
             (not (object-stacked ?o))
             (not (robot-holding ?o))
         )
      :effect
         (and
             (not (object-in ?o ?r))
             (not (robot-holding ?o))
             (robot-holding ?o)
         )
   )

   (:action Open a Small Receptacle
      :parameters (
         ?r - smallReceptacle
         ?f - furnitureAppliance
         ?o - householdObject
      )
      :precondition
         (and
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
             (object-in ?r ?f)
             (not (object-stacked ?r))
             (object-pickupable ?r)
             (not (robot-holding ?o))  ; where ?o is any householdObject
         )
      :effect
         (and
             (not (object-in ?r ?f))
             (robot-holding ?r)
         )
   )

   (:action Close a Small Receptacle
      :parameters (
         ?r - smallReceptacle
         ?f - furnitureAppliance
         ?o - householdObject
      )
      :precondition
         (and
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
             (object-in ?r ?f)
             (not (object-stacked ?r))
             (robot-holding ?o) ; the robot must be holding an object
         )
      :effect
         (and
             (not (object-in ?r ?f)) ; the receptacle is no longer in the open state
             (object-in ?r ?f) ; assuming the receptacle is still in the same place after closing
         )
   )

   (:action Mash Food with a Blender
      :parameters (
         ?f - furnitureAppliance
         ?h - householdObject
      )
      :precondition
         (and
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
             (object-pickupable ?h)
             (not (object-stacked ?h))
             (object-in ?h ?f)
             (robot-holding ?h)
         )
      :effect
         (and
             (not (robot-holding ?h))
             (object-in ?h ?f)
             (not (object-pickupable ?h))
         )
   )

   (:action Wash an Object
      :parameters (
         ?o - householdObject
         ?s - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?s)
             (furnitureAppliance-clear ?s)
             (robot-holding ?o)
             (object-pickupable ?o)
         )
      :effect
         (and
             (not (robot-holding ?o))
             (object-in ?o ?s)
         )
   )

   (:action Wipe a Surface
      :parameters (
         ?f - furnitureAppliance
         ?c - smallReceptacle
      )
      :precondition
         (and
             (robot-at ?f)
             (furnitureAppliance-clear ?f)
             (robot-holding ?c)
             (object-pickupable ?c)
         )
      :effect
         (and
             (not (robot-holding ?c))
             (not (object-pickupable ?c))
             (object-in ?c ?f)
             (dirty ?c)
         )
   )

   (:action Vacuum a Carpet
      :parameters (
         ?v - householdObject
         ?c - furnitureAppliance
      )
      :precondition
         (and
             (robot-holding ?v)
             (robot-at ?c)
             (furnitureAppliance-clear ?c)
             (not (dirty ?v))
         )
      :effect
         (and
             (not (robot-holding ?v))
             (dirty ?v)
         )
   )

   (:action Empty a Vacuum Cleaner
      :parameters (
         ?v - householdObject
         ?t - smallReceptacle
         ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-holding ?v)
             (robot-at ?f)  ; The robot is at the furniture appliance
             (furnitureAppliance-clear ?f)
             (object-in ?v ?f)
         )
      :effect
         (and
             (not (object-in ?v ?f))
             (dirty ?t)  ; The trash can is now dirty after emptying the vacuum cleaner
             (robot-holding ?v)
         )
   )
)