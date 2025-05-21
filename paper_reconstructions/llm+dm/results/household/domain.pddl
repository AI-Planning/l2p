(define (domain household)
   (:requirements
      :typing :negative-preconditions :disjunctive-preconditions :conditional-effects :equality)

   (:types 
      furnitureAppliance householdObject - object
      smallReceptacle - householdObject
   )

   (:predicates 
      (robot-at ?f - furnitureAppliance)
      (furnitureAppliance-clear ?f - furnitureAppliance)
      (pickupable ?o - householdObject)
      (stacked ?o - householdObject)
      (openable ?f - furnitureAppliance)
      (robot-holding ?o - householdObject)
      (opened ?f - furnitureAppliance)
      (sliced ?o - householdObject ?k - householdObject)
      (heated ?r - smallReceptacle ?m - furnitureAppliance)
      (dirty ?c - householdObject)
   )

   (:action go-to
      :parameters (
         ?from ?to - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?from) ; The robot is currently at the starting furniture or appliance
             (furnitureAppliance-clear ?to) ; The destination furniture or appliance is clear for navigation
         )
      :effect
         (and
             (not (robot-at ?from)) ; The robot is no longer at the starting furniture or appliance
             (robot-at ?to) ; The robot is now at the destination furniture or appliance
         )
   )

   (:action pick-up
      :parameters (
         ?o - householdObject ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?f) ; The robot is at the location of the furniture or appliance
             (furnitureAppliance-clear ?f) ; The furniture or appliance is clear for manipulation
             (pickupable ?o) ; The object is pickupable by the robot
             (not (stacked ?o)) ; The object is not stacked on top of other household items
             (or
                 (openable ?f) ; The furniture or appliance is openable
                 (not (openable ?f)) ; The furniture or appliance is not openable
             )
         )
      :effect
         (and
             (not (furnitureAppliance-clear ?f)) ; The furniture or appliance is no longer clear after picking up the object
             (not (pickupable ?o)) ; The object is no longer pickupable as it is now held by the robot
             (robot-holding ?o) ; The robot is now holding the object
         )
   )

   (:action put-on-or-in
      :parameters (
         ?o - householdObject ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-holding ?o) ; The robot must be holding the object
             (robot-at ?f) ; The robot must be at the location of the furniture or appliance
             (furnitureAppliance-clear ?f) ; The furniture or appliance must be clear for placing the object
             (or
                 (openable ?f) ; The furniture or appliance must be openable if it is to be opened
                 (not (openable ?f)) ; If it is not openable, this condition is satisfied
             )
         )
      :effect
         (and
             (not (robot-holding ?o)) ; The robot is no longer holding the object
             (not (furnitureAppliance-clear ?f)) ; The furniture or appliance is no longer clear
             (stacked ?o) ; The object is now placed on or in the furniture or appliance
         )
   )

   (:action stack
      :parameters (
         ?o1 ?o2 - householdObject ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-holding ?o1) ; The robot is holding object_1
             (pickupable ?o1) ; object_1 is a pickupable household object
             (pickupable ?o2) ; object_2 is a pickupable household object
             (furnitureAppliance-clear ?f) ; The furniture piece has a clear surface
             (not (stacked ?o2)) ; object_2 is not already stacked on another object
             (robot-at ?f) ; The robot is at the location of the furniture piece
         )
      :effect
         (and
             (not (robot-holding ?o1)) ; The robot is no longer holding object_1
             (stacked ?o1) ; object_1 is now stacked
             (not (furnitureAppliance-clear ?f)) ; The furniture piece is no longer clear
         )
   )

   (:action unstack
      :parameters (
         ?o1 ?o2 - householdObject ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?f) ; The robot is at some furniture or appliance
             (furnitureAppliance-clear ?f) ; The furniture or appliance is clear
             (stacked ?o1) ; The object to unstack is stacked on another object
             (pickupable ?o1) ; The object to unstack is pickupable
             (not (robot-holding ?o2)) ; The robot is not holding the object below
         )
      :effect
         (and
             (not (stacked ?o1)) ; The object is no longer stacked
             (not (pickupable ?o1)) ; The object is now being held, so it's not pickupable
             (robot-holding ?o1) ; The robot is now holding the unstacked object
         )
   )

   (:action open
      :parameters (
         ?f - furnitureAppliance ?o - householdObject
      )
      :precondition
         (and
             (robot-at ?f) ; The robot is at the location of the furniture or appliance
             (openable ?f) ; The furniture or appliance is openable
             (furnitureAppliance-clear ?f) ; The furniture or appliance is clear for opening
             (not (robot-holding ?o)) ; The robot is not holding any object
         )
      :effect
         (and
             (not (furnitureAppliance-clear ?f)) ; The furniture or appliance is no longer clear after opening
             (opened ?f) ; The furniture or appliance is now opened
         )
   )

   (:action close
      :parameters (
         ?f - furnitureAppliance ?o - householdObject
      )
      :precondition
         (and
             (robot-at ?f) ; The robot is at the location of the furniture or appliance
             (opened ?f) ; The furniture or appliance is currently open
             (furnitureAppliance-clear ?f) ; The furniture or appliance is clear for manipulation
             (not (robot-holding ?o)) ; The robot is not holding any household object
         )
      :effect
         (and
             (not (opened ?f)) ; The furniture or appliance is now closed
             (furnitureAppliance-clear ?f) ; The furniture or appliance remains clear after closing
         )
   )

   (:action toggle-on
      :parameters (
         ?a - householdObject ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?f) ; The robot is at the location of the furniture appliance
             (furnitureAppliance-clear ?f) ; The furniture appliance is clear for manipulation
             (pickupable ?a) ; The appliance is a small household object that can be picked up
             (not (robot-holding ?a)) ; The robot is not holding any other object
         )
      :effect
         (and
             (not (pickupable ?a)) ; The appliance is no longer pickupable after being toggled
             (robot-holding ?a) ; The robot is now holding the appliance
             (opened ?f) ; The appliance is toggled on, indicating it is now opened
         )
   )

   (:action toggle-off
      :parameters (
         ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-holding ?f) ; The robot must be holding the appliance
             (pickupable ?f) ; The appliance must be a pickupable household object
             (openable ?f) ; The appliance must be openable to toggle it off
         )
      :effect
         (and
             (not (robot-holding ?f)) ; The robot is no longer holding the appliance
             (not (opened ?f)) ; The appliance is toggled off (considered as not opened)
         )
   )

   (:action slice
      :parameters (
         ?o ?k - householdObject ?c - furnitureAppliance
      )
      :precondition
         (and
             (robot-holding ?k) ; The robot is holding the knife
             (pickupable ?o) ; The object to slice is a household object that can be picked up
             (stacked ?o) ; The object to slice is stackable (indicating it can be placed on the cutting board)
             (robot-at ?c) ; The robot is at the location of the cutting board
             (furnitureAppliance-clear ?c) ; The cutting board must be clear for slicing
             (not (stacked ?k)) ; The knife should not be stacked (it should be held)
         )
      :effect
         (and
             (not (robot-holding ?k)) ; The robot is no longer holding the knife
             (not (stacked ?o)) ; The object is now sliced and is no longer stackable
             (sliced ?o ?k) ; The object has been sliced with the knife
         )
   )

   (:action heat-microwave
      :parameters (
         ?m - furnitureAppliance ?r - smallReceptacle
      )
      :precondition
         (and
             (robot-at ?m) ; The robot is at the microwave
             (furnitureAppliance-clear ?m) ; The microwave is clear for operation
             (robot-holding ?r) ; The robot is holding the small receptacle
             (opened ?m) ; The microwave door is closed
             (pickupable ?r) ; The small receptacle is pickupable
         )
      :effect
         (and
             (not (robot-holding ?r)) ; The robot is no longer holding the small receptacle
             (not (pickupable ?r)) ; The small receptacle is no longer pickupable after heating
             (not (opened ?m)) ; The microwave door is now closed
             (heated ?r ?m) ; The food in the small receptacle has been heated in the microwave
         )
   )

   (:action heat-pan
      :parameters (
         ?f - householdObject ?p - smallReceptacle ?s - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?s) ; The robot is at the stove burner
             (furnitureAppliance-clear ?s) ; The stove burner is clear
             (robot-holding ?p) ; The robot is holding the pan
             (pickupable ?f) ; The food is pickupable
             (not (heated ?p ?s)) ; The pan is not already heated on the stove
         )
      :effect
         (and
             (not (robot-holding ?p)) ; The robot is no longer holding the pan
             (not (pickupable ?f)) ; The food is no longer pickupable after heating
             (heated ?p ?s) ; The pan is now heated on the stove burner
         )
   )

   (:action transfer-food
      :parameters (
         ?f1 ?f2 - furnitureAppliance ?r1 ?r2 - smallReceptacle ?o - householdObject
      )
      :precondition
         (and
             (robot-at ?f1) ; The robot must be at the location of receptacle_1
             (furnitureAppliance-clear ?f1) ; The furniture piece must be clear
             (furnitureAppliance-clear ?f2) ; The furniture piece must be clear
             (opened ?f1) ; receptacle_1 must be open (checking the furniture appliance)
             (opened ?f2) ; receptacle_2 must be open (checking the furniture appliance)
             (not (stacked ?r1)) ; receptacle_1 must not be stacked on another object
             (not (stacked ?r2)) ; receptacle_2 must not be stacked on another object
             (robot-holding ?o) ; The robot must be holding the food object
             (pickupable ?o) ; The food object must be pickupable
             (heated ?r1 ?f1) ; receptacle_1 must be heated and on a flat surface
         )
      :effect
         (and
             (not (robot-holding ?o)) ; The robot is no longer holding the food object
             (sliced ?o ?r2) ; The food object is now transferred to receptacle_2
             (not (opened ?f1)) ; receptacle_1 is no longer open (checking the furniture appliance)
             (opened ?f2) ; receptacle_2 remains open (checking the furniture appliance)
         )
   )

   (:action put-on-or-in-small
      :parameters (
         ?o - householdObject ?r - smallReceptacle ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-holding ?o) ; The robot must be holding the object
             (pickupable ?o) ; The object must be pickupable
             (furnitureAppliance-clear ?f) ; The furniture must be clear for manipulation
             (openable ?f) ; The furniture piece must be openable
             (opened ?f) ; The furniture piece must be opened
             (not (stacked ?r)) ; The receptacle must not be stacked on other objects
             (robot-at ?f) ; The robot must be at the furniture piece
         )
      :effect
         (and
             (not (robot-holding ?o)) ; The robot is no longer holding the object
             (sliced ?o ?r) ; The object is now in the receptacle
         )
   )

   (:action pick-up-small
      :parameters (
         ?o - householdObject ?r - smallReceptacle ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?f) ; The robot is at the location of the furniture piece
             (furnitureAppliance-clear ?f) ; The furniture piece is clear for manipulation
             (pickupable ?o) ; The object is pickable
             (not (stacked ?r)) ; The receptacle is not stacked on other objects
             (openable ?f) ; The furniture piece is openable
             (opened ?f) ; The furniture piece is opened to access the receptacle
             (not (robot-holding ?o)) ; The robot is not holding any object
         )
      :effect
         (and
             (not (opened ?f)) ; The furniture piece is now closed after picking up the object
             (robot-holding ?o) ; The robot is now holding the object
             (not (pickupable ?o)) ; The object is no longer available to be picked up
         )
   )

   (:action open-small
      :parameters (
         ?r - smallReceptacle ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?f) ; The robot is at the location of the furniture piece
             (furnitureAppliance-clear ?f) ; The furniture piece is clear for manipulation
             (openable ?f) ; The furniture piece is openable
             (pickupable ?r) ; The receptacle is a household object that can be picked up
             (not (stacked ?r)) ; The receptacle is not stacked on top of other objects
         )
      :effect
         (and
             (not (robot-holding ?r)) ; The robot is not holding the receptacle anymore
             (opened ?f) ; The receptacle is now opened, indicating the furniture piece is opened
         )
   )

   (:action close-small
      :parameters (
         ?r - smallReceptacle ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-at ?f) ; The robot is at the location of the furniture piece
             (furnitureAppliance-clear ?f) ; The furniture piece is clear for manipulation
             (pickupable ?r) ; The receptacle is a household object that can be picked up
             (openable ?r) ; The receptacle is openable
             (not (stacked ?r)) ; The receptacle is not stacked on top of other objects
             (opened ?f) ; The furniture appliance is currently open
         )
      :effect
         (and
             (not (opened ?f)) ; The receptacle is now closed
             (robot-holding ?r) ; The robot is holding the receptacle while closing it
         )
   )

   (:action mash-food
      :parameters (
         ?b - smallReceptacle ?f - furnitureAppliance ?o - householdObject
      )
      :precondition
         (and
             (robot-at ?f) ; The robot is at the location of the blender
             (furnitureAppliance-clear ?f) ; The blender is clear for use
             (robot-holding ?o) ; The robot is holding the food to be mashed
             (sliced ?o ?o) ; The food has been sliced beforehand, using itself as the second argument
         )
      :effect
         (and
             (not (robot-holding ?o)) ; The robot is no longer holding the food
             (heated ?b ?f) ; The food is now heated in the receptacle
         )
   )

   (:action wash
      :parameters (
         ?o - householdObject ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-holding ?o) ; The robot must be holding the object to wash
             (pickupable ?o) ; The object must be a washable household object
             (robot-at ?f) ; The robot must be at the location of the sink or basin
             (furnitureAppliance-clear ?f) ; The sink or basin must be clear for washing
         )
      :effect
         (and
             (not (robot-holding ?o)) ; The robot is no longer holding the object after washing
             (sliced ?o ?o) ; The object is now considered washed (sliced can represent a state change)
             (not (pickupable ?o)) ; The object is no longer pickupable until it is dried or put away
         )
   )

   (:action wipe
      :parameters (
         ?f - furnitureAppliance ?c - householdObject
      )
      :precondition
         (and
             (robot-at ?f) ; The robot is at the location of the furniture or appliance
             (furnitureAppliance-clear ?f) ; The surface of the furniture or appliance is clear for cleaning
             (robot-holding ?c) ; The robot is holding the cloth
             (pickupable ?c) ; The cloth is a pickupable household object
         )
      :effect
         (and
             (not (robot-holding ?c)) ; The robot is no longer holding the cloth
             (not (furnitureAppliance-clear ?f)) ; The surface of the furniture or appliance is now dirty
             (dirty ?c) ; The cloth is now dirty after cleaning
         )
   )

   (:action vacuum
      :parameters (
         ?v ?c - householdObject ?f - furnitureAppliance
      )
      :precondition
         (and
             (robot-holding ?v) ; The robot is holding the vacuum cleaner
             (pickupable ?v) ; The vacuum cleaner is a small household item that can be picked up
             (robot-at ?f) ; The robot is at the location of the furniture appliance
             (furnitureAppliance-clear ?f) ; The furniture appliance is clear for vacuuming
             (not (dirty ?c)) ; The carpet is not clean (indicating the dust bin is not full)
         )
      :effect
         (and
             (not (robot-holding ?v)) ; The robot is no longer holding the vacuum cleaner
             (dirty ?c) ; The carpet is now considered dirty after vacuuming
         )
   )

   (:action empty-vacuum
      :parameters (
         ?v - householdObject ?t - furnitureAppliance
      )
      :precondition
         (and
             (robot-holding ?v) ; The robot must be holding the vacuum cleaner
             (robot-at ?t) ; The robot must be standing next to the trash can
             (openable ?t) ; The trash can must be openable
             (opened ?t) ; The trash can must be opened
         )
      :effect
         (and
             (not (dirty ?v)) ; The dust bin of the vacuum cleaner is now empty
             (robot-holding ?v) ; The robot is still holding the vacuum cleaner
         )
   )
)