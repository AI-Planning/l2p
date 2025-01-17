(define (domain household)
   (:requirements
      :strips :typing :equality :negative-preconditions :disjunctive-preconditions :universal-preconditions :conditional-effects)

   (:types 
      robot - object
      furniture - object
      flat_surface - furniture
      appliance - object
      small_item - object
   )

   (:predicates 
      (at ?r - robot ?loc - object) ;  true if the robot ?r is at the location ?loc
      (is_furniture ?loc - object) ;  true if the location ?loc is a piece of furniture
      (is_appliance ?loc - object) ;  true if the location ?loc is an appliance
      (on ?item - small_item ?surface - flat_surface) ;  true if the item is on the flat surface
      (gripper_empty ?r - robot) ;  true if the robot's gripper is empty
      (holding ?r - robot ?item - small_item) ;  true if the robot is holding the item
      (appliance_open ?a - appliance) ;  true if the appliance ?a is open
      (appliance_closed ?a - appliance) ;  true if the appliance ?a is closed
      (drawer_open ?f - furniture) ;  true if the drawer on the furniture ?f is open
      (drawer_closed ?f - furniture) ;  true if the drawer on the furniture ?f is closed
      (appliance_on ?a - appliance) ;  true if the appliance ?a is on
      (appliance_off ?a - appliance) ;  true if the appliance ?a is off
   )

   (:action navigate_to
      :parameters (
         
      )
      :precondition
         (and
             (at ?r ?current_loc) ; The robot is at its current location
             (or (is_furniture ?loc) (is_appliance ?loc)) ; The target location is either furniture or an appliance
         )
      :effect
         (and
             (not (at ?r ?current_loc)) ; The robot is no longer at the current location
             (at ?r ?loc) ; The robot is now at the target location
         )
   )

   (:action pick_up
      :parameters (
         ?r - robot
         ?item - small_item
         ?surface - flat_surface
      )
      :precondition
         (and
             (at ?r ?surface) ; The robot is at the location of the flat surface
             (on ?item ?surface) ; The item is on the flat surface
             (gripper_empty ?r) ; The robot's gripper is empty
         )
      :effect
         (and
             (not (on ?item ?surface)) ; The item is no longer on the flat surface
             (not (gripper_empty ?r)) ; The robot's gripper is no longer empty
             (holding ?r ?item) ; The robot is now holding the item
         )
   )

   (:action put_down
      :parameters (
         ?r - robot
         ?item - small_item
         ?surface - flat_surface
      )
      :precondition
         (and
             (at ?r ?surface) ; The robot is at the location of the flat surface
             (holding ?r ?item) ; The robot is holding the item
             (is_furniture ?surface) ; Ensure the surface is a flat surface
         )
      :effect
         (and
             (on ?item ?surface) ; The item is now on the flat surface
             (not (holding ?r ?item)) ; The robot is no longer holding the item
             (gripper_empty ?r) ; The robot's gripper is now empty
         )
   )

   (:action open_appliance
      :parameters (
         ?r - robot
         ?a - appliance
      )
      :precondition
         (and
             (at ?r ?a) ; The robot is at the location of the appliance
             (is_appliance ?a) ; The location is an appliance
             (gripper_empty ?r) ; The robot's gripper is empty
             (appliance_closed ?a) ; The appliance is currently closed
         )
      :effect
         (and
             (not (appliance_closed ?a)) ; The appliance is no longer closed
             (appliance_open ?a) ; The appliance is now open
         )
   )

   (:action close_appliance
      :parameters (
         ?r - robot
         ?a - appliance
      )
      :precondition
         (and
             (at ?r ?a) ; The robot is at the location of the appliance
             (gripper_empty ?r) ; The robot's gripper is empty
             (appliance_open ?a) ; The appliance is open
         )
      :effect
         (and
             (not (appliance_open ?a)) ; The appliance is no longer open
             (appliance_closed ?a) ; The appliance is now closed
         )
   )

   (:action open_drawer
      :parameters (
         ?r - robot
         ?f - furniture
      )
      :precondition
         (and
             (at ?r ?f) ; The robot is at the location of the furniture
             (gripper_empty ?r) ; The robot's gripper is empty
             (not (drawer_open ?f)) ; The drawer is not already open
         )
      :effect
         (and
             (drawer_open ?f) ; The drawer on the furniture is now open
         )
   )

   (:action close_drawer
      :parameters (
         ?r - robot
         ?f - furniture
      )
      :precondition
         (and
             (at ?r ?f) ; The robot is at the location of the furniture
             (gripper_empty ?r) ; The robot's gripper is empty
             (drawer_open ?f) ; The drawer on the furniture is open
         )
      :effect
         (and
             (not (drawer_open ?f)) ; The drawer on the furniture is no longer open
             (drawer_closed ?f) ; The drawer on the furniture is now closed
         )
   )

   (:action turn_on_appliance
      :parameters (
         ?r - robot
         ?a - appliance
      )
      :precondition
         (and
             (at ?r ?a) ; The robot is at the location of the appliance
             (appliance_off ?a) ; The appliance is currently off
         )
      :effect
         (and
             (not (appliance_off ?a)) ; The appliance is no longer off
             (appliance_on ?a) ; The appliance is now on
         )
   )

   (:action turn_off_appliance
      :parameters (
         ?r - robot
         ?a - appliance
      )
      :precondition
         (and
             (at ?r ?a) ; The robot is at the location of the appliance
             (appliance_on ?a) ; The appliance is currently on
         )
      :effect
         (and
             (not (appliance_on ?a)) ; The appliance is no longer on
             (appliance_off ?a) ; The appliance is now off
         )
   )

   (:action check_state
      :parameters (
         ?r - robot
         ?loc - object
      )
      :precondition
         (and
             (at ?r ?loc) ; The robot is at the location of the object
             (or (is_furniture ?loc) (is_appliance ?loc)) ; The object is either furniture or an appliance
         )
      :effect
         (and
             ; No effects as checking state is a non-modifying action
         )
   )
)