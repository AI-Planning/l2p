(define (domain tyreworld)
   (:requirements
      :strips :typing :equality :negative-preconditions :disjunctive-preconditions :universal-preconditions :conditional-effects)

   (:types 
      vehicle - object
      hub - vehicle
      tyre - object
      flat_tyre - tyre
      spare_tyre - tyre
      tool - object
      wrench - tool
      jack - tool
      pump - tool
   )

   (:predicates 
      (is_robot ?r - object) ;  true if the object ?r is a robot
      (in_boot ?t - tool ?v - vehicle) ;  true if the tool ?t is in the boot of the vehicle ?v
      (at_vehicle ?r - object ?v - vehicle) ;  true if the robot ?r is at the vehicle ?v
      (has_tool ?r - object ?t - tool) ;  true if the robot ?r has the tool ?t
      (accessible_hub ?h - hub) ;  true if the hub ?h is accessible for the robot to work on
      (nut_undone ?h - hub) ;  true if the nut on the hub ?h is undone
      (jacked_up ?v - vehicle) ;  true if the vehicle ?v is jacked up
      (stable_position ?v - vehicle) ;  true if the vehicle ?v is in a stable position
      (on_hub ?t - tyre ?h - hub) ;  true if the tyre ?t is on the hub ?h
      (available ?t - tyre) ;  true if the tyre ?t is available for handling
      (available_for_disposal ?t - tyre) ;  true if the tyre ?t is available for disposal
      (nut_done_up ?h - hub) ;  true if the nut on the hub ?h is done up
      (inflated ?t - spare_tyre) ;  true if the spare tyre ?t is inflated
   )

   (:action fetch_tool
      :parameters (
         ?r - object
         ?t - tool
         ?v - vehicle
      )
      :precondition
         (and
             (is_robot ?r) ; The object ?r is a robot
             (at_vehicle ?r ?v) ; The robot is at the vehicle
             (in_boot ?t ?v) ; The tool is in the boot of the vehicle
         )
      :effect
         (and
             (not (in_boot ?t ?v)) ; The tool is no longer in the boot
             (has_tool ?r ?t) ; The robot now has the tool
         )
   )

   (:action undo_nut
      :parameters (
         ?r - object
         ?w - wrench
         ?h - hub
      )
      :precondition
         (and
             (is_robot ?r) ; The object is a robot
             (has_tool ?r ?w) ; The robot has the wrench
             (accessible_hub ?h) ; The hub is accessible
         )
      :effect
         (and
             (nut_undone ?h) ; The nut on the hub is now undone
         )
   )

   (:action jack_up_vehicle
      :parameters (
         ?r - object
         ?j - jack
         ?v - vehicle
      )
      :precondition
         (and
             (has_tool ?r ?j) ; The robot has the jack
             (at_vehicle ?r ?v) ; The robot is at the vehicle
             (stable_position ?v) ; The vehicle is in a stable position
         )
      :effect
         (and
             (not (stable_position ?v)) ; The vehicle is no longer in a stable position
             (jacked_up ?v) ; The vehicle is now jacked up
         )
   )

   (:action remove_flat_tyre
      :parameters (
         ?r - object
         ?h - hub
         ?ft - flat_tyre
         ?v - vehicle
      )
      :precondition
         (and
             (nut_undone ?h) ; The nut on the hub is undone
             (jacked_up ?v) ; The vehicle is jacked up
             (accessible_hub ?h) ; The hub is accessible
             (at_vehicle ?r ?v) ; The robot is at the vehicle
         )
      :effect
         (and
             (not (on_hub ?ft ?h)) ; The flat tyre is no longer on the hub
             (available_for_disposal ?ft) ; The flat tyre is available for disposal
         )
   )

   (:action install_spare_tyre
      :parameters (
         ?r - object
         ?s - spare_tyre
         ?h - hub
         ?f - flat_tyre
      )
      :precondition
         (and
             (not (on_hub ?f ?h)) ; The flat tyre is not on the hub
             (available ?s) ; The spare tyre is available for installation
         )
      :effect
         (and
             (on_hub ?s ?h) ; The spare tyre is now on the hub
             (not (available ?s)) ; The spare tyre is no longer available for handling
         )
   )

   (:action do_up_nut
      :parameters (
         ?r - object
         ?w - wrench
         ?h - hub
         ?s - spare_tyre
         ?v - vehicle
      )
      :precondition
         (and
             (is_robot ?r) ; The object is a robot
             (has_tool ?r ?w) ; The robot has the wrench
             (on_hub ?s ?h) ; The spare tyre is installed on the hub
             (accessible_hub ?h) ; The hub is accessible
             (stable_position ?v) ; The vehicle is in a stable position
         )
      :effect
         (and
             (not (nut_undone ?h)) ; The nut is no longer undone
             (nut_done_up ?h) ; The nut is done up on the hub
         )
   )

   (:action lower_vehicle
      :parameters (
         ?r - object
         ?j - jack
         ?v - vehicle
      )
      :precondition
         (and
             (jacked_up ?v) ; The vehicle is currently jacked up
             (has_tool ?r ?j) ; The robot has the jack in its possession
         )
      :effect
         (and
             (not (jacked_up ?v)) ; The vehicle is no longer jacked up
             (stable_position ?v) ; The vehicle is now in a stable position
         )
   )

   (:action inflate_tyre
      :parameters (
         ?r - object
         ?p - pump
         ?t - spare_tyre
         ?h - hub
      )
      :precondition
         (and
             (has_tool ?r ?p) ; The robot has the pump
             (on_hub ?t ?h) ; The spare tyre is installed on the hub
         )
      :effect
         (and
             (inflated ?t) ; The spare tyre is now inflated
         )
   )
)