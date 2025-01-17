(define
   (problem tyreworld_problem)
   (:domain tyreworld)

   (:objects 
      robot1 - object
      vehicle1 - vehicle
      hub1 - hub
      flat_tyre1 - flat_tyre
      spare_tyre1 - spare_tyre
      wrench1 - wrench
      jack1 - jack
      pump1 - pump
   )

   (:init
      (is_robot robot1)
      (at_vehicle robot1 vehicle1)
      (has_tool robot1 wrench1)
      (has_tool robot1 jack1)
      (has_tool robot1 pump1)
      (accessible_hub hub1)
      (nut_undone hub1)
      (jacked_up vehicle1)
      (stable_position vehicle1)
      (on_hub flat_tyre1 hub1)
      (available spare_tyre1)
   )

   (:goal
      (and 
         (available_for_disposal flat_tyre1) 
         (on_hub spare_tyre1 hub1) 
         (nut_done_up hub1) 
         (inflated spare_tyre1) 
         (stable_position vehicle1) 
      )
   )

)