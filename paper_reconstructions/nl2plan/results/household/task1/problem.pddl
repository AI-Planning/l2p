(define
   (problem household_problem)
   (:domain household)

   (:objects 
      robot1 - robot
      cabinet_1 - furniture
      cabinet_2 - furniture
      drawer_1 - furniture
      drawer_2 - furniture
      countertop_1 - flat_surface
      dining_table_1 - flat_surface
      side_table_1 - flat_surface
      fridge_1 - appliance
      blender_1 - appliance
      lamp_1 - appliance
      humidifier_1 - appliance
      cup_1 - small_item
      plate_1 - small_item
      cutting_board_1 - small_item
      apple_1 - small_item
      book_1 - small_item
      book_2 - small_item
      mug_1 - small_item
   )

   (:init
      (at robot1 cabinet_1)
      (holding robot1 mug_1)
      (drawer_open drawer_1)
      (drawer_open drawer_2)
      (appliance_open fridge_1)
      (drawer_closed cabinet_1)
      (drawer_open cabinet_2)
      (on cup_1 cabinet_2)
      (on plate_1 drawer_2)
      (on cutting_board_1 countertop_1)
      (on blender_1 dining_table_1)
      (appliance_off blender_1)
      (on lamp_1 side_table_1)
      (appliance_off lamp_1)
      (on humidifier_1 side_table_1)
      (appliance_on humidifier_1)
      (on apple_1 cabinet_2)
      (on book_1 dining_table_1)
      (on book_2 dining_table_1)
   )

   (:goal
      (and 
         (appliance_on lamp_1) 
         (appliance_off humidifier_1) 
      )
   )

)