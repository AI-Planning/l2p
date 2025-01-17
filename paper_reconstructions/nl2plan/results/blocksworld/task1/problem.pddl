(define
   (problem blocksworld_problem)
   (:domain blocksworld)

   (:objects 
      blue - block
      red - block
      yellow - block
      green - block
      table1 - table
      robot_arm1 - object
   )

   (:init
      (on_block blue red)
      (on_block red yellow)
      (on_table yellow table1)
      (on_table green table1)
      (clear blue)
      (clear green)
      (arm_free robot_arm1)
      (is_robot_arm robot_arm1)
      (is_table table1)
   )

   (:goal
      (and 
         (on_block red green) 
         (on_block blue red) 
         (on_table yellow table1) 
         (on_table green table1) 
      )
   )

)