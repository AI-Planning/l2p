(define
   (problem blocksworld_problem)
   (:domain blocksworld)

   (:objects 
      blue - block
      red - block
      green - block
      yellow - block
      table1 - table
      robot_arm1 - object
   )

   (:init
      (is_robot_arm robot_arm1)
      (clear blue)
      (clear green)
      (arm_free robot_arm1)
      (on blue red)
      (on green yellow)
      (on yellow table1)
      (table_space_available table1)
   )

   (:goal
      (and 
         (on green red) 
         (on blue green) 
         (on yellow blue) 
         (clear yellow) 
      )
   )

)