(define
   (problem grippers_problem)
   (:domain grippers)

   (:objects 
      robot1 - robot
      robot2 - robot
      robot3 - robot
      lgripper1 - gripper
      rgripper1 - gripper
      lgripper2 - gripper
      rgripper2 - gripper
      lgripper3 - gripper
      rgripper3 - gripper
      room1 - room
      room2 - room
      room3 - room
      room4 - room
      ball1 - object
      ball2 - object
      ball3 - object
      ball4 - object
   )

   (:init
      (at-robby robot1 room4)
      (at-robby robot2 room4)
      (at-robby robot3 room1)
      (free robot1 rgripper1)
      (free robot1 lgripper1)
      (free robot2 rgripper2)
      (free robot2 lgripper2)
      (free robot3 rgripper3)
      (free robot3 lgripper3)
      (at ball1 room1)
      (at ball2 room1)
      (at ball3 room1)
      (at ball4 room2)
   )

   (:goal
      (and 
         (at ball1 room1) 
         (at ball2 room1) 
         (at ball3 room3) 
         (at ball4 room2) 
      )
   )

)