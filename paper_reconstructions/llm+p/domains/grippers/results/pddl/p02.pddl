(define
   (problem grippers_problem)
   (:domain grippers)

   (:objects 
      robot1 - robot
      robot2 - robot
      rgripper1 - gripper
      lgripper1 - gripper
      rgripper2 - gripper
      lgripper2 - gripper
      room1 - room
      room2 - room
      room3 - room
      ball1 - object
      ball2 - object
      ball3 - object
      ball4 - object
   )

   (:init
      (at-robby robot1 room2)
      (free robot1 rgripper1)
      (free robot1 lgripper1)
      (at-robby robot2 room3)
      (free robot2 rgripper2)
      (free robot2 lgripper2)
      (at ball1 room3)
      (at ball2 room1)
      (at ball3 room1)
      (at ball4 room3)
   )

   (:goal
      (and 
         (at ball1 room2) 
         (at ball2 room2) 
         (at ball3 room3) 
         (at ball4 room3) 
      )
   )

)