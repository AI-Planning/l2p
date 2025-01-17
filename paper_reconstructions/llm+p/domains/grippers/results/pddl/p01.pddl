(define
   (problem grippers_problem)
   (:domain grippers)

   (:objects 
      robot1 - robot
      robot2 - robot
      lgripper1 - gripper
      rgripper1 - gripper
      lgripper2 - gripper
      rgripper2 - gripper
      room1 - room
      room2 - room
      ball1 - object
      ball2 - object
   )

   (:init
      (at-robby robot1 room1)
      (free robot1 rgripper1)
      (free robot1 lgripper1)
      (at-robby robot2 room1)
      (free robot2 rgripper2)
      (free robot2 lgripper2)
      (at ball1 room1)
      (at ball2 room1)
   )

   (:goal
      (and 
         (at ball1 room1) 
         (at ball2 room1) 
      )
   )

)