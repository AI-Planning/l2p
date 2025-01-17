(define
   (problem blocksworld-4ops_problem)
   (:domain blocksworld)

   (:objects 
      b1 - block
      b2 - block
      b3 - block
      b4 - block
   )

   (:init
      (on b1 b4)
      (on b4 b2)
      (on b3 b1)
      (on-table b2)
      (clear b3)
      (arm-empty )
   )

   (:goal
      (and 
         (on b1 b2) 
         (on b2 b3) 
         (on b3 b4) 
      )
   )

)