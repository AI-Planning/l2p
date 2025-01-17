(define
   (problem blocksworld-4ops_problem)
   (:domain blocksworld)

   (:objects 
      b1 - block
      b2 - block
      b3 - block
   )

   (:init
      (on b1 b3)
      (on b3 b2)
      (on-table b2)
      (clear b1)
      (arm-empty )
   )

   (:goal
      (and 
         (on b2 b3) 
         (on b3 b1) 
      )
   )

)