(define
   (problem blocksworld-4ops_problem)
   (:domain blocksworld)

   (:objects 
      b1 - block
      b2 - block
      b3 - block
   )

   (:init
      (on b2 b3)
      (on b3 b1)
      (on-table b1)
      (clear b2)
      (arm-empty )
   )

   (:goal
      (and 
         (on b2 b3)
         (on b3 b1)
      )
   )
)