(define
   (problem blocksworld_problem)
   (:domain blocksworld)

   (:objects 
      red1 - block
      red2 - block
      blue1 - block
      green1 - block
      green2 - block
   )

   (:init
      (on_block red1 green1)
      (on_block red2 green2)
      (on_block blue1 green1)
      (on_block green1 green2)
      (stable red1)
      (stable red2)
      (stable blue1)
      (stable green1)
      (stable green2)
      (held red1)
   )

   (:goal
      (and 
         (on_block green1 red1) 
         (on_block green2 red2) 
         (on_block green1 blue1) 
         (on_block green2 green1) 
         (stable red1) 
         (stable red2) 
         (stable blue1) 
         (stable green1) 
         (stable green2) 
      )
   )

)