(define
   (problem blocksworld_problem)
   (:domain blocksworld)

   (:objects 
      blue_block - block
      red_block - block
      yellow_block - block
      green_block - block
      fixed_table - table
   )

   (:init
      (on blue_block red_block)
      (on red_block yellow_block)
      (on_table yellow_block)
      (on_table green_block)
      (clear green_block)
      (not (clear yellow_block))
      (not (clear blue_block))
      (not (clear red_block))
   )

   (:goal
      (and 
         (on red_block green_block)
         (clear green_block)
      )
   )
)