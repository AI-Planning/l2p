(define
   (problem problem-01)
   (:domain simple-blocks)

   (:objects 
      sketchbook
      sweatshirt
      keyboard
      novel
   )

   (:init
      (ontable sketchbook)
      (on sweatshirt sketchbook)
      (on keyboard sweatshirt)
      (on novel keyboard)
      (clear novel)
   )

   (:goal
      (and 
         (on keyboard sketchbook)
         (ontable sweatshirt)
         (clear sweatshirt)
      )
   )
)