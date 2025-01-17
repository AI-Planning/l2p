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
      (noteq sketchbook sweatshirt)
      (noteq sketchbook keyboard)
      (noteq sketchbook novel)
      (noteq sweatshirt keyboard)
      (noteq sweatshirt novel)
      (noteq keyboard novel)
   )

   (:goal
      (and 
         (on keyboard sketchbook) 
         (ontable sweatshirt) 
      )
   )

)