(define
   (problem problem-03)
   (:domain simple-blocks)

   (:objects 
      mouse-pad
      hacksaw
      saucepan
      raincoat
   )

   (:init
      (ontable raincoat)
      (clear raincoat)
      (ontable hacksaw)
      (clear hacksaw)
      (ontable saucepan)
      (on mouse-pad saucepan)
      (clear mouse-pad)
      (noteq mouse-pad hacksaw)
      (noteq mouse-pad saucepan)
      (noteq mouse-pad raincoat)
      (noteq hacksaw saucepan)
      (noteq hacksaw raincoat)
      (noteq saucepan raincoat)
   )

   (:goal
      (and 
         (ontable mouse-pad) 
         (ontable saucepan) 
         (clear hacksaw) 
         (on raincoat mouse-pad) 
         (on hacksaw raincoat) 
         (clear saucepan) 
      )
   )

)