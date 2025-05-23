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