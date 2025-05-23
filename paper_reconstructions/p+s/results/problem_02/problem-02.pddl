(define
   (problem problem-02)
   (:domain simple-blocks)

   (:objects 
      newspaper
      accordion
      saucepan
      peacoat
   )

   (:init
      (ontable accordion)
      (on newspaper accordion)
      (on saucepan newspaper)
      (clear saucepan)
      (ontable peacoat)
      (clear peacoat)
   )

   (:goal
      (and 
         (clear newspaper)
         (ontable accordion)
         (clear saucepan)
         (ontable saucepan)
         (on newspaper accordion)
         (ontable peacoat)
         (clear peacoat)
      )
   )
)