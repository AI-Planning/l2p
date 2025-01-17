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
      (noteq newspaper accordion)
      (noteq newspaper saucepan)
      (noteq newspaper peacoat)
      (noteq accordion saucepan)
      (noteq accordion peacoat)
      (noteq saucepan peacoat)
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