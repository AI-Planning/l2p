(define
   (problem tyreworld_problem)
   (:domain tyreworld)

   (:objects 
      wrench - tool
      jack - tool
      pump - tool
      the-hub1 - hub
      the-hub2 - hub
      the-hub3 - hub
      nuts1 - nut
      nuts2 - nut
      nuts3 - nut
      r1 - wheel
      r2 - wheel
      r3 - wheel
      w1 - wheel
      w2 - wheel
      w3 - wheel
   )

   (:init
      (in jack boot)
      (in pump boot)
      (in wrench boot)
      (unlocked boot)
      (closed boot)
      (intact r1)
      (in r1 boot)
      (not-inflated r1)
      (intact r2)
      (in r2 boot)
      (not-inflated r2)
      (intact r3)
      (in r3 boot)
      (not-inflated r3)
      (on w1 the-hub1)
      (on-ground the-hub1)
      (tight nuts1 the-hub1)
      (fastened the-hub1)
      (on w2 the-hub2)
      (on-ground the-hub2)
      (tight nuts2 the-hub2)
      (fastened the-hub2)
      (on w3 the-hub3)
      (on-ground the-hub3)
      (tight nuts3 the-hub3)
      (fastened the-hub3)
   )

   (:goal
      (and 
         (on r1 the-hub1) 
         (inflated r1) 
         (tight nuts1 the-hub1) 
         (in w1 boot) 
         (on r2 the-hub2) 
         (inflated r2) 
         (tight nuts2 the-hub2) 
         (in w2 boot) 
         (on r3 the-hub3) 
         (inflated r3) 
         (tight nuts3 the-hub3) 
         (in w3 boot) 
         (in wrench boot) 
         (in jack boot) 
         (in pump boot) 
         (closed boot) 
      )
   )

)