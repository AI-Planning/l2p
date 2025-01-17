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
      the-hub4 - hub
      the-hub5 - hub
      the-hub6 - hub
      the-hub7 - hub
      the-hub8 - hub
      the-hub9 - hub
      nuts1 - nut
      nuts2 - nut
      nuts3 - nut
      nuts4 - nut
      nuts5 - nut
      nuts6 - nut
      nuts7 - nut
      nuts8 - nut
      nuts9 - nut
      boot - container
      r1 - wheel
      w1 - wheel
      r2 - wheel
      w2 - wheel
      r3 - wheel
      w3 - wheel
      r4 - wheel
      w4 - wheel
      r5 - wheel
      w5 - wheel
      r6 - wheel
      w6 - wheel
      r7 - wheel
      w7 - wheel
      r8 - wheel
      w8 - wheel
      r9 - wheel
      w9 - wheel
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
      (intact r4)
      (in r4 boot)
      (not-inflated r4)
      (intact r5)
      (in r5 boot)
      (not-inflated r5)
      (intact r6)
      (in r6 boot)
      (not-inflated r6)
      (intact r7)
      (in r7 boot)
      (not-inflated r7)
      (intact r8)
      (in r8 boot)
      (not-inflated r8)
      (intact r9)
      (in r9 boot)
      (not-inflated r9)
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
      (on w4 the-hub4)
      (on-ground the-hub4)
      (tight nuts4 the-hub4)
      (fastened the-hub4)
      (on w5 the-hub5)
      (on-ground the-hub5)
      (tight nuts5 the-hub5)
      (fastened the-hub5)
      (on w6 the-hub6)
      (on-ground the-hub6)
      (tight nuts6 the-hub6)
      (fastened the-hub6)
      (on w7 the-hub7)
      (on-ground the-hub7)
      (tight nuts7 the-hub7)
      (fastened the-hub7)
      (on w8 the-hub8)
      (on-ground the-hub8)
      (tight nuts8 the-hub8)
      (fastened the-hub8)
      (on w9 the-hub9)
      (on-ground the-hub9)
      (tight nuts9 the-hub9)
      (fastened the-hub9)
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
         (on r4 the-hub4) 
         (inflated r4) 
         (tight nuts4 the-hub4) 
         (in w4 boot) 
         (on r5 the-hub5) 
         (inflated r5) 
         (tight nuts5 the-hub5) 
         (in w5 boot) 
         (on r6 the-hub6) 
         (inflated r6) 
         (tight nuts6 the-hub6) 
         (in w6 boot) 
         (on r7 the-hub7) 
         (inflated r7) 
         (tight nuts7 the-hub7) 
         (in w7 boot) 
         (on r8 the-hub8) 
         (inflated r8) 
         (tight nuts8 the-hub8) 
         (in w8 boot) 
         (on r9 the-hub9) 
         (inflated r9) 
         (tight nuts9 the-hub9) 
         (in w9 boot) 
         (in wrench boot) 
         (in jack boot) 
         (in pump boot) 
         (closed boot) 
      )
   )

)