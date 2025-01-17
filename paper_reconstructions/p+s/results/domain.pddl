(define (domain simple-blocks)
  (:requirements :strips)
  (:predicates (on ?x ?y)
               (clear ?x)
               (ontable ?x)
               )
  (:action stack
             :parameters (?obj ?oldunder ?newunder)
             :precondition 
             (and (clear ?newunder) (clear ?obj) (on ?obj ?oldunder))
             :effect
             (and  (not (clear ?newunder))
                   (on ?obj ?newunder)
                   (not (on ?obj ?oldunder))
                   (clear ?oldunder)))
  (:action unstack
             :parameters (?sob ?sunderob)
             :precondition (and (on ?sob ?sunderob) (clear ?sob))
             :effect
             (and  (clear ?sunderob)
                   (ontable ?sob)
                   (not (on ?sob ?sunderob))))
  (:action stackfromtable 
            :parameters (?obj ?newunder)
            :precondition (and (clear ?obj) (clear ?newunder) (ontable ?obj))
            :effect
            (and (not (clear ?newunder))
                 (on ?obj ?newunder)
                 (not (ontable ?obj)))))