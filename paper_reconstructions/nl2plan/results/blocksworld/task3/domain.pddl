(define (domain blocksworld)
   (:requirements
      :strips :typing :equality :negative-preconditions :disjunctive-preconditions :universal-preconditions :conditional-effects)

   (:types 
      movable_object - object
      block - movable_object
      table - object
   )

   (:predicates 
      (on_table ?b - block) ;  true if the block ?b is on the table
      (on_block ?b1 - block ?b2 - block) ;  true if block ?b1 is on top of block ?b2
      (held ?b - block) ;  true if the block ?b is currently held by the robot arm
      (stable ?b - block) ;  true if the block ?b is stable and can support another block on top of it.
   )

   (:action pick_block
      :parameters (
         ?b - block
      )
      :precondition
         (and
             (or 
                 (on_table ?b) ; The block is on the table
                 (exists (?b2 - block) (on_block ?b ?b2)) ; The block is on top of another block
             )
             (not (held ?b)) ; The block is not currently held
         )
      :effect
         (and
             (held ?b) ; The block is now held by the robot arm
             (not (on_table ?b)) ; The block is no longer on the table
             (not (on_block ?b ?b2)) ; The block is no longer on top of another block
         )
   )

   (:action place_on_table
      :parameters (
         ?b - block
      )
      :precondition
         (and
             (held ?b) ; The block is currently held by the robot arm
         )
      :effect
         (and
             (not (held ?b)) ; The block is no longer held by the robot arm
             (on_table ?b) ; The block is now on the table
         )
   )

   (:action place_on_block
      :parameters (
         ?b1 - block
         ?b2 - block
      )
      :precondition
         (and
             (held ?b1) ; The block ?b1 is currently held by the robot arm
             (not (held ?b2)) ; The block ?b2 is not currently held
             (stable ?b1) ; The block ?b1 must also be stable
             (stable ?b2) ; The block ?b2 is stable
         )
      :effect
         (and
             (not (held ?b1)) ; The block ?b1 is no longer held by the robot arm
             (on_block ?b1 ?b2) ; The block ?b1 is now on top of block ?b2
         )
   )

   (:action release_block
      :parameters (
         ?b - block
      )
      :precondition
         (and
             (held ?b) ; The block is currently held by the robot arm
             (stable ?b) ; The block is stable and can be safely placed
         )
      :effect
         (and
             (not (held ?b)) ; The block is no longer held by the robot arm
             (on_table ?b) ; The block is placed on the table after being released
         )
   )
)