(define (domain blocksworld)
   (:requirements
      :strips :typing :equality :negative-preconditions :disjunctive-preconditions :universal-preconditions :conditional-effects)

   (:types 
      movable_object - object
      block - movable_object
      surface - object
      table - surface
   )

   (:predicates 
      (is_robot_arm ?r - object) ;  true if the object ?r is a robot arm
      (clear ?b - block) ;  true if no other block is on top of block ?b
      (arm_free ?r - object) ;  true if the robot arm ?r is not holding any block
      (on ?b - block ?s - surface) ;  true if block ?b is on surface ?s
      (holding ?r - object ?b - block) ;  true if the robot arm ?r is holding block ?b
      (table_space_available ?t - table) ;  true if the table ?t has space available for a block
      (on_block ?b1 - block ?b2 - block) ;  true if block ?b1 is on block ?b2
   )

   (:action pick_up_block
      :parameters (
         ?r - object
         ?b - block
         ?s - surface
      )
      :precondition
         (and
             (is_robot_arm ?r) ; The object ?r is a robot arm
             (clear ?b) ; The block ?b is clear (no block on top of it)
             (arm_free ?r) ; The robot arm ?r is free (not holding any block)
             (on ?b ?s) ; The block ?b is on the surface ?s
         )
      :effect
         (and
             (not (on ?b ?s)) ; The block ?b is no longer on the surface ?s
             (holding ?r ?b) ; The robot arm ?r is now holding the block ?b
             (not (arm_free ?r)) ; The robot arm ?r is no longer free
         )
   )

   (:action place_block_on_table
      :parameters (
         ?r - object
         ?b - block
         ?t - table
      )
      :precondition
         (and
             (is_robot_arm ?r) ; The object ?r is a robot arm
             (holding ?r ?b) ; The robot arm ?r is holding the block ?b
             (table_space_available ?t) ; The table ?t has space available for the block
         )
      :effect
         (and
             (not (holding ?r ?b)) ; The robot arm ?r is no longer holding the block ?b
             (on ?b ?t) ; The block ?b is now on the table ?t
             (arm_free ?r) ; The robot arm ?r is now free
         )
   )

   (:action place_block_on_block
      :parameters (
         ?r - object
         ?b1 - block
         ?b2 - block
      )
      :precondition
         (and
             (is_robot_arm ?r) ; The object ?r is a robot arm
             (holding ?r ?b1) ; The robot arm ?r is holding block ?b1
             (clear ?b2) ; The block ?b2 is clear, i.e., no other block is on top of it
         )
      :effect
         (and
             (not (holding ?r ?b1)) ; The robot arm ?r is no longer holding block ?b1
             (on_block ?b1 ?b2) ; Block ?b1 is now on block ?b2
             (clear ?b1) ; Block ?b1 is clear after being placed
             (not (clear ?b2)) ; Block ?b2 is no longer clear after ?b1 is placed on it
         )
   )
)