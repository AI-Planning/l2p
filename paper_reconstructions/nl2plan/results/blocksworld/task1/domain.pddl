(define (domain blocksworld)
   (:requirements
      :strips :typing :equality :negative-preconditions :disjunctive-preconditions :universal-preconditions :conditional-effects)

   (:types 
      block - object
      table - object
      robot_arm - object
   )

   (:predicates 
      (clear ?b - block) ;  true if no block is on top of block ?b
      (on_table ?b - block) ;  true if block ?b is on the table
      (on_block ?b1 - block ?b2 - block) ;  true if block ?b1 is on block ?b2
      (arm_free ?r - robot_arm) ;  true if the robot arm ?r is not holding any block
      (holding ?r - robot_arm ?b - block) ;  true if the robot arm ?r is holding block ?b
      (is_robot_arm ?o - object) ;  
      (is_table ?o - object) ;  
   )

   (:action pick_up_block
      :parameters (
         ?r - robot_arm
         ?b - block
         ?b2 - block
      )
      :precondition
         (and
             (arm_free ?r) ; The robot arm is not holding any block
             (clear ?b) ; No block is on top of block ?b
             (or
                 (on_table ?b) ; Block ?b is on the table
                 (exists (?b2 - block) (on_block ?b ?b2)) ; Block ?b is on another block ?b2
             )
         )
      :effect
         (and
             (not (arm_free ?r)) ; The robot arm is now holding a block
             (holding ?r ?b) ; The robot arm is holding block ?b
             (not (on_table ?b)) ; Block ?b is no longer on the table
             (forall (?b2 - block) (when (on_block ?b ?b2) (not (on_block ?b ?b2)))) ; If block ?b was on block ?b2, it is no longer on it
         )
   )

   (:action place_block_on_table
      :parameters (
         ?r - object
         ?b - block
         ?t - object
      )
      :precondition
         (and
             (holding ?r ?b) ; The robot arm is holding the block
             (clear ?t) ; The table has space for the block
             (is_robot_arm ?r) ; The object ?r is a robot arm
             (is_table ?t) ; The object ?t is a table
         )
      :effect
         (and
             (not (holding ?r ?b)) ; The robot arm is no longer holding the block
             (on_table ?b) ; The block is now on the table
             (arm_free ?r) ; The robot arm is now free
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
             (is_robot_arm ?r) ; ?r is a robot arm
             (holding ?r ?b1) ; The robot arm is holding block ?b1
             (clear ?b2) ; Block ?b2 is clear, meaning no block is on top of it
         )
      :effect
         (and
             (not (holding ?r ?b1)) ; The robot arm is no longer holding block ?b1
             (on_block ?b1 ?b2) ; Block ?b1 is now on block ?b2
             (not (clear ?b2)) ; Block ?b2 is no longer clear
             (arm_free ?r) ; The robot arm is now free
         )
   )
)