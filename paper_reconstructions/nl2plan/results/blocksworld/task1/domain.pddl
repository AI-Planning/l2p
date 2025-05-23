(define (domain blocksworld)
   (:requirements
      :strips :typing :equality :negative-preconditions :disjunctive-preconditions :universal-preconditions :conditional-effects)

   (:types 
      block table - physical_object
   )

   (:predicates 
      (clear ?b - block)
      (holding ?b - block)
      (on_table ?b - block)
      (on ?b1 - block ?b2 - block)
   )
)