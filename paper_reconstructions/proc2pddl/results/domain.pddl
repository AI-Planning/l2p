(define (domain survive_deserted_island)
   (:requirements
     :strips :typing :equality :negative-preconditions :disjunctive-preconditions :universal-preconditions :conditional-effects)

   (:types 
      item - object
      water - item
      wood - item
      fire - item
      rock - item
      leaves - item
      tinder - item
      raft - item
      vines - item
      spear - item
      fish - item
      location - object
      beach - location
      jungle - location
      ocean - location
      treetop - location
      human - object
      player - human
      survivor - human
      direction - object
   )

   (:predicates 
      (treated ?water - water)
      (groove ?wood - wood)
      (at ?obj - object ?loc - location)
      (inventory ?player ?item)
      (connected ?loc1 - location ?dir - direction ?loc2 - location)
      (has_water_source ?loc - location)
      (has_wood ?loc - location)
      (can_light_fire ?loc - location)
      (has_fire ?loc - location)
      (has_shelter ?loc - location)
      (drank ?water - water)
      (has_friend ?survivor - survivor)
      (has_escaped ?player - player)
      (at_ocean ?loc - location)
      (is_safe ?loc -location)
      (has_fish ?loc - location)
      (cooked ?item - item)
   )

   (:action go
      :parameters (
         ?player - human
         ?from - location
         ?to - location
      )
      :precondition
         (and
             (at ?player ?from) ; The player is at the current location
             (connected ?from ?to) ; The current location is connected to the new location
         )
      :effect
         (and
             (not (at ?player ?from)) ; The player is no longer at the current location
             (at ?player ?to) ; The player is now at the new adjacent location
         )
   )

   (:action get
      :parameters (
         ?player - human
         ?item - object
         ?loc - location
      )
      :precondition
         (and
             (at ?player ?loc) ; The player is at the current location
             (at ?item ?loc) ; The item is present in the current location
         )
      :effect
         (and
             (not (at ?item ?loc)) ; The item is no longer in the environment
             (inventory ?player ?item) ; The item is now in the player's inventory
         )
   )

   (:action get_water
      :parameters (
         ?player - human
         ?water - water
         ?loc - location
      )
      :precondition
         (and
             (at ?player ?loc) ; The player is at the current location
             (has_water_source ?loc) ; The location has a water source
         )
      :effect
         (and
             (inventory ?player ?water) ; The water is now in the player's inventory
         )
   )

   (:action chop_wood
      :parameters (
         ?player - human
         ?wood - wood
         ?loc - location
      )
      :precondition
         (and
             (at ?player ?loc) ; The player is at the current location
             (has_wood ?loc) ; The location has wood
         )
      :effect
         (and
             (inventory ?player ?wood) ; The wood is now in the player's inventory
         )
   )

   (:action carve_groove
      :parameters (
         ?player - human
         ?wood - wood
      )
      :precondition
         (and
             (inventory ?player ?wood) ; The player has wood in their inventory
         )
      :effect
         (and
             (groove ?wood) ; The wood now has a groove
         )
   )

   (:action light_fire
      :parameters (
         ?player - human
         ?loc - location
      )
      :precondition
         (and
             (at ?player ?loc) ; The player is at the current location
             (can_light_fire ?loc) ; The location is safe for lighting a fire
         )
      :effect
         (and
             (has_fire ?loc) ; A fire is now lit at the location
         )
   )

   (:action build_shelter
      :parameters (
         ?player - human
         ?wood - wood
         ?loc - location
      )
      :precondition
         (and
             (inventory ?player ?wood) ; The player has wood in their inventory
             (is_safe ?loc) ; The location is safe to make shelter
         )
      :effect
         (and
             (has_shelter ?loc) ; A shelter is now built at the location
         )
   )

   (:action clean_water
      :parameters (
         ?player - human
         ?water - water
         ?loc - location
      )
      :precondition
         (and
             (inventory ?player ?water) ; The player has untreated water in their inventory
             (has_fire ?loc) ; There is a fire lit at the location
         )
      :effect
         (and
             (treated ?water) ; The water is now treated and safe for drinking
         )
   )

   (:action drink_water
      :parameters (
         ?player - human
         ?water - water
      )
      :precondition
         (and
             (inventory ?player ?water) ; The player has water in their inventory
         )
      :effect
         (and
             (not (inventory ?player ?water)) ; The water is removed from the inventory
             (drank ?player) ; The player has consumed the water
         )
   )

   (:action find_other_survivors
      :parameters (
         ?player - human
      )
      :precondition
         (and
             ; The player must navigate to various locations (no specific predicates needed)
         )
      :effect
         (and
             (has_friend ?player) ; The player may have found a survivor
         )
   )

   (:action build_raft
      :parameters (
         ?player - human
         ?wood - wood
         ?vines - vines
      )
      :precondition
         (and
             (inventory ?player ?wood) ; The player has wood in their inventory
             (inventory ?player ?vines) ; The player has vines in their inventory
         )
      :effect
         (and
             (has_escaped ?player) ; The player has built a raft and is ready to escape
         )
   )

   (:action make_weapon
      :parameters (
         ?player - human
         ?wood - wood
         ?rock - rock
         ?spear - spear
      )
      :precondition
         (and
             (inventory ?player ?wood) ; The player has wood in their inventory
             (inventory ?player ?rock) ; The player has a sharp rock in their inventory
         )
      :effect
         (and
             (inventory ?player ?spear) ; The spear is now in the player's inventory
         )
   )

   (:action hunt_fish
      :parameters (
         ?player - human
         ?fish - fish
         ?loc - location
      )
      :precondition
         (and
             (at ?player ?loc) ; The player is at the current location
             (has_fish ?loc) ; The location has fish
         )
      :effect
         (and
             (inventory ?player ?fish) ; The fish is now in the player's inventory
         )
   )

   (:action cook_fish
      :parameters (
         ?player - human
         ?fish - fish
         ?loc - location
      )
      :precondition
         (and
             (inventory ?player ?fish) ; The player has fish in their inventory
             (has_fire ?loc) ; There is a fire lit at the location
         )
      :effect
         (and
             (cooked ?fish) ; The fish is now cooked and safe for consumption
         )
   )
)