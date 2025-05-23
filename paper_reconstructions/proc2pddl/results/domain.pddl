(define (domain survive_deserted_island)
    (:requirements :conditional-effects :disjunctive-preconditions :equality :negative-preconditions :strips :typing :universal-preconditions)
    (:types
        direction human item location - object
        player survivor - human
        fire fish leaves raft rock spear tinder vines water wood - item
        beach jungle ocean treetop - location
    )
    (:predicates (at ?player - player ?loc - location)  (at_ocean ?loc - location)  (can_light_fire ?loc - location)  (connected ?loc1 - location ?dir - direction ?loc2 - location)  (cooked ?item - item)  (drank ?water - water)  (groove ?wood - wood)  (has_escaped ?player - player)  (has_fire ?loc - location)  (has_fish ?loc - location)  (has_friend ?survivor - survivor)  (has_shelter ?loc - location)  (has_water_source ?loc - location)  (has_wood ?loc - location)  (inventory ?player ?item)  (is_safe ?loc - location)  (treated ?water - water))
    (:action build_raft
        :parameters (?player - player ?raft - raft ?loc - location)
        :precondition (and (at ?player ?loc) (at_ocean ?loc))
        :effect (has_escaped ?player)
    )
     (:action build_shelter
        :parameters (?player - player ?loc - location)
        :precondition (and (at ?player ?loc) (is_safe ?loc))
        :effect (has_shelter ?loc)
    )
     (:action carve_groove
        :parameters (?player - player ?wood - wood)
        :precondition (inventory ?player ?wood)
        :effect (groove ?wood)
    )
     (:action chop_wood
        :parameters (?player - player ?wood - wood ?loc - location)
        :precondition (and (at ?player ?loc) (has_wood ?loc))
        :effect (inventory ?player ?wood)
    )
     (:action clean_water
        :parameters (?player - player ?water - water ?loc - location)
        :precondition (and (at ?player ?loc) (inventory ?player ?water) (not (treated ?water)) (has_fire ?loc))
        :effect (treated ?water)
    )
     (:action cook_fish
        :parameters (?player - player ?fish - fish ?loc - location)
        :precondition (and (at ?player ?loc) (inventory ?player ?fish) (has_fire ?loc))
        :effect (cooked ?fish)
    )
     (:action drink_water
        :parameters (?player - player ?water - water)
        :precondition (inventory ?player ?water)
        :effect (drank ?water)
    )
     (:action find_other_survivors
        :parameters (?player - player ?survivor - survivor ?loc - location)
        :precondition (at ?player ?loc)
        :effect (has_friend ?survivor)
    )
     (:action get
        :parameters (?player - player ?item - item ?loc - location)
        :precondition (and (at ?player ?loc) (at ?item ?loc))
        :effect (and (not (at ?item ?loc)) (inventory ?player ?item))
    )
     (:action get_water
        :parameters (?player - player ?water - water ?loc - location)
        :precondition (and (at ?player ?loc) (has_water_source ?loc))
        :effect (inventory ?player ?water)
    )
     (:action go
        :parameters (?player - player ?from - location ?to - location ?dir - direction)
        :precondition (and (at ?player ?from) (connected ?from ?dir ?to))
        :effect (and (not (at ?player ?from)) (at ?player ?to))
    )
     (:action hunt_fish
        :parameters (?player - player ?fish - fish ?loc - location)
        :precondition (and (at ?player ?loc) (has_fish ?loc) (inventory ?player ?spear))
        :effect (inventory ?player ?fish)
    )
     (:action light_fire
        :parameters (?player - player ?loc - location)
        :precondition (and (at ?player ?loc) (can_light_fire ?loc))
        :effect (has_fire ?loc)
    )
     (:action make_weapon
        :parameters (?player - player ?spear - spear)
        :precondition (inventory ?player ?wood)
        :effect (inventory ?player ?spear)
    )
)