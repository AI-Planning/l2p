(define (problem test_domain_problem)
    (:domain test_domain)
    (:objects arm - arm blue green red yellow - block table - table)
    (:init (clear blue) (clear green) (clear yellow) (empty arm) (on blue red) (on green table) (on red yellow) (on yellow table))
    (:goal (on red green))
)