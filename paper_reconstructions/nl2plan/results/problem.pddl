(define (problem test_problem)
    (:domain test_domain)
    (:objects blue_block green_block red_block yellow_block - block table1 - table)
    (:init (clear green_block) (clear yellow_block) (on blue_block red_block) (on green_block table1) (on red_block yellow_block) (on yellow_block table1))
    (:goal (on red_block green_block))
)