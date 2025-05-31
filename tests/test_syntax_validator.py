from l2p import *
import unittest, textwrap


class TestSyntaxValidator(unittest.TestCase):
    def setUp(self):
        self.syntax_validator = SyntaxValidator()

    def test_validate_params(self):

        types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
        }

        params = OrderedDict([("?a", "arm"), ("?top", "block"), ("?bottom", "block")])

        # case 1: parameters are typed and types available
        flag, _ = self.syntax_validator.validate_params(parameters=params, types=types)
        self.assertEqual(flag, True)

        # case 2: parameters are untyped and types available
        untyped_params = OrderedDict([("?a", "arm"), ("?top", ""), ("?bottom", "")])
        flag, _ = self.syntax_validator.validate_params(
            parameters=untyped_params, types=types
        )
        self.assertEqual(flag, True)

        # case 3: no types available
        flag, _ = self.syntax_validator.validate_params(parameters=params)
        self.assertEqual(flag, False)

        # case 3: parameter types not found in types
        incorrect_params = OrderedDict(
            [("?a", "table"), ("?top", "block"), ("?bottom", "block")]
        )
        flag, _ = self.syntax_validator.validate_params(
            parameters=incorrect_params, types=types
        )
        self.assertEqual(flag, False)

        # case 4: parameter does not contain `?` in front of it
        missing_char_params = OrderedDict(
            [("a", "arm"), ("?top", "block"), ("bottom", "block")]
        )
        flag, _ = self.syntax_validator.validate_params(
            parameters=missing_char_params, types=types
        )
        self.assertEqual(flag, False)

    def test_validate_types_predicates(self):

        types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "table": "table that blocks sits on",
        }

        predicates = [
            Predicate(
                {
                    "name": "on_top",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "(on_top ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("?b2", "block")]),
                    "clean": "(on_top ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2",
                }
            )
        ]

        # case 1: correct predicates and types
        flag, _ = self.syntax_validator.validate_types_predicates(
            predicates=predicates, types=types
        )
        self.assertEqual(flag, True)

        # case 2: no types
        flag, _ = self.syntax_validator.validate_types_predicates(predicates=predicates)
        self.assertEqual(flag, True)

        # case 3: predicate name is the same as a type
        incorrect_predicates = [
            Predicate(
                {
                    "name": "on_top",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "(on_top ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("?b2", "block")]),
                    "clean": "(on_top ?b1 - block ?b2 - block)",
                }
            ),
            Predicate(
                {
                    "name": "under",
                    "desc": "true if the block ?b1 is under block ?b2",
                    "raw": "(under ?b1 - block ?b2 - block): true if the block ?b1 is under block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("?b2", "block")]),
                    "clean": "(under ?b1 - block ?b2 - block)",
                }
            ),
        ]

        types = {
            "on_top": "true if the block ?b1 is on top of the block ?b2",
            "under": "true if the block ?b1 is under block ?b2",
            "table": "table that blocks sits on",
        }

        flag, _ = self.syntax_validator.validate_types_predicates(
            predicates=incorrect_predicates, types=types
        )
        self.assertEqual(flag, False)

    def test_validate_duplicate_predicates(self):

        # case 1: duplicate predicate names but same parameters (identical predicates)
        predicates_1 = [
            Predicate(
                {
                    "name": "on_top",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "(on_top ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("?b2", "block")]),
                    "clean": "(on_top ?b1 - block ?b2 - block)",
                }
            ),
            Predicate(
                {
                    "name": "under",
                    "desc": "true if the block ?b1 is under block ?b2",
                    "raw": "(under ?b1 - block ?b2 - block): true if the block ?b1 is under block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("?b2", "block")]),
                    "clean": "(under ?b1 - block ?b2 - block)",
                }
            ),
        ]

        predicates_2 = [
            Predicate(
                {
                    "name": "on_top",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "(on_top ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("?b2", "block")]),
                    "clean": "(on_top ?b1 - block ?b2 - block)",
                }
            )
        ]

        flag, _ = self.syntax_validator.validate_duplicate_predicates(
            curr_predicates=predicates_1, new_predicates=predicates_2
        )
        self.assertEqual(flag, True)

        # case 2: duplicate predicate names but different parameters
        predicates_3 = [
            Predicate(
                {
                    "name": "on_top",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "(on_top ?b2 - block ?b1 - block): true if the block ?b1 is on top of the block ?b2",
                    "params": OrderedDict([("?b2", "block"), ("?b1", "block")]),
                    "clean": "(on_top ?b2 - block ?b1 - block)",
                }
            )
        ]

        flag, _ = self.syntax_validator.validate_duplicate_predicates(
            curr_predicates=predicates_1, new_predicates=predicates_3
        )
        self.assertEqual(flag, False)

        # case 3: different predicates
        predicates_4 = [
            Predicate(
                {
                    "name": "next_to",
                    "desc": "true if the block ?b1 is next to block ?b2",
                    "raw": "(next_to ?b1 - block ?b2 - block): true if the block ?b1 is next_to block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("?b2", "block")]),
                    "clean": "(next_to ?b1 - block ?b2 - block)",
                }
            )
        ]
        flag, _ = self.syntax_validator.validate_duplicate_predicates(
            curr_predicates=predicates_1, new_predicates=predicates_4
        )
        self.assertEqual(flag, True)

    def test_validate_format_predicate(self):

        # case 1: predicates are formatted correctly w/ correct predicate name, parameter string syntax
        predicates = [
            Predicate(
                {
                    "name": "on_top",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "(on_top ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("?b2", "block")]),
                    "clean": "(on_top ?b1 - block ?b2 - block)",
                }
            )
        ]

        types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "table": "table that blocks sits on",
        }

        flag, _ = self.syntax_validator.validate_format_predicates(
            predicates=predicates, types=types
        )
        self.assertEqual(flag, True)

        # case 2: predicate name is missing / starts with incorrect `?`
        pred_missing_name = [
            Predicate(
                {
                    "name": "",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "(?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("?b2", "block")]),
                    "clean": "(?b1 - block ?b2 - block)",
                }
            )
        ]

        flag, _ = self.syntax_validator.validate_format_predicates(
            predicates=pred_missing_name, types=types
        )
        self.assertEqual(flag, False)

        # case 3: parameter variables do not start with `?` syntax
        pred_incorrect_var = [
            Predicate(
                {
                    "name": "on_top",
                    "desc": "true if the block b1 is on top of the block ?b2",
                    "raw": "(on_top b1 - block ?b2 - block): true if the block b1 is on top of the block ?b2",
                    "params": OrderedDict([("b1", "block"), ("?b2", "block")]),
                    "clean": "(on_top b1 - block ?b2 - block)",
                }
            )
        ]

        flag, _ = self.syntax_validator.validate_format_predicates(
            predicates=pred_incorrect_var, types=types
        )
        self.assertEqual(flag, False)

        # other placement
        pred_incorrect_var = [
            Predicate(
                {
                    "name": "on_top",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "(on_top ?b1 - block b2 - block): true if the block ?b1 is on top of the block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("b2", "block")]),
                    "clean": "(on_top ?b1 - block b2 - block)",
                }
            )
        ]

        flag, _ = self.syntax_validator.validate_format_predicates(
            predicates=pred_incorrect_var, types=types
        )
        self.assertEqual(flag, False)

        # case 4: missing `-` syntax or using some other random character
        pred_incorrect_char = [
            Predicate(
                {
                    "name": "on_top",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "(on_top ?b1 - block ?b2 - ): true if the block ?b1 is on top of the block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("?b2", "")]),
                    "clean": "(on_top ?b1 - block ?b2 - )",
                }
            )
        ]

        flag, _ = self.syntax_validator.validate_format_predicates(
            predicates=pred_incorrect_char, types=types
        )
        self.assertEqual(flag, False)

        pred_incorrect_char = [
            Predicate(
                {
                    "name": "on_top",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "(on_top ?b1 | block ?b2 | block): true if the block ?b1 is on top of the block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("?b2", "block")]),
                    "clean": "(on_top ?b1 | block ?b2 | block)",
                }
            )
        ]

        flag, _ = self.syntax_validator.validate_format_predicates(
            predicates=pred_incorrect_char, types=types
        )
        self.assertEqual(flag, False)

        # case 5: variable is untyped (allowed in PDDL)
        pred_untyped = [
            Predicate(
                {
                    "name": "on_top",
                    "desc": "true if the block ?b1 is on top of ?x",
                    "raw": "(on_top ?b1 - block ?x): true if the block ?b1 is on top of ?x",
                    "params": OrderedDict([("?b1", "block"), ("?x", "")]),
                    "clean": "(on_top ?b1 - block ?x)",
                }
            ),
            Predicate(
                {
                    "name": "bottom",
                    "desc": "",
                    "raw": "(bottom ?b1 - block ?x ?y ?z): ",
                    "params": OrderedDict(
                        [("?b1", "block"), ("?x", ""), ("?y", ""), ("?z", "")]
                    ),
                    "clean": "(bottom ?b1 - block ?x ?y ?z)",
                }
            ),
        ]

        flag, _ = self.syntax_validator.validate_format_predicates(
            predicates=pred_untyped, types=types
        )
        self.assertEqual(flag, True)

        # case 6: parameter types are not found in types list
        pred_incorrect_param_type = [
            Predicate(
                {
                    "name": "on_top",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "(on_top ?b1 - box ?b2 - triangle): true if the box ?b1 is on top of the triangle ?b2",
                    "params": OrderedDict([("?b1", "box"), ("?b2", "triangle")]),
                    "clean": "(on_top ?b1 - box ?b2 - triangle)",
                }
            )
        ]

        flag, _ = self.syntax_validator.validate_format_predicates(
            predicates=pred_incorrect_param_type, types=types
        )
        self.assertEqual(flag, False)

        # no types exist
        flag, _ = self.syntax_validator.validate_format_predicates(
            predicates=pred_incorrect_param_type
        )
        self.assertEqual(flag, False)

    def test_validate_pddl_action(self):

        # case 1: correct implementation – predicate aligns with action parameters and types
        types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "table": "table that blocks sits on",
        }

        predicates = [
            Predicate(
                {
                    "name": "on",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "(on ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("?b2", "block")]),
                    "clean": "(on ?b1 - block ?b2 - block)",
                }
            ),
            Predicate(
                {
                    "name": "holding",
                    "desc": "true if arm is holding a block",
                    "raw": "(holding ?a - arm ?b - block): true if arm is holding a block",
                    "params": OrderedDict([("?a", "arm"), ("?b", "block")]),
                    "clean": "(holding ?a - arm ?b - block)",
                }
            ),
            Predicate(
                {
                    "name": "clear",
                    "desc": "true if a block does not have anything on top of it",
                    "raw": "(clear ?b - block): true if a block does not have anything on top of it",
                    "params": OrderedDict([("?b", "block")]),
                    "clean": "(clear ?b - block)",
                }
            ),
        ]

        precond_str = "( and      ( holding ?a ?b1 )  ; The arm is holding the top block      (clear ?b2 )  ; The bottom block is clear  )"
        params_info = (
            OrderedDict({"?b1": "block", "?b2": "block", "?a": "arm"}),
            [
                "- ?b1 - block: The block being stacked on top",
                "- ?b2 - block: The block being stacked upon",
                "- ?a - arm: The arm performing the stacking action",
            ],
        )

        flag, _ = self.syntax_validator.validate_pddl_action(
            pddl=precond_str,
            predicates=predicates,
            action_params=params_info[0],
            types=types,
            part="preconditions",
        )
        self.assertEqual(flag, True)

        # case 2: incorrect number of predicate parameters in pddl component
        precond_str = "( and      ( holding ?b1 )  ; The arm is holding the top block      (clear ?b2 )  ; The bottom block is clear  )"
        flag, _ = self.syntax_validator.validate_pddl_action(
            pddl=precond_str,
            predicates=predicates,
            action_params=params_info[0],
            types=types,
            part="preconditions",
        )
        self.assertEqual(flag, False)

        # case 3: predicate parameters include object types
        precond_str = "( and      ( holding ?a - arm ?b1 - block)  ; The arm is holding the top block      (clear ?b2 )  ; The bottom block is clear  )"
        flag, _ = self.syntax_validator.validate_pddl_action(
            pddl=precond_str,
            predicates=predicates,
            action_params=params_info[0],
            types=types,
            part="preconditions",
        )
        self.assertEqual(flag, False)

        # case 3: parameters declared in predicate not found in action parameter
        precond_str = "( and      ( holding ?a ?b1)  ; The arm is holding the top block      (clear ?rock )  ; The bottom block is clear  )"

        flag, _ = self.syntax_validator.validate_pddl_action(
            pddl=precond_str,
            predicates=predicates,
            action_params=params_info[0],
            types=types,
            part="preconditions",
        )
        self.assertEqual(flag, False)

        # case 4: check if declared predicate object types align with original predicate types
        params_info = (
            OrderedDict({"?b1": "block", "?b2": "block", "?a": "arm"}),
            [
                "- ?b1 - block: The block being stacked on top",
                "- ?b2 - block: The block being stacked upon",
                "- ?a - arm: The arm performing the stacking action",
            ],
        )

        # `clear` predicate parameter should be type `block` but instead `arm`
        precond_str = "( and      ( holding ?a ?b1)  ; The arm is holding the top block      (clear ?a )  ; The bottom block is clear  )"

        flag, _ = self.syntax_validator.validate_pddl_action(
            pddl=precond_str,
            predicates=predicates,
            action_params=params_info[0],
            types=types,
            part="preconditions",
        )
        self.assertEqual(flag, False)

        # case 5: types is empty but action parameter or predicate in pddl contains types
        types = {}

        predicates = [
            Predicate(
                {
                    "name": "on",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "(on ?b1 ?b2): true if the block ?b1 is on top of the block ?b2",
                    "params": OrderedDict([("?b1", ""), ("?b2", "")]),
                    "clean": "(on ?b1 ?b2)",
                }
            ),
            Predicate(
                {
                    "name": "holding",
                    "desc": "true if arm is holding a block",
                    "raw": "(holding ?a - arm ?b - block): true if arm is holding a block",
                    "params": OrderedDict([("?a", "arm"), ("?b", "block")]),
                    "clean": "(holding ?a - arm ?b - block)",
                }
            ),
            Predicate(
                {
                    "name": "clear",
                    "desc": "true if a block does not have anything on top of it",
                    "raw": "(clear ?b): true if a block does not have anything on top of it",
                    "params": OrderedDict([("?b", "")]),
                    "clean": "(clear ?b)",
                }
            ),
        ]

        predicates_with_types = [
            Predicate(
                {
                    "name": "on",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "(on ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("?b2", "block")]),
                    "clean": "(on ?b1 - block ?b2 - block)",
                }
            ),
            Predicate(
                {
                    "name": "holding",
                    "desc": "true if arm is holding a block",
                    "raw": "(holding ?a - arm ?b - block): true if arm is holding a block",
                    "params": OrderedDict([("?a", "arm"), ("?b", "block")]),
                    "clean": "(holding ?a - arm ?b - block)",
                }
            ),
            Predicate(
                {
                    "name": "clear",
                    "desc": "true if a block does not have anything on top of it",
                    "raw": "(clear ?b - block): true if a block does not have anything on top of it",
                    "params": OrderedDict([("?b", "block")]),
                    "clean": "(clear ?b - block)",
                }
            ),
        ]

        precond_str = "( and      ( holding ?a ?b1 )  ; The arm is holding the top block      (clear ?b2 )  ; The bottom block is clear  )"
        params_info = (
            OrderedDict({"?b1": "block", "?b2": "block", "?a": "arm"}),
            [
                "- ?b1 - block: The block being stacked on top",
                "- ?b2 - block: The block being stacked upon",
                "- ?a - arm: The arm performing the stacking action",
            ],
        )

        flag, _ = self.syntax_validator.validate_pddl_action(
            pddl=precond_str,
            predicates=predicates,
            action_params=params_info[0],
            types=types,
            part="preconditions",
        )
        self.assertEqual(flag, False)

        flag, _ = self.syntax_validator.validate_pddl_action(
            pddl=precond_str,
            predicates=predicates_with_types,
            action_params=params_info[0],
            types=types,
            part="preconditions",
        )
        self.assertEqual(flag, False)

        params_info = (
            OrderedDict({"?b1": "", "?b2": "block", "?a": ""}),
            [
                "- ?b1: The block being stacked on top",
                "- ?b2 - block: The block being stacked upon",
                "- ?a: The arm performing the stacking action",
            ],
        )

        flag, _ = self.syntax_validator.validate_pddl_action(
            pddl=precond_str,
            predicates=predicates_with_types,
            action_params=params_info[0],
            types=types,
            part="preconditions",
        )
        self.assertEqual(flag, False)

    def test_validate_usage_action(self):

        predicates = [
            {
                "name": "on",
                "desc": "true if the block ?b1 is on top of the block ?b2",
                "raw": "(on ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2",
                "params": OrderedDict([("?b1", "block"), ("?b2", "block")]),
                "clean": "(on ?b1 - block ?b2 - block)",
            },
            {
                "name": "holding",
                "desc": "true if arm is holding a block",
                "raw": "(holding ?a - arm ?b - block): true if arm is holding a block",
                "params": OrderedDict([("?a", "arm"), ("?b", "block")]),
                "clean": "(holding ?a - arm ?b - block)",
            },
            {
                "name": "clear",
                "desc": "true if a block does not have anything on top of it",
                "raw": "(clear ?b - block): true if a block does not have anything on top of it",
                "params": OrderedDict([("?b", "block")]),
                "clean": "(clear ?b - block)",
            },
        ]

        functions = [
            {
                "name": "weight",
                "desc": "weight of the block",
                "raw": "(weight ?b1 - block): weight of the block",
                "params": OrderedDict([("?b1", "block")]),
                "clean": "(weight ?b1 - block)",
            },
            {
                "name": "battery-level",
                "desc": "battery level of arm",
                "raw": "(battery-level ?a - arm): battery level of arm",
                "params": OrderedDict([("?a", "arm")]),
                "clean": "(battery-level ?a - arm)",
            },
        ]

        types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "table": "table that blocks sits on",
        }

        # case 1: correct implementation
        llm_response = textwrap.dedent(
            """
            ### Action Parameters
            ```
            - ?a - arm: The robotic arm
            - ?b1 - block: The block being moved
            - ?b2 - block: The block it's being stacked upon
            ``` 

            ### Action Preconditions
            ```
            (and
            
                ; normal predicates
                (holding ?a ?b1)
                (clear ?b2)
                
                ;; object equality
                (not (= ?b1 ?b2))
                
                ;; comparison operators on functions
                (<= (weight ?b1) 30)
                (>= (weight ?b2) 10)
                (< (battery-level ?a) 100)
                (> (battery-level ?a) (* 2 (weight ?b1))) ;; comparison with nested arithmetic (battery-level)
                
                ;; universal quantifier
                (forall (?x - block)
                    (or
                        (clear ?x)
                        (< (weight ?x) 50)
                    )
                )
                
                ;; existential quantifier
                (exists (?y - block)
                    (and
                        (not (= ?y ?b1))
                        (> (weight ?y) 25)
                        (clear ?y)
                    )
                )
            )
            ```

            ### Action Effects
            ```
            (and
                (on ?b1 ?b2)
                
                ;; `not` logical connective
                (not (clear ?b2))
                (not (holding ?a ?b1))
                
                ;; conditional effect
                (when (>= (weight ?b1) 20)
                    (and
                        (scale-down (battery-level ?a) 1.5)
                        (when (< (battery-level ?a) 100)
                            (assign (battery-level ?a) 100)
                        )
                    )
                )
                
                ;; assignment oeprators
                (increase (battery-level ?a) 5)
                (decrease (weight ?b1) 10)
                (scale-up (battery-level ?a) 2.5)
                (scale-down (battery-level ?a) 5)
            )
            ```
            """
        )

        flag, _ = self.syntax_validator.validate_usage_action(
            llm_response=llm_response,
            curr_predicates=predicates,
            types=types,
            functions=functions,
        )

        self.assertEqual(flag, True)

        # case 2: incorrect predicate usage
        invalid_predicates = [
            Predicate(
                {
                    "name": "unstack",
                    "desc": "true if a block does not have anything on top of it",
                    "raw": "(unstack ?b - block): true if a block does not have anything on top of it",
                    "params": OrderedDict([("?b", "block")]),
                    "clean": "(unstack ?b - block)",
                }
            ),
            Predicate(
                {
                    "name": "stack",
                    "desc": "true if a block does not have anything on top of it",
                    "raw": "(stack ?b - block): true if a block does not have anything on top of it",
                    "params": OrderedDict([("?b", "block")]),
                    "clean": "(stack ?b - block)",
                }
            ),
        ]

        flag, _ = self.syntax_validator.validate_usage_action(
            llm_response=llm_response, curr_predicates=invalid_predicates, types=types
        )

        self.assertEqual(flag, False)

        # case 3: incorrect function usage
        invalid_functions = [
            {
                "name": "weight",
                "desc": "weight of the block",
                "raw": "(weight ?b1 - block): weight of the block",
                "params": OrderedDict([("?b1", "block")]),
                "clean": "(weight ?b1 - block)",
            }
        ]

        flag, _ = self.syntax_validator.validate_usage_action(
            llm_response=llm_response,
            curr_predicates=predicates,
            types=types,
            functions=invalid_functions,
        )

        self.assertEqual(flag, False)

        # case 4: incorrect quantifier usage (invalid arguments)
        llm_response = textwrap.dedent(
            """
            ### Action Parameters
            ```
            - ?a - arm: The robotic arm
            - ?b1 - block: The block being moved
            - ?b2 - block: The block it's being stacked upon
            ``` 

            ### Action Preconditions
            ```
            (and
                (forall ?x - block
                    (or
                        (clear ?x)
                        (< (weight ?x) 50)
                    )
                )
            )
            ```

            ### Action Effects
            ```
            (and)
            ```
            """
        )

        flag, _ = self.syntax_validator.validate_usage_action(
            llm_response=llm_response,
            curr_predicates=predicates,
            types=types,
            functions=functions,
        )

        self.assertEqual(flag, False)

        llm_response = textwrap.dedent(
            """
            ### Action Parameters
            ```
            - ?a - arm: The robotic arm
            - ?b1 - block: The block being moved
            - ?b2 - block: The block it's being stacked upon
            ``` 

            ### Action Preconditions
            ```
            (and
                (exists (?y - block)
                        (not (= ?y ?b1))
                        (> (weight ?y) 25)
                        (clear ?y)
                )
            )
            ```

            ### Action Effects
            ```
            (and)
            ```
            """
        )

        flag, _ = self.syntax_validator.validate_usage_action(
            llm_response=llm_response,
            curr_predicates=predicates,
            types=types,
            functions=functions,
        )

        self.assertEqual(flag, False)

        # case 5: incorrect conditional effect usage
        llm_response = textwrap.dedent(
            """
            ### Action Parameters
            ```
            - ?a - arm: The robotic arm
            - ?b1 - block: The block being moved
            - ?b2 - block: The block it's being stacked upon
            ``` 

            ### Action Preconditions
            ```
            (and
                (when (on ?b1 ?b2) ;; THIS SHOULD FAIL
                        (clear ?b1)
                )
            )
            ```

            ### Action Effects
            ```
            (and
                (when on ?b1 ?b2 ;; THIS SHOULD FAILURE DUE TO PDDL MALFORMITY
                        (clear ?b1)
                )
            )
            ```
            """
        )
        flag, msg = self.syntax_validator.validate_usage_action(
            llm_response=llm_response,
            curr_predicates=predicates,
            types=types,
            functions=functions,
        )

        self.assertEqual(flag, False)

        # case 6: predicate used as function - vice versa
        llm_response = textwrap.dedent(
            """
            ### Action Parameters
            ```
            - ?a - arm: The robotic arm
            - ?b1 - block: The block being moved
            - ?b2 - block: The block it's being stacked upon
            ``` 

            ### Action Preconditions
            ```
            (and
                (< (holding ?a ?b1) 100)
            )
            ```

            ### Action Effects
            ```
            (and)
            ```
            """
        )

        flag, _ = self.syntax_validator.validate_usage_action(
            llm_response=llm_response,
            curr_predicates=predicates,
            types=types,
            functions=functions,
        )

        self.assertEqual(flag, False)

        llm_response = textwrap.dedent(
            """
            ### Action Parameters
            ```
            - ?a - arm: The robotic arm
            - ?b1 - block: The block being moved
            - ?b2 - block: The block it's being stacked upon
            ``` 

            ### Action Preconditions
            ```
            (and
                (battery-level ?a)
            )
            ```

            ### Action Effects
            ```
            (and)
            ```
            """
        )

        flag, _ = self.syntax_validator.validate_usage_action(
            llm_response=llm_response,
            curr_predicates=predicates,
            types=types,
            functions=functions,
        )

        self.assertEqual(flag, False)

        # case 7: incorrect object equality usage
        llm_response = textwrap.dedent(
            """
            ### Action Parameters
            ```
            - ?left - arm: left robot arm
            - ?right - arm: right robot arm
            - ?b1 - block: The block being moved
            - ?b2 - block: The block it's being stacked upon
            ``` 

            ### Action Preconditions
            ```
            (and
                (not (= ?left ?right))
                (= ?b2 5) ;; incorrect
            )
            ```

            ### Action Effects
            ```
            (and)
            ```
            """
        )

        flag, _ = self.syntax_validator.validate_usage_action(
            llm_response=llm_response,
            curr_predicates=predicates,
            types=types,
            functions=functions,
        )

        self.assertEqual(flag, False)

        llm_response = textwrap.dedent(
            """
            ### Action Parameters
            ```
            - ?a - arm: The robotic arm
            - ?b1 - block: The block being moved
            - ?b2 - block: The block it's being stacked upon
            ``` 

            ### Action Preconditions
            ```
            (and
                (not (= ?a ?b1))
                (= ?b2 5) ;; incorrect
            )
            ```

            ### Action Effects
            ```
            (and)
            ```
            """
        )

        flag, _ = self.syntax_validator.validate_usage_action(
            llm_response=llm_response,
            curr_predicates=predicates,
            types=types,
            functions=functions,
        )

        self.assertEqual(flag, False)

    def test_validate_overflow_predicates(self):

        llm_response = textwrap.dedent(
            """
            ### Action Parameters
            ```
            - ?b1 - block: The block being stacked on top
            - ?b2 - block: The block being stacked upon
            - ?a - arm: The arm performing the stacking action
            ```

            ### Action Preconditions
            ```
            (and
                (holding ?a ?b1) ; The arm is holding the top block
                (clear ?b2) ; The bottom block is clear
            )
            ```

            ### Action Effects
            ```
            (and
                (not (holding ?a ?b1)) ; The arm is no longer holding the top block
                (on ?b1 ?b2) ; The top block is now on the bottom block
                (not (clear ?b2)) ; The bottom block is no longer clear
            )
            ```

            ### New Predicates
            ```
            ```
            """
        )

        # case 1: predicates under limit
        flag, _ = self.syntax_validator.validate_overflow_predicates(
            llm_response=llm_response, limit=10
        )
        self.assertEqual(flag, True)

        # case 2: precondition predicates exceed limit
        flag, _ = self.syntax_validator.validate_overflow_predicates(
            llm_response=llm_response, limit=1
        )
        self.assertEqual(flag, False)

        # case 3: effect predicates exceed limit
        flag, _ = self.syntax_validator.validate_overflow_predicates(
            llm_response=llm_response, limit=2
        )
        self.assertEqual(flag, False)

    def test_validate_task_objects(self):
        types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "table": "table that blocks sits on",
        }

        type_hierarchy = [
            {"arm": "'arm for a robot'", "children": []},
            {
                "block": "block that can be stacked and unstacked",
                "children": [
                    {
                        "heavy_block": "a heavy block that cannot be picked up",
                        "children": [],
                    },
                    {
                        "light_block": "a light block that can be picked up",
                        "children": [],
                    },
                ],
            },
            {"table": "table that blocks sits on", "children": []},
        ]

        objects = {
            "blue_block": "",
            "red_block": "block",
            "yellow_block": "block",
            "green_block": "block",
        }

        # case 1: task objects are assigned correct types
        flag, _ = self.syntax_validator.validate_task_objects(
            objects=objects, types=types
        )
        self.assertEqual(flag, True)

        objects = {
            "blue_block": "",
            "red_block": "light_block",
            "yellow_block": "heavy_block",
            "green_block": "block",
        }

        flag, _ = self.syntax_validator.validate_task_objects(
            objects=objects, types=type_hierarchy
        )
        self.assertEqual(flag, True)

        # extra case: no types assigned to objects
        objects = {
            "blue_block": "",
            "red_block": "",
            "yellow_block": "",
            "green_block": "",
        }
        flag, _ = self.syntax_validator.validate_task_objects(
            objects=objects, types=type_hierarchy
        )
        self.assertEqual(flag, True)

        # case 2: task object contains type not found in types
        objects = {
            "blue_block": "",
            "red_block": "rock",
            "yellow_block": "triangle",
            "green_block": "block",
        }

        flag, _ = self.syntax_validator.validate_task_objects(
            objects=objects, types=types
        )
        self.assertEqual(flag, False)

        flag, _ = self.syntax_validator.validate_task_objects(
            objects=objects, types=type_hierarchy
        )
        self.assertEqual(flag, False)

        # case 3: task object has same name as type
        objects = {"block": "", "light_block": "", "table": "", "arm": ""}
        flag, _ = self.syntax_validator.validate_task_objects(
            objects=objects, types=types
        )
        self.assertEqual(flag, False)

        flag, _ = self.syntax_validator.validate_task_objects(
            objects=objects, types=type_hierarchy
        )
        self.assertEqual(flag, False)

        # case 4: no types
        flag, _ = self.syntax_validator.validate_task_objects(objects=objects)
        self.assertEqual(flag, True)

        objects = {
            "blue_block": "",
            "red_block": "rock",
            "yellow_block": "triangle",
            "green_block": "block",
        }

        flag, s = self.syntax_validator.validate_task_objects(objects=objects)
        self.assertEqual(flag, False)

    def test_validate_task_states(self):
        initial_states = [
            {
                "pred_name": "on_top",
                "params": ["blue_block", "red_block"],
                "neg": False,
            },
            {
                "pred_name": "on_top",
                "params": ["red_block", "yellow_block"],
                "neg": False,
            },
            {"pred_name": "on_table", "params": ["yellow_block"], "neg": False},
            {"pred_name": "on_table", "params": ["green_block"], "neg": False},
            {"pred_name": "clear", "params": ["yellow_block"], "neg": False},
            {"pred_name": "clear", "params": ["green_block"], "neg": False},
            {"pred_name": "clear", "params": ["red_block"], "neg": True},
            # {'func_name': 'weight', 'params': ['blue_block'], 'value': 100, 'op': '='}
        ]

        objects = {
            "blue_block": "block",
            "red_block": "block",
            "yellow_block": "block",
            "green_block": "block",
        }

        predicates = [
            Predicate(
                {
                    "name": "on_top",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "(on_top ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("?b2", "block")]),
                    "clean": "(on_top ?b1 - block ?b2 - block)",
                }
            ),
            Predicate(
                {
                    "name": "on_table",
                    "desc": "true if the block ?b is on table",
                    "raw": "(on_table ?b - block): true if the block ?b is on table",
                    "params": OrderedDict([("?b", "block")]),
                    "clean": "(on_table ?b - block)",
                }
            ),
            Predicate(
                {
                    "name": "holding",
                    "desc": "true if arm is holding a block",
                    "raw": "(holding ?a - arm ?b - block): true if arm is holding a block",
                    "params": OrderedDict([("?a", "arm"), ("?b", "block")]),
                    "clean": "(holding ?a - arm ?b - block)",
                }
            ),
            Predicate(
                {
                    "name": "clear",
                    "desc": "true if a block does not have anything on top of it",
                    "raw": "(clear ?b - block): true if a block does not have anything on top of it",
                    "params": OrderedDict([("?b", "block")]),
                    "clean": "(clear ?b - block)",
                }
            ),
        ]

        # case 1: correct task states, objects, and predicates
        flag, _ = self.syntax_validator.validate_task_states(
            states=initial_states,
            objects=objects,
            predicates=predicates,
            state_type="initial",
        )
        self.assertEqual(flag, True)

        # case 2: predicates in states not found in predicate definition
        initial_states = [
            {"pred_name": "throw", "params": ["blue_block", "red_block"], "neg": False},
            {"pred_name": "on", "params": ["red_block", "yellow_block"], "neg": False},
            {"pred_name": "on_table", "params": ["yellow_block"], "neg": False},
        ]

        flag, _ = self.syntax_validator.validate_task_states(
            states=initial_states,
            objects=objects,
            predicates=predicates,
            state_type="initial",
        )
        self.assertEqual(flag, False)

        # case 3: object variables in states are not found in task object list
        initial_states = [
            {
                "pred_name": "on_top",
                "params": ["purple_block", "red_block"],
                "neg": False,
            },
            {"pred_name": "on_top", "params": ["red_block", "arm"], "neg": False},
        ]

        flag, _ = self.syntax_validator.validate_task_states(
            states=initial_states,
            objects=objects,
            predicates=predicates,
            state_type="initial",
        )
        self.assertEqual(flag, False)

        # case 4: placement of the task predicate parameter types do not align w/ predicate definition parameter types
        initial_states = [
            {
                "pred_name": "on_top",
                "params": ["blue_block", "red_block"],
                "neg": False,
            },
            {
                "pred_name": "holding",
                "params": ["blue_block", "red_block"],
                "neg": False,
            },
        ]

        objects = {
            "blue_block": "block",
            "red_block": "block",
            "yellow_block": "block",
            "green_block": "block",
            "a": "arm",
        }

        flag, _ = self.syntax_validator.validate_task_states(
            states=initial_states,
            objects=objects,
            predicates=predicates,
            state_type="initial",
        )
        self.assertEqual(flag, False)

    def test_validate_header(self):

        # case 1: correct output structure
        llm_response = textwrap.dedent(
            """
            ### Action Parameters
            ```
            ```

            ### Action Preconditions
            ```
            ```

            ### Action Effects
            ```
            ```

            ### New Predicates
            ```
            ```
            """
        )

        flag, _ = self.syntax_validator.validate_header(llm_response=llm_response)
        self.assertEqual(flag, True)

        # case 2: correct output structure (w/ removed headers)
        llm_response = textwrap.dedent(
            """
            ### Action Preconditions
            ```
            ```

            ### Action Effects
            ```
            ```
            """
        )

        self.syntax_validator.headers = ["Action Preconditions", "Action Effects"]

        flag, _ = self.syntax_validator.validate_header(llm_response=llm_response)
        self.assertEqual(flag, True)

        # case 3: header not declared in llm output

        self.syntax_validator.headers = [
            "Action Parameters",
            "Action Preconditions",
            "Action Effects",
        ]

        flag, _ = self.syntax_validator.validate_header(llm_response=llm_response)
        self.assertEqual(flag, False)

        # case 4: header contains incorrect closing code blocks
        llm_response = textwrap.dedent(
            """
            ### Action Preconditions
            ```

            ### Action Effects
            ```
            ```
            """
        )

        self.syntax_validator.headers = ["Action Preconditions", "Action Effects"]

        flag, _ = self.syntax_validator.validate_header(llm_response=llm_response)
        self.assertEqual(flag, False)

    def test_unsupported_keywords(self):

        # case 1: llm response does not contain unsupported keywords
        llm_response = textwrap.dedent(
            """
            ### Action Preconditions
            ```
            (at ?r ?from)
            (connected ?from ?to)
            ```

            ### Action Effects
            ```
            (not (at ?r ?from))
            (at ?r ?to)
            ```
            """
        )

        self.syntax_validator.unsupported_keywords = ["forall", "when", "implies"]

        flag, _ = self.syntax_validator.validate_unsupported_keywords(
            llm_response=llm_response
        )
        self.assertEqual(flag, True)

        # case 2: no unsupported keywords declared

        self.syntax_validator.unsupported_keywords = []

        flag, _ = self.syntax_validator.validate_unsupported_keywords(
            llm_response=llm_response
        )
        self.assertEqual(flag, True)

        # case 3: llm response contains unsupported keywords
        llm_response = textwrap.dedent(
            """
            ### Action Preconditions
            ```
            (at ?r ?from)
            (connected ?from ?to)
            (forall (?o - obstacle) (not (blocked ?to ?o)))
            ```

            ### Action Effects
            ```
            (not (at ?r ?from))
            (at ?r ?to)
            (when (carrying ?r)
                (update-location ?r ?to)
            )
            ```
            """
        )

        self.syntax_validator.unsupported_keywords = ["forall", "when", "implies"]

        flag, _ = self.syntax_validator.validate_unsupported_keywords(
            llm_response=llm_response
        )
        self.assertEqual(flag, False)

    def test_validate_duplicate_headers(self):

        # case 1: correct llm response containing only 1 declaration per header
        llm_response = textwrap.dedent(
            """
            ### Action Parameters
            ```
            ?x - block
            ?y - block
            ```

            ### Action Preconditions
            ```
            (clear ?x)
            (clear ?y)
            (holding ?x)
            ```

            ### Action Effects
            ```
            (not (clear ?y))
            (not (holding ?x))
            (clear ?x)
            (on ?x ?y)
            (handempty)
            ```

            ### New Predicates
            ```
            (on ?x ?y) - block x is on block y
            (clear ?x) - block x has nothing on top
            (holding ?x) - the robot is holding block x
            (handempty) - the robot's hand is empty
            ```
            """
        )

        self.syntax_validator.headers = [
            "Action Parameters",
            "Action Preconditions",
            "Action Effects",
            "New Predicates",
        ]

        flag, _ = self.syntax_validator.validate_duplicate_headers(
            llm_response=llm_response
        )
        self.assertEqual(flag, True)

        # case 2: duplicate headers in llm response
        llm_response = textwrap.dedent(
            """
            ### Action Parameters
            ```
            ?x - block
            ?y - block
            ```

            ### Action Parameters
            ```
            ?x - block
            ```

            ### Action Effects
            ```
            (not (clear ?y))
            (not (holding ?x))
            (clear ?x)
            (on ?x ?y)
            (handempty)
            ```

            ### New Predicates
            ```
            (on ?x ?y) - block x is on block y
            (clear ?x) - block x has nothing on top
            (holding ?x) - the robot is holding block x
            (handempty) - the robot's hand is empty
            ```

            ### New Predicates
            ```
            (on ?x ?y) - block x is on block y
            (handempty) - the robot's hand is empty
            ```
            """
        )

        flag, _ = self.syntax_validator.validate_duplicate_headers(
            llm_response=llm_response
        )
        self.assertEqual(flag, False)

    def test_validate_type(self):

        # case 1: claimed type matches target type or its sub-types
        types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "table": "table that blocks sits on",
        }

        type_hierarchy = [
            {"arm": "'arm for a robot'", "children": []},
            {
                "block": "block that can be stacked and unstacked",
                "children": [
                    {
                        "heavy_block": "a heavy block that cannot be picked up",
                        "children": [],
                    },
                    {
                        "light_block": "a light block that can be picked up",
                        "children": [],
                    },
                ],
            },
            {"table": "table that blocks sits on", "children": []},
        ]

        target_type, claimed_type = "arm", "arm"

        flag, _ = self.syntax_validator.validate_type(
            target_type=target_type, claimed_type=claimed_type, types=types
        )
        self.assertEqual(flag, True)

        target_type, claimed_type = "block", "light_block"

        flag, _ = self.syntax_validator.validate_type(
            target_type=target_type, claimed_type=claimed_type, types=type_hierarchy
        )
        self.assertEqual(flag, True)

        # case 2: claimed type is not found in target type or its sub-types
        target_type, claimed_type = "block", "ball"
        flag, _ = self.syntax_validator.validate_type(
            target_type=target_type, claimed_type=claimed_type, types=types
        )
        self.assertEqual(flag, False)

        flag, _ = self.syntax_validator.validate_type(
            target_type=target_type, claimed_type=claimed_type, types=type_hierarchy
        )
        self.assertEqual(flag, False)

        # case 3: target type is not found in :types definition
        target_type, claimed_type = "rock", "ball"
        flag, _ = self.syntax_validator.validate_type(
            target_type=target_type, claimed_type=claimed_type, types=type_hierarchy
        )
        self.assertEqual(flag, False)

    def test_validate_format_types(self):

        # case 1: correct types
        types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "table": "table that blocks sits on",
        }

        type_hierarchy = [
            {"arm": "'arm for a robot'", "children": []},
            {
                "block": "block that can be stacked and unstacked",
                "children": [
                    {
                        "heavy_block": "a heavy block that cannot be picked up",
                        "children": [],
                    },
                    {
                        "light_block": "a light block that can be picked up",
                        "children": [],
                    },
                ],
            },
            {"table": "table that blocks sits on", "children": []},
        ]

        flag, _ = self.syntax_validator.validate_format_types(types=types)
        self.assertEqual(flag, True)
        flag, _ = self.syntax_validator.validate_format_types(types=type_hierarchy)
        self.assertEqual(flag, True)

        # case 2: types contain `?` character
        types = {
            "?arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "?table": "table that blocks sits on",
        }

        type_hierarchy = [
            {"arm": "'arm for a robot'", "children": []},
            {
                "block": "block that can be stacked and unstacked",
                "children": [
                    {
                        "?heavy_block": "a heavy block that cannot be picked up",
                        "children": [],
                    },
                    {
                        "?light_block": "a light block that can be picked up",
                        "children": [],
                    },
                ],
            },
            {"?table": "table that blocks sits on", "children": []},
        ]

        flag, _ = self.syntax_validator.validate_format_types(types=types)
        self.assertEqual(flag, False)

        flag, _ = self.syntax_validator.validate_format_types(types=type_hierarchy)
        self.assertEqual(flag, False)

    def test_validate_cyclic_types(self):

        # case 1: type_hierarchy is acyclic
        type_hierarchy = [
            {
                "block": "",
                "children": [
                    {"light_block": "", "children": []},
                    {"heavy_block": "", "children": []},
                ],
            },
            {
                "arm": "",
                "children": [
                    {
                        "robot_arm": "",
                        "children": [
                            {"little_arm": "", "children": []},
                            {"big_arm": "", "children": []},
                        ],
                    },
                ],
            },
        ]

        types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "table": "table that blocks sits on",
        }

        flag, _ = self.syntax_validator.validate_cyclic_types(type_hierarchy)
        self.assertEqual(flag, True)

        flag, _ = self.syntax_validator.validate_cyclic_types(types)
        self.assertEqual(flag, True)

        # case 2: type hierarchy is cyclic within its branches
        type_hierarchy = [
            {
                "block": "",
                "children": [
                    {"light_block": "", "children": []},
                    {"heavy_block": "", "children": []},
                ],
            },
            {
                "arm": "",
                "children": [
                    {
                        "robot_arm": "",
                        "children": [
                            {
                                "little_arm": "",
                                "children": [{"arm": "", "children": ""}],
                            },
                            {"big_arm": "", "children": []},
                        ],
                    },
                ],
            },
            {"table": "", "children": []},
        ]

        flag, _ = self.syntax_validator.validate_cyclic_types(type_hierarchy)
        self.assertEqual(flag, False)

        # case 3: type hierarchy is cyclic in its parent types
        type_hierarchy = [
            {
                "block": "",
                "children": [
                    {"arm": "", "children": []},
                ],
            },
            {
                "arm": "",
                "children": [
                    {"block": "", "children": []},
                ],
            },
        ]

        flag, _ = self.syntax_validator.validate_cyclic_types(type_hierarchy)
        self.assertEqual(flag, False)

    def test_validate_constant_types(self):

        constants = {
            "robot1": "robot",
            "robot2": "robot",
            "charging-station": "location",
            "storage-room": "location",
            "loading-dock": "location",
        }

        types = {"robot": "a robot", "location": "location to be at"}

        self.syntax_validator.error_types = ["validate_constant_types"]

        flag, _ = self.syntax_validator.validate_constant_types(
            constants=constants, types=types
        )

        self.assertEqual(flag, True)

        constants = {
            "robot1": "robot",
            "robot2": "robo",
            "charging-station": "location",
            "storage-room": "location",
            "loading-dock": "location",
        }

        flag, _ = self.syntax_validator.validate_constant_types(
            constants=constants, types=types
        )

        self.assertEqual(flag, False)


if __name__ == "__main__":
    unittest.main()
