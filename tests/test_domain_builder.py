import unittest, textwrap
from collections import OrderedDict
from l2p import *
from .mock_llm import MockLLM


class TestDomainBuilder(unittest.TestCase):
    def setUp(self):
        self.domain_builder = DomainBuilder()
        self.syntax_validator = SyntaxValidator()
        self.mock_llm = MockLLM()

    def normalize(self, string):
        return "\n".join(
            line.strip() for line in textwrap.dedent(string).strip().splitlines()
        )

    def test_formalize_types(self):

        self.syntax_validator.error_types = ["validate_format_types"]
        self.mock_llm.output = textwrap.dedent(
            """
            ### TYPES
            ```
            {
                "arm": "arm for a robot",
                "block": "block that can be stacked and unstacked",
                "table": "table that blocks sits on",
            }
            ```
            """
        )

        expected_types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "table": "table that blocks sits on",
        }

        types, _, validation_info = self.domain_builder.formalize_types(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            syntax_validator=self.syntax_validator,
        )

        self.assertEqual(expected_types, types)
        self.assertEqual(validation_info[0], True)

    def test_formalize_type_hierarchy(self):
        self.syntax_validator.error_types = [
            "validate_format_types",
            "validate_cyclic_types",
        ]
        self.mock_llm.output = textwrap.dedent(
            """
            ### TYPES
            ```
            [
                {
                    "arm": "'arm for a robot'",
                    "children": []
                },
                {
                    "block": "block that can be stacked and unstacked",
                    "children": [
                        {
                            "heavy_block": "a heavy block that cannot be picked up",
                            "children": []
                        },
                        {
                            "light_block": "a light block that can be picked up",
                            "children": []
                        }
                    ]
                },
                {
                    "table": "table that blocks sits on",
                    "children": []
                }
            ]
            ```
            """
        )

        expected_types = [
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

        types, _, validation_info = self.domain_builder.formalize_type_hierarchy(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            syntax_validator=self.syntax_validator,
        )

        self.assertEqual(expected_types, types)
        self.assertEqual(validation_info[0], True)

    def test_formalize_functions(self):

        self.syntax_validator.error_types = ["validate_format_functions"]

        self.mock_llm.output = textwrap.dedent(
            """
            ## DISTRACTION TEXT
            ### FUNCTIONS
            ```
            (battery-level ?a - arm): battery level of robot arm
            (weight ?b - block): weight of a block
            ```
            ## DISTRACTION TEXT
            """
        )

        types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "table": "table that blocks sits on",
        }

        functions, _, validation_info = self.domain_builder.formalize_functions(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            types=types,
            syntax_validator=self.syntax_validator,
        )

        exp_functions = [
            {
                "name": "battery-level",
                "desc": "battery level of robot arm",
                "raw": "(battery-level ?a - arm): battery level of robot arm",
                "params": OrderedDict([("?a", "arm")]),
                "clean": "(battery-level ?a - arm)",
            },
            {
                "name": "weight",
                "desc": "weight of a block",
                "raw": "(weight ?b - block): weight of a block",
                "params": OrderedDict([("?b", "block")]),
                "clean": "(weight ?b - block)",
            },
        ]

        self.assertEqual(validation_info[0], True)
        self.assertEqual(exp_functions, functions)

    def test_formalize_pddl_action(self):

        self.syntax_validator.unsupported_keywords = []

        self.syntax_validator.error_types = [
            "validate_header",
            "validate_duplicate_headers",
            "validate_unsupported_keywords",
            "validate_params",
            "validate_duplicate_predicates",
            "validate_types_predicates",
            "validate_format_predicates",
            "validate_usage_action",
        ]
        self.mock_llm.output = textwrap.dedent(
            """
            ## DISTRACTION TEXT
            ### Action Parameters
            ```
            - ?b1 - block: The block being stacked on top
            - ?b2 - block: The block being stacked upon
            - ?a - arm: The arm performing the stacking action
            ```

            ## DISTRACTION TEXT

            ### Action Preconditions
            ```
            (and
                (holding ?a ?b1) ; The arm is holding the top block
                (clear ?b2) ; The bottom block is clear
                (= (weight ?b1) (weight ?b2))
                (forall (?block1 ?block2 - block ?a1 - arm)
                    (and
                    (clear ?block1)
                    (= (battery-level ?a) 100)
                    (>= (weight ?block1) 10)
                    (<= (weight ?b1) -100)
                    (< (weight ?block2) 20)
                    (> (battery-level ?a1) 10)
                    )
                )
            )
            ```

            ## DISTRACTION TEXT

            ### Action Effects
            ```
            (and
                (not (holding ?a ?b1)) ; The arm is no longer holding the top block
                (on ?b1 ?b2) ; The top block is now on the bottom block
                (not (clear ?b2)) ; The bottom block is no longer clear
                (when (clear ?b1)
                    (and
                    (clear ?b2)
                    (clear ?b1)
                    )
                )
                (increase (battery-level ?a) 200)
                (increase (battery-level ?a) (* (weight ?b1) 100))
                (decrease (battery-level ?a) (* 0.05 (weight ?b2)))
                (assign (battery-level ?a) (* (weight ?b1) 2))
                (scale-up (battery-level ?a) 2)
                (scale-down (battery-level ?a) (weight ?b1))
            )
            ```

            ## DISTRACTION TEXT

            ### New Predicates
            ```
            - (on ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2
            - (holding ?a - arm ?b - block): true if arm is holding a block
            - (clear ?b - block): true if a block does not have anything on top of it
            ```

            ## DISTRACTION TEXT
            """
        )

        exp_predicates = [
            Predicate(
                {
                    "name": "on",
                    "desc": "true if the block ?b1 is on top of the block ?b2",
                    "raw": "- (on ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2",
                    "params": OrderedDict([("?b1", "block"), ("?b2", "block")]),
                    "clean": "(on ?b1 - block ?b2 - block)",
                }
            ),
            Predicate(
                {
                    "name": "holding",
                    "desc": "true if arm is holding a block",
                    "raw": "- (holding ?a - arm ?b - block): true if arm is holding a block",
                    "params": OrderedDict([("?a", "arm"), ("?b", "block")]),
                    "clean": "(holding ?a - arm ?b - block)",
                }
            ),
            Predicate(
                {
                    "name": "clear",
                    "desc": "true if a block does not have anything on top of it",
                    "raw": "- (clear ?b - block): true if a block does not have anything on top of it",
                    "params": OrderedDict([("?b", "block")]),
                    "clean": "(clear ?b - block)",
                }
            ),
        ]

        exp_action = {
            "name": "stack",
            "params": OrderedDict([("?b1", "block"), ("?b2", "block"), ("?a", "arm")]),
            "preconditions": "(and\n    (holding ?a ?b1) ; The arm is holding the top block\n    (clear ?b2) ; The bottom block is clear\n    (= (weight ?b1) (weight ?b2))\n    (forall (?block1 ?block2 - block ?a1 - arm)\n        (and\n        (clear ?block1)\n        (= (battery-level ?a) 100)\n        (>= (weight ?block1) 10)\n        (<= (weight ?b1) -100)\n        (< (weight ?block2) 20)\n        (> (battery-level ?a1) 10)\n        )\n    )\n)",
            "effects": "(and\n    (not (holding ?a ?b1)) ; The arm is no longer holding the top block\n    (on ?b1 ?b2) ; The top block is now on the bottom block\n    (not (clear ?b2)) ; The bottom block is no longer clear\n    (when (clear ?b1)\n        (and\n        (clear ?b2)\n        (clear ?b1)\n        )\n    )\n    (increase (battery-level ?a) 200)\n    (increase (battery-level ?a) (* (weight ?b1) 100))\n    (decrease (battery-level ?a) (* 0.05 (weight ?b2)))\n    (assign (battery-level ?a) (* (weight ?b1) 2))\n    (scale-up (battery-level ?a) 2)\n    (scale-down (battery-level ?a) (weight ?b1))\n)",
        }

        types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "table": "table that blocks sits on",
        }

        functions = [
            {
                "name": "battery-level",
                "desc": "battery level of robot arm",
                "raw": "(battery-level ?a - arm): battery level of robot arm",
                "params": OrderedDict([("?a", "arm")]),
                "clean": "(battery-level ?a - arm)",
            },
            {
                "name": "weight",
                "desc": "weight of a block",
                "raw": "(weight ?b - block): weight of a block",
                "params": OrderedDict([("?b", "block")]),
                "clean": "(weight ?b - block)",
            },
        ]

        action, new_predicates, _, validation_info = (
            self.domain_builder.formalize_pddl_action(
                model=self.mock_llm,
                domain_desc="",
                prompt_template="",
                types=types,
                functions=functions,
                syntax_validator=self.syntax_validator,
                action_name="stack",
                extract_new_preds=True,
            )
        )

        self.assertEqual(exp_action, action)
        self.assertEqual(exp_predicates, new_predicates)
        self.assertEqual(validation_info[0], True)

    def test_formalize_parameters(self):

        self.syntax_validator.headers = ["Action Parameters"]
        self.syntax_validator.error_types = [
            "validate_header",
            "validate_duplicate_headers",
            "validate_unsupported_keywords",
            "validate_params",
        ]

        self.mock_llm.output = textwrap.dedent(
            """
            ## DISTRACTION TEXT

            ### Action Parameters
            ```
            - ?b1 - block: the block that is being stacked on top
            - ?b2 - block: the block that is being stacked upon
            - ?a - arm: the arm of the robot performing the action
            ```

            ## DISTRACTION TEXT
            """
        )

        types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "table": "table that blocks sits on",
        }

        exp_params = OrderedDict({"?b1": "block", "?b2": "block", "?a": "arm"})

        params, _, _, validation_info = self.domain_builder.formalize_parameters(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            action_name="stack",
            types=types,
            syntax_validator=self.syntax_validator,
        )

        self.assertEqual(exp_params, params)
        self.assertEqual(validation_info[0], True)

    def test_formalize_preconditions(self):
        self.syntax_validator.headers = ["Action Preconditions"]
        self.syntax_validator.error_types = [
            "validate_header",
            "validate_duplicate_headers",
            "validate_unsupported_keywords",
            "validate_params",
        ]

        self.mock_llm.output = textwrap.dedent(
            """
            ## DISTRACTION TEXT

            ### Action Preconditions
            ```
            (and
                (holding ?a ?b1) ; The arm is holding the top block
                (clear ?b2) ; The bottom block is clear
            )
            ```

            ## DISTRACTION TEXT
            """
        )

        types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "table": "table that blocks sits on",
        }

        params = OrderedDict({"?b1": "block", "?b2": "block", "?a": "arm"})

        exp_preconditions = textwrap.dedent(
            """
            (and
                (holding ?a ?b1) ; The arm is holding the top block
                (clear ?b2) ; The bottom block is clear
            )
            """
        )

        preconditions, _, _, validation_info = (
            self.domain_builder.formalize_preconditions(
                model=self.mock_llm,
                domain_desc="",
                prompt_template="",
                action_name="stack",
                params=params,
                types=types,
                syntax_validator=self.syntax_validator,
                extract_new_preds=False,
            )
        )

        self.assertEqual(exp_preconditions.strip(), preconditions.strip())
        self.assertEqual(validation_info[0], True)

    def test_formalize_effects(self):
        self.syntax_validator.headers = ["Action Effects"]
        self.syntax_validator.error_types = [
            "validate_header",
            "validate_duplicate_headers",
            "validate_unsupported_keywords",
            "validate_params",
        ]

        self.mock_llm.output = textwrap.dedent(
            """
            ## DISTRACTION TEXT

            ### Action Preconditions
            ```
            (and
                (holding ?a ?b1) ; The arm is holding the top block
                (clear ?b2) ; The bottom block is clear
            )
            ```

            ## DISTRACTION TEXT

            ### Action Effects
            ```
            (and
                (not (holding ?a ?b1)) ; The arm is no longer holding the top block
                (on ?b1 ?b2) ; The top block is now on the bottom block
                (not (clear ?b2)) ; The bottom block is no longer clear
            )
            ```
            """
        )

        types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "table": "table that blocks sits on",
        }

        params = OrderedDict({"?b1": "block", "?b2": "block", "?a": "arm"})

        exp_effects = textwrap.dedent(
            """
            (and
                (not (holding ?a ?b1)) ; The arm is no longer holding the top block
                (on ?b1 ?b2) ; The top block is now on the bottom block
                (not (clear ?b2)) ; The bottom block is no longer clear
            )
            """
        )

        effects, _, _, validation_info = self.domain_builder.formalize_effects(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            action_name="stack",
            params=params,
            types=types,
            syntax_validator=self.syntax_validator,
            extract_new_preds=False,
        )

        self.assertEqual(self.normalize(exp_effects), self.normalize(effects))
        self.assertEqual(validation_info[0], True)

    def test_formalize_predicates(self):
        self.syntax_validator.headers = ["New Predicates"]
        self.syntax_validator.error_types = [
            "validate_types_predicates",
            "validate_format_predicates",
            "validate_duplicate_predicates",
        ]

        self.mock_llm.output = textwrap.dedent(
            """
            ## DISTRACTION TEXT

            ### New Predicates
            ```
            - (holding ?a - arm ?b - block): true if arm is holding a block
            - (clear ?b - block): true if a block does not have anything on top of it
            ```

            ## DISTRACTION TEXT
            """
        )

        types = {
            "arm": "arm for a robot",
            "block": "block that can be stacked and unstacked",
            "table": "table that blocks sits on",
        }

        predicates = [
            {
                "name": "on",
                "desc": "true if the block ?b1 is on top of the block ?b2",
                "raw": "- (on ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2",
                "params": OrderedDict({"?b1": "block", "?b2": "block"}),
                "clean": "(on ?b1 - block ?b2 - block)",
            }
        ]

        new_predicates, _, validation_info = self.domain_builder.formalize_predicates(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            types=types,
            predicates=predicates,
            syntax_validator=self.syntax_validator,
        )

        predicates.extend(new_predicates)

        exp_predicates = [
            {
                "name": "holding",
                "desc": "true if arm is holding a block",
                "raw": "- (holding ?a - arm ?b - block): true if arm is holding a block",
                "params": OrderedDict({"?a": "arm", "?b": "block"}),
                "clean": "(holding ?a - arm ?b - block)",
            },
            {
                "name": "clear",
                "desc": "true if a block does not have anything on top of it",
                "raw": "- (clear ?b - block): true if a block does not have anything on top of it",
                "params": OrderedDict({"?b": "block"}),
                "clean": "(clear ?b - block)",
            },
            {
                "name": "on",
                "desc": "true if the block ?b1 is on top of the block ?b2",
                "raw": "- (on ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2",
                "params": OrderedDict({"?b1": "block", "?b2": "block"}),
                "clean": "(on ?b1 - block ?b2 - block)",
            },
        ]

        self.assertCountEqual(exp_predicates, predicates)
        self.assertEqual(validation_info[0], True)

    def test_formalize_function(self):

        self.syntax_validator.error_types = ["validate_format_functions"]
        self.mock_llm.output = textwrap.dedent(
            """
            ## DISTRACTION TEXT

            ### FUNCTIONS
            ```
            - (battery-level ?r - robot): battery level of robot
            - (humidity ?loc - location): humidity of location
            ```

            ## DISTRACTION TEXT
            """
        )

        types = {"robot": "a robot", "location": "location to be at"}

        functions, _, validation_info = self.domain_builder.formalize_functions(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            types=types,
            syntax_validator=self.syntax_validator,
        )

        exp_functions = [
            {
                "name": "battery-level",
                "desc": "battery level of robot",
                "raw": "- (battery-level ?r - robot): battery level of robot",
                "params": OrderedDict([("?r", "robot")]),
                "clean": "(battery-level ?r - robot)",
            },
            {
                "name": "humidity",
                "desc": "humidity of location",
                "raw": "- (humidity ?loc - location): humidity of location",
                "params": OrderedDict([("?loc", "location")]),
                "clean": "(humidity ?loc - location)",
            },
        ]

        self.assertEqual(validation_info[0], True)
        self.assertEqual(exp_functions, functions)

    def test_formalize_constants(self):
        self.syntax_validator.error_types = ["validate_constant_types"]

        self.mock_llm.output = textwrap.dedent(
            """
            ### DISTRACTION TEXT
            ### CONSTANTS
            ```
            {
                'robot1': 'robot',
                'robot2': 'robot',
                'charging-station': 'location',
                'storage-room': 'location',
                'loading-dock': 'location',
            }
            ```
            ### DISTRACTION TEXT
            """
        )

        types = {"robot": "a robot", "location": "location to be at"}

        exp_constants = {
            "robot1": "robot",
            "robot2": "robot",
            "charging-station": "location",
            "storage-room": "location",
            "loading-dock": "location",
        }

        constants, _, validation_info = self.domain_builder.formalize_constants(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            types=types,
            constants=None,
            syntax_validator=self.syntax_validator,
        )

        self.assertEqual(exp_constants, constants)
        self.assertEqual(validation_info[0], True)

    def test_formalize_domain_level_specs(self):

        self.mock_llm.output = textwrap.dedent(
            """
            ### TYPES
            ```
            [
                {
                    "arm": "'arm for a robot'",
                    "children": []
                },
                {
                    "block": "block that can be stacked and unstacked",
                    "children": [
                        {
                            "heavy_block": "a heavy block that cannot be picked up",
                            "children": []
                        },
                        {
                            "light_block": "a light block that can be picked up",
                            "children": []
                        }
                    ]
                },
                {
                    "table": "table that blocks sits on",
                    "children": []
                }
            ]
            ```
            
            ### CONSTANTS
            ```
            {
                'robot1': 'robot',
                'robot2': 'robot',
                'charging-station': 'location',
                'storage-room': 'location',
                'loading-dock': 'location',
            }
            ```
            
            ### New Predicates
            ```
            - (holding ?a - arm ?b - block): true if arm is holding a block
            - (clear ?b - block): true if a block does not have anything on top of it
            ```
            
            ### FUNCTIONS
            ```
            - (battery-level ?r - robot): battery level of robot
            - (humidity ?loc - location): humidity of location
            ```
            
            """
        )

        results, _, _ = self.domain_builder.formalize_domain_level_specs(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            formalize_types=True,
            formalize_constants=True,
            formalize_predicates=True,
            formalize_functions=True,
        )

        exp_types = [
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

        exp_constants = {
            "robot1": "robot",
            "robot2": "robot",
            "charging-station": "location",
            "storage-room": "location",
            "loading-dock": "location",
        }

        exp_preds = [
            {
                "name": "holding",
                "desc": "true if arm is holding a block",
                "raw": "- (holding ?a - arm ?b - block): true if arm is holding a block",
                "params": OrderedDict([("?a", "arm"), ("?b", "block")]),
                "clean": "(holding ?a - arm ?b - block)",
            },
            {
                "name": "clear",
                "desc": "true if a block does not have anything on top of it",
                "raw": "- (clear ?b - block): true if a block does not have anything on top of it",
                "params": OrderedDict([("?b", "block")]),
                "clean": "(clear ?b - block)",
            },
        ]

        exp_func = [
            {
                "name": "battery-level",
                "desc": "battery level of robot",
                "raw": "- (battery-level ?r - robot): battery level of robot",
                "params": OrderedDict([("?r", "robot")]),
                "clean": "(battery-level ?r - robot)",
            },
            {
                "name": "humidity",
                "desc": "humidity of location",
                "raw": "- (humidity ?loc - location): humidity of location",
                "params": OrderedDict([("?loc", "location")]),
                "clean": "(humidity ?loc - location)",
            },
        ]

        self.assertEqual(results["types"], exp_types)
        self.assertEqual(results["constants"], exp_constants)
        self.assertEqual(results["predicates"], exp_preds)
        self.assertEqual(results["functions"], exp_func)

    def test_generate_domain(self):

        domain = "test_domain"

        types = {"robot": "a robot", "location": "location to be at"}

        constants = {
            "robot1": "robot",
            "robot2": "robot",
            "charging-station": "location",
            "storage-room": "location",
            "loading-dock": "location",
        }

        predicates = [
            {
                "name": "at",
                "desc": "true if robot is at a location",
                "raw": "- (at ?r - robot ?l - location): true if robot is at a location",
                "params": OrderedDict({"?r": "robot", "?l": "location"}),
                "clean": "(at ?r - robot ?l - location)",
            },
            {
                "name": "connected",
                "desc": "true if location ?l1 is connected to location ?l2",
                "raw": "- (connected ?l1 ?l2 - location): true if location ?l1 is connected to location ?l2",
                "params": OrderedDict({"?l1": "location", "?l2": "location"}),
                "clean": "(connected ?l1 ?l2 - location)",
            },
        ]

        functions = [
            {
                "name": "battery-level",
                "desc": "battery level of robot",
                "raw": "(battery-level ?r - robot): battery level of robot",
                "params": OrderedDict([("?r", "robot")]),
                "clean": "(battery-level ?r - robot)",
            },
            {
                "name": "humidity",
                "desc": "humidity of location",
                "raw": "(humidity ?loc - location): humidity of location",
                "params": OrderedDict([("?loc", "location")]),
                "clean": "(humidity ?loc - location)",
            },
        ]

        actions = [
            {
                "name": "move",
                "params": {"?r": "robot", "?l1": "location", "?l2": "location"},
                "preconditions": "(and (at ?r ?l1) (connected ?l1 ?l2))",
                "effects": "(and (not (at ?r ?l1)) (at ?r ?l2))",
            },
            {
                "name": "pick",
                "params": {"?r": "robot", "?l": "location"},
                "preconditions": "(and (at ?r ?l) (not (holding ?r)))",
                "effects": "(holding ?r)",
            },
        ]

        requirements = [":strips", ":typing", ":numeric-fluents"]

        expected_output = textwrap.dedent(
            """
            (define (domain test_domain)
                (:requirements
                    :negative-preconditions :numeric-fluents :strips :typing)

                (:types 
                    location 
                    robot
                )
                
                (:constants 
                    robot1 - robot
                    robot2 - robot
                    charging-station - location
                    storage-room - location
                    loading-dock - location
                )

                (:predicates 
                    (at ?r - robot ?l - location)
                    (connected ?l1 ?l2 - location)
                )
                
                (:functions 
                    (battery-level ?r - robot)
                    (humidity ?loc - location)
                )

                (:action move
                    :parameters (
                        ?r - robot ?l1 ?l2 - location
                    )
                    :precondition
                        (and (at ?r ?l1) (connected ?l1 ?l2))
                    :effect
                        (and (not (at ?r ?l1)) (at ?r ?l2))
                )

                (:action pick
                    :parameters (
                        ?r - robot ?l - location
                    )
                    :precondition
                        (and (at ?r ?l) (not (holding ?r)))
                    :effect
                        (holding ?r)
                )
            )
            """
        )

        result = self.domain_builder.generate_domain(
            domain_name=domain,
            types=types,
            constants=constants,
            predicates=predicates,
            functions=functions,
            actions=actions,
            # requirements=requirements,
        )

        self.assertEqual(self.normalize(result), self.normalize(expected_output))


if __name__ == "__main__":
    unittest.main()
