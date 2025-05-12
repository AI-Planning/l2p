import unittest, textwrap
from collections import OrderedDict
from l2p.domain_builder import DomainBuilder
from l2p.utils.pddl_validator import SyntaxValidator
from l2p.utils.pddl_types import Predicate
from .mock_llm import MockLLM

class TestDomainBuilder(unittest.TestCase):
    def setUp(self):
        self.domain_builder = DomainBuilder()
        self.syntax_validator = SyntaxValidator()
        self.mock_llm = MockLLM()
        
    def normalize(self, string):
        return "\n".join(line.strip() for line in textwrap.dedent(string).strip().splitlines())

    def test_extract_types(self):

        self.syntax_validator.error_types = ['validate_format_types']
        self.mock_llm.output = textwrap.dedent(
            """
            ## OUTPUT
            {
                "arm": "arm for a robot",
                "block": "block that can be stacked and unstacked",
                "table": "table that blocks sits on",
            }
            """
        )

        expected_types = {
            'arm': 'arm for a robot', 'block': 
            'block that can be stacked and unstacked', 
            'table': 'table that blocks sits on'
            }
        
        types, _, validation_info = self.domain_builder.extract_types(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            syntax_validator=self.syntax_validator,
        )

        self.assertEqual(expected_types, types)
        self.assertEqual(validation_info[0], True)


    def test_extract_type_hierarchy(self):
        self.syntax_validator.error_types = ['validate_format_types', 'validate_cyclic_types']
        self.mock_llm.output = textwrap.dedent(
            """
            ## OUTPUT
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
            """)

        expected_types = [
            {"arm": "'arm for a robot'", "children": []},
            {"block": "block that can be stacked and unstacked",
                "children": [{"heavy_block": "a heavy block that cannot be picked up","children": []},
                             {"light_block": "a light block that can be picked up","children": []}]},
            {"table": "table that blocks sits on","children": []}
            ]
        
        types, _, validation_info = self.domain_builder.extract_type_hierarchy(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            syntax_validator=self.syntax_validator,
        )

        self.assertEqual(expected_types, types)
        self.assertEqual(validation_info[0], True)


    def test_extract_pddl_action(self):

        self.syntax_validator.error_types = [
            'validate_header', 'validate_duplicate_headers', 'validate_unsupported_keywords',
            'validate_params', 'validate_duplicate_predicates', 'validate_types_predicates',
            'validate_format_predicates', 'validate_usage_predicates'
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
            Predicate({'name': 'on', 
                    'desc': 'true if the block ?b1 is on top of the block ?b2', 
                    'raw': '- (on ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2', 
                    'params': OrderedDict([('?b1', 'block'), ('?b2', 'block')]), 
                    'clean': '(on ?b1 - block ?b2 - block)'}),
            Predicate({'name': 'holding', 
                    'desc': 'true if arm is holding a block', 
                    'raw': '- (holding ?a - arm ?b - block): true if arm is holding a block', 
                    'params': OrderedDict([('?a', 'arm'), ('?b', 'block')]), 
                    'clean': '(holding ?a - arm ?b - block)'}),
            Predicate({'name': 'clear', 
                    'desc': 'true if a block does not have anything on top of it', 
                    'raw': '- (clear ?b - block): true if a block does not have anything on top of it', 
                    'params': OrderedDict([('?b', 'block')]), 
                    'clean': '(clear ?b - block)'}),
            ]
        
        exp_action = {
            'name': 'stack', 
            'params': OrderedDict({'?b1': 'block', '?b2': 'block', '?a': 'arm'}), 
            'preconditions': '(and\n    (holding ?a ?b1) ; The arm is holding the top block\n    (clear ?b2) ; The bottom block is clear\n)', 'effects': '(and\n    (not (holding ?a ?b1)) ; The arm is no longer holding the top block\n    (on ?b1 ?b2) ; The top block is now on the bottom block\n    (not (clear ?b2)) ; The bottom block is no longer clear\n)'}

        types = {
            'arm': 'arm for a robot', 'block': 
            'block that can be stacked and unstacked', 
            'table': 'table that blocks sits on'
            }
        
        action, new_predicates, _, validation_info = self.domain_builder.extract_pddl_action(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            types=types,
            syntax_validator=self.syntax_validator,
            action_name="stack"
        )

        self.assertEqual(exp_action, action)
        self.assertEqual(exp_predicates, new_predicates)
        self.assertEqual(validation_info[0], True)


    def test_extract_parameters(self):

        self.syntax_validator.headers = ['Action Parameters']
        self.syntax_validator.error_types = [
            'validate_header', 'validate_duplicate_headers', 
            'validate_unsupported_keywords', 'validate_params'
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
            'arm': 'arm for a robot', 'block': 
            'block that can be stacked and unstacked', 
            'table': 'table that blocks sits on'
            }
        
        exp_params = OrderedDict({'?b1': 'block', '?b2': 'block', '?a': 'arm'})
        
        params, _, _, validation_info = self.domain_builder.extract_parameters(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            action_name="stack",
            types=types,
            syntax_validator=self.syntax_validator
        )

        self.assertEqual(exp_params, params)
        self.assertEqual(validation_info[0], True)


    def test_extract_preconditions(self):
        self.syntax_validator.headers = ['Action Preconditions']
        self.syntax_validator.error_types = [
            'validate_header', 'validate_duplicate_headers', 
            'validate_unsupported_keywords', 'validate_params'
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
            'arm': 'arm for a robot', 'block': 
            'block that can be stacked and unstacked', 
            'table': 'table that blocks sits on'
            }
        
        params = OrderedDict({'?b1': 'block', '?b2': 'block', '?a': 'arm'})

        exp_preconditions = textwrap.dedent(
            """
            (and
                (holding ?a ?b1) ; The arm is holding the top block
                (clear ?b2) ; The bottom block is clear
            )
            """
        )
        
        preconditions, _, _, validation_info = self.domain_builder.extract_preconditions(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            action_name="stack",
            params=params,
            types=types,
            syntax_validator=self.syntax_validator,
            parse_predicates=False
        )

        self.assertEqual(exp_preconditions.strip(), preconditions.strip())
        self.assertEqual(validation_info[0], True)


    def test_extract_effects(self):
        self.syntax_validator.headers = ['Action Effects']
        self.syntax_validator.error_types = [
            'validate_header', 'validate_duplicate_headers', 
            'validate_unsupported_keywords', 'validate_params'
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
            'arm': 'arm for a robot', 'block': 
            'block that can be stacked and unstacked', 
            'table': 'table that blocks sits on'
            }
        
        params = OrderedDict({'?b1': 'block', '?b2': 'block', '?a': 'arm'})

        exp_effects = textwrap.dedent(
            """
            (and
                (not (holding ?a ?b1)) ; The arm is no longer holding the top block
                (on ?b1 ?b2) ; The top block is now on the bottom block
                (not (clear ?b2)) ; The bottom block is no longer clear
            )
            """
        )
        
        effects, _, _, validation_info = self.domain_builder.extract_effects(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            action_name="stack",
            params=params,
            types=types,
            syntax_validator=self.syntax_validator,
            parse_predicates=False
        )

        self.assertEqual(self.normalize(exp_effects),self.normalize(effects))
        self.assertEqual(validation_info[0], True)


    def test_extract_predicates(self):
        self.syntax_validator.headers = ['New Predicates']
        self.syntax_validator.error_types = [
            'validate_types_predicates', 'validate_format_predicates', 
            'validate_duplicate_predicates', 
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
            'arm': 'arm for a robot', 'block': 
            'block that can be stacked and unstacked', 
            'table': 'table that blocks sits on'
            }
        
        predicates = [
            {'name': 'on', 
             'desc': 'true if the block ?b1 is on top of the block ?b2', 
             'raw': '- (on ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2', 
             'params': OrderedDict({'?b1': 'block', '?b2': 'block'}), 
             'clean': '(on ?b1 - block ?b2 - block)'}
             ]
        
        new_predicates, _, validation_info = self.domain_builder.extract_predicates(
            model=self.mock_llm,
            domain_desc="",
            prompt_template="",
            types=types,
            predicates=predicates,
            syntax_validator=self.syntax_validator
        )

        predicates.extend(new_predicates)

        exp_predicates = [
            {'name': 'holding', 
             'desc': 'true if arm is holding a block', 
             'raw': '- (holding ?a - arm ?b - block): true if arm is holding a block', 
             'params': OrderedDict({'?a': 'arm', '?b': 'block'}), 
             'clean': '(holding ?a - arm ?b - block)'}, 
            {'name': 'clear', 
             'desc': 'true if a block does not have anything on top of it', 
             'raw': '- (clear ?b - block): true if a block does not have anything on top of it', 
             'params': OrderedDict({'?b': 'block'}), 
             'clean': '(clear ?b - block)'},
            {'name': 'on', 
             'desc': 'true if the block ?b1 is on top of the block ?b2', 
             'raw': '- (on ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2', 
             'params': OrderedDict({'?b1': 'block', '?b2': 'block'}), 
             'clean': '(on ?b1 - block ?b2 - block)'}
        ]

        self.assertCountEqual(exp_predicates, predicates)
        self.assertEqual(validation_info[0], True)


    def test_generate_domain(self):

        domain = "test_domain"
        
        types = {
            'robot': 'a robot',
            'location': 'location to be at'
        }

        predicates = [
            {'name': 'at', 
             'desc': 'true if robot is at a location', 
             'raw': '- (at ?r - robot ?l - location): true if robot is at a location', 
             'params': OrderedDict({'?r': 'robot', '?l': 'location'}), 
             'clean': '(at ?r - robot ?l - location)'}, 
            {'name': 'connected', 
             'desc': 'true if location ?l1 is connected to location ?l2', 
             'raw': '- (connected ?l1 ?l2 - location): true if location ?l1 is connected to location ?l2', 
             'params': OrderedDict({'?l1': 'location', '?l2': 'location'}),
             'clean': '(connected ?l1 ?l2 - location)'}
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

        requirements = [":strips", ":typing"]

        expected_output = textwrap.dedent(
            """
            (define (domain test_domain)
                (:requirements
                    :strips :typing)

                (:types 
                    location robot - object
                )

                (:predicates 
                    (at ?r - robot ?l - location)
                    (connected ?l1 ?l2 - location)
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
            predicates=predicates,
            actions=actions,
            requirements=requirements,
        )

        self.assertEqual(self.normalize(result), self.normalize(expected_output))


if __name__ == "__main__":
    unittest.main()
