import unittest, textwrap
from l2p.domain_builder import DomainBuilder
from l2p.utils.pddl_parser import load_file
from .mock_llm import MockLLM


class TestDomainBuilder(unittest.TestCase):
    def setUp(self):
        self.domain_builder = DomainBuilder()

    def test_extract_type(self):
        mock_llm_1 = MockLLM(
            [
                load_file(
                    "tests/test_prompts/test_domain_builder/test_extract_type/01.txt"
                )
            ]
        )
        mock_llm_2 = MockLLM(
            [
                load_file(
                    "tests/test_prompts/test_domain_builder/test_extract_type/02.txt"
                )
            ]
        )
        mock_llm_3 = MockLLM(
            [
                load_file(
                    "tests/test_prompts/test_domain_builder/test_extract_type/03.txt"
                )
            ]
        )

        types, _ = self.domain_builder.extract_type(
            model=mock_llm_1,
            domain_desc="Blocksworld is a...",
            prompt_template="Prompt template placeholder",
        )
        self.assertEqual(
            types,
            {
                "object": "Object is always root, everything is an object",
                "children": [
                    {
                        "arm": "mechanical arm that picks up and stacks blocks on other blocks or table.",
                        "children": [],
                    },
                    {
                        "block": "colored block that can be stacked or stacked on other blocks or table.",
                        "children": [],
                    },
                    {
                        "table": "surface where the blocks can be placed on top of.",
                        "children": [],
                    },
                ],
            },
        )

        with self.assertRaises(RuntimeError) as context:
            types, _ = self.domain_builder.extract_type(
                model=mock_llm_2,
                domain_desc="Blocksworld is a...",
                prompt_template="Prompt template placeholder",
            )
        self.assertIn("Max retries exceeded", str(context.exception))

    def test_extract_type_hierarchy(self):

        mock_llm_1 = MockLLM(
            [
                load_file(
                    "tests/test_prompts/test_domain_builder/test_extract_type_hierarchy/01.txt"
                )
            ]
        )
        mock_llm_2 = MockLLM(
            [
                load_file(
                    "tests/test_prompts/test_domain_builder/test_extract_type_hierarchy/02.txt"
                )
            ]
        )
        mock_llm_3 = MockLLM(
            [
                load_file(
                    "tests/test_prompts/test_domain_builder/test_extract_type_hierarchy/03.txt"
                )
            ]
        )

        expected_hierarchy = {
            "object": "Object is always root, everything is an object",
            "children": [
                {
                    "worker": "A type of object consisting of humans who do things.",
                    "children": [
                        {"administrator": "A type of worker.", "children": []},
                        {"general_worker": "A type of worker.", "children": []},
                    ],
                },
                {
                    "order": "A type of object consisting of instructions.",
                    "children": [],
                },
                {"vehicle": "A type of object consisting of vehicles.", "children": []},
                {
                    "house_component": "A type of object consisting of the components of a house.",
                    "children": [
                        {"wall": "A type of house_component.", "children": []},
                        {"floor": "A type of house_component.", "children": []},
                        {"roof": "A type of house_component.", "children": []},
                    ],
                },
                {
                    "location": "A type of object consisting of places which can be visited.",
                    "children": [
                        {
                            "house": "A type of location. ",
                            "children": [
                                {"mansion": "A type of house.", "children": []},
                                {"library": "A type of house.", "children": []},
                            ],
                        },
                        {"depot": "A type of location.", "children": []},
                    ],
                },
            ],
        }

        type_hierarchy, _ = self.domain_builder.extract_type_hierarchy(
            model=mock_llm_1,
            domain_desc="HouseConstruction is...",
            prompt_template="Prompt template placeholder",
        )
        self.assertEqual(type_hierarchy, expected_hierarchy)

        with self.assertRaises(RuntimeError) as context:
            type_hierarchy, _ = self.domain_builder.extract_type_hierarchy(
                model=mock_llm_2,
                domain_desc="HouseConstruction is...",
                prompt_template="Prompt template placeholder",
            )
        self.assertIn("Max retries exceeded", str(context.exception))

    def test_extract_pddl_action(self):
        pass

    def test_extract_pddl_actions(self):
        pass

    def test_extract_parameters(self):
        pass

    def test_extract_preconditions(self):
        pass

    def test_extract_effects(self):
        pass

    def test_extract_predicates(self):
        pass

    def test_generate_domain(self):

        domain = "test_domain"
        types = "robot location"
        predicates = "(at ?r - robot ?l - location)\n(connected ?l1 ?l2 - location)"
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
            """\
            (define (domain test_domain)
               (:requirements
                  :strips :typing)

               (:types 
                  robot location
               )

               (:predicates 
                  (at ?r - robot ?l - location)
                  (connected ?l1 ?l2 - location)
               )

               (:action move
                  :parameters (
                     ?r - robot
                     ?l1 - location
                     ?l2 - location
                  )
                  :precondition
                     (and (at ?r ?l1) (connected ?l1 ?l2))
                  :effect
                     (and (not (at ?r ?l1)) (at ?r ?l2))
               )

               (:action pick
                  :parameters (
                     ?r - robot
                     ?l - location
                  )
                  :precondition
                     (and (at ?r ?l) (not (holding ?r)))
                  :effect
                     (holding ?r)
               )
            )
        """
        ).strip()

        result = self.domain_builder.generate_domain(
            domain=domain,
            types=types,
            predicates=predicates,
            actions=actions,
            requirements=requirements,
        )

        self.assertEqual(result.strip(), expected_output.strip())


if __name__ == "__main__":
    unittest.main()
