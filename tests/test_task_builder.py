import unittest, textwrap
from l2p.task_builder import TaskBuilder
from l2p.utils import *
from l2p.utils.pddl_validator import SyntaxValidator
from l2p.utils.pddl_types import Predicate
from .mock_llm import MockLLM


class TestTaskBuilder(unittest.TestCase):
    def setUp(self):
        self.task_builder = TaskBuilder()
        self.syntax_validator = SyntaxValidator()
        self.mock_llm = MockLLM()

    def normalize(self, string):
        return "\n".join(
            line.strip() for line in textwrap.dedent(string).strip().splitlines()
        )

    def test_formalize_objects(self):

        self.syntax_validator.headers = ["OBJECTS"]
        self.syntax_validator.error_types = [
            "validate_header",
            "validate_duplicate_headers",
            "validate_unsupported_keywords",
            "validate_task_objects",
        ]
        self.mock_llm.output = textwrap.dedent(
            """
            ## DISTRACTION TEXT
            ### OBJECTS
            ```
            object1 - type_1
            object2 - type_2
            object3 - type_3
            ```
            ## DISTRACTION TEXT
            """
        )

        types = {"type_1": "", "type_2": "", "type_3": ""}

        objects, _, validation_info = self.task_builder.formalize_objects(
            model=self.mock_llm,
            problem_desc="",
            prompt_template="",
            types=types,
            syntax_validator=self.syntax_validator,
        )

        exp_objects = {"object1": "type_1", "object2": "type_2", "object3": "type_3"}

        self.assertEqual(validation_info[0], True)
        self.assertEqual(exp_objects, objects)

    def test_formalize_initial_state(self):

        self.syntax_validator.headers = ["INITIAL"]
        self.syntax_validator.error_types = [
            "validate_header",
            "validate_duplicate_headers",
            "validate_unsupported_keywords",
            "validate_task_states",
        ]
        self.mock_llm.output = textwrap.dedent(
            """
            ## DISTRACTION TEXT
            ### INITIAL
            ```
            (predicate_1 object1 object2) ; comment for initial state predicate 1
            (predicate_2 object3 object1) ; comment for initial state predicate 2
            (predicate_3 object1) ; comment for initial state predicate 3
            ```
            ## DISTRACTION TEXT
            """
        )

        types = {"type_1": "", "type_2": "", "type_3": ""}
        objects = {"object1": "type_1", "object2": "type_2", "object3": "type_3"}

        predicates = [
            Predicate(
                {
                    "name": "predicate_1",
                    "desc": "",
                    "raw": "(predicate_1 ?t1 ?t2): description",
                    "params": OrderedDict([("?t1", "type_1"), ("?t2", "type_2")]),
                    "clean": "(predicate_1 ?t1 ?t2",
                }
            ),
            Predicate(
                {
                    "name": "predicate_2",
                    "desc": "",
                    "raw": "(predicate_2 ?t1 ?t2): description",
                    "params": OrderedDict([("?t1", "type_3"), ("?t2", "type_1")]),
                    "clean": "(predicate_2 ?t1 ?t2",
                }
            ),
            Predicate(
                {
                    "name": "predicate_3",
                    "desc": "",
                    "raw": "(predicate_3 ?t1): description",
                    "params": OrderedDict([("?t1", "type_1")]),
                    "clean": "(predicate_3 ?t1",
                }
            ),
        ]

        initial, _, validation_info = self.task_builder.formalize_initial_state(
            model=self.mock_llm,
            problem_desc="",
            prompt_template="",
            types=types,
            predicates=predicates,
            objects=objects,
            syntax_validator=self.syntax_validator,
        )

        exp_initial = [
            {
                "pred_name": "predicate_1",
                "params": ["object1", "object2"],
                "neg": False,
            },
            {
                "pred_name": "predicate_2",
                "params": ["object3", "object1"],
                "neg": False,
            },
            {"pred_name": "predicate_3", "params": ["object1"], "neg": False},
        ]

        self.assertEqual(validation_info[0], True)
        self.assertEqual(exp_initial, initial)

    def test_formalize_task(self):
        self.syntax_validator.headers = ["INITIAL"]
        self.syntax_validator.error_types = [
            "validate_header",
            "validate_duplicate_headers",
            "validate_unsupported_keywords",
            "validate_task_states",
            "validate_task_objects",
            "validate_task_states",
        ]
        self.mock_llm.output = textwrap.dedent(
            """
            ## DISTRACTION TEXT
            ### OBJECTS
            ```
            object1 - type_1
            object2 - type_2
            object3 - type_3
            ```

            ## DISTRACTION TEXT
            ### INITIAL
            ```
            (predicate_1 object1 object2) ; comment for initial state predicate 1
            (predicate_2 object3 object1) ; comment for initial state predicate 2
            ```

            ## DISTRACTION TEXT
            ### GOAL
            ```
            (predicate_3 object1) ; comment for initial state predicate 3
            (not (predicate_2 object3 object1))
            ```
            ## DISTRACTION TEXT
            """
        )

        types = {"type_1": "", "type_2": "", "type_3": ""}

        predicates = [
            Predicate(
                {
                    "name": "predicate_1",
                    "desc": "",
                    "raw": "(predicate_1 ?t1 ?t2): description",
                    "params": OrderedDict([("?t1", "type_1"), ("?t2", "type_2")]),
                    "clean": "(predicate_1 ?t1 ?t2",
                }
            ),
            Predicate(
                {
                    "name": "predicate_2",
                    "desc": "",
                    "raw": "(predicate_2 ?t1 ?t2): description",
                    "params": OrderedDict([("?t1", "type_3"), ("?t2", "type_1")]),
                    "clean": "(predicate_2 ?t1 ?t2",
                }
            ),
            Predicate(
                {
                    "name": "predicate_3",
                    "desc": "",
                    "raw": "(predicate_3 ?t1): description",
                    "params": OrderedDict([("?t1", "type_1")]),
                    "clean": "(predicate_3 ?t1",
                }
            ),
        ]

        objects, initial, goal, _, validation_info = self.task_builder.formalize_task(
            model=self.mock_llm,
            problem_desc="",
            prompt_template="",
            types=types,
            predicates=predicates,
            syntax_validator=self.syntax_validator,
        )

        exp_objects = {"object1": "type_1", "object2": "type_2", "object3": "type_3"}

        exp_initial = [
            {
                "pred_name": "predicate_1",
                "params": ["object1", "object2"],
                "neg": False,
            },
            {
                "pred_name": "predicate_2",
                "params": ["object3", "object1"],
                "neg": False,
            },
        ]

        exp_goal = [
            {"pred_name": "predicate_3", "params": ["object1"], "neg": False},
            {"pred_name": "predicate_2", "params": ["object3", "object1"], "neg": True},
        ]

        self.assertEqual(validation_info[0], True)
        self.assertEqual(exp_objects, objects)
        self.assertEqual(exp_initial, initial)
        self.assertEqual(exp_goal, goal)

    def test_generate_task(self):

        objects = {"object1": "type_1", "object2": "type_2", "object3": "type_3"}
        initial = [
            {
                "pred_name": "predicate_1",
                "params": ["object1", "object2"],
                "neg": False,
            },
            {
                "pred_name": "predicate_2",
                "params": ["object3", "object1"],
                "neg": False,
            },
        ]
        goal = [{"pred_name": "predicate_3", "params": ["object1"], "neg": False}]

        expected_output = textwrap.dedent(
            """
            (define
                (problem test_problem)
                (:domain test_domain)

                (:objects 
                    object1 - type_1
                    object2 - type_2
                    object3 - type_3
                )

                (:init
                    (predicate_1 object1 object2)
                    (predicate_2 object3 object1)
                )

                (:goal
                    (and 
                        (predicate_3 object1) 
                    )
                )
            )
            """
        )

        result = self.task_builder.generate_task(
            domain_name="test_domain",
            problem_name="test_problem",
            objects=objects,
            initial=initial,
            goal=goal,
        )

        self.assertEqual(self.normalize(result), self.normalize(expected_output))


if __name__ == "__main__":
    unittest.main()
