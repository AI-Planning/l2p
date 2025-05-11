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

    # TODO: implement test task builder functions
    def test_extract_objects(self):
        pass

    def test_extract_initial_state(self):
        pass

    def test_extract_goal_state(self):
        pass

    def test_extract_task(self):
        pass

    def test_extract_nl_conditions(self):
        pass

    def test_generate_task(self):
        pass

    def test_format_action(self):
        pass

    def test_format_objects(self):
        pass

    def test_format_initial(self):
        pass

    def test_format_goal(self):
        pass


if __name__ == "__main__":
    unittest.main()
