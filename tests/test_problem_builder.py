import unittest
import textwrap
import io
from contextlib import redirect_stdout

from l2p import ProblemBuilder, DomainBuilder
from l2p.utils.pddl_types import *
from tests.mock_llm import MockLLM


class TestProblemBuilderConstruction(unittest.TestCase):
    """Tests for ProblemBuilder __init__."""

    def test_default_construction(self):
        pb = ProblemBuilder()
        self.assertEqual(pb.problem_details.name, "problem-placeholder")
        self.assertEqual(pb.problem_details.domain_name, "domain-placeholder")

    def test_with_kwargs(self):
        pb = ProblemBuilder(name="prob1", domain_name="dom1")
        self.assertEqual(pb.problem_details.name, "prob1")
        self.assertEqual(pb.problem_details.domain_name, "dom1")

    def test_with_problem_details(self):
        pd = ProblemDetails(name="provided-prob", domain_name="provided-dom")
        pb = ProblemBuilder(problem_details=pd)
        self.assertIs(pb.problem_details, pd)

    def test_with_domain_details_sets_domain_name(self):
        dd = DomainDetails(name="robot-domain")
        pb = ProblemBuilder(domain_details=dd)
        self.assertEqual(pb.problem_details.domain_name, "robot-domain")

    def test_with_domain_details_and_explicit_name(self):
        dd = DomainDetails(name="robot-domain")
        pb = ProblemBuilder(name="prob1", domain_name="explicit-dom", domain_details=dd)
        self.assertEqual(pb.problem_details.name, "prob1")
        self.assertEqual(pb.problem_details.domain_name, "explicit-dom")


class TestProblemBuilderFormalize(unittest.TestCase):
    """Tests for ProblemBuilder.formalize_component()."""

    def setUp(self):
        self.builder = ProblemBuilder()
        self.mock = MockLLM()
        self.prompt = "Extract PDDL problem components:\n{problem_desc}\n{context_injection}"

    # ---- OBJECTS (list of items) ----

    def test_formalize_single_object(self):
        self.mock.output = textwrap.dedent("""\
        <objects>
        [
            {"name": "rover1", "type": "rover", "desc": "main rover"}
        ]
        </objects>""")
        result, output = self.builder.formalize_component(
            model=self.mock, component_class=PDDLObject,
            prompt_template=self.prompt, problem_desc="test")
        # Single item is unwrapped
        self.assertIsInstance(result, PDDLObject)
        self.assertEqual(result.name, "rover1")
        self.assertEqual(result.type, "rover")

    def test_formalize_multiple_objects(self):
        self.mock.output = textwrap.dedent("""\
        <objects>
        [
            {"name": "rover1", "type": "rover", "desc": null},
            {"name": "rover2", "type": "rover", "desc": null},
            {"name": "base", "type": "location", "desc": "base station"}
        ]
        </objects>""")
        result, _ = self.builder.formalize_component(
            model=self.mock, component_class=PDDLObject,
            prompt_template=self.prompt, problem_desc="test")
        # Multiple items stays as list
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].name, "rover1")
        self.assertEqual(result[2].type, "location")

    def test_formalize_empty_objects(self):
        self.mock.output = "<objects>\n[]\n</objects>"
        result, _ = self.builder.formalize_component(
            model=self.mock, component_class=PDDLObject,
            prompt_template=self.prompt, problem_desc="empty")
        self.assertEqual(result, [])

    def test_formalize_objects_with_mixed_types(self):
        self.mock.output = textwrap.dedent("""\
        <objects>
        [
            {"name": "a", "type": "block", "desc": null},
            {"name": "b", "type": "block", "desc": null},
            {"name": "arm1", "type": "arm", "desc": null}
        ]
        </objects>""")
        result, _ = self.builder.formalize_component(
            model=self.mock, component_class=PDDLObject,
            prompt_template=self.prompt, problem_desc="test")
        self.assertEqual(len(result), 3)
        types = {o.name: o.type for o in result}
        self.assertEqual(types["a"], "block")
        self.assertEqual(types["arm1"], "arm")

    # ---- INITIAL STATE (single object, unwrapped) ----

    def test_formalize_initial_state(self):
        self.mock.output = textwrap.dedent("""\
        <initial_states>
        {
            "facts": ["(at rover1 base)", "(= (battery rover1) 100.0)"],
            "timed_facts": [],
            "desc": null
        }
        </initial_states>""")
        result, _ = self.builder.formalize_component(
            model=self.mock, component_class=InitialState,
            prompt_template=self.prompt, problem_desc="test")
        self.assertIsInstance(result, InitialState)
        self.assertEqual(len(result.facts), 2)
        self.assertIn("(at rover1 base)", result.facts)

    def test_formalize_initial_state_with_timed_facts(self):
        self.mock.output = textwrap.dedent("""\
        <initial_states>
        {
            "facts": ["(clear a)"],
            "timed_facts": [
                {"time": 10.0, "fact": "(blackout)", "desc": null}
            ],
            "desc": null
        }
        </initial_states>""")
        result, _ = self.builder.formalize_component(
            model=self.mock, component_class=InitialState,
            prompt_template=self.prompt, problem_desc="test")
        self.assertEqual(len(result.timed_facts), 1)
        self.assertEqual(result.timed_facts[0].time, 10.0)
        self.assertEqual(result.timed_facts[0].fact, "(blackout)")

    def test_formalize_empty_initial_state(self):
        self.mock.output = textwrap.dedent("""\
        <initial_states>
        {"facts": [], "timed_facts": [], "desc": null}
        </initial_states>""")
        result, _ = self.builder.formalize_component(
            model=self.mock, component_class=InitialState,
            prompt_template=self.prompt, problem_desc="empty")
        self.assertEqual(len(result.facts), 0)
        self.assertEqual(len(result.timed_facts), 0)

    # ---- GOAL STATE (single object, unwrapped) ----

    def test_formalize_goal_state(self):
        self.mock.output = textwrap.dedent("""\
        <goal_states>
        {
            "conditions": ["(at rover1 waypoint3)", "(data-transmitted)"],
            "desc": null
        }
        </goal_states>""")
        result, _ = self.builder.formalize_component(
            model=self.mock, component_class=GoalState,
            prompt_template=self.prompt, problem_desc="test")
        self.assertIsInstance(result, GoalState)
        self.assertEqual(len(result.conditions), 2)
        self.assertIn("(at rover1 waypoint3)", result.conditions)

    def test_formalize_goal_with_logical_operators(self):
        self.mock.output = textwrap.dedent("""\
        <goal_states>
        {
            "conditions": [
                {"operator": "or", "conditions": ["(has-rock a)", "(has-soil a)"]},
                {"operator": "not", "condition": "(broken a)"}
            ],
            "desc": null
        }
        </goal_states>""")
        result, _ = self.builder.formalize_component(
            model=self.mock, component_class=GoalState,
            prompt_template=self.prompt, problem_desc="test")
        self.assertEqual(len(result.conditions), 2)
        first = result.conditions[0]
        self.assertIsInstance(first, dict)
        self.assertEqual(first["operator"], "or")

    def test_formalize_empty_goal(self):
        self.mock.output = textwrap.dedent("""\
        <goal_states>
        {"conditions": [], "desc": null}
        </goal_states>""")
        result, _ = self.builder.formalize_component(
            model=self.mock, component_class=GoalState,
            prompt_template=self.prompt, problem_desc="empty")
        self.assertEqual(len(result.conditions), 0)

    # ---- METRIC (single object, unwrapped) ----

    def test_formalize_metric_minimize(self):
        self.mock.output = textwrap.dedent("""\
        <metrics>
        {"optimization": "minimize", "expression": "total-time", "desc": null}
        </metrics>""")
        result, _ = self.builder.formalize_component(
            model=self.mock, component_class=Metric,
            prompt_template=self.prompt, problem_desc="test")
        self.assertIsInstance(result, Metric)
        self.assertEqual(result.optimization, "minimize")
        self.assertEqual(result.expression, "total-time")

    def test_formalize_metric_maximize(self):
        self.mock.output = textwrap.dedent("""\
        <metrics>
        {"optimization": "maximize", "expression": "(battery rover1)", "desc": null}
        </metrics>""")
        result, _ = self.builder.formalize_component(
            model=self.mock, component_class=Metric,
            prompt_template=self.prompt, problem_desc="test")
        self.assertEqual(result.optimization, "maximize")
        self.assertEqual(result.expression, "(battery rover1)")

    # ---- MULTI-CLASS EXTRACTION ----

    def test_formalize_multiple_components(self):
        self.mock.output = textwrap.dedent("""\
        <objects>
        [
            {"name": "r1", "type": "robot", "desc": null}
        ]
        </objects>
        <initial_states>
        {"facts": ["(at r1 base)"], "timed_facts": [], "desc": null}
        </initial_states>""")
        results, _ = self.builder.formalize_component(
            model=self.mock,
            component_class=[PDDLObject, InitialState],
            prompt_template="Extract problem components",
            problem_desc="test")
        self.assertIsInstance(results, dict)
        self.assertIn(PDDLObject, results)
        self.assertIn(InitialState, results)
        # PDDLObject should be a list
        self.assertIsInstance(results[PDDLObject], list)
        # InitialState should be a list
        self.assertIsInstance(results[InitialState], list)

    # ---- EDGE CASES ----

    def test_llm_output_preserved(self):
        self.mock.output = "<objects>\n[{\"name\": \"r1\", \"type\": \"robot\", \"desc\": null}]\n</objects>"
        result, llm_output = self.builder.formalize_component(
            model=self.mock, component_class=PDDLObject,
            prompt_template=self.prompt, problem_desc="test")
        self.assertIn("r1", llm_output)

    def test_retry_eventually_succeeds(self):
        class RetryMockLLM(MockLLM):
            def __init__(self):
                super().__init__()
                self.count = 0
            def query(self, prompt):
                self.count += 1
                if self.count == 1:
                    return "bad output"
                return "<objects>\n[{\"name\": \"r1\", \"type\": \"robot\", \"desc\": null}]\n</objects>"
        retry_mock = RetryMockLLM()
        result, _ = self.builder.formalize_component(
            model=retry_mock, component_class=PDDLObject,
            prompt_template=self.prompt, problem_desc="test", max_retries=3)
        self.assertEqual(result.name, "r1")


class TestProblemBuilderGenerateProblem(unittest.TestCase):
    """Tests for ProblemBuilder.generate_problem()."""

    def setUp(self):
        self.builder = ProblemBuilder()

    def normalize(self, s):
        return "\n".join(l.strip() for l in textwrap.dedent(s).strip().splitlines())

    def test_generate_problem_basic(self):
        problem = ProblemDetails(
            name="test-problem",
            domain_name="test-domain",
            objects=[
                PDDLObject(name="rover1", type="rover"),
                PDDLObject(name="base", type="location"),
            ],
            initial_state=InitialState(facts=["(at rover1 base)"]),
            goal_state=GoalState(conditions=["(at rover1 base)"])
        )
        result = self.builder.generate_problem(problem)
        self.assertIn("(define (problem test-problem)", result)
        self.assertIn("(:domain test-domain)", result)
        self.assertIn(":objects", result)
        self.assertIn("rover1 - rover", result)
        self.assertIn(":init", result)
        self.assertIn(":goal", result)

    def test_generate_problem_with_all_sections(self):
        problem = ProblemDetails(
            name="prob1",
            domain_name="dom1",
            objects=[
                PDDLObject(name="a", type="block"),
                PDDLObject(name="b", type="block"),
                PDDLObject(name="c", type="block"),
            ],
            initial_state=InitialState(
                facts=["(ontable a)", "(ontable b)", "(clear a)", "(clear b)"],
                timed_facts=[TimedFact(time=5.0, fact="(blackout)")]
            ),
            goal_state=GoalState(conditions=["(on a b)", "(on b c)"]),
            metric=Metric(optimization="minimize", expression="total-time"),
            constraint=[Constraint(condition={"operator": "always", "condition": "(> (battery r) 0)"})]
        )
        result = self.builder.generate_problem(problem)
        self.assertIn(":constraints", result)
        self.assertIn(":metric", result)
        self.assertIn("minimize", result)
        self.assertIn("total-time", result)
        self.assertIn("(at 5.0", result)  # timed fact

    def test_generate_problem_no_init_warning(self):
        problem = ProblemDetails(name="p", domain_name="d",
                                 goal_state=GoalState(conditions=["(p)"]))
        f = io.StringIO()
        with redirect_stdout(f):
            result = self.builder.generate_problem(problem)
        self.assertIn("WARNING", f.getvalue())

    def test_generate_problem_no_goal_warning(self):
        problem = ProblemDetails(name="p", domain_name="d",
                                 initial_state=InitialState(facts=["(p)"]))
        f = io.StringIO()
        with redirect_stdout(f):
            result = self.builder.generate_problem(problem)
        self.assertIn("WARNING", f.getvalue())

    def test_generate_problem_pddl_structure(self):
        problem = ProblemDetails(
            name="p1",
            domain_name="d1",
            objects=[PDDLObject(name="o1", type="t1")],
            initial_state=InitialState(facts=["(p o1)"]),
            goal_state=GoalState(conditions=["(q o1)"])
        )
        result = self.builder.generate_problem(problem)
        self.assertTrue(result.startswith("(define"))
        self.assertTrue(result.strip().endswith(")"))
        self.assertIn("(:domain d1)", result)

    def test_generate_problem_objects_grouped_by_type(self):
        problem = ProblemDetails(
            name="p",
            domain_name="d",
            objects=[
                PDDLObject(name="a", type="block"),
                PDDLObject(name="b", type="block"),
                PDDLObject(name="arm1", type="arm"),
            ],
            initial_state=InitialState(facts=[]),
            goal_state=GoalState(conditions=[])
        )
        result = self.builder.generate_problem(problem)
        # blocks should be grouped: "a b - block"
        self.assertIn("a b - block", result)
        self.assertIn("arm1 - arm", result)

    def test_generate_problem_with_goal_condition_block(self):
        problem = ProblemDetails(
            name="p",
            domain_name="d",
            objects=[PDDLObject(name="o", type="t")],
            initial_state=InitialState(facts=["(p o)"]),
            goal_state=GoalState(conditions=[{"operator": "not", "condition": "(p o)"}])
        )
        result = self.builder.generate_problem(problem)
        self.assertIn("(not", result)

    def test_generate_problem_goal_wrap_in_and(self):
        problem = ProblemDetails(
            name="p",
            domain_name="d",
            objects=[],
            initial_state=InitialState(facts=[]),
            goal_state=GoalState(conditions=["(p)", "(q)"])
        )
        result = self.builder.generate_problem(problem)
        # multiple conditions should be wrapped in and
        self.assertIn("(and", result)
        self.assertIn("(p)", result)
        self.assertIn("(q)", result)


class TestProblemBuilderSetters(unittest.TestCase):
    """Tests for ProblemBuilder set_* methods."""

    def setUp(self):
        self.builder = ProblemBuilder()

    def test_set_problem_name(self):
        self.builder.set_problem_name("my-problem")
        self.assertEqual(self.builder.problem_details.name, "my-problem")

    def test_set_problem_name_default(self):
        self.builder.set_problem_name()
        self.assertEqual(self.builder.problem_details.name, "problem-placeholder")

    def test_set_problem_desc(self):
        self.builder.set_problem_desc("A test problem")
        self.assertEqual(self.builder.problem_details.desc, "A test problem")

    def test_set_problem_desc_none(self):
        self.builder.set_problem_desc()
        self.assertIsNone(self.builder.problem_details.desc)

    def test_set_objects(self):
        o = PDDLObject(name="r1", type="robot")
        self.builder.set_objects(o)
        self.assertEqual(len(self.builder.problem_details.objects), 1)

    def test_set_objects_append(self):
        self.builder.set_objects(PDDLObject(name="a", type="t"))
        self.builder.set_objects(PDDLObject(name="b", type="t"), append=True)
        self.assertEqual(len(self.builder.problem_details.objects), 2)

    def test_set_objects_overwrite(self):
        self.builder.set_objects(PDDLObject(name="a", type="t"))
        self.builder.set_objects(PDDLObject(name="b", type="t"))
        self.assertEqual(len(self.builder.problem_details.objects), 1)

    def test_set_objects_none_clears(self):
        self.builder.set_objects(PDDLObject(name="a", type="t"))
        self.builder.set_objects(None)
        self.assertEqual(len(self.builder.problem_details.objects), 0)

    def test_set_initial_states(self):
        init = InitialState(facts=["(p)"])
        self.builder.set_initial_states(init)
        self.assertEqual(len(self.builder.problem_details.initial_state.facts), 1)

    def test_set_goal_states(self):
        goal = GoalState(conditions=["(q)"])
        self.builder.set_goal_states(goal)
        self.assertEqual(len(self.builder.problem_details.goal_state.conditions), 1)

    def test_set_metric(self):
        m = Metric(optimization="minimize", expression="total-time")
        self.builder.set_metric(m)
        self.assertEqual(self.builder.problem_details.metric.optimization, "minimize")

    def test_set_constraints(self):
        c = Constraint(condition={"operator": "always", "condition": "(p ?r)"})
        self.builder.set_constraints(c)
        self.assertEqual(len(self.builder.problem_details.constraint), 1)

    def test_set_objects_list(self):
        objs = [PDDLObject(name="a", type="t"), PDDLObject(name="b", type="t")]
        self.builder.set_objects(objs)
        self.assertEqual(len(self.builder.problem_details.objects), 2)


if __name__ == "__main__":
    unittest.main()
