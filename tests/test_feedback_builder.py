import json
import textwrap
import unittest

from l2p import FeedbackBuilder
from l2p.utils.pddl_types import PDDLType
from tests.mock_llm import MockLLM


class TestFeedbackBuilderDiagnose(unittest.TestCase):
    """Tests for FeedbackBuilder.diagnose()."""

    def setUp(self):
        self.fb = FeedbackBuilder()
        self.mock = MockLLM()

    def test_diagnose_parses_output(self):
        self.mock.output = textwrap.dedent("""\
        <diagnosis>
        {
            "summary": "Missing type declaration",
            "identified_errors": [
                {
                    "error_type": "UndeclaredType",
                    "location_in_json": "params[0].type",
                    "validator_message": "Type 'location' not found",
                    "root_cause_analysis": "Forgot to add location to types"
                }
            ],
            "repair_plan": [
                "Add location to the types list",
                "Re-generate the component"
            ]
        }
        </diagnosis>""")
        result, raw = self.fb.diagnose(
            model=self.mock,
            description="a robot domain",
            errors="Type 'location' not found",
            generated_output='[{"name": "rover", "parent": "vehicle"}]'
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(result["summary"], "Missing type declaration")
        self.assertEqual(len(result["identified_errors"]), 1)
        self.assertEqual(result["identified_errors"][0]["error_type"], "UndeclaredType")
        self.assertEqual(len(result["repair_plan"]), 2)
        self.assertIn("<diagnosis>", raw)

    def test_diagnose_with_context(self):
        self.mock.output = textwrap.dedent("""\
        <diagnosis>
        {"summary": "OK", "identified_errors": [], "repair_plan": []}
        </diagnosis>""")
        result, _ = self.fb.diagnose(
            model=self.mock,
            description="test",
            errors="some error",
            generated_output="output",
            types=[PDDLType(name="robot", parent="object")]
        )
        self.assertEqual(result["summary"], "OK")

    def test_diagnose_custom_prompt(self):
        self.mock.output = textwrap.dedent("""\
        <diagnosis>
        {"summary": "custom", "identified_errors": [], "repair_plan": []}
        </diagnosis>""")
        result, _ = self.fb.diagnose(
            model=self.mock,
            description="test",
            errors="err",
            generated_output="out",
            prompt_template="Custom template {description} {errors} {generated_output} {context}"
        )
        self.assertEqual(result["summary"], "custom")

    def test_diagnose_missing_xml_tag(self):
        self.mock.output = "no diagnosis tag here"
        with self.assertRaises(RuntimeError) as ctx:
            self.fb.diagnose(
                model=self.mock,
                description="test",
                errors="err",
                generated_output="out",
                max_retries=1
            )
        self.assertIn("Max retries", str(ctx.exception))

    def test_diagnose_retry_then_succeeds(self):
        class RetryMock(MockLLM):
            def __init__(self):
                super().__init__()
                self.count = 0
            def query(self, prompt):
                self.count += 1
                if self.count <= 1:
                    return "bad output"
                return '<diagnosis>{"summary": "ok", "identified_errors": [], "repair_plan": []}</diagnosis>'
        retry_mock = RetryMock()
        result, _ = self.fb.diagnose(
            model=retry_mock,
            description="test",
            errors="err",
            generated_output="out",
            max_retries=3
        )
        self.assertEqual(result["summary"], "ok")
        self.assertEqual(retry_mock.count, 2)


class TestFeedbackBuilderEvaluate(unittest.TestCase):
    """Tests for FeedbackBuilder.evaluate()."""

    def setUp(self):
        self.fb = FeedbackBuilder()
        self.mock = MockLLM()

    def test_evaluate_parses_output(self):
        self.mock.output = textwrap.dedent("""\
        <evaluation>
        {
            "score": 8,
            "is_passing": true,
            "critique": [],
            "missing_elements": []
        }
        </evaluation>""")
        result, raw = self.fb.evaluate(
            model=self.mock,
            description="a robot domain",
            generated_output="some PDDL output"
        )
        self.assertEqual(result["score"], 8)
        self.assertTrue(result["is_passing"])
        self.assertIn("<evaluation>", raw)

    def test_evaluate_failing(self):
        self.mock.output = textwrap.dedent("""\
        <evaluation>
        {
            "score": 3,
            "is_passing": false,
            "critique": ["Missing required predicates"],
            "missing_elements": ["(at ?r ?l) predicate"]
        }
        </evaluation>""")
        result, _ = self.fb.evaluate(
            model=self.mock,
            description="test",
            generated_output="output"
        )
        self.assertFalse(result["is_passing"])
        self.assertEqual(len(result["critique"]), 1)

    def test_evaluate_with_context(self):
        self.mock.output = textwrap.dedent("""\
        <evaluation>
        {"score": 10, "is_passing": true, "critique": [], "missing_elements": []}
        </evaluation>""")
        result, _ = self.fb.evaluate(
            model=self.mock,
            description="test",
            generated_output="out",
            predicates=["pred1"]
        )
        self.assertEqual(result["score"], 10)


class TestFeedbackBuilderReflect(unittest.TestCase):
    """Tests for FeedbackBuilder.reflect()."""

    def setUp(self):
        self.fb = FeedbackBuilder()
        self.mock = MockLLM()

    def test_reflect_parses_output(self):
        self.mock.output = textwrap.dedent("""\
        <reflection>
        {
            "context": "Failed to generate types",
            "lesson_learned": "Always declare parent types before subtypes",
            "anti_pattern": "Declaring rover before vehicle",
            "correct_pattern": "Declare vehicle first, then rover : vehicle"
        }
        </reflection>""")
        result, raw = self.fb.reflect(
            model=self.mock,
            description="rover domain",
            diagnosis="Missing vehicle type",
            generated_output="some output"
        )
        self.assertEqual(result["lesson_learned"], "Always declare parent types before subtypes")
        self.assertEqual(result["anti_pattern"], "Declaring rover before vehicle")
        self.assertIn("<reflection>", raw)

    def test_reflect_with_context(self):
        self.mock.output = textwrap.dedent("""\
        <reflection>
        {"context": "t", "lesson_learned": "l", "anti_pattern": "a", "correct_pattern": "c"}
        </reflection>""")
        result, _ = self.fb.reflect(
            model=self.mock,
            description="d",
            diagnosis="diag",
            generated_output="out",
            types=[PDDLType(name="robot", parent="object")]
        )
        self.assertEqual(result["lesson_learned"], "l")


class TestFeedbackBuilderRevise(unittest.TestCase):
    """Tests for FeedbackBuilder.revise()."""

    def setUp(self):
        self.fb = FeedbackBuilder()
        self.mock = MockLLM()

    def test_revise_returns_raw_json_string(self):
        self.mock.output = textwrap.dedent("""\
        <types>
        [
            {"name": "robot", "parent": "object", "desc": "a robot"},
            {"name": "location", "parent": "object", "desc": "a location"}
        ]
        </types>""")
        corrected, raw = self.fb.revise(
            model=self.mock,
            description="rover domain",
            repair_plan="Add location type",
            generated_output="[{\"name\": \"robot\"}]",
            xml_tag="types"
        )
        self.assertIsInstance(corrected, str)
        parsed = json.loads(corrected)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["name"], "robot")
        self.assertEqual(parsed[1]["name"], "location")

    def test_revise_respects_dynamic_xml_tag(self):
        self.mock.output = '<predicates>\n[{"name": "at", "params": []}]\n</predicates>'
        corrected, _ = self.fb.revise(
            model=self.mock,
            description="test",
            repair_plan="fix it",
            generated_output="old",
            xml_tag="predicates"
        )
        parsed = json.loads(corrected)
        self.assertEqual(parsed[0]["name"], "at")

    def test_revise_with_context(self):
        self.mock.output = '<actions>\n[]\n</actions>'
        corrected, _ = self.fb.revise(
            model=self.mock,
            description="d",
            repair_plan="p",
            generated_output="out",
            xml_tag="actions",
            predicates=["existing_pred"]
        )
        self.assertEqual(json.loads(corrected), [])


class TestFeedbackBuilderSelect(unittest.TestCase):
    """Tests for FeedbackBuilder.select_best()."""

    def setUp(self):
        self.fb = FeedbackBuilder()
        self.mock = MockLLM()

    def test_select_best_parses_output(self):
        self.mock.output = textwrap.dedent("""\
        <selection>
        {
            "best_candidate_id": "candidate_2",
            "selection_reasoning": "Candidate 2 has correct type declarations",
            "rejected_candidates_flaws": {
                "candidate_1": "Missing location type",
                "candidate_3": "Duplicate predicate declarations"
            }
        }
        </selection>""")
        result, raw = self.fb.select_best(
            model=self.mock,
            original_prompt="generate types",
            candidates="candidate_1: ..., candidate_2: ..., candidate_3: ..."
        )
        self.assertEqual(result["best_candidate_id"], "candidate_2")
        self.assertIn("candidate_1", result["rejected_candidates_flaws"])
        self.assertIn("<selection>", raw)

    def test_select_best_with_context(self):
        self.mock.output = textwrap.dedent("""\
        <selection>
        {"best_candidate_id": "a", "selection_reasoning": "r", "rejected_candidates_flaws": {}}
        </selection>""")
        result, _ = self.fb.select_best(
            model=self.mock,
            original_prompt="p",
            candidates="candidates",
            types=[PDDLType(name="robot", parent="object")]
        )
        self.assertEqual(result["best_candidate_id"], "a")


class TestFeedbackBuilderPlanDiagnosis(unittest.TestCase):
    """Tests for FeedbackBuilder.plan_diagnosis()."""

    def setUp(self):
        self.fb = FeedbackBuilder()
        self.mock = MockLLM()

    def test_plan_diagnosis_parses_output(self):
        self.mock.output = textwrap.dedent("""\
        <plan_diagnosis>
        {
            "failure_point": "Robot cannot reach waypoint3 because it is not connected",
            "suspected_component": "Problem / Initial State",
            "recommended_fix": [
                "Add (connected waypoint2 waypoint3) to the initial state"
            ]
        }
        </plan_diagnosis>""")
        result, raw = self.fb.plan_diagnosis(
            model=self.mock,
            domain_pddl="(define (domain test) ...)",
            problem_pddl="(define (problem p) ...)",
            planner_output="ff: goal can be simplified to FALSE"
        )
        self.assertIn("Robot cannot reach", result["failure_point"])
        self.assertEqual(len(result["recommended_fix"]), 1)
        self.assertIn("<plan_diagnosis>", raw)


class TestFeedbackBuilderPlanEvaluate(unittest.TestCase):
    """Tests for FeedbackBuilder.plan_evaluate()."""

    def setUp(self):
        self.fb = FeedbackBuilder()
        self.mock = MockLLM()

    def test_plan_evaluate_parses_output(self):
        self.mock.output = textwrap.dedent("""\
        <plan_evaluation>
        {
            "is_aligned": true,
            "semantic_analysis": "Plan correctly moves the rover to the target",
            "identified_loopholes": [],
            "domain_improvement_suggestions": []
        }
        </plan_evaluation>""")
        result, raw = self.fb.plan_evaluate(
            model=self.mock,
            description="rover should reach waypoint3",
            domain_pddl="(define (domain test) ...)",
            problem_pddl="(define (problem p) ...)",
            plan="0.0: (move rover1 wp1 wp2) [1.0]"
        )
        self.assertTrue(result["is_aligned"])
        self.assertIn("rover", result["semantic_analysis"])
        self.assertIn("<plan_evaluation>", raw)

    def test_plan_evaluate_not_aligned(self):
        self.mock.output = textwrap.dedent("""\
        <plan_evaluation>
        {
            "is_aligned": false,
            "semantic_analysis": "Plan uses a loophole",
            "identified_loopholes": ["Robot teleports without moving"],
            "domain_improvement_suggestions": ["Add (connected ?from ?to) precondition"]
        }
        </plan_evaluation>""")
        result, _ = self.fb.plan_evaluate(
            model=self.mock,
            description="robot should only move along connected paths",
            domain_pddl="...",
            problem_pddl="...",
            plan="..."
        )
        self.assertFalse(result["is_aligned"])
        self.assertEqual(len(result["identified_loopholes"]), 1)

    def test_plan_evaluate_with_context(self):
        self.mock.output = textwrap.dedent("""\
        <plan_evaluation>
        {"is_aligned": true, "semantic_analysis": "ok", "identified_loopholes": [], "domain_improvement_suggestions": []}
        </plan_evaluation>""")
        result, _ = self.fb.plan_evaluate(
            model=self.mock,
            description="d",
            domain_pddl="dom",
            problem_pddl="prob",
            plan="plan"
        )
        self.assertTrue(result["is_aligned"])


class TestFeedbackBuilderHooks(unittest.TestCase):
    """Tests for the three extension hooks."""

    def setUp(self):
        self.mock = MockLLM()

    def test_resolve_template_custom_returned_as_is(self):
        fb = FeedbackBuilder()
        result = fb.resolve_template("diagnosis", prompt_template="custom template")
        self.assertEqual(result, "custom template")

    def test_resolve_template_default_found(self):
        fb = FeedbackBuilder()
        result = fb.resolve_template("diagnosis")
        self.assertIn("## ROLE", result)
        self.assertIn("Diagnostic", result)

    def test_resolve_template_missing_raises(self):
        fb = FeedbackBuilder()
        with self.assertRaises(ValueError) as ctx:
            fb.resolve_template("nonexistent_feedback")
        self.assertIn("No default template", str(ctx.exception))

    def test_build_prompt_fills_placeholders(self):
        fb = FeedbackBuilder()
        result = fb.build_prompt("Hello {name}!", name="World")
        self.assertEqual(result, "Hello World!")

    def test_build_prompt_unknown_placeholder_unchanged(self):
        fb = FeedbackBuilder()
        result = fb.build_prompt("Hello {name}!", other="value")
        self.assertEqual(result, "Hello {name}!")

    def test_parse_result_extracts_and_parses_json(self):
        fb = FeedbackBuilder()
        result = fb.parse_result(
            '<diagnosis>{"summary": "ok"}</diagnosis>',
            "diagnosis"
        )
        self.assertEqual(result, {"summary": "ok"})

    def test_parse_result_missing_tag_raises(self):
        fb = FeedbackBuilder()
        with self.assertRaises(ValueError):
            fb.parse_result("no xml here", "diagnosis")

    def test_build_prompt_empty_template(self):
        fb = FeedbackBuilder()
        result = fb.build_prompt("", key="value")
        self.assertEqual(result, "")


class TestFeedbackBuilderCustomSubclass(unittest.TestCase):
    """Tests for subclassing FeedbackBuilder."""

    def setUp(self):
        self.mock = MockLLM()

    def test_custom_method(self):
        class MyFeedback(FeedbackBuilder):
            def token_count(self, model, text):
                raw = model.query(f"Count tokens in: {text}")
                return {"token_count": len(raw.split())}, raw

        fb = MyFeedback()
        self.mock.output = "this is a test response"
        result, raw = fb.token_count(self.mock, "hello world")
        self.assertEqual(result["token_count"], 5)
        self.assertEqual(raw, "this is a test response")

    def test_override_parse_result(self):
        class CustomParse(FeedbackBuilder):
            def parse_result(self, llm_output, xml_tag):
                return {"custom": "parsed", "length": len(llm_output)}

        fb = CustomParse()
        self.mock.output = "<diagnosis>some data</diagnosis>"
        result, _ = fb.diagnose(
            model=self.mock,
            description="test",
            errors="err",
            generated_output="out",
            max_retries=1
        )
        self.assertEqual(result["custom"], "parsed")
        self.assertEqual(result["length"], len("<diagnosis>some data</diagnosis>"))

    def test_override_build_prompt(self):
        class CustomPrompt(FeedbackBuilder):
            def build_prompt(self, template, **placeholders):
                return f"CUSTOM: {placeholders.get('description', '')}"

            def parse_result(self, llm_output, xml_tag):
                return {"prompt_used": llm_output}

        class TrackMock(MockLLM):
            def __init__(self):
                super().__init__()
                self.last_prompt = ""
            def query(self, prompt):
                self.last_prompt = prompt
                return '<evaluation>{"prompt_used": "ok"}</evaluation>'

        fb = CustomPrompt()
        track = TrackMock()
        result, _ = fb.evaluate(
            model=track,
            description="my desc",
            generated_output="out",
            max_retries=1
        )
        self.assertIn("CUSTOM: my desc", track.last_prompt)

    def test_override_resolve_template(self):
        class CustomResolve(FeedbackBuilder):
            def resolve_template(self, feedback_type, prompt_template=None):
                return "Always this template"

        fb = CustomResolve()
        # Should use the overridden template resolution
        self.mock.output = "<diagnosis>{}</diagnosis>"
        result, _ = fb.diagnose(
            model=self.mock,
            description="test",
            errors="err",
            generated_output="out",
            max_retries=1
        )
        self.assertIsInstance(result, dict)


class TestFeedbackBuilderEdgeCases(unittest.TestCase):
    """Edge cases and error handling."""

    def setUp(self):
        self.fb = FeedbackBuilder()
        self.mock = MockLLM()

    def test_max_retries_exceeded_raises(self):
        self.mock.output = "always bad output"
        with self.assertRaises(RuntimeError) as ctx:
            self.fb.diagnose(
                model=self.mock,
                description="test",
                errors="err",
                generated_output="out",
                max_retries=2
            )
        self.assertIn("Max retries", str(ctx.exception))

    def test_no_model_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.fb.diagnose(
                model=None,
                description="test",
                errors="err",
                generated_output="out"
            )
        self.assertIn("LLM instance must be provided", str(ctx.exception))

    def test_non_json_in_xml(self):
        self.mock.output = "<diagnosis>not json</diagnosis>"
        with self.assertRaises(RuntimeError):
            self.fb.diagnose(
                model=self.mock,
                description="test",
                errors="err",
                generated_output="out",
                max_retries=1
            )

    def test_llm_output_preserved_in_result(self):
        expected_raw = "<diagnosis>\n{\"summary\": \"test\", \"identified_errors\": [], \"repair_plan\": []}\n</diagnosis>"
        self.mock.output = expected_raw
        _, raw = self.fb.diagnose(
            model=self.mock,
            description="test",
            errors="err",
            generated_output="out"
        )
        self.assertEqual(raw, expected_raw)

    def test_empty_diagnosis_result(self):
        self.mock.output = "<diagnosis>{}</diagnosis>"
        result, _ = self.fb.diagnose(
            model=self.mock,
            description="test",
            errors="err",
            generated_output="out",
            max_retries=1
        )
        self.assertEqual(result, {})

    def test_revise_empty_xml_tag(self):
        self.mock.output = "<anything>[]</anything>"
        corrected, _ = self.fb.revise(
            model=self.mock,
            description="test",
            repair_plan="fix",
            generated_output="old",
            xml_tag="anything"
        )
        self.assertEqual(corrected, "[]")


if __name__ == "__main__":
    unittest.main()
