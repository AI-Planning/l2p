import json
import textwrap
import unittest

from l2p import FeedbackBuilder
from l2p.utils.pddl_types import PDDLType, Predicate, Parameter
from tests.mock_llm import MockLLM


class TestFeedbackBuilderLlmDiagnose(unittest.TestCase):
    """Tests for FeedbackBuilder.llm_diagnose()."""

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

        artifact = PDDLType(name="rover", parent="vehicle")

        result, raw = self.fb.llm_diagnose(
            model=self.mock,
            artifact=artifact,
            description="a robot domain",
            errors="Type 'location' not found",
        )

        parsed = json.loads(result)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["summary"], "Missing type declaration")
        self.assertEqual(len(parsed["identified_errors"]), 1)
        self.assertEqual(parsed["identified_errors"][0]["error_type"], "UndeclaredType")
        self.assertEqual(len(parsed["repair_plan"]), 2)
        self.assertIn("<diagnosis>", raw)

    def test_diagnose_with_context(self):
        self.mock.output = textwrap.dedent("""\
        <diagnosis>
        {"summary": "OK", "identified_errors": [], "repair_plan": []}
        </diagnosis>""")

        artifact = PDDLType(name="rover", parent="vehicle")
        result, _ = self.fb.llm_diagnose(
            model=self.mock,
            artifact=artifact,
            description="test",
            errors="some error",
            types=[PDDLType(name="robot", parent="object")],
        )

        parsed = json.loads(result)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["summary"], "OK")

    def test_diagnose_custom_prompt(self):
        self.mock.output = textwrap.dedent("""\
        <diagnosis>
        {"summary": "custom", "identified_errors": [], "repair_plan": []}
        </diagnosis>""")

        artifact = PDDLType(name="rover", parent="vehicle")
        result, _ = self.fb.llm_diagnose(
            model=self.mock,
            artifact=artifact,
            description="test",
            errors="err",
            prompt_template="Custom template {description} {errors} {artifact} {context}",
        )
        parsed = json.loads(result)
        self.assertEqual(parsed["summary"], "custom")

    def test_diagnose_missing_xml_tag(self):
        self.mock.output = "no diagnosis tag here"
        with self.assertRaises(RuntimeError) as ctx:
            self.fb.llm_diagnose(
                model=self.mock,
                description="test",
                errors="err",
                artifact=PDDLType(name="rover", parent="vehicle"),
                max_retries=1,
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
        result, _ = self.fb.llm_diagnose(
            model=retry_mock,
            description="test",
            errors="err",
            artifact=PDDLType(name="rover", parent="vehicle"),
            max_retries=3,
        )
        parsed = json.loads(result)
        self.assertEqual(parsed["summary"], "ok")
        self.assertEqual(retry_mock.count, 2)


class TestFeedbackBuilderLlmEvaluate(unittest.TestCase):
    """Tests for FeedbackBuilder.llm_evaluate()."""

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
        result, raw = self.fb.llm_evaluate(
            model=self.mock,
            artifact=PDDLType(name="rover", parent="vehicle"),
            description="a robot domain",
        )
        parsed = json.loads(result)
        self.assertEqual(parsed["score"], 8)
        self.assertTrue(parsed["is_passing"])
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
        result, _ = self.fb.llm_evaluate(
            model=self.mock,
            artifact=PDDLType(name="rover", parent="vehicle"),
            description="test",
        )
        parsed = json.loads(result)
        self.assertFalse(parsed["is_passing"])
        self.assertEqual(len(parsed["critique"]), 1)

    def test_evaluate_with_context(self):
        self.mock.output = textwrap.dedent("""\
        <evaluation>
        {"score": 10, "is_passing": true, "critique": [], "missing_elements": []}
        </evaluation>""")
        result, _ = self.fb.llm_evaluate(
            model=self.mock,
            artifact=PDDLType(name="rover", parent="vehicle"),
            description="test",
            predicates=[
                Predicate(name="at", params=[Parameter(variable="?r", type="rover")])
            ],
        )
        parsed = json.loads(result)
        self.assertEqual(parsed["score"], 10)


class TestFeedbackBuilderLlmReflect(unittest.TestCase):
    """Tests for FeedbackBuilder.llm_reflect()."""

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
        result, raw = self.fb.llm_reflect(
            model=self.mock,
            artifact=PDDLType(name="rover", parent="vehicle"),
            description="rover domain",
            diagnosis="Missing vehicle type",
        )
        parsed = json.loads(result)
        self.assertEqual(
            parsed["lesson_learned"],
            "Always declare parent types before subtypes",
        )
        self.assertEqual(parsed["anti_pattern"], "Declaring rover before vehicle")
        self.assertIn("<reflection>", raw)

    def test_reflect_with_context(self):
        self.mock.output = textwrap.dedent("""\
        <reflection>
        {"context": "t", "lesson_learned": "l", "anti_pattern": "a", "correct_pattern": "c"}
        </reflection>""")
        result, _ = self.fb.llm_reflect(
            model=self.mock,
            artifact=PDDLType(name="rover", parent="vehicle"),
            description="d",
            diagnosis="diag",
            types=[PDDLType(name="robot", parent="object")],
        )
        parsed = json.loads(result)
        self.assertEqual(parsed["lesson_learned"], "l")


class TestFeedbackBuilderLlmRevise(unittest.TestCase):
    """Tests for FeedbackBuilder.llm_revise()."""

    def setUp(self):
        self.fb = FeedbackBuilder()
        self.mock = MockLLM()

    def test_revise_parses_into_pydantic_models(self):
        self.mock.output = textwrap.dedent("""\
        <types>
        [
            {"name": "robot", "parent": "object", "desc": "a robot"},
            {"name": "location", "parent": "object", "desc": "a location"}
        ]
        </types>""")
        corrected, raw = self.fb.llm_revise(
            model=self.mock,
            artifact=PDDLType(name="rover", parent="vehicle"),
            component_class=PDDLType,
            description="rover domain",
            diagnosis="Add location type",
        )
        self.assertIsInstance(corrected, dict)
        self.assertIn(PDDLType, corrected)
        self.assertIsInstance(corrected[PDDLType], list)
        self.assertEqual(len(corrected[PDDLType]), 2)
        self.assertEqual(corrected[PDDLType][0].name, "robot")
        self.assertEqual(corrected[PDDLType][1].name, "location")

    def test_revise_with_multiple_component_classes(self):
        self.mock.output = textwrap.dedent("""\
        <types>
        [{"name": "robot", "parent": "object"}]
        </types>
        <predicates>
        [{"name": "at", "params": [{"variable": "?r", "type": "robot"}]}]
        </predicates>""")
        corrected, _ = self.fb.llm_revise(
            model=self.mock,
            artifact=PDDLType(name="rover", parent="vehicle"),
            component_class=[PDDLType, Predicate],
            description="test",
            diagnosis="fix hierarchy and add predicate",
        )
        self.assertIn(PDDLType, corrected)
        self.assertIn(Predicate, corrected)
        self.assertEqual(len(corrected[PDDLType]), 1)
        self.assertEqual(len(corrected[Predicate]), 1)
        self.assertEqual(corrected[Predicate][0].name, "at")

    def test_revise_with_context(self):
        self.mock.output = textwrap.dedent("""\
        <types>
        []
        </types>""")
        corrected, _ = self.fb.llm_revise(
            model=self.mock,
            artifact=PDDLType(name="rover", parent="vehicle"),
            component_class=PDDLType,
            description="d",
            diagnosis="p",
            predicates=[Predicate(name="at", params=[])],
        )
        self.assertEqual(corrected[PDDLType], [])

    def test_revise_missing_xml_tag_raises(self):
        self.mock.output = "no tags here"
        with self.assertRaises(RuntimeError) as ctx:
            self.fb.llm_revise(
                model=self.mock,
                artifact=PDDLType(name="rover", parent="vehicle"),
                component_class=PDDLType,
                description="test",
                diagnosis="fix",
                max_retries=1,
            )
        self.assertIn("Max retries", str(ctx.exception))


class TestFeedbackBuilderLlmSelect(unittest.TestCase):
    """Tests for FeedbackBuilder.llm_select()."""

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
        candidates = [
            PDDLType(name="robot", parent="object"),
            PDDLType(name="robot", parent="object", desc="with location"),
            PDDLType(name="robot", parent="vehicle"),
        ]
        result, raw = self.fb.llm_select(
            model=self.mock,
            description="generate types",
            candidates=candidates,
        )
        parsed = json.loads(result)
        self.assertEqual(parsed["best_candidate_id"], "candidate_2")
        self.assertIn("candidate_1", parsed["rejected_candidates_flaws"])
        self.assertIn("<selection>", raw)

    def test_select_best_with_context(self):
        self.mock.output = textwrap.dedent("""\
        <selection>
        {"best_candidate_id": "a", "selection_reasoning": "r", "rejected_candidates_flaws": {}}
        </selection>""")
        candidates = [PDDLType(name="robot", parent="object")]
        result, _ = self.fb.llm_select(
            model=self.mock,
            description="p",
            candidates=candidates,
            types=[PDDLType(name="robot", parent="object")],
        )
        parsed = json.loads(result)
        self.assertEqual(parsed["best_candidate_id"], "a")


class TestFeedbackBuilderLlmDiagnosePlan(unittest.TestCase):
    """Tests for FeedbackBuilder.llm_diagnose_plan()."""

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
        result, raw = self.fb.llm_diagnose_plan(
            model=self.mock,
            domain="(define (domain test) ...)",
            problem="(define (problem p) ...)",
            plan_error="ff: goal can be simplified to FALSE",
        )
        parsed = json.loads(result)
        self.assertIn("Robot cannot reach", parsed["failure_point"])
        self.assertEqual(len(parsed["recommended_fix"]), 1)
        self.assertIn("<plan_diagnosis>", raw)


class TestFeedbackBuilderLlmEvaluatePlan(unittest.TestCase):
    """Tests for FeedbackBuilder.llm_evaluate_plan()."""

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
        result, raw = self.fb.llm_evaluate_plan(
            model=self.mock,
            description="rover should reach waypoint3",
            domain="(define (domain test) ...)",
            problem="(define (problem p) ...)",
            plan="0.0: (move rover1 wp1 wp2) [1.0]",
        )
        parsed = json.loads(result)
        self.assertTrue(parsed["is_aligned"])
        self.assertIn("rover", parsed["semantic_analysis"])
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
        result, _ = self.fb.llm_evaluate_plan(
            model=self.mock,
            description="robot should only move along connected paths",
            domain="...",
            problem="...",
            plan="...",
        )
        parsed = json.loads(result)
        self.assertFalse(parsed["is_aligned"])
        self.assertEqual(len(parsed["identified_loopholes"]), 1)

    def test_plan_evaluate_with_context(self):
        self.mock.output = textwrap.dedent("""\
        <plan_evaluation>
        {"is_aligned": true, "semantic_analysis": "ok", "identified_loopholes": [], "domain_improvement_suggestions": []}
        </plan_evaluation>""")
        result, _ = self.fb.llm_evaluate_plan(
            model=self.mock,
            description="d",
            domain="dom",
            problem="prob",
            plan="plan",
        )
        parsed = json.loads(result)
        self.assertTrue(parsed["is_aligned"])


class TestFeedbackBuilderNormalizeArtifacts(unittest.TestCase):
    """Tests for FeedbackBuilder.normalize_artifacts()."""

    def setUp(self):
        self.fb = FeedbackBuilder()

    def test_normalize_single_model(self):
        artifact = PDDLType(name="rover", parent="vehicle")
        result = self.fb.normalize_artifacts(artifact)
        self.assertIn("<types>", result)
        self.assertIn("</types>", result)
        self.assertIn("rover", result)

    def test_normalize_list_of_models(self):
        artifacts = [
            PDDLType(name="rover", parent="vehicle"),
            PDDLType(name="location", parent="object"),
        ]
        result = self.fb.normalize_artifacts(artifacts)
        self.assertIn("<types>", result)
        self.assertIn("rover", result)
        self.assertIn("location", result)

    def test_normalize_grouped_by_type(self):
        artifacts = [
            PDDLType(name="rover", parent="vehicle"),
            Predicate(name="at", params=[]),
        ]
        result = self.fb.normalize_artifacts(artifacts)
        self.assertIn("<types>", result)
        self.assertIn("<predicates>", result)
        self.assertIn("rover", result)
        self.assertIn("at", result)

    def test_normalize_empty_list_raises(self):
        with self.assertRaises(ValueError):
            self.fb.normalize_artifacts([])

    def test_normalize_unsupported_type_raises(self):
        with self.assertRaises(ValueError):
            self.fb.normalize_artifacts("unsupported string")


class TestFeedbackBuilderCustomPrompt(unittest.TestCase):
    """Tests for custom prompt_template override."""

    def setUp(self):
        self.fb = FeedbackBuilder()
        self.mock = MockLLM()

    def test_custom_template_in_evaluate(self):
        self.mock.output = "<evaluation>\n{}\n</evaluation>"
        result, _ = self.fb.llm_evaluate(
            model=self.mock,
            artifact=PDDLType(name="rover", parent="vehicle"),
            description="test",
            prompt_template="Custom: {description} {artifact} {context}",
        )
        self.assertEqual(result, "{}")


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

    def test_custom_subclass_uses_inherited_methods(self):
        class CustomEval(FeedbackBuilder):
            pass

        fb = CustomEval()
        self.mock.output = textwrap.dedent("""\
        <evaluation>
        {"score": 7, "is_passing": true, "critique": [], "missing_elements": []}
        </evaluation>""")
        result, _ = fb.llm_evaluate(
            model=self.mock,
            artifact=PDDLType(name="rover", parent="vehicle"),
            description="test",
        )
        parsed = json.loads(result)
        self.assertEqual(parsed["score"], 7)


class TestFeedbackBuilderEdgeCases(unittest.TestCase):
    """Edge cases and error handling."""

    def setUp(self):
        self.fb = FeedbackBuilder()
        self.mock = MockLLM()

    def test_max_retries_exceeded_raises(self):
        self.mock.output = "always bad output"
        with self.assertRaises(RuntimeError) as ctx:
            self.fb.llm_diagnose(
                model=self.mock,
                artifact=PDDLType(name="rover", parent="vehicle"),
                description="test",
                errors="err",
                max_retries=2,
            )
        self.assertIn("Max retries", str(ctx.exception))

    def test_no_model_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.fb.llm_diagnose(
                model=None,
                artifact=PDDLType(name="rover", parent="vehicle"),
                description="test",
                errors="err",
            )
        self.assertIn("LLM instance must be provided", str(ctx.exception))

    def test_non_json_in_xml_returns_raw_string(self):
        raw_text = "plain text inside tags, not JSON"
        self.mock.output = f"<diagnosis>{raw_text}</diagnosis>"
        result, _ = self.fb.llm_diagnose(
            model=self.mock,
            artifact=PDDLType(name="rover", parent="vehicle"),
            description="test",
            errors="err",
            max_retries=1,
        )
        self.assertEqual(result, raw_text)

    def test_llm_output_preserved_in_result(self):
        expected_raw = '<diagnosis>\n{"summary": "test", "identified_errors": [], "repair_plan": []}\n</diagnosis>'
        self.mock.output = expected_raw
        _, raw = self.fb.llm_diagnose(
            model=self.mock,
            artifact=PDDLType(name="rover", parent="vehicle"),
            description="test",
            errors="err",
        )
        self.assertEqual(raw, expected_raw)

    def test_empty_diagnosis_result(self):
        self.mock.output = "<diagnosis>{}</diagnosis>"
        result, _ = self.fb.llm_diagnose(
            model=self.mock,
            artifact=PDDLType(name="rover", parent="vehicle"),
            description="test",
            errors="err",
            max_retries=1,
        )
        self.assertEqual(result, "{}")

    def test_revise_retry_on_bad_output(self):
        class RetryReviseMock(MockLLM):
            def __init__(self):
                super().__init__()
                self.count = 0

            def query(self, prompt):
                self.count += 1
                if self.count <= 1:
                    return "bad output"
                return '<types>\n[{"name": "fixed", "parent": "object"}]\n</types>'

        retry = RetryReviseMock()
        corrected, _ = self.fb.llm_revise(
            model=retry,
            artifact=PDDLType(name="rover", parent="vehicle"),
            component_class=PDDLType,
            description="test",
            diagnosis="fix it",
            max_retries=3,
        )
        self.assertEqual(corrected[PDDLType][0].name, "fixed")
        self.assertEqual(retry.count, 2)


if __name__ == "__main__":
    unittest.main()
