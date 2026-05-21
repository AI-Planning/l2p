import unittest
from pydantic import BaseModel
from l2p.validators.base import (
    ValidationResult,
    FunctionalRule,
    SyntaxValidator,
    _extract_symbols,
    _verify_symbols,
    get_ordinal,
)
from l2p.utils.pddl_types import (
    PDDLType,
    Predicate,
    Parameter,
    Action,
    ActionPrecondition,
    ActionEffect,
)


class TestValidationResult(unittest.TestCase):

    def test_initial_state(self):
        r = ValidationResult()
        self.assertTrue(r.valid)
        self.assertEqual(r.errors, [])
        self.assertEqual(r.warnings, [])

    def test_add_error_flips_valid(self):
        r = ValidationResult()
        r.add_error("fatal")
        self.assertFalse(r.valid)
        self.assertEqual(r.errors, ["fatal"])

    def test_multiple_errors_cumulate(self):
        r = ValidationResult()
        r.add_error("e1")
        r.add_error("e2")
        self.assertFalse(r.valid)
        self.assertEqual(len(r.errors), 2)

    def test_add_warning_preserves_valid(self):
        r = ValidationResult()
        r.add_warning("caution")
        self.assertTrue(r.valid)
        self.assertEqual(r.warnings, ["caution"])

    def test_mixed_error_warning(self):
        r = ValidationResult()
        r.add_warning("warn")
        r.add_error("err")
        self.assertFalse(r.valid)
        self.assertEqual(len(r.errors), 1)
        self.assertEqual(len(r.warnings), 1)


class TestFunctionalRule(unittest.TestCase):

    def test_wraps_function(self):
        def my_fn(target, context):
            res = ValidationResult()
            if target == "bad":
                res.add_error("bad")
            return res

        rule = FunctionalRule(name="test", targets=[PDDLType], func=my_fn)
        self.assertEqual(rule.name, "test")
        self.assertEqual(rule.target_models, [PDDLType])

        self.assertTrue(rule.validate("good", {}).valid)
        self.assertFalse(rule.validate("bad", {}).valid)

    def test_multiple_targets(self):
        def dummy(t, c):
            return ValidationResult()

        rule = FunctionalRule(name="multi", targets=[PDDLType, Predicate], func=dummy)
        self.assertIn(PDDLType, rule.target_models)
        self.assertIn(Predicate, rule.target_models)


class TestSyntaxValidator(unittest.TestCase):

    def test_register_and_run(self):
        v = SyntaxValidator()

        def check_name(target, context):
            r = ValidationResult()
            if not getattr(target, "name", ""):
                r.add_error("empty name")
            return r

        v.register_rule(FunctionalRule("chk", [PDDLType], check_name))

        self.assertTrue(v.validate_component(PDDLType(name="r", parent="o"), {}).valid)
        self.assertFalse(v.validate_component(PDDLType(name="", parent="o"), {}).valid)

    def test_no_applicable_rules_returns_valid(self):
        v = SyntaxValidator()

        def dummy(t, c):
            return ValidationResult()

        v.register_rule(FunctionalRule("pred_only", [Predicate], dummy))
        r = v.validate_component(PDDLType(name="r", parent="o"), {})
        self.assertTrue(r.valid)

    def test_multiple_rules_on_same_target(self):
        v = SyntaxValidator()

        def r1(t, c):
            res = ValidationResult()
            if not getattr(t, "name", ""):
                res.add_error("no name")
            return res

        def r2(t, c):
            res = ValidationResult()
            if not getattr(t, "parent", ""):
                res.add_error("no parent")
            return res

        v.register_rule(FunctionalRule("r1", [PDDLType], r1))
        v.register_rule(FunctionalRule("r2", [PDDLType], r2))

        self.assertTrue(v.validate_component(PDDLType(name="r", parent="o"), {}).valid)
        self.assertFalse(v.validate_component(PDDLType(name="", parent="o"), {}).valid)
        self.assertFalse(v.validate_component(PDDLType(name="r", parent=""), {}).valid)
        r = v.validate_component(PDDLType(name="", parent=""), {})
        self.assertFalse(r.valid)
        self.assertEqual(len(r.errors), 2)

    def test_warnings_aggregated(self):
        v = SyntaxValidator()

        def warn(t, c):
            res = ValidationResult()
            res.add_warning("warning")
            return res

        v.register_rule(FunctionalRule("w", [PDDLType], warn))
        r = v.validate_component(PDDLType(name="x", parent="y"), {})
        self.assertEqual(len(r.warnings), 1)


class TestExtractSymbols(unittest.TestCase):

    def test_from_simple_string(self):
        s = _extract_symbols("(at robot1 location1)")
        self.assertIn("at", s)

    def test_from_string_multiple(self):
        s = _extract_symbols("(and (holding ?a ?b) (clear ?b2))")
        for sym in ("and", "holding", "clear"):
            self.assertIn(sym, s)

    def test_from_dict(self):
        data = {"operator": "and", "conditions": ["(holding ?a)", "(clear ?b)"]}
        s = _extract_symbols(data)
        self.assertIn("holding", s)
        self.assertIn("clear", s)

    def test_from_nested_dict(self):
        data = {
            "operator": "not",
            "condition": {
                "operator": "and",
                "conditions": ["(pred1 ?x)", "(pred2 ?y)"],
            },
        }
        s = _extract_symbols(data)
        self.assertIn("pred1", s)
        self.assertIn("pred2", s)

    def test_from_list(self):
        s = _extract_symbols(["(move ?a)", "(stop ?b)"])
        self.assertIn("move", s)
        self.assertIn("stop", s)

    def test_from_empty(self):
        self.assertEqual(_extract_symbols(""), set())
        self.assertEqual(_extract_symbols([]), set())
        self.assertEqual(_extract_symbols({}), set())

    def test_from_pydantic_model(self):
        action = Action(
            name="move",
            params=[Parameter(variable="?r", type="robot")],
            preconditions=ActionPrecondition(conditions=["(at ?r ?l)"]),
            effects=ActionEffect(add=["(moved ?r)"]),
        )
        s = _extract_symbols(action)
        self.assertIn("at", s)
        self.assertIn("moved", s)


class TestVerifySymbols(unittest.TestCase):

    def test_all_declared(self):
        ctx = {
            Predicate: [
                Predicate(name="at", params=[Parameter(variable="?r", type="robot")]),
                Predicate(
                    name="clear", params=[Parameter(variable="?b", type="block")]
                ),
            ]
        }
        self.assertTrue(_verify_symbols({"at", "clear"}, ctx, "loc").valid)

    def test_undeclared_symbol(self):
        ctx = {Predicate: [Predicate(name="at", params=[])]}
        r = _verify_symbols({"at", "undefined_pred"}, ctx, "test")
        self.assertFalse(r.valid)
        self.assertIn("undefined_pred", r.errors[0])

    def test_pddl_keywords_are_allowed(self):
        ctx = {Predicate: []}
        keywords = {
            "and",
            "or",
            "not",
            "forall",
            "when",
            "increase",
            "decrease",
            "assign",
        }
        self.assertTrue(_verify_symbols(keywords, ctx, "test").valid)

    def test_mixed_valid_and_invalid(self):
        ctx = {
            Predicate: [
                Predicate(name="at", params=[]),
                Predicate(name="holding", params=[]),
            ]
        }
        r = _verify_symbols({"at", "holding", "fly"}, ctx, "test")
        self.assertFalse(r.valid)
        self.assertEqual(len(r.errors), 1)
        self.assertIn("fly", r.errors[0])

    def test_empty_symbols(self):
        ctx = {Predicate: []}
        self.assertTrue(_verify_symbols(set(), ctx, "test").valid)


class TestGetOrdinal(unittest.TestCase):

    def test_ordinals(self):
        cases = {
            1: "1st",
            2: "2nd",
            3: "3rd",
            4: "4th",
            11: "11th",
            12: "12th",
            13: "13th",
            21: "21st",
            22: "22nd",
            23: "23rd",
            100: "100th",
            101: "101st",
            111: "111th",
        }
        for n, expected in cases.items():
            self.assertEqual(get_ordinal(n), expected)


if __name__ == "__main__":
    unittest.main()
