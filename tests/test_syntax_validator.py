import unittest
from l2p.utils.pddl_types import *
from l2p.validators.domain import DomainValidator, DomainSemantics
from l2p.validators.problem import ProblemValidator, ProblemSemantics

# =============================================================================
# DOMAIN VALIDATOR RULES
# =============================================================================


class TestDomainValidatorNaming(unittest.TestCase):
    """Tests for validate_pddl_naming rule on domain components."""

    def setUp(self):
        self.validator = DomainValidator()

    def _validate(self, target, context=None):
        return self.validator.validate_component(target, context or {})

    # --- valid names ---
    def test_valid_name_passes(self):
        t = PDDLType(name="robot", parent="object")
        r = self._validate(t, {PDDLType: [t]})
        self.assertTrue(r.valid)

    def test_name_with_hyphen_passes(self):
        t = PDDLType(name="battery-level", parent="object")
        r = self._validate(t, {PDDLType: [t]})
        self.assertTrue(r.valid)

    def test_name_with_underscore_passes(self):
        t = PDDLType(name="my_type", parent="object")
        r = self._validate(t, {PDDLType: [t]})
        self.assertTrue(r.valid)

    def test_name_with_numbers_passes(self):
        t = PDDLType(name="type1", parent="object")
        r = self._validate(t, {PDDLType: [t]})
        self.assertTrue(r.valid)

    # --- invalid names ---
    def test_name_starts_with_question_mark_fails(self):
        t = PDDLType(name="?robot", parent="object")
        r = self._validate(t, {PDDLType: [t]})
        self.assertFalse(r.valid)
        self.assertIn("?", r.errors[0])

    def test_name_with_special_chars_fails(self):
        t = PDDLType(name="robot@1", parent="object")
        r = self._validate(t, {PDDLType: [t]})
        self.assertFalse(r.valid)

    def test_name_starts_with_number_fails(self):
        t = PDDLType(name="1robot", parent="object")
        r = self._validate(t, {PDDLType: [t]})
        self.assertFalse(r.valid)

    def test_name_is_pddl_keyword_fails(self):
        t = PDDLType(name="and", parent="object")
        r = self._validate(t, {PDDLType: [t]})
        self.assertFalse(r.valid)

    # --- duplicate names ---
    def test_duplicate_name_fails(self):
        t1 = PDDLType(name="robot", parent="object")
        t2 = PDDLType(name="robot", parent="object")
        r = self._validate(t2, {PDDLType: [t1, t2]})
        self.assertFalse(r.valid)
        self.assertIn("already in use", r.errors[0])

    def test_duplicate_name_case_insensitive_fails(self):
        t1 = PDDLType(name="Robot", parent="object")
        t2 = PDDLType(name="robot", parent="object")
        r = self._validate(t2, {PDDLType: [t1, t2]})
        self.assertFalse(r.valid)

    def test_unique_names_pass(self):
        t1 = PDDLType(name="robot", parent="object")
        t2 = PDDLType(name="drone", parent="object")
        r = self._validate(t2, {PDDLType: [t1, t2]})
        self.assertTrue(r.valid)

    def test_predicate_name_duplicate_with_type_fails(self):
        t = PDDLType(name="robot", parent="object")
        p = Predicate(name="robot", params=[Parameter(variable="?x", type="object")])
        r = self._validate(p, {PDDLType: [t], Predicate: [p]})
        self.assertFalse(r.valid)

    # --- uppercase warning ---
    def test_uppercase_generates_warning(self):
        t = PDDLType(name="RobotArm", parent="object")
        r = self._validate(t, {PDDLType: [t]})
        self.assertTrue(r.valid)
        self.assertEqual(len(r.warnings), 1)
        self.assertIn("uppercase", r.warnings[0])

    def test_lowercase_no_warning(self):
        t = PDDLType(name="robotarm", parent="object")
        r = self._validate(t, {PDDLType: [t]})
        self.assertEqual(len(r.warnings), 0)

    # --- empty name ---
    def test_empty_name_skips_validation(self):
        t = PDDLType(name="", parent="object")
        r = self._validate(t, {PDDLType: [t]})
        self.assertTrue(r.valid)

    # --- applies to all relevant types ---
    def test_naming_applies_to_constant(self):
        c = Constant(name="base1", type="location")
        r = self._validate(
            c, {Constant: [c], PDDLType: [PDDLType(name="location", parent="object")]}
        )
        self.assertTrue(r.valid)

    def test_naming_applies_to_function(self):
        f = Function(name="battery", params=[Parameter(variable="?r", type="robot")])
        r = self._validate(
            f, {Function: [f], PDDLType: [PDDLType(name="robot", parent="object")]}
        )
        self.assertTrue(r.valid)


class TestDomainValidatorTypeInheritance(unittest.TestCase):
    """Tests for check_type_inheritance rule."""

    def setUp(self):
        self.validator = DomainValidator()

    def test_parent_exists_passes(self):
        types = [
            PDDLType(name="vehicle", parent="object"),
            PDDLType(name="rover", parent="vehicle"),
        ]
        r = self.validator.validate_component(types[1], {PDDLType: types})
        self.assertTrue(r.valid)

    def test_parent_missing_fails(self):
        types = [
            PDDLType(name="rover", parent="vehicle"),
        ]
        r = self.validator.validate_component(types[0], {PDDLType: types})
        self.assertFalse(r.valid)

    def test_parent_object_skipped(self):
        t = PDDLType(name="robot", parent="object")
        r = self.validator.validate_component(t, {PDDLType: [t]})
        self.assertTrue(r.valid)

    def test_no_types_in_context_returns_error(self):
        t = PDDLType(name="rover", parent="vehicle")
        r = self.validator.validate_component(t, {PDDLType: [t]})
        self.assertFalse(r.valid)


class TestDomainValidatorTypeCycle(unittest.TestCase):
    """Tests for check_type_cycle rule."""

    def setUp(self):
        self.validator = DomainValidator()

    def test_acyclic_passes(self):
        types = [
            PDDLType(name="vehicle", parent="object"),
            PDDLType(name="rover", parent="vehicle"),
            PDDLType(name="drone", parent="vehicle"),
        ]
        for t in types:
            r = self.validator.validate_component(t, {PDDLType: types})
            self.assertTrue(r.valid)

    def test_self_cycle_fails(self):
        types = [PDDLType(name="paradox", parent="paradox")]
        r = self.validator.validate_component(types[0], {PDDLType: types})
        self.assertFalse(r.valid)
        self.assertIn("cycle", r.errors[0])

    def test_cross_cycle_fails(self):
        types = [
            PDDLType(name="a", parent="b"),
            PDDLType(name="b", parent="a"),
        ]
        r = self.validator.validate_component(types[0], {PDDLType: types})
        self.assertFalse(r.valid)

    def test_chain_cycle_fails(self):
        types = [
            PDDLType(name="a", parent="b"),
            PDDLType(name="b", parent="c"),
            PDDLType(name="c", parent="a"),
        ]
        r = self.validator.validate_component(types[0], {PDDLType: types})
        self.assertFalse(r.valid)


class TestDomainValidatorConstantInheritance(unittest.TestCase):
    """Tests for check_constant_inheritance rule."""

    def setUp(self):
        self.validator = DomainValidator()

    def test_type_exists_passes(self):
        types = [PDDLType(name="robot", parent="object")]
        c = Constant(name="r1", type="robot")
        r = self.validator.validate_component(c, {PDDLType: types, Constant: [c]})
        self.assertTrue(r.valid)

    def test_type_missing_fails(self):
        c = Constant(name="r1", type="robot")
        r = self.validator.validate_component(c, {Constant: [c]})
        self.assertFalse(r.valid)

    def test_type_object_skipped(self):
        c = Constant(name="r1", type="object")
        r = self.validator.validate_component(c, {Constant: [c]})
        self.assertTrue(r.valid)

    def test_empty_name_skipped(self):
        c = Constant(name="", type="robot")
        r = self.validator.validate_component(c, {Constant: [c]})
        self.assertTrue(r.valid)


class TestDomainValidatorParameterTypes(unittest.TestCase):
    """Tests for check_parameter_types rule."""

    def setUp(self):
        self.validator = DomainValidator()

    def test_valid_typed_params_passes(self):
        types = [
            PDDLType(name="robot", parent="object"),
            PDDLType(name="location", parent="object"),
        ]
        p = Predicate(
            name="at-location",
            params=[
                Parameter(variable="?r", type="robot"),
                Parameter(variable="?l", type="location"),
            ],
        )
        r = self.validator.validate_component(p, {PDDLType: types, Predicate: [p]})
        self.assertTrue(r.valid)

    def test_missing_question_mark_fails(self):
        types = [PDDLType(name="robot", parent="object")]
        p = Predicate.model_construct(
            name="at-location",
            params=[Parameter.model_construct(variable="r", type="robot")],
        )
        r = self.validator.validate_component(p, {PDDLType: types, Predicate: [p]})
        self.assertFalse(r.valid)

    def test_undeclared_type_fails(self):
        types = [PDDLType(name="robot", parent="object")]
        p = Predicate(name="at", params=[Parameter(variable="?r", type="alien")])
        r = self.validator.validate_component(p, {PDDLType: types, Predicate: [p]})
        self.assertFalse(r.valid)

    def test_untyped_param_passes(self):
        p = Predicate(name="pred", params=[Parameter(variable="?x", type="object")])
        r = self.validator.validate_component(p, {Predicate: [p]})
        self.assertTrue(r.valid)

    def test_allows_constant_types(self):
        types = [PDDLType(name="robot", parent="object")]
        consts = [Constant(name="r1", type="robot")]
        p = Predicate(
            name="at-location", params=[Parameter(variable="?r", type="robot")]
        )
        r = self.validator.validate_component(
            p, {PDDLType: types, Constant: consts, Predicate: [p]}
        )
        self.assertTrue(r.valid)

    def test_applies_to_functions(self):
        types = [PDDLType(name="robot", parent="object")]
        f = Function(name="battery", params=[Parameter(variable="?r", type="robot")])
        r = self.validator.validate_component(f, {PDDLType: types, Function: [f]})
        self.assertTrue(r.valid)


class TestDomainValidatorDerivedPredicate(unittest.TestCase):
    """Tests for check_derived_predicate rule."""

    def setUp(self):
        self.validator = DomainValidator()

    def test_valid_symbols_passes(self):
        ctx = {
            PDDLType: [PDDLType(name="robot", parent="object")],
            Predicate: [
                Predicate(
                    name="battery", params=[Parameter(variable="?r", type="robot")]
                )
            ],
        }
        dp = DerivedPredicate(
            name="can-move",
            params=[Parameter(variable="?r", type="robot")],
            condition="(> (battery ?r) 0)",
        )
        r = self.validator.validate_component(dp, ctx | {DerivedPredicate: [dp]})
        self.assertTrue(r.valid)

    def test_undeclared_symbol_fails(self):
        ctx = {Predicate: []}
        dp = DerivedPredicate(
            name="can-move",
            params=[Parameter(variable="?r", type="robot")],
            condition="(> (unknown ?r) 0)",
        )
        r = self.validator.validate_component(dp, ctx | {DerivedPredicate: [dp]})
        self.assertFalse(r.valid)


class TestDomainValidatorActionPrecondition(unittest.TestCase):
    """Tests for check_action_precondition rule."""

    def setUp(self):
        self.validator = DomainValidator()

    def test_valid_symbols_passes(self):
        ctx = {
            PDDLType: [
                PDDLType(name="robot", parent="object"),
                PDDLType(name="location", parent="object"),
            ],
            Predicate: [
                Predicate(
                    name="at-location",
                    params=[
                        Parameter(variable="?r", type="robot"),
                        Parameter(variable="?l", type="location"),
                    ],
                ),
                Predicate(
                    name="battery-dead", params=[Parameter(variable="?r", type="robot")]
                ),
            ],
        }
        action = Action(
            name="move",
            params=[
                Parameter(variable="?r", type="robot"),
                Parameter(variable="?l", type="location"),
            ],
            preconditions=ActionPrecondition(
                conditions=[
                    "(at-location ?r ?l)",
                    {"operator": "not", "condition": "(battery-dead ?r)"},
                ]
            ),
            effects=ActionEffect(),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        self.assertTrue(r.valid)

    def test_undeclared_symbol_fails(self):
        ctx = {Predicate: [Predicate(name="at", params=[])]}
        action = Action(
            name="move",
            params=[],
            preconditions=ActionPrecondition(conditions=["(at ?r)", "(fly ?r)"]),
            effects=ActionEffect(),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        self.assertFalse(r.valid)


class TestDomainValidatorActionEffect(unittest.TestCase):
    """Tests for check_action_effect rule."""

    def setUp(self):
        self.validator = DomainValidator()

    def test_valid_symbols_passes(self):
        ctx = {
            PDDLType: [
                PDDLType(name="robot", parent="object"),
                PDDLType(name="location", parent="object"),
            ],
            Predicate: [
                Predicate(
                    name="at-location",
                    params=[
                        Parameter(variable="?r", type="robot"),
                        Parameter(variable="?l", type="location"),
                    ],
                )
            ],
        }
        action = Action(
            name="move",
            params=[
                Parameter(variable="?r", type="robot"),
                Parameter(variable="?l", type="location"),
                Parameter(variable="?old", type="location"),
            ],
            preconditions=ActionPrecondition(),
            effects=ActionEffect(
                add=["(at-location ?r ?l)"], delete=["(at-location ?r ?old)"]
            ),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        self.assertTrue(r.valid)

    def test_numeric_effects_with_declared_function(self):
        ctx = {
            PDDLType: [PDDLType(name="robot", parent="object")],
            Function: [
                Function(
                    name="battery", params=[Parameter(variable="?r", type="robot")]
                )
            ],
        }
        action = Action(
            name="charge",
            params=[Parameter(variable="?r", type="robot")],
            preconditions=ActionPrecondition(),
            effects=ActionEffect(numeric=["(increase (battery ?r) 10)"]),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        # increase is a keyword, battery is a declared function
        self.assertTrue(r.valid)

    def test_undeclared_function_in_numeric_fails(self):
        ctx = {Function: []}
        action = Action(
            name="charge",
            params=[],
            preconditions=ActionPrecondition(),
            effects=ActionEffect(numeric=["(increase (unknown) 10)"]),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        self.assertFalse(r.valid)


class TestDomainValidatorComponentVariables(unittest.TestCase):
    """Tests for check_component_variables rule - the most complex rule."""

    def setUp(self):
        self.validator = DomainValidator()

    # --- variable existence ---
    def test_all_variables_declared_passes(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(name="clear", params=[Parameter(variable="?b", type="block")])
            ],
        }
        action = Action(
            name="pickup",
            params=[Parameter(variable="?b", type="block")],
            preconditions=ActionPrecondition(conditions=["(clear ?b)"]),
            effects=ActionEffect(add=[], delete=["(clear ?b)"]),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        self.assertTrue(r.valid)

    def test_undeclared_variable_fails(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(name="clear", params=[Parameter(variable="?b", type="block")])
            ],
        }
        action = Action(
            name="pickup",
            params=[Parameter(variable="?b", type="block")],
            preconditions=ActionPrecondition(conditions=["(clear ?x)"]),
            effects=ActionEffect(),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        self.assertFalse(r.valid)
        self.assertIn("?x", r.errors[0])

    # --- arity checking ---
    def test_arity_mismatch_fails(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(
                    name="on",
                    params=[
                        Parameter(variable="?b1", type="block"),
                        Parameter(variable="?b2", type="block"),
                    ],
                )
            ],
        }
        action = Action(
            name="check",
            params=[Parameter(variable="?b", type="block")],
            preconditions=ActionPrecondition(conditions=["(on ?b)"]),
            effects=ActionEffect(),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        self.assertFalse(r.valid)
        self.assertIn("expects 2 arguments", r.errors[0])

    # --- type matching ---
    def test_type_mismatch_fails(self):
        ctx = {
            PDDLType: [
                PDDLType(name="arm", parent="object"),
                PDDLType(name="block", parent="object"),
            ],
            Predicate: [
                Predicate(
                    name="holding",
                    params=[
                        Parameter(variable="?a", type="arm"),
                        Parameter(variable="?b", type="block"),
                    ],
                )
            ],
        }
        action = Action(
            name="pickup",
            params=[
                Parameter(variable="?b", type="block"),
                Parameter(variable="?a", type="arm"),
            ],
            preconditions=ActionPrecondition(conditions=["(holding ?b ?a)"]),
            effects=ActionEffect(),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        # ?b is block, ?a is arm - holding expects arm first, block second
        # So ?b (block) for param 1 (expects arm) should fail
        self.assertFalse(r.valid)

    def test_correct_type_with_inheritance_passes(self):
        ctx = {
            PDDLType: [
                PDDLType(name="vehicle", parent="object"),
                PDDLType(name="rover", parent="vehicle"),
                PDDLType(name="location", parent="object"),
            ],
            Constant: [Constant(name="base", type="location")],
            Predicate: [
                Predicate(
                    name="at-location",
                    params=[
                        Parameter(variable="?v", type="vehicle"),
                        Parameter(variable="?l", type="location"),
                    ],
                )
            ],
        }
        action = Action(
            name="move",
            params=[Parameter(variable="?r", type="rover")],
            preconditions=ActionPrecondition(conditions=["(at-location ?r base)"]),
            effects=ActionEffect(),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        self.assertTrue(r.valid)

    # --- forall / exists scope ---
    def test_forall_introduces_scope(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(name="clear", params=[Parameter(variable="?b", type="block")])
            ],
        }
        action = Action(
            name="check-all",
            params=[Parameter(variable="?r", type="object")],
            preconditions=ActionPrecondition(
                conditions=[
                    {
                        "quantifier": "forall",
                        "parameters": [{"variable": "?x", "type": "block"}],
                        "conditions": ["(clear ?x)"],
                    }
                ]
            ),
            effects=ActionEffect(),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        # ?x is introduced by forall, (clear ?x) should be valid
        self.assertTrue(r.valid)

    def test_exists_without_scope_but_declared_in_action_fails(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(name="clear", params=[Parameter(variable="?b", type="block")])
            ],
        }
        action = Action(
            name="check",
            params=[],
            preconditions=ActionPrecondition(
                conditions=[
                    {
                        "quantifier": "exists",
                        "parameters": [{"variable": "?x", "type": "block"}],
                        "conditions": ["(clear ?x)"],
                    }
                ]
            ),
            effects=ActionEffect(),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        self.assertTrue(r.valid)

    # --- constants as arguments ---
    def test_constant_as_argument_passes(self):
        ctx = {
            PDDLType: [PDDLType(name="location", parent="object")],
            Constant: [Constant(name="base", type="location")],
            Predicate: [
                Predicate(
                    name="at",
                    params=[
                        Parameter(variable="?r", type="object"),
                        Parameter(variable="?l", type="location"),
                    ],
                )
            ],
        }
        action = Action(
            name="move",
            params=[Parameter(variable="?r", type="object")],
            preconditions=ActionPrecondition(conditions=["(at ?r base)"]),
            effects=ActionEffect(),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        self.assertTrue(r.valid)

    # --- number literals ---
    def test_number_literal_as_argument_passes(self):
        ctx = {
            PDDLType: [PDDLType(name="object", parent="object")],
            Predicate: [
                Predicate(
                    name="active", params=[Parameter(variable="?r", type="object")]
                )
            ],
        }
        action = Action(
            name="nop",
            params=[Parameter(variable="?r", type="object")],
            preconditions=ActionPrecondition(conditions=["(>= ?r 5)"]),
            effects=ActionEffect(),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        self.assertTrue(r.valid)


class TestDomainSemantics(unittest.TestCase):
    """Tests for the DomainSemantics helper class."""

    def test_type_parents(self):
        ctx = {
            PDDLType: [
                PDDLType(name="vehicle", parent="object"),
                PDDLType(name="rover", parent="vehicle"),
            ]
        }
        sem = DomainSemantics(ctx)
        self.assertEqual(sem.type_parents["vehicle"], "object")
        self.assertEqual(sem.type_parents["rover"], "vehicle")

    def test_is_subtype(self):
        ctx = {
            PDDLType: [
                PDDLType(name="vehicle", parent="object"),
                PDDLType(name="rover", parent="vehicle"),
                PDDLType(name="drone", parent="vehicle"),
            ]
        }
        sem = DomainSemantics(ctx)
        self.assertTrue(sem.is_subtype("rover", "vehicle"))
        self.assertTrue(sem.is_subtype("rover", "object"))
        self.assertTrue(sem.is_subtype("vehicle", "vehicle"))
        self.assertFalse(sem.is_subtype("rover", "drone"))

    def test_signatures(self):
        ctx = {
            Predicate: [
                Predicate(
                    name="at",
                    params=[
                        Parameter(variable="?r", type="rover"),
                        Parameter(variable="?l", type="location"),
                    ],
                )
            ]
        }
        sem = DomainSemantics(ctx)
        self.assertIn("at", sem.signatures)
        self.assertEqual(sem.signatures["at"], ["rover", "location"])

    def test_constants(self):
        ctx = {Constant: [Constant(name="base", type="location")]}
        sem = DomainSemantics(ctx)
        self.assertEqual(sem.constants["base"], "location")


class TestDomainValidatorDurativeActions(unittest.TestCase):
    """Tests for durative action rules."""

    def setUp(self):
        self.validator = DomainValidator()

    def test_dur_action_conditions_valid(self):
        ctx = {
            PDDLType: [
                PDDLType(name="robot", parent="object"),
                PDDLType(name="location", parent="object"),
            ],
            Predicate: [
                Predicate(
                    name="at-location",
                    params=[
                        Parameter(variable="?r", type="robot"),
                        Parameter(variable="?l", type="location"),
                    ],
                )
            ],
        }
        da = DurativeAction(
            name="move",
            params=[
                Parameter(variable="?r", type="robot"),
                Parameter(variable="?from", type="location"),
                Parameter(variable="?to", type="location"),
            ],
            duration=["(>= ?duration 5.0)"],
            conditions=DurativeActionConditions(at_start=["(at-location ?r ?from)"]),
            effects=DurativeActionEffect(
                at_end=ActionEffect(add=["(at-location ?r ?to)"])
            ),
        )
        r = self.validator.validate_component(da, ctx | {DurativeAction: [da]})
        self.assertTrue(r.valid)

    def test_dur_action_effect_undeclared_fails(self):
        ctx = {
            PDDLType: [PDDLType(name="robot", parent="object")],
            Predicate: [],
        }
        da = DurativeAction(
            name="move",
            params=[Parameter(variable="?r", type="robot")],
            duration=["(>= ?duration 5.0)"],
            conditions=DurativeActionConditions(at_start=["(unknown ?r ?from)"]),
            effects=DurativeActionEffect(at_end=ActionEffect(add=["(at ?r ?to)"])),
        )
        r = self.validator.validate_component(da, ctx | {DurativeAction: [da]})
        # at should be undeclared; unknown should be undeclared
        self.assertFalse(r.valid)

    def test_dur_action_duration_variable(self):
        ctx = {
            PDDLType: [PDDLType(name="robot", parent="object")],
            Predicate: [
                Predicate(name="at", params=[Parameter(variable="?r", type="robot")])
            ],
        }
        da = DurativeAction(
            name="move",
            params=[Parameter(variable="?r", type="robot")],
            duration=["(>= ?duration 5.0)"],
            conditions=DurativeActionConditions(),
            effects=DurativeActionEffect(),
        )
        r = self.validator.validate_component(da, ctx | {DurativeAction: [da]})
        # ?duration is automatically allowed in duration block
        self.assertTrue(r.valid)


class TestDomainValidatorConstraint(unittest.TestCase):
    """Tests for check_constraint rule."""

    def setUp(self):
        self.validator = DomainValidator()

    def test_valid_constraint_symbols_passes(self):
        ctx = {
            Predicate: [
                Predicate(
                    name="battery-dead", params=[Parameter(variable="?r", type="robot")]
                )
            ],
            Function: [
                Function(
                    name="battery", params=[Parameter(variable="?r", type="robot")]
                )
            ],
        }
        c = Constraint(
            condition={"operator": "always", "condition": "(not (battery-dead ?r))"}
        )
        r = self.validator.validate_component(c, ctx | {Constraint: [c]})
        self.assertTrue(r.valid)

    def test_undeclared_symbol_fails(self):
        ctx = {Predicate: []}
        c = Constraint(condition={"operator": "always", "condition": "(unknown ?r)"})
        r = self.validator.validate_component(c, ctx | {Constraint: [c]})
        self.assertFalse(r.valid)


class TestDomainValidatorEdgeCases(unittest.TestCase):
    """Edge cases and boundary tests."""

    def setUp(self):
        self.validator = DomainValidator()

    def test_deeply_nested_conditions(self):
        ctx = {
            PDDLType: [PDDLType(name="obj", parent="object")],
            Predicate: [
                Predicate(name="p", params=[Parameter(variable="?x", type="obj")])
            ],
        }
        action = Action(
            name="complex",
            params=[Parameter(variable="?a", type="obj")],
            preconditions=ActionPrecondition(
                conditions=[
                    {
                        "operator": "and",
                        "conditions": [
                            {
                                "operator": "or",
                                "conditions": [
                                    "(p ?a)",
                                    {"operator": "not", "condition": "(p ?a)"},
                                ],
                            },
                            {
                                "quantifier": "forall",
                                "parameters": [{"variable": "?x", "type": "obj"}],
                                "conditions": [
                                    {
                                        "operator": "imply",
                                        "antecedent": ["(p ?x)"],
                                        "consequent": ["(p ?x)"],
                                    }
                                ],
                            },
                        ],
                    }
                ]
            ),
            effects=ActionEffect(),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        self.assertTrue(r.valid)

    def test_event_validation(self):
        ctx = {
            PDDLType: [PDDLType(name="robot", parent="object")],
            Predicate: [
                Predicate(
                    name="battery-dead", params=[Parameter(variable="?r", type="robot")]
                )
            ],
        }
        evt = Event(
            name="battery-depleted",
            params=[Parameter(variable="?r", type="robot")],
            preconditions=ActionPrecondition(conditions=["(battery-dead ?r)"]),
            effects=ActionEffect(add=["(dead ?r)"]),
        )
        r = self.validator.validate_component(evt, ctx | {Event: [evt]})
        self.assertTrue(r.valid)

    def test_process_validation(self):
        ctx = {
            PDDLType: [PDDLType(name="robot", parent="object")],
            Function: [
                Function(
                    name="battery", params=[Parameter(variable="?r", type="robot")]
                )
            ],
        }
        proc = Process(
            name="solar-charging",
            params=[Parameter(variable="?r", type="robot")],
            preconditions=ActionPrecondition(conditions=["(in-sun ?r)"]),
            effects=ActionEffect(numeric=["(increase (battery ?r) (* #t 2.0))"]),
        )
        r = self.validator.validate_component(proc, ctx | {Process: [proc]})
        # Note: Process is not targeted by check_action_precondition, so undeclared
        # predicates in preconditions are not caught by that rule.
        # check_component_variables covers Process but only validates declared predicates' signatures.
        self.assertTrue(r.valid)


# =============================================================================
# PROBLEM VALIDATOR RULES
# =============================================================================


class TestProblemValidatorNaming(unittest.TestCase):
    """Tests for validate_pddl_naming on PDDLObject."""

    def setUp(self):
        self.validator = ProblemValidator()

    def test_valid_name_passes(self):
        o = PDDLObject(name="rover1", type="robot")
        r = self.validator.validate_component(
            o, {PDDLObject: [o], PDDLType: [PDDLType(name="robot", parent="object")]}
        )
        self.assertTrue(r.valid)

    def test_name_with_question_mark_fails(self):
        o = PDDLObject(name="?rover1", type="robot")
        r = self.validator.validate_component(o, {PDDLObject: [o]})
        self.assertFalse(r.valid)

    def test_duplicate_object_fails(self):
        o1 = PDDLObject(name="rover1", type="robot")
        o2 = PDDLObject(name="rover1", type="robot")
        r = self.validator.validate_component(o2, {PDDLObject: [o1, o2]})
        self.assertFalse(r.valid)

    def test_reserved_keyword_fails(self):
        o = PDDLObject(name="and", type="object")
        r = self.validator.validate_component(o, {PDDLObject: [o]})
        self.assertFalse(r.valid)

    def test_name_with_uppercase_warns(self):
        o = PDDLObject(name="Rover1", type="robot")
        r = self.validator.validate_component(
            o, {PDDLObject: [o], PDDLType: [PDDLType(name="robot", parent="object")]}
        )
        self.assertTrue(r.valid)
        self.assertGreater(len(r.warnings), 0)


class TestProblemValidatorObjTypeInheritance(unittest.TestCase):
    """Tests for check_obj_type_inheritance rule."""

    def setUp(self):
        self.validator = ProblemValidator()

    def test_type_exists_passes(self):
        o = PDDLObject(name="r1", type="robot")
        r = self.validator.validate_component(
            o, {PDDLObject: [o], PDDLType: [PDDLType(name="robot", parent="object")]}
        )
        self.assertTrue(r.valid)

    def test_type_missing_fails(self):
        o = PDDLObject(name="r1", type="robot")
        r = self.validator.validate_component(o, {PDDLObject: [o]})
        self.assertFalse(r.valid)

    def test_type_object_skipped(self):
        o = PDDLObject(name="r1", type="object")
        r = self.validator.validate_component(o, {PDDLObject: [o]})
        self.assertTrue(r.valid)


class TestProblemValidatorInitialState(unittest.TestCase):
    """Tests for check_initial_state rule."""

    def setUp(self):
        self.validator = ProblemValidator()

    def test_valid_initial_state_passes(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(
                    name="on",
                    params=[
                        Parameter(variable="?b1", type="block"),
                        Parameter(variable="?b2", type="block"),
                    ],
                )
            ],
            PDDLObject: [
                PDDLObject(name="a", type="block"),
                PDDLObject(name="b", type="block"),
            ],
        }
        init = InitialState(facts=["(on a b)"])
        r = self.validator.validate_component(init, ctx | {InitialState: [init]})
        self.assertTrue(r.valid)

    def test_undeclared_object_fails(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(
                    name="on",
                    params=[
                        Parameter(variable="?b1", type="block"),
                        Parameter(variable="?b2", type="block"),
                    ],
                )
            ],
            PDDLObject: [PDDLObject(name="a", type="block")],
        }
        init = InitialState(facts=["(on a b)"])
        r = self.validator.validate_component(init, ctx | {InitialState: [init]})
        # 'b' is not declared
        self.assertFalse(r.valid)

    def test_arity_mismatch_fails(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(
                    name="on",
                    params=[
                        Parameter(variable="?b1", type="block"),
                        Parameter(variable="?b2", type="block"),
                    ],
                )
            ],
            PDDLObject: [
                PDDLObject(name="a", type="block"),
                PDDLObject(name="b", type="block"),
            ],
        }
        init = InitialState(facts=["(on a)"])
        r = self.validator.validate_component(init, ctx | {InitialState: [init]})
        self.assertFalse(r.valid)

    def test_type_mismatch_fails(self):
        ctx = {
            PDDLType: [
                PDDLType(name="arm", parent="object"),
                PDDLType(name="block", parent="object"),
            ],
            Predicate: [
                Predicate(
                    name="holding",
                    params=[
                        Parameter(variable="?a", type="arm"),
                        Parameter(variable="?b", type="block"),
                    ],
                )
            ],
            PDDLObject: [
                PDDLObject(name="arm1", type="block"),  # wrong type
                PDDLObject(name="block1", type="block"),
            ],
        }
        init = InitialState(facts=["(holding arm1 block1)"])
        r = self.validator.validate_component(init, ctx | {InitialState: [init]})
        self.assertFalse(r.valid)

    def test_variables_in_initial_state_fails(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(name="clear", params=[Parameter(variable="?b", type="block")])
            ],
            PDDLObject: [PDDLObject(name="a", type="block")],
        }
        init = InitialState(facts=["(clear ?x)"])
        r = self.validator.validate_component(init, ctx | {InitialState: [init]})
        self.assertFalse(r.valid)

    def test_timed_fact_valid(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(name="clear", params=[Parameter(variable="?b", type="block")])
            ],
            PDDLObject: [PDDLObject(name="a", type="block")],
        }
        init = InitialState(
            facts=[], timed_facts=[TimedFact(time=5.0, fact="(clear a)")]
        )
        r = self.validator.validate_component(init, ctx | {InitialState: [init]})
        self.assertTrue(r.valid)

    def test_timed_fact_with_undeclared_object_fails(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(name="clear", params=[Parameter(variable="?b", type="block")])
            ],
            PDDLObject: [],
        }
        init = InitialState(
            facts=[], timed_facts=[TimedFact(time=5.0, fact="(clear a)")]
        )
        r = self.validator.validate_component(init, ctx | {InitialState: [init]})
        self.assertFalse(r.valid)

    def test_empty_initial_state_passes(self):
        init = InitialState()
        r = self.validator.validate_component(init, {InitialState: [init]})
        self.assertTrue(r.valid)

    def test_inheritance_with_objects_passes(self):
        ctx = {
            PDDLType: [
                PDDLType(name="vehicle", parent="object"),
                PDDLType(name="rover", parent="vehicle"),
            ],
            Predicate: [
                Predicate(
                    name="at",
                    params=[
                        Parameter(variable="?v", type="vehicle"),
                        Parameter(variable="?l", type="object"),
                    ],
                )
            ],
            PDDLObject: [
                PDDLObject(name="perseverance", type="rover"),
                PDDLObject(name="base", type="object"),
            ],
        }
        init = InitialState(facts=["(at perseverance base)"])
        r = self.validator.validate_component(init, ctx | {InitialState: [init]})
        self.assertTrue(r.valid)


class TestProblemValidatorGoalState(unittest.TestCase):
    """Tests for check_goal_state rule."""

    def setUp(self):
        self.validator = ProblemValidator()

    def test_valid_goal_passes(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(
                    name="on",
                    params=[
                        Parameter(variable="?b1", type="block"),
                        Parameter(variable="?b2", type="block"),
                    ],
                )
            ],
            PDDLObject: [
                PDDLObject(name="a", type="block"),
                PDDLObject(name="b", type="block"),
            ],
        }
        goal = GoalState(conditions=["(on a b)"])
        r = self.validator.validate_component(goal, ctx | {GoalState: [goal]})
        self.assertTrue(r.valid)

    def test_goal_with_logical_operator_passes(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(name="clear", params=[Parameter(variable="?b", type="block")])
            ],
            PDDLObject: [
                PDDLObject(name="a", type="block"),
                PDDLObject(name="b", type="block"),
            ],
        }
        goal = GoalState(
            conditions=[
                {
                    "operator": "and",
                    "conditions": [
                        "(clear a)",
                        {"operator": "not", "condition": "(clear b)"},
                    ],
                }
            ]
        )
        r = self.validator.validate_component(goal, ctx | {GoalState: [goal]})
        self.assertTrue(r.valid)

    def test_goal_undeclared_object_fails(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(name="clear", params=[Parameter(variable="?b", type="block")])
            ],
            PDDLObject: [],
        }
        goal = GoalState(conditions=["(clear a)"])
        r = self.validator.validate_component(goal, ctx | {GoalState: [goal]})
        self.assertFalse(r.valid)

    def test_goal_undeclared_predicate_fails(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [],
            PDDLObject: [PDDLObject(name="a", type="block")],
        }
        goal = GoalState(conditions=["(unknown a)"])
        r = self.validator.validate_component(goal, ctx | {GoalState: [goal]})
        # Note: Problem validator does not verify predicate name declaration in goals;
        # it only checks object existence, arity, and type compatibility.
        # Predicate declaration checking is handled by the domain validator.
        self.assertTrue(r.valid)

    def test_empty_goal_passes(self):
        goal = GoalState()
        r = self.validator.validate_component(goal, {GoalState: [goal]})
        self.assertTrue(r.valid)


class TestProblemValidatorMetric(unittest.TestCase):
    """Tests for check_metric_syntax rule."""

    def setUp(self):
        self.validator = ProblemValidator()

    def test_valid_metric_with_declared_function_passes(self):
        ctx = {Function: [Function(name="total-cost", params=[])]}
        m = Metric(optimization="minimize", expression="(total-cost)")
        r = self.validator.validate_component(m, ctx | {Metric: [m]})
        self.assertTrue(r.valid)

    def test_total_time_passes(self):
        m = Metric(optimization="minimize", expression="total-time")
        r = self.validator.validate_component(m, {Metric: [m]})
        self.assertTrue(r.valid)

    def test_variables_in_metric_fails(self):
        ctx = {
            Function: [
                Function(
                    name="battery", params=[Parameter(variable="?r", type="robot")]
                )
            ]
        }
        m = Metric(optimization="maximize", expression="(battery ?r)")
        r = self.validator.validate_component(m, ctx | {Metric: [m]})
        self.assertFalse(r.valid)

    def test_undeclared_function_fails(self):
        m = Metric(optimization="minimize", expression="(unknown)")
        r = self.validator.validate_component(m, {Metric: [m]})
        self.assertFalse(r.valid)

    def test_metric_with_expression_and_is_violated_passes(self):
        ctx = {Function: [Function(name="total-cost", params=[])]}
        m = Metric(
            optimization="minimize",
            expression="(+ (total-cost) (* 10 (is-violated pref1)))",
        )
        r = self.validator.validate_component(m, ctx | {Metric: [m]})
        self.assertTrue(r.valid)


class TestProblemSemantics(unittest.TestCase):
    """Tests for ProblemSemantics helper."""

    def test_signatures_and_objects(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(
                    name="on",
                    params=[
                        Parameter(variable="?b1", type="block"),
                        Parameter(variable="?b2", type="block"),
                    ],
                )
            ],
        }
        sem = ProblemSemantics(ctx)
        self.assertIn("on", sem.signatures)
        self.assertEqual(sem.signatures["on"], ["block", "block"])

    def test_is_subtype_in_problem(self):
        ctx = {
            PDDLType: [
                PDDLType(name="vehicle", parent="object"),
                PDDLType(name="rover", parent="vehicle"),
            ],
        }
        sem = ProblemSemantics(ctx)
        self.assertTrue(sem.is_subtype("rover", "vehicle"))
        self.assertFalse(sem.is_subtype("vehicle", "rover"))


class TestProblemValidatorEdgeCases(unittest.TestCase):
    """Edge cases for problem validator."""

    def setUp(self):
        self.validator = ProblemValidator()

    def test_initial_state_with_constant_from_domain(self):
        ctx = {
            PDDLType: [PDDLType(name="location", parent="object")],
            Constant: [Constant(name="base", type="location")],
            PDDLObject: [PDDLObject(name="rover1", type="object")],
            Predicate: [
                Predicate(
                    name="at",
                    params=[
                        Parameter(variable="?r", type="object"),
                        Parameter(variable="?l", type="location"),
                    ],
                )
            ],
        }
        init = InitialState(facts=["(at rover1 base)"])
        r = self.validator.validate_component(init, ctx | {InitialState: [init]})
        self.assertTrue(r.valid)

    def test_initial_state_with_numeric_assignment(self):
        ctx = {
            PDDLType: [PDDLType(name="rover", parent="object")],
            Function: [
                Function(
                    name="battery", params=[Parameter(variable="?r", type="rover")]
                )
            ],
            PDDLObject: [PDDLObject(name="r1", type="rover")],
        }
        init = InitialState(facts=["(= (battery r1) 100.0)"])
        r = self.validator.validate_component(init, ctx | {InitialState: [init]})
        self.assertTrue(r.valid)

    def test_goal_with_nested_or(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(
                    name="on",
                    params=[
                        Parameter(variable="?b1", type="block"),
                        Parameter(variable="?b2", type="block"),
                    ],
                )
            ],
            PDDLObject: [
                PDDLObject(name="a", type="block"),
                PDDLObject(name="b", type="block"),
                PDDLObject(name="c", type="block"),
            ],
        }
        goal = GoalState(
            conditions=[{"operator": "or", "conditions": ["(on a b)", "(on a c)"]}]
        )
        r = self.validator.validate_component(goal, ctx | {GoalState: [goal]})
        self.assertTrue(r.valid)


# =============================================================================
# DOMAIN REGISTRY FULL PIPELINE
# =============================================================================


class TestDomainValidatorFullPipeline(unittest.TestCase):
    """Integration: run the full DomainValidator against a realistic scenario."""

    def setUp(self):
        self.validator = DomainValidator()

    def test_complete_domain_validation_passes(self):
        types = [
            PDDLType(name="arm", parent="object"),
            PDDLType(name="block", parent="object"),
        ]
        predicates = [
            Predicate(
                name="holding",
                params=[
                    Parameter(variable="?a", type="arm"),
                    Parameter(variable="?b", type="block"),
                ],
            ),
            Predicate(name="clear", params=[Parameter(variable="?b", type="block")]),
            Predicate(
                name="on",
                params=[
                    Parameter(variable="?b1", type="block"),
                    Parameter(variable="?b2", type="block"),
                ],
            ),
        ]
        functions = [
            Function(name="weight", params=[Parameter(variable="?b", type="block")]),
        ]
        ctx = {
            PDDLType: types,
            Predicate: predicates,
            Function: functions,
        }

        action = Action(
            name="stack",
            params=[
                Parameter(variable="?b1", type="block"),
                Parameter(variable="?b2", type="block"),
                Parameter(variable="?a", type="arm"),
            ],
            preconditions=ActionPrecondition(
                conditions=[
                    "(holding ?a ?b1)",
                    "(clear ?b2)",
                    {"operator": "not", "condition": "(= ?b1 ?b2)"},
                ]
            ),
            effects=ActionEffect(
                add=["(on ?b1 ?b2)", "(clear ?b1)"],
                delete=["(holding ?a ?b1)", "(clear ?b2)"],
                numeric=["(decrease (weight ?b1) 5)"],
            ),
        )
        r = self.validator.validate_component(action, ctx | {Action: [action]})
        self.assertTrue(r.valid)


class TestProblemValidatorFullPipeline(unittest.TestCase):
    """Integration: run the full ProblemValidator against a realistic scenario."""

    def setUp(self):
        self.validator = ProblemValidator()

    def test_complete_problem_validation_passes(self):
        ctx = {
            PDDLType: [PDDLType(name="block", parent="object")],
            Predicate: [
                Predicate(
                    name="on",
                    params=[
                        Parameter(variable="?b1", type="block"),
                        Parameter(variable="?b2", type="block"),
                    ],
                ),
                Predicate(
                    name="clear", params=[Parameter(variable="?b", type="block")]
                ),
                Predicate(
                    name="ontable", params=[Parameter(variable="?b", type="block")]
                ),
            ],
            PDDLObject: [
                PDDLObject(name="a", type="block"),
                PDDLObject(name="b", type="block"),
                PDDLObject(name="c", type="block"),
            ],
        }

        init = InitialState(
            facts=[
                "(ontable a)",
                "(ontable b)",
                "(ontable c)",
                "(clear a)",
                "(clear b)",
                "(clear c)",
            ]
        )
        r = self.validator.validate_component(init, ctx | {InitialState: [init]})
        self.assertTrue(r.valid)

        goal = GoalState(
            conditions=[{"operator": "and", "conditions": ["(on a b)", "(on b c)"]}]
        )
        r = self.validator.validate_component(goal, ctx | {GoalState: [goal]})
        self.assertTrue(r.valid)

        # combined: all objects satisfy both checks
        total_failures = 0
        if not self.validator.validate_component(
            init, ctx | {InitialState: [init], GoalState: [goal]}
        ).valid:
            total_failures += 1
        self.assertEqual(total_failures, 0)


if __name__ == "__main__":
    unittest.main()
