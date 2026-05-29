import unittest
import textwrap
import io
from contextlib import redirect_stdout

from l2p import DomainBuilder
from l2p.utils.pddl_types import *
from tests.mock_llm import MockLLM


class TestDomainBuilderFormalize(unittest.TestCase):
    """Tests for DomainBuilder.formalize_component()."""

    def setUp(self):
        self.builder = DomainBuilder()
        self.mock = MockLLM()
        self.prompt = "Extract PDDL from:\n{domain_desc}\n{context}"

    # ---- TYPES ----

    def test_formalize_types(self):
        self.mock.output = textwrap.dedent("""\
        <types>
        [
            {"name": "arm", "parent": "object", "desc": "robot arm"},
            {"name": "block", "parent": "object", "desc": "stackable block"},
            {"name": "table", "parent": "object", "desc": "blocks sit on table"}
        ]
        </types>""")
        results, output = self.builder.formalize_component(
            model=self.mock,
            component_class=PDDLType,
            prompt_template=self.prompt,
            domain_desc="blocksworld",
        )
        self.assertIn(PDDLType, results)
        types = results[PDDLType]
        self.assertEqual(len(types), 3)
        self.assertEqual(types[0].name, "arm")
        self.assertEqual(types[1].name, "block")
        self.assertEqual(types[2].name, "table")

    def test_formalize_types_with_hierarchy(self):
        self.mock.output = textwrap.dedent("""\
        <types>
        [
            {"name": "vehicle", "parent": "object", "desc": null},
            {"name": "rover", "parent": "vehicle", "desc": "planetary rover"},
            {"name": "drone", "parent": "vehicle", "desc": "flying drone"}
        ]
        </types>""")
        results, _ = self.builder.formalize_component(
            model=self.mock,
            component_class=PDDLType,
            prompt_template=self.prompt,
            domain_desc="rovers",
        )
        types = results[PDDLType]
        self.assertEqual(len(types), 3)
        rover = next(t for t in types if t.name == "rover")
        self.assertEqual(rover.parent, "vehicle")

    def test_formalize_empty_types(self):
        self.mock.output = "<types>\n[]\n</types>"
        results, _ = self.builder.formalize_component(
            model=self.mock,
            component_class=PDDLType,
            prompt_template=self.prompt,
            domain_desc="empty",
        )
        self.assertEqual(results[PDDLType], [])

    # ---- PREDICATES ----

    def test_formalize_predicates(self):
        self.mock.output = textwrap.dedent("""\
        <predicates>
        [
            {"name": "at", "params": [{"variable": "?r", "type": "rover"}, {"variable": "?l", "type": "location"}], "desc": null},
            {"name": "connected", "params": [{"variable": "?from", "type": "location"}, {"variable": "?to", "type": "location"}], "desc": null},
            {"name": "empty", "params": [], "desc": "global boolean"}
        ]
        </predicates>""")
        results, _ = self.builder.formalize_component(
            model=self.mock,
            component_class=Predicate,
            prompt_template=self.prompt,
            domain_desc="test",
        )
        preds = results[Predicate]
        self.assertEqual(len(preds), 3)
        at = next(p for p in preds if p.name == "at")
        self.assertEqual(len(at.params), 2)
        self.assertEqual(at.params[0].variable, "?r")
        empty = next(p for p in preds if p.name == "empty")
        self.assertEqual(len(empty.params), 0)

    def test_formalize_predicate_with_untyped_params(self):
        self.mock.output = textwrap.dedent("""\
        <predicates>
        [
            {"name": "pred", "params": [{"variable": "?x", "type": "object"}, {"variable": "?y", "type": "object"}], "desc": null}
        ]
        </predicates>""")
        results, _ = self.builder.formalize_component(
            model=self.mock,
            component_class=Predicate,
            prompt_template=self.prompt,
            domain_desc="test",
        )
        preds = results[Predicate]
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].params[0].type, "object")

    # ---- FUNCTIONS ----

    def test_formalize_functions(self):
        self.mock.output = textwrap.dedent("""\
        <functions>
        [
            {"name": "battery-level", "params": [{"variable": "?r", "type": "rover"}], "desc": "battery level"},
            {"name": "total-cost", "params": [], "desc": null}
        ]
        </functions>""")
        results, _ = self.builder.formalize_component(
            model=self.mock,
            component_class=Function,
            prompt_template=self.prompt,
            domain_desc="test",
        )
        funcs = results[Function]
        self.assertEqual(len(funcs), 2)
        self.assertEqual(funcs[0].name, "battery-level")
        self.assertEqual(funcs[1].name, "total-cost")
        self.assertEqual(len(funcs[1].params), 0)

    # ---- CONSTANTS ----

    def test_formalize_constants(self):
        self.mock.output = textwrap.dedent("""\
        <constants>
        [
            {"name": "robot1", "type": "robot", "desc": null},
            {"name": "station", "type": "location", "desc": "charging station"}
        ]
        </constants>""")
        results, _ = self.builder.formalize_component(
            model=self.mock,
            component_class=Constant,
            prompt_template=self.prompt,
            domain_desc="test",
        )
        consts = results[Constant]
        self.assertEqual(len(consts), 2)
        self.assertEqual(consts[0].name, "robot1")
        self.assertEqual(consts[0].type, "robot")

    # ---- REQUIREMENTS ----

    def test_formalize_requirements(self):
        self.mock.output = textwrap.dedent("""\
        <requirements>
        [
            {"name": ":strips", "desc": null},
            {"name": ":typing", "desc": null},
            {"name": ":negative-preconditions", "desc": null}
        ]
        </requirements>""")
        results, _ = self.builder.formalize_component(
            model=self.mock,
            component_class=Requirement,
            prompt_template=self.prompt,
            domain_desc="test",
        )
        reqs = results[Requirement]
        self.assertEqual(len(reqs), 3)
        self.assertEqual(reqs[0].name, ":strips")
        self.assertEqual(reqs[2].name, ":negative-preconditions")

    # ---- ACTIONS ----

    def test_formalize_action_simple(self):
        self.mock.output = textwrap.dedent("""\
        <actions>
        [
            {
                "name": "move",
                "params": [{"variable": "?r", "type": "rover"}, {"variable": "?from", "type": "waypoint"}, {"variable": "?to", "type": "waypoint"}],
                "preconditions": {"conditions": ["(at ?r ?from)"], "desc": null},
                "effects": {"add": ["(at ?r ?to)"], "delete": ["(at ?r ?from)"], "numeric": [], "conditional": [], "desc": null},
                "desc": null
            }
        ]
        </actions>""")
        results, _ = self.builder.formalize_component(
            model=self.mock,
            component_class=Action,
            prompt_template=self.prompt,
            domain_desc="test",
        )
        actions = results[Action]
        self.assertEqual(len(actions), 1)
        a = actions[0]
        self.assertEqual(a.name, "move")
        self.assertEqual(len(a.params), 3)

    def test_formalize_action_with_conditional_and_numeric_effects(self):
        self.mock.output = textwrap.dedent("""\
        <actions>
        [
            {
                "name": "drive",
                "params": [{"variable": "?r", "type": "rover"}],
                "preconditions": {"conditions": [{"operator": "not", "condition": "(battery-dead ?r)"}], "desc": null},
                "effects": {
                    "add": ["(at ?r ?l)"],
                    "delete": [],
                    "numeric": ["(decrease (battery ?r) 5.0)"],
                    "conditional": [
                        {
                            "condition": ["(has-camera ?r)"],
                            "effect": {"add": ["(photo-taken ?r)"], "delete": [], "numeric": []}
                        }
                    ],
                    "desc": null
                },
                "desc": null
            }
        ]
        </actions>""")
        results, _ = self.builder.formalize_component(
            model=self.mock,
            component_class=Action,
            prompt_template=self.prompt,
            domain_desc="test",
        )
        a = results[Action][0]
        self.assertTrue(len(a.preconditions.conditions) > 0)
        self.assertEqual(len(a.effects.numeric), 1)
        self.assertEqual(len(a.effects.conditional), 1)
        ce = a.effects.conditional[0]
        self.assertIn("(has-camera ?r)", ce.condition)
        self.assertIn("(photo-taken ?r)", ce.effect["add"])

    def test_formalize_action_empty_preconditions(self):
        self.mock.output = textwrap.dedent("""\
        <actions>
        [
            {
                "name": "nop",
                "params": [],
                "preconditions": {"conditions": [], "desc": null},
                "effects": {"add": [], "delete": [], "numeric": [], "conditional": [], "desc": null},
                "desc": null
            }
        ]
        </actions>""")
        results, _ = self.builder.formalize_component(
            model=self.mock,
            component_class=Action,
            prompt_template=self.prompt,
            domain_desc="test",
        )
        a = results[Action][0]
        self.assertEqual(len(a.preconditions.conditions), 0)
        self.assertEqual(len(a.effects.add), 0)

    # ---- MULTI-CLASS EXTRACTION ----

    def test_formalize_multiple_components(self):
        self.mock.output = textwrap.dedent("""\
        <types>
        [
            {"name": "robot", "parent": "object", "desc": null}
        ]
        </types>
        <predicates>
        [
            {"name": "at", "params": [{"variable": "?r", "type": "robot"}], "desc": null}
        ]
        </predicates>""")
        results, _ = self.builder.formalize_component(
            model=self.mock,
            component_class=[PDDLType, Predicate],
            prompt_template="Extract types and predicates",
            domain_desc="test",
        )
        self.assertIn(PDDLType, results)
        self.assertIn(Predicate, results)
        self.assertEqual(len(results[PDDLType]), 1)
        self.assertEqual(len(results[Predicate]), 1)

    # ---- LLM OUTPUT PRESERVED ----

    def test_llm_output_preserved(self):
        self.mock.output = (
            '<types>\n[{"name": "robot", "parent": "object", "desc": null}]\n</types>'
        )
        results, llm_output = self.builder.formalize_component(
            model=self.mock,
            component_class=PDDLType,
            prompt_template=self.prompt,
            domain_desc="test",
        )
        self.assertIn("robot", llm_output)

    # ---- RETRY ON FAILURE ----

    def test_retry_eventually_succeeds(self):
        """Simulates first attempt failing (bad XML), second succeeding."""

        class RetryMock(MockLLM):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def query(self, prompt):
                self.call_count += 1
                if self.call_count == 1:
                    return "bad output"
                return '<types>\n[{"name": "robot", "parent": "object", "desc": null}]\n</types>'

        retry_mock = RetryMock()
        results, _ = self.builder.formalize_component(
            model=retry_mock,
            component_class=PDDLType,
            prompt_template=self.prompt,
            domain_desc="test",
            max_retries=3,
        )
        self.assertEqual(len(results[PDDLType]), 1)
        self.assertEqual(results[PDDLType][0].name, "robot")


class TestDomainBuilderGenerateDomain(unittest.TestCase):
    """Tests for DomainBuilder.generate_domain()."""

    def setUp(self):
        self.builder = DomainBuilder()

    def normalize(self, s):
        return "\n".join(l.strip() for l in textwrap.dedent(s).strip().splitlines())

    def test_generate_domain_basic(self):
        domain = DomainDetails(
            name="test-domain",
            requirements=[Requirement(name=":strips"), Requirement(name=":typing")],
            types=[PDDLType(name="robot", parent="object")],
            predicates=[
                Predicate(name="at", params=[Parameter(variable="?r", type="robot")])
            ],
            actions=[
                Action(
                    name="nop",
                    params=[],
                    preconditions=ActionPrecondition(),
                    effects=ActionEffect(),
                )
            ],
        )
        result = self.builder.generate_domain(domain)
        self.assertIn("(define (domain test-domain)", result)
        self.assertIn(":strips", result)
        self.assertIn(":typing", result)
        self.assertIn(":types", result)
        self.assertIn(":predicates", result)
        self.assertIn("(:action nop", result)

    def test_generate_domain_with_all_sections(self):
        domain = DomainDetails(
            name="full-domain",
            requirements=[Requirement(name=":strips"), Requirement(name=":typing")],
            types=[
                PDDLType(name="arm", parent="object"),
                PDDLType(name="block", parent="object"),
            ],
            constants=[Constant(name="table1", type="object")],
            predicates=[
                Predicate(
                    name="on",
                    params=[
                        Parameter(variable="?b1", type="block"),
                        Parameter(variable="?b2", type="block"),
                    ],
                ),
                Predicate(
                    name="holding",
                    params=[
                        Parameter(variable="?a", type="arm"),
                        Parameter(variable="?b", type="block"),
                    ],
                ),
                Predicate(
                    name="clear", params=[Parameter(variable="?b", type="block")]
                ),
            ],
            functions=[
                Function(name="weight", params=[Parameter(variable="?b", type="block")])
            ],
            actions=[
                Action(
                    name="stack",
                    params=[
                        Parameter(variable="?b1", type="block"),
                        Parameter(variable="?b2", type="block"),
                        Parameter(variable="?a", type="arm"),
                    ],
                    preconditions=ActionPrecondition(
                        conditions=["(holding ?a ?b1)", "(clear ?b2)"]
                    ),
                    effects=ActionEffect(
                        add=["(on ?b1 ?b2)"],
                        delete=["(holding ?a ?b1)", "(clear ?b2)"],
                    ),
                )
            ],
        )
        result = self.builder.generate_domain(domain)
        self.assertIn(":constants", result)
        self.assertIn(":functions", result)
        self.assertIn("(:action stack", result)
        self.assertIn("?b1 - block", result)

    def test_generate_domain_no_predicates_warning(self):
        domain = DomainDetails(name="warn-domain")
        f = io.StringIO()
        with redirect_stdout(f):
            result = self.builder.generate_domain(domain)
        self.assertIn("WARNING", f.getvalue())
        self.assertIn("(define (domain warn-domain)", result)

    def test_generate_domain_no_actions_warning(self):
        domain = DomainDetails(
            name="no-actions", predicates=[Predicate(name="p", params=[])]
        )
        f = io.StringIO()
        with redirect_stdout(f):
            result = self.builder.generate_domain(domain)
        self.assertIn("WARNING", f.getvalue())

    def test_generate_domain_with_derived_predicates(self):
        domain = DomainDetails(
            name="derived",
            requirements=[
                Requirement(name=":strips"),
                Requirement(name=":derived-predicates"),
            ],
            predicates=[
                Predicate(
                    name="battery-low", params=[Parameter(variable="?r", type="object")]
                )
            ],
            derived_predicates=[
                DerivedPredicate(
                    name="needs-charge",
                    params=[Parameter(variable="?r", type="object")],
                    condition="(battery-low ?r)",
                )
            ],
        )
        result = self.builder.generate_domain(domain)
        self.assertIn(":derived", result)
        self.assertIn("needs-charge", result)

    def test_generate_domain_with_constraint(self):
        domain = DomainDetails(
            name="constrained",
            requirements=[
                Requirement(name=":strips"),
                Requirement(name=":constraints"),
            ],
            predicates=[
                Predicate(
                    name="active", params=[Parameter(variable="?r", type="object")]
                )
            ],
            constraint=[
                Constraint(condition={"operator": "always", "condition": "(active ?r)"})
            ],
        )
        result = self.builder.generate_domain(domain)
        self.assertIn(":constraints", result)
        self.assertIn("always", result)

    def test_generate_domain_no_requirements_auto(self):
        """If no requirements are set, generate_requirements is called internally."""
        domain = DomainDetails(
            name="auto-req",
            types=[PDDLType(name="robot", parent="object")],
            predicates=[
                Predicate(name="at", params=[Parameter(variable="?r", type="robot")])
            ],
            actions=[
                Action(
                    name="nop",
                    params=[],
                    preconditions=ActionPrecondition(),
                    effects=ActionEffect(),
                )
            ],
        )
        result = self.builder.generate_domain(domain)
        self.assertIn(":typing", result)
        self.assertIn(":strips", result)

    def test_generated_pddl_output_format(self):
        """Check the structure of the generated PDDL output."""
        domain = DomainDetails(
            name="format-check",
            types=[PDDLType(name="obj", parent="object")],
            predicates=[
                Predicate(name="p", params=[Parameter(variable="?x", type="obj")])
            ],
            actions=[
                Action(
                    name="act",
                    params=[Parameter(variable="?x", type="obj")],
                    preconditions=ActionPrecondition(conditions=["(p ?x)"]),
                    effects=ActionEffect(add=["(p ?x)"], delete=[]),
                )
            ],
        )
        result = self.builder.generate_domain(domain)
        # Should start with define
        self.assertTrue(result.startswith("(define"))
        # Should end with )
        self.assertTrue(result.strip().endswith(")"))
        # Should contain action
        self.assertIn("(:action act", result)


class TestDomainBuilderGenerateRequirements(unittest.TestCase):
    """Tests for DomainBuilder.generate_requirements()."""

    def setUp(self):
        self.builder = DomainBuilder()

    def _names(self, reqs):
        return sorted(r.name for r in reqs)

    def test_baseline_strips(self):
        d = DomainDetails(name="t")
        n = self._names(self.builder.generate_requirements(d))
        self.assertIn(":strips", n)

    def test_typing(self):
        d = DomainDetails(name="t", types=[PDDLType(name="r", parent="object")])
        n = self._names(self.builder.generate_requirements(d))
        self.assertIn(":typing", n)

    def test_numeric_fluents(self):
        d = DomainDetails(name="t", functions=[Function(name="b", params=[])])
        n = self._names(self.builder.generate_requirements(d))
        self.assertIn(":numeric-fluents", n)

    def test_derived_predicates(self):
        dp = DerivedPredicate(
            name="can",
            params=[Parameter(variable="?x", type="o")],
            condition="(> (b ?x) 0)",
        )
        d = DomainDetails(name="t", derived_predicates=[dp])
        n = self._names(self.builder.generate_requirements(d))
        self.assertIn(":derived-predicates", n)

    def test_durative_actions(self):
        da = DurativeAction(
            name="fly",
            params=[Parameter(variable="?d", type="d")],
            duration=[">= ?duration 5.0"],
            conditions=DurativeActionConditions(),
            effects=DurativeActionEffect(),
        )
        d = DomainDetails(name="t", durative_actions=[da])
        n = self._names(self.builder.generate_requirements(d))
        self.assertIn(":durative-actions", n)

    def test_negative_preconditions(self):
        a = Action(
            name="go",
            params=[],
            preconditions=ActionPrecondition(
                conditions=[{"operator": "not", "condition": "(blocked)"}]
            ),
            effects=ActionEffect(),
        )
        d = DomainDetails(name="t", actions=[a])
        n = self._names(self.builder.generate_requirements(d))
        self.assertIn(":negative-preconditions", n)

    def test_conditional_effects(self):
        ce = ConditionalEffect(
            condition=["(has-item)"], effect={"add": [], "delete": [], "numeric": []}
        )
        a = Action(
            name="proc",
            params=[],
            preconditions=ActionPrecondition(),
            effects=ActionEffect(conditional=[ce]),
        )
        d = DomainDetails(name="t", actions=[a])
        n = self._names(self.builder.generate_requirements(d))
        self.assertIn(":conditional-effects", n)

    def test_equality(self):
        a = Action(
            name="eq",
            params=[],
            preconditions=ActionPrecondition(conditions=["(>= ?x ?y)"]),
            effects=ActionEffect(),
        )
        d = DomainDetails(name="t", actions=[a])
        n = self._names(self.builder.generate_requirements(d))
        self.assertIn(":equality", n)

    def test_constraints_from_constraint_block(self):
        c = Constraint(condition={"operator": "always", "condition": "(> (b ?r) 0)"})
        d = DomainDetails(name="t", constraint=[c])
        n = self._names(self.builder.generate_requirements(d))
        self.assertIn(":constraints", n)

    def test_quantified_preconditions(self):
        a1 = Action(
            name="all",
            params=[],
            preconditions=ActionPrecondition(
                conditions=[
                    {
                        "quantifier": "forall",
                        "parameters": [{"variable": "?x", "type": "t"}],
                        "conditions": ["(p ?x)"],
                    }
                ]
            ),
            effects=ActionEffect(),
        )
        a2 = Action(
            name="some",
            params=[],
            preconditions=ActionPrecondition(
                conditions=[
                    {
                        "quantifier": "exists",
                        "parameters": [{"variable": "?y", "type": "t"}],
                        "conditions": ["(p ?y)"],
                    }
                ]
            ),
            effects=ActionEffect(),
        )
        d = DomainDetails(name="t", actions=[a1, a2])
        n = self._names(self.builder.generate_requirements(d))
        self.assertIn(":quantified-preconditions", n)
        self.assertNotIn(":existential-preconditions", n)
        self.assertNotIn(":universal-preconditions", n)

    def test_timed_initial_literals(self):
        problem = ProblemDetails(
            name="p",
            domain_name="d",
            initial_state=InitialState(
                timed_facts=[TimedFact(time=5.0, fact="(event)")]
            ),
        )
        d = DomainDetails(name="t")
        n = self._names(self.builder.generate_requirements(d, problem_details=problem))
        self.assertIn(":timed-initial-literals", n)

    def test_action_costs(self):
        f = Function(name="total-cost", params=[])
        a = Action(
            name="do",
            params=[],
            preconditions=ActionPrecondition(),
            effects=ActionEffect(numeric=["(increase (total-cost) 1)"]),
        )
        d = DomainDetails(name="t", functions=[f], actions=[a])
        n = self._names(self.builder.generate_requirements(d))
        self.assertIn(":action-costs", n)

    def test_durative_inequalities(self):
        da = DurativeAction(
            name="fly",
            params=[Parameter(variable="?d", type="d")],
            duration=["(< ?duration 10.0)"],
            conditions=DurativeActionConditions(),
            effects=DurativeActionEffect(),
        )
        d = DomainDetails(name="t", durative_actions=[da])
        n = self._names(self.builder.generate_requirements(d))
        self.assertIn(":durative-inequalities", n)

    def test_events_and_processes(self):
        evt = Event(
            name="e",
            params=[],
            preconditions=ActionPrecondition(),
            effects=ActionEffect(),
        )
        d = DomainDetails(name="t", events=[evt])
        n = self._names(self.builder.generate_requirements(d))
        self.assertIn(":time", n)


class TestDomainBuilderSetters(unittest.TestCase):
    """Tests for DomainBuilder set_* methods."""

    def setUp(self):
        self.builder = DomainBuilder()

    def test_set_types(self):
        t = PDDLType(name="robot", parent="object")
        self.builder.set_types(t)
        self.assertEqual(len(self.builder.domain_details.types), 1)
        self.assertEqual(self.builder.domain_details.types[0].name, "robot")

    def test_set_types_append(self):
        self.builder.set_types(PDDLType(name="a", parent="object"))
        self.builder.set_types(PDDLType(name="b", parent="object"), append=True)
        self.assertEqual(len(self.builder.domain_details.types), 2)

    def test_set_types_overwrite(self):
        self.builder.set_types(PDDLType(name="a", parent="object"))
        self.builder.set_types(PDDLType(name="b", parent="object"))
        self.assertEqual(len(self.builder.domain_details.types), 1)
        self.assertEqual(self.builder.domain_details.types[0].name, "b")

    def test_set_types_list(self):
        types = [
            PDDLType(name="a", parent="object"),
            PDDLType(name="b", parent="object"),
        ]
        self.builder.set_types(types)
        self.assertEqual(len(self.builder.domain_details.types), 2)

    def test_set_types_none_clears(self):
        self.builder.set_types(PDDLType(name="a", parent="object"))
        self.builder.set_types(None)
        self.assertEqual(len(self.builder.domain_details.types), 0)

    def test_set_predicates(self):
        p = Predicate(name="at", params=[Parameter(variable="?r", type="robot")])
        self.builder.set_predicates(p)
        self.assertEqual(len(self.builder.domain_details.predicates), 1)

    def test_set_actions(self):
        a = Action(
            name="move",
            params=[],
            preconditions=ActionPrecondition(),
            effects=ActionEffect(),
        )
        self.builder.set_actions(a)
        self.assertEqual(len(self.builder.domain_details.actions), 1)

    def test_set_domain_name(self):
        self.builder.set_domain_name("my-domain")
        self.assertEqual(self.builder.domain_details.name, "my-domain")

    def test_set_domain_name_default(self):
        self.builder.set_domain_name()
        self.assertEqual(self.builder.domain_details.name, "domain-placeholder")

    def test_set_constants(self):
        c = Constant(name="base", type="location")
        self.builder.set_constants(c)
        self.assertEqual(len(self.builder.domain_details.constants), 1)

    def test_set_functions(self):
        f = Function(name="battery", params=[Parameter(variable="?r", type="robot")])
        self.builder.set_functions(f)
        self.assertEqual(len(self.builder.domain_details.functions), 1)

    def test_set_derived_predicates(self):
        dp = DerivedPredicate(
            name="can",
            params=[Parameter(variable="?r", type="robot")],
            condition="(> (b ?r) 0)",
        )
        self.builder.set_derived_predicates(dp)
        self.assertEqual(len(self.builder.domain_details.derived_predicates), 1)

    def test_set_constraints(self):
        c = Constraint(condition={"operator": "always", "condition": "(p ?r)"})
        self.builder.set_constraints(c)
        self.assertEqual(len(self.builder.domain_details.constraint), 1)

    def test_set_requirements(self):
        r = Requirement(name=":strips")
        self.builder.set_requirements(r)
        self.assertEqual(len(self.builder.domain_details.requirements), 1)

    def test_set_append_multiple_types(self):
        types = [
            PDDLType(name="a", parent="object"),
            PDDLType(name="b", parent="object"),
        ]
        self.builder.set_types(types)
        self.builder.set_types(PDDLType(name="c", parent="object"), append=True)
        self.assertEqual(len(self.builder.domain_details.types), 3)

    def test_set_events(self):
        e = Event(
            name="e",
            params=[],
            preconditions=ActionPrecondition(),
            effects=ActionEffect(),
        )
        self.builder.set_events(e)
        self.assertEqual(len(self.builder.domain_details.events), 1)

    def test_set_processes(self):
        p = Process(
            name="p",
            params=[],
            preconditions=ActionPrecondition(),
            effects=ActionEffect(),
        )
        self.builder.set_processes(p)
        self.assertEqual(len(self.builder.domain_details.processes), 1)

    def test_set_durative_actions(self):
        da = DurativeAction(
            name="fly",
            params=[Parameter(variable="?d", type="d")],
            duration=[">= ?duration 5.0"],
            conditions=DurativeActionConditions(),
            effects=DurativeActionEffect(),
        )
        self.builder.set_durative_actions(da)
        self.assertEqual(len(self.builder.domain_details.durative_actions), 1)

    def test_set_domain_desc(self):
        self.builder.set_domain_desc("A robot domain")
        self.assertEqual(self.builder.domain_details.desc, "A robot domain")

    def test_set_domain_desc_none(self):
        self.builder.set_domain_desc()
        self.assertIsNone(self.builder.domain_details.desc)


class TestDomainBuilderConstruction(unittest.TestCase):
    """Tests for DomainBuilder __init__."""

    def test_default_construction(self):
        b = DomainBuilder()
        self.assertEqual(b.domain_details.name, "domain-placeholder")

    def test_with_kwargs(self):
        b = DomainBuilder(name="my-domain")
        self.assertEqual(b.domain_details.name, "my-domain")

    def test_with_domain_details(self):
        dd = DomainDetails(name="provided-domain")
        b = DomainBuilder(domain_details=dd)
        self.assertIs(b.domain_details, dd)

    def test_with_problem_details(self):
        pd = ProblemDetails(name="prob", domain_name="dom")
        b = DomainBuilder(problem_details=pd)
        self.assertEqual(b.domain_details.name, "domain-placeholder")
        self.assertIs(b.problem_details, pd)


if __name__ == "__main__":
    unittest.main()
