import textwrap
import unittest

from l2p.utils.pddl_parser import parse_domain_pddl, parse_problem_pddl


class TestParseDomainBasic(unittest.TestCase):

    def test_minimal_domain(self):
        pddl = textwrap.dedent("""\
        (define (domain empty)
            (:requirements :strips)
            (:predicates (p))
            (:action a :parameters () :precondition (p) :effect (p))
        )""")
        dd = parse_domain_pddl(pddl)
        self.assertEqual(dd.name, "empty")
        self.assertEqual(len(dd.requirements), 1)
        self.assertEqual(dd.requirements[0].name, ":strips")

    def test_domain_with_types(self):
        pddl = textwrap.dedent("""\
        (define (domain typed)
            (:requirements :strips :typing)
            (:types robot location)
            (:constants base - location)
            (:predicates (at ?r - robot ?l - location))
            (:action nop :parameters (?r - robot) :precondition (at ?r base) :effect (at ?r base))
        )""")
        dd = parse_domain_pddl(pddl)
        self.assertEqual(len(dd.types), 2)
        type_names = {t.name for t in dd.types}
        self.assertIn("robot", type_names)
        self.assertIn("location", type_names)
        for t in dd.types:
            self.assertEqual(t.parent, "object")

    def test_domain_with_hierarchy(self):
        pddl = textwrap.dedent("""\
        (define (domain hier)
            (:requirements :strips :typing)
            (:types drone rover)
            (:predicates (active ?v - drone))
            (:action nop :parameters (?v - drone) :precondition (active ?v) :effect (active ?v))
        )""")
        dd = parse_domain_pddl(pddl)
        self.assertEqual(len(dd.types), 2)
        parents = {t.name: t.parent for t in dd.types}
        self.assertIn("drone", parents)
        self.assertIn("rover", parents)
        self.assertEqual(parents["drone"], "object")

    def test_domain_with_constants(self):
        pddl = textwrap.dedent("""\
        (define (domain consts)
            (:requirements :strips :typing)
            (:types location)
            (:constants base - location outpost - location)
            (:predicates (at ?l - location))
            (:action nop :parameters () :precondition (at base) :effect (at outpost))
        )""")
        dd = parse_domain_pddl(pddl)
        self.assertEqual(len(dd.constants), 2)
        const_map = {c.name: c.type for c in dd.constants}
        self.assertEqual(const_map["base"], "location")
        self.assertEqual(const_map["outpost"], "location")

    def test_domain_no_optional_sections(self):
        pddl = textwrap.dedent("""\
        (define (domain bare)
            (:requirements :strips)
            (:predicates (p))
            (:action a :parameters () :precondition (p) :effect (p))
        )""")
        dd = parse_domain_pddl(pddl)
        self.assertEqual(dd.types, [])
        self.assertEqual(dd.constants, [])
        self.assertEqual(dd.functions, [])
        self.assertEqual(dd.derived_predicates, [])

    def test_domain_untyped_params_default_to_object(self):
        pddl = textwrap.dedent("""\
        (define (domain untyped)
            (:requirements :strips)
            (:predicates (p ?x))
            (:action a :parameters (?x) :precondition (p ?x) :effect (p ?x))
        )""")
        dd = parse_domain_pddl(pddl)
        self.assertEqual(len(dd.predicates), 1)
        self.assertEqual(dd.predicates[0].params[0].type, "object")
        self.assertEqual(dd.actions[0].params[0].type, "object")


class TestParseDomainPreconditions(unittest.TestCase):

    def test_and_precondition(self):
        pddl = textwrap.dedent("""\
        (define (domain test)
            (:requirements :strips)
            (:predicates (p) (q))
            (:action a :parameters () :precondition (and (p) (q)) :effect (p))
        )""")
        dd = parse_domain_pddl(pddl)
        conds = dd.actions[0].preconditions.conditions
        self.assertEqual(len(conds), 2)
        self.assertEqual(conds, ["(p)", "(q)"])

    def test_negative_precondition(self):
        pddl = textwrap.dedent("""\
        (define (domain test)
            (:requirements :strips :negative-preconditions)
            (:predicates (p) (q))
            (:action a :parameters () :precondition (and (p) (not (q))) :effect (p))
        )""")
        dd = parse_domain_pddl(pddl)
        conds = dd.actions[0].preconditions.conditions
        self.assertEqual(len(conds), 2)
        self.assertIsInstance(conds[0], str)
        self.assertEqual(conds[0], "(p)")
        self.assertIsInstance(conds[1], dict)
        self.assertEqual(conds[1]["operator"], "not")
        self.assertEqual(conds[1]["condition"], "(q)")

    def test_or_precondition(self):
        pddl = textwrap.dedent("""\
        (define (domain test)
            (:requirements :strips :disjunctive-preconditions)
            (:predicates (p) (q))
            (:action a :parameters () :precondition (or (p) (q)) :effect (p))
        )""")
        dd = parse_domain_pddl(pddl)
        cond = dd.actions[0].preconditions.conditions[0]
        self.assertIsInstance(cond, dict)
        self.assertEqual(cond["operator"], "or")
        self.assertEqual(len(cond["conditions"]), 2)

    def test_imply_precondition(self):
        pddl = textwrap.dedent("""\
        (define (domain test)
            (:requirements :adl)
            (:types obj)
            (:predicates (p ?x - obj) (q ?x - obj))
            (:action a :parameters (?x - obj) :precondition (imply (p ?x) (q ?x)) :effect (p ?x))
        )""")
        dd = parse_domain_pddl(pddl)
        cond = dd.actions[0].preconditions.conditions[0]
        self.assertIsInstance(cond, dict)
        self.assertEqual(cond["operator"], "imply")

    def test_forall_precondition(self):
        pddl = textwrap.dedent("""\
        (define (domain test)
            (:requirements :strips :typing :universal-preconditions)
            (:types block)
            (:predicates (clear ?b - block))
            (:action a :parameters (?b - block)
                :precondition (and (clear ?b) (forall (?x - block) (clear ?x)))
                :effect (clear ?b))
        )""")
        dd = parse_domain_pddl(pddl)
        conds = dd.actions[0].preconditions.conditions
        self.assertEqual(len(conds), 2)
        forall = conds[1]
        self.assertIsInstance(forall, dict)
        self.assertEqual(forall["quantifier"], "forall")
        self.assertEqual(len(forall["parameters"]), 1)
        self.assertEqual(forall["parameters"][0]["variable"], "?x")
        self.assertEqual(forall["parameters"][0]["type"], "block")

    def test_exists_precondition(self):
        pddl = textwrap.dedent("""\
        (define (domain test)
            (:requirements :strips :typing :existential-preconditions)
            (:types block)
            (:predicates (clear ?b - block))
            (:action a :parameters (?b - block)
                :precondition (exists (?x - block) (clear ?x))
                :effect (clear ?b))
        )""")
        dd = parse_domain_pddl(pddl)
        cond = dd.actions[0].preconditions.conditions[0]
        self.assertEqual(cond["quantifier"], "exists")

    def test_single_predicate_precondition_no_and(self):
        pddl = textwrap.dedent("""\
        (define (domain test)
            (:requirements :strips)
            (:predicates (p))
            (:action a :parameters () :precondition (p) :effect (p))
        )""")
        dd = parse_domain_pddl(pddl)
        self.assertEqual(len(dd.actions[0].preconditions.conditions), 1)
        self.assertEqual(dd.actions[0].preconditions.conditions[0], "(p)")


class TestParseDomainEffects(unittest.TestCase):

    def test_add_and_delete_effects(self):
        pddl = textwrap.dedent("""\
        (define (domain test)
            (:requirements :strips)
            (:predicates (p) (q))
            (:action a :parameters () :precondition (p) :effect (and (q) (not (p))))
        )""")
        dd = parse_domain_pddl(pddl)
        eff = dd.actions[0].effects
        self.assertEqual(eff.add, ["(q)"])
        self.assertEqual(eff.delete, ["(p)"])
        self.assertEqual(eff.numeric, [])
        self.assertEqual(eff.conditional, [])

    def test_conditional_effect(self):
        pddl = textwrap.dedent("""\
        (define (domain test)
            (:requirements :strips :conditional-effects)
            (:predicates (p) (q) (r))
            (:action a :parameters () :precondition (p)
                :effect (and (q) (when (p) (r))))
        )""")
        dd = parse_domain_pddl(pddl)
        eff = dd.actions[0].effects
        self.assertIn("(q)", eff.add)
        self.assertEqual(len(eff.conditional), 1)
        ce = eff.conditional[0]
        self.assertEqual(len(ce.condition), 1)
        self.assertEqual(ce.condition[0], "(p)")
        self.assertIn("(r)", ce.effect["add"])

    def test_numeric_effects(self):
        pddl = textwrap.dedent("""\
        (define (domain test)
            (:requirements :strips :typing :numeric-fluents)
            (:types robot location)
            (:constants base - location)
            (:predicates (at ?r - robot ?l - location))
            (:functions (battery ?r - robot) (capacity))
            (:action charge :parameters (?r - robot)
                :precondition (at ?r base)
                :effect (and (increase (capacity) 1) (assign (battery ?r) 100)))
        )""")
        dd = parse_domain_pddl(pddl)
        eff = dd.actions[0].effects
        self.assertTrue(len(eff.numeric) >= 1)
        self.assertIn("(increase (capacity) 1)", eff.numeric)
        self.assertIn("(assign (battery ?r) 100)", eff.numeric)

    def test_empty_effect(self):
        pddl = textwrap.dedent("""\
        (define (domain test)
            (:requirements :strips)
            (:predicates (p))
            (:action a :parameters () :precondition (p) :effect (p))
        )""")
        dd = parse_domain_pddl(pddl)
        eff = dd.actions[0].effects
        self.assertIn("(p)", eff.add)


class TestParseDomainDerived(unittest.TestCase):

    def test_derived_predicate(self):
        pddl = textwrap.dedent("""\
        (define (domain test)
            (:requirements :strips :typing :derived-predicates)
            (:types robot)
            (:predicates (battery-low ?r - robot) (charging ?r - robot) (needs-charge ?r - robot))
            (:derived (needs-charge ?r - robot) (and (battery-low ?r) (not (charging ?r))))
            (:action a :parameters (?r - robot) :precondition (battery-low ?r) :effect (charging ?r))
        )""")
        dd = parse_domain_pddl(pddl)
        self.assertEqual(len(dd.derived_predicates), 1)
        dp = dd.derived_predicates[0]
        self.assertEqual(dp.name, "needs-charge")
        self.assertEqual(dp.params[0].variable, "?r")
        self.assertIsInstance(dp.condition, dict)
        self.assertEqual(dp.condition["operator"], "and")


class TestParseProblemBasic(unittest.TestCase):

    def test_minimal_problem(self):
        domain_pddl = textwrap.dedent("""\
        (define (domain d) (:requirements :strips) (:predicates (p)) (:action a :parameters () :precondition (p) :effect (p)))
        """)
        problem_pddl = textwrap.dedent("""\
        (define (problem p) (:domain d) (:objects) (:init (p)) (:goal (p)))
        """)
        pd = parse_problem_pddl(problem_pddl)
        self.assertEqual(pd.name, "p")
        self.assertEqual(pd.domain_name, "d")

    def test_problem_with_objects(self):
        problem_pddl = textwrap.dedent("""\
        (define (problem prob) (:domain d)
            (:objects rover1 - robot rover2 - robot base - location)
            (:init (at rover1 base)) (:goal (at rover1 base)))
        """)
        pd = parse_problem_pddl(problem_pddl)
        self.assertEqual(len(pd.objects), 3)
        obj_map = {o.name: o.type for o in pd.objects}
        self.assertEqual(obj_map["rover1"], "robot")
        self.assertEqual(obj_map["rover2"], "robot")
        self.assertEqual(obj_map["base"], "location")

    def test_problem_init_facts(self):
        problem_pddl = textwrap.dedent("""\
        (define (problem prob) (:domain d)
            (:objects a - block b - block)
            (:init (ontable a) (ontable b) (clear a) (clear b))
            (:goal (ontable a)))
        """)
        pd = parse_problem_pddl(problem_pddl)
        self.assertEqual(len(pd.initial_state.facts), 4)
        self.assertIn("(ontable a)", pd.initial_state.facts)
        self.assertIn("(clear b)", pd.initial_state.facts)

    def test_problem_goal_and(self):
        problem_pddl = textwrap.dedent("""\
        (define (problem prob) (:domain d)
            (:objects a - block b - block)
            (:init (ontable a) (ontable b))
            (:goal (and (ontable a) (ontable b))))
        """)
        pd = parse_problem_pddl(problem_pddl)
        self.assertEqual(len(pd.goal_state.conditions), 2)
        self.assertEqual(pd.goal_state.conditions[0], "(ontable a)")

    def test_problem_goal_single_predicate(self):
        problem_pddl = textwrap.dedent("""\
        (define (problem prob) (:domain d)
            (:objects a - block) (:init (ontable a))
            (:goal (ontable a)))
        """)
        pd = parse_problem_pddl(problem_pddl)
        self.assertEqual(len(pd.goal_state.conditions), 1)
        self.assertEqual(pd.goal_state.conditions[0], "(ontable a)")

    def test_problem_with_metric(self):
        problem_pddl = textwrap.dedent("""\
        (define (problem prob) (:domain d)
            (:objects) (:init) (:goal (p))
            (:metric minimize (total-time)))
        """)
        pd = parse_problem_pddl(problem_pddl)
        self.assertIsNotNone(pd.metric)
        self.assertEqual(pd.metric.optimization, "minimize")
        self.assertEqual(pd.metric.expression, "(total-time)")

    def test_problem_no_metric(self):
        problem_pddl = textwrap.dedent("""\
        (define (problem prob) (:domain d)
            (:objects) (:init) (:goal (p)))
        """)
        pd = parse_problem_pddl(problem_pddl)
        self.assertIsNone(pd.metric)


class TestParseRoundtrip(unittest.TestCase):
    """Parse a PDDL string, then re-generate it, checking structural fidelity."""

    def test_domain_roundtrip(self):
        original = textwrap.dedent("""\
        (define (domain logistics)
            (:requirements :strips :typing :negative-preconditions)
            (:types location)
            (:constants base - location)
            (:predicates (at ?t - location) (connected ?from - location ?to - location))
            (:action move
                :parameters (?from - location ?to - location)
                :precondition (and (at ?from) (connected ?from ?to) (not (at ?to)))
                :effect (and (not (at ?from)) (at ?to)))
        )""")
        dd = parse_domain_pddl(original)
        self.assertEqual(dd.name, "logistics")
        self.assertEqual(len(dd.types), 1)
        self.assertEqual(dd.types[0].name, "location")
        self.assertEqual(len(dd.constants), 1)
        self.assertEqual(dd.constants[0].name, "base")
        self.assertEqual(len(dd.actions), 1)
        action = dd.actions[0]
        self.assertEqual(action.name, "move")
        self.assertEqual(len(action.params), 2)
        self.assertTrue(
            any(
                isinstance(c, dict) and c.get("operator") == "not"
                for c in action.preconditions.conditions
            )
        )

    def test_problem_roundtrip(self):
        problem_pddl = textwrap.dedent("""\
        (define (problem p1) (:domain d)
            (:objects r1 - robot wp1 - location)
            (:init (at r1 wp1) (= (battery r1) 100))
            (:goal (and (at r1 wp1) (>= (battery r1) 50)))
        )""")
        pd = parse_problem_pddl(problem_pddl)
        self.assertEqual(pd.name, "p1")
        self.assertEqual(pd.domain_name, "d")
        self.assertEqual(len(pd.objects), 2)
        self.assertEqual(len(pd.initial_state.facts), 2)
        self.assertEqual(len(pd.goal_state.conditions), 2)


class TestParseEdgeCases(unittest.TestCase):

    def test_large_real_world_domain(self):
        """Logistics domain from the test fixtures."""
        pddl = textwrap.dedent("""\
        (define (domain logistics-strips)
            (:requirements :strips)
            (:predicates (AIRPLANE ?airplane) (AIRPORT ?airport) (CITY ?city)
                         (LOCATION ?loc) (OBJ ?obj) (TRUCK ?truck)
                         (at ?obj ?loc) (in ?obj1 ?obj2) (in-city ?obj ?city))
            (:action DRIVE-TRUCK
                :parameters (?truck ?loc-from ?loc-to ?city)
                :precondition (and (TRUCK ?truck) (LOCATION ?loc-from) (LOCATION ?loc-to)
                                   (CITY ?city) (at ?truck ?loc-from)
                                   (in-city ?loc-from ?city) (in-city ?loc-to ?city))
                :effect (and (not (at ?truck ?loc-from)) (at ?truck ?loc-to)))
        )""")
        dd = parse_domain_pddl(pddl)
        self.assertEqual(len(dd.predicates), 9)
        self.assertEqual(len(dd.actions), 1)
        self.assertEqual(dd.actions[0].name, "DRIVE-TRUCK")
        self.assertEqual(len(dd.actions[0].preconditions.conditions), 7)

    def test_malformed_pddl_raises(self):
        with self.assertRaises(Exception):
            parse_domain_pddl("this is not valid pddl")


if __name__ == "__main__":
    unittest.main()
