"""
PDDL Parsing: LLM Output → Pydantic Models

Two parsing directions:

1. Structured Output Extraction (LLM → Pydantic)
------------------------------------------------
Extracts JSON from LLM XML-tagged responses and validates them against L2P's
Pydantic models.

    blocks = parse_xml_tags(llm_output, "predicates")
    predicates = parse_component(blocks, Predicate, "predicates")
    single = parse_element(blocks, DomainDetails, "domain")

- ``parse_xml_tags`` — extracts content between ``<tag>...</tag>`` blocks
- ``parse_component`` — validates a list of model instances from JSON arrays
- ``parse_element`` — validates a single model instance from a JSON object

2. Raw PDDL String Parsing (PDDL → Pydantic)
----------------------------------------------
Converts standard PDDL syntax into L2P models using the ``pddl`` library:

    details = parse_domain_pddl(pddl_str)
    problem = parse_problem_pddl(pddl_str)
"""

import os
import re
import tempfile
from typing import Dict, List, Optional, Type, TypeVar

from pydantic import TypeAdapter, ValidationError
from pddl import parse_domain, parse_problem
from pddl.core import Action as PddlAction
from pddl.logic.base import And, Or, Not, Imply, ForallCondition, ExistsCondition
from pddl.logic.predicates import Predicate as PddlPredicate
from pddl.logic.effects import When, Forall as ForallEffect
from pddl.logic.functions import (
    Increase,
    Decrease,
    Assign,
    ScaleUp,
    ScaleDown,
    GreaterThan,
    LesserThan,
    GreaterEqualThan,
    LesserEqualThan,
)
from pddl.logic.terms import Constant as Variable
from l2p.utils.pddl_types import *

T = TypeVar("T")


# =============================================================================
# LLM OUTPUT EXTRACTION FUNCTIONS
# =============================================================================


def parse_xml_tags(llm_output: str, tag_name: str) -> List[str]:
    """
    Parses out XML tags to extract list of components.
    Args:
        llm_output (str): String output from LLM
        tag_name (str): Specific XML tag to extract blocks from
    Returns:
        Single list of content within XML blocks
    """
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, llm_output, flags=re.DOTALL)
    return [match.strip() for match in matches]


def parse_component(
    raw_blocks: List[str], model_class: Type[T], tag_name: str
) -> List[T]:
    """
    Processes list of content blocks containing specific model class. Returns first valid
    JSON object pertaining to that specific model class.
    Args:
        raw_blocks (List[str]): Content blocks (likely extracted from `parse_xml_tags`) to filter
        model_class (Type[T]): Specific model class to validate content block
        tag_name (str): Specific XML tag to extract blocks from
    Returns:
        List of JSON object models matching the specific model class
    """

    list_adapter = TypeAdapter(List[model_class])
    collected_errors = []

    # iterate over each raw content blocks
    for block in raw_blocks:
        try:
            # returns list of all content blocks containing correct json block
            return list_adapter.validate_json(block)
        except (ValidationError, ValueError) as list_err:
            # return single object as fallback
            try:
                single_obj = model_class.model_validate_json(block)
                return [single_obj]
            except (ValidationError, ValueError) as single_err:
                collected_errors.append(str(list_err))
                continue

    # return diagnostic message if no classes match
    import inspect as _inspect

    class_source = _inspect.getsource(model_class)
    error_message = (
        f"Error: The JSON provided inside <{tag_name}> failed validation.\n\n"
        f"--- YOUR ERRORS ---\n"
        f"{collected_errors[0] if collected_errors else 'No parsable blocks found.'}\n\n"
        f"--- EXPECTED CLASS DEFINITION FOR THE LIST ---\n"
        f"Please ensure your output is a JSON array `[...]` containing objects "
        f"that match this exact structure:\n\n"
        f"```python\n{class_source}```"
    )
    raise ValueError(error_message)


def parse_element(raw_blocks: List[str], model_class: Type[T], tag_name: str) -> T:
    """
    Processes list of content blocks containing specific model class that returns a single
    JSON object pertaining to that specific model class.
    Args:
        raw_blocks (List[str]): Content blocks (likely extracted from `parse_xml_tags`) to filter
        model_class (Type[T]): Specific model class to validate content block
        tag_name (str): Specific XML tag to extract blocks from
    Returns:
        A single JSON object model matching the specific model class
    """
    collected_errors = []

    for block in raw_blocks:
        try:
            return model_class.model_validate_json(block)
        except (ValidationError, ValueError) as e:
            collected_errors.append(str(e))
            continue

    import inspect as _inspect

    class_source = _inspect.getsource(model_class)
    error_message = (
        f"Error: The JSON provided inside <{tag_name}> failed validation.\n\n"
        f"--- YOUR ERRORS ---\n"
        f"{collected_errors[0]}\n\n"
        f"--- EXPECTED CLASS DEFINITION ---\n"
        f"Please ensure your output is a single JSON object `{{...}}` matching "
        f"this exact Pydantic model structure:\n\n"
        f"```python\n{class_source}```"
    )
    raise ValueError(error_message)


# =============================================================================
# PDDL STRING PARSING - convert raw PDDL into L2P Pydantic models
# **Side Note** - These functions leverage PDDL parser: https://github.com/AI-Planning/pddl
# CURRENTLY DOES NOT SUPPORT THE FOLLOWING
#   - Durative Actions
#   - Processes and Events
#   - Constraints
# =============================================================================


def parse_domain_pddl(domain_str: str) -> DomainDetails:
    """
    Parse a raw PDDL domain string into a :class:`DomainDetails` model.
    Args:
        domain_str (str): PDDL domain as a string
    Returns:
        DomainDetails: Pydantic L2P BaseModel
    """
    domain = write_temp_pddl(domain_str, parse_domain)

    return DomainDetails(
        name=domain.name,
        requirements=_convert_requirements(domain.requirements),
        types=_convert_types(domain.types),
        constants=_convert_constants(domain.constants),
        predicates=_convert_predicates(domain.predicates),
        functions=_convert_functions(domain.functions),
        derived_predicates=_convert_derived_predicates(
            getattr(domain, "derived_predicates", frozenset())
        ),
        actions=_convert_actions(domain.actions),
    )


def parse_problem_pddl(pddl_str: str) -> ProblemDetails:
    """
    Parse a raw PDDL problem string into a :class:`ProblemDetails` model.
    Args:
        pddl_str (str): PDDL domain as a string
    Returns:
        ProblemDetails: Pydantic L2P BaseModel
    """
    problem = write_temp_pddl(pddl_str, parse_problem)

    return ProblemDetails(
        name=problem.name,
        domain_name=problem.domain_name,
        objects=_convert_objects(problem.objects),
        initial_state=_convert_initial_state(problem.init),
        goal_state=_convert_goal_state(problem.goal),
        metric=_convert_metric(problem.metric) if hasattr(problem, "metric") else None,
    )


def write_temp_pddl(pddl_str: str, parser_func):
    """Write *pddl_str* to a temp file, parse it with *parser_func*, clean up."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as f:
        f.write(pddl_str)
        f.flush()
        fname = f.name
    try:
        return parser_func(fname)
    finally:
        os.unlink(fname)


# ---------------------------------------------------------------------------
# Domain component converters (from PDDL parser to L2P models)
# ---------------------------------------------------------------------------


def _convert_requirements(reqs: frozenset) -> List[Requirement]:
    """Converts pddl requirements class into L2P class."""
    return [Requirement(name=str(r)) for r in reqs]


def _convert_types(types: Dict) -> List[PDDLType]:
    result: List[PDDLType] = []
    for name, parent in types.items():
        result.append(
            PDDLType(
                name=name,
                parent=str(parent) if parent else "object",
            )
        )
    return result


def _convert_constants(constants: frozenset) -> List[Constant]:
    """Converts pddl constants class into L2P class."""
    result: List[Constant] = []
    for c in constants:
        type_str = next(iter(c.type_tags)) if c.type_tags else "object"
        result.append(Constant(name=c.name, type=type_str))
    return result


def _convert_predicates(predicates: frozenset) -> List[Predicate]:
    """Converts pddl predicates class into L2P class."""
    result: List[Predicate] = []
    for p in predicates:
        result.append(
            Predicate(
                name=p.name,
                params=_make_params(p.terms),
            )
        )
    return result


def _convert_functions(functions: Dict) -> List[Function]:
    """Converts pddl functions class into L2P class."""
    result: List[Function] = []
    for func, _return_type in functions.items():
        result.append(
            Function(
                name=func.name,
                params=_make_params(func.terms),
            )
        )
    return result


def _convert_derived_predicates(derived: frozenset) -> List[DerivedPredicate]:
    """Converts pddl derived predicates class into L2P class."""
    result: List[DerivedPredicate] = []
    for dp in derived:
        pred = dp.predicate
        result.append(
            DerivedPredicate(
                name=pred.name,
                params=_make_params(pred.terms),
                condition=_convert_condition(dp.condition),
            )
        )
    return result


def _convert_actions(actions: frozenset) -> List[Action]:
    """Converts pddl actions class into L2P class."""
    result: List[Action] = []
    for a in actions:
        result.append(_convert_single_action(a))
    return result


def _convert_single_action(a: PddlAction) -> Action:
    """Converts pddl action class into L2P class."""
    return Action(
        name=a.name,
        params=_make_params(a.parameters),
        preconditions=_convert_precondition(a.precondition),
        effects=_parse_effects(a.effect),
    )


def _convert_precondition(formula) -> ActionPrecondition:
    """
    Convert a pddl precondition formula into an ActionPrecondition.

    ``And`` at the top level is flattened — each operand becomes a separate
    entry in ``conditions``.  All other formulae produce a single entry.
    """
    if isinstance(formula, And):
        conditions = [_convert_condition(op) for op in formula.operands]
    else:
        conditions = [_convert_condition(formula)]
    return ActionPrecondition(conditions=conditions)


def _convert_condition(formula) -> LogicalCondition:
    """
    Recursively convert a pddl formula object to a ``LogicalCondition``
    (``str`` or ``dict``).
    """

    # --- logical connectives ---
    if isinstance(formula, And):
        return {
            "operator": "and",
            "conditions": [_convert_condition(op) for op in formula.operands],
        }

    if isinstance(formula, Or):
        return {
            "operator": "or",
            "conditions": [_convert_condition(op) for op in formula.operands],
        }

    if isinstance(formula, Not):
        return {
            "operator": "not",
            "condition": _convert_condition(formula.argument),
        }

    if isinstance(formula, Imply):
        # Imply stores antecedent + consequent as operands[0] / operands[1]
        return {
            "operator": "imply",
            "antecedent": [_convert_condition(formula.operands[0])],
            "consequent": [_convert_condition(formula.operands[1])],
        }

    # --- quantifiers ---
    if isinstance(formula, ForallCondition):
        return {
            "quantifier": "forall",
            "parameters": _make_params_from_variables(list(formula.variables)),
            "conditions": [_convert_condition(formula.condition)],
        }

    if isinstance(formula, ExistsCondition):
        return {
            "quantifier": "exists",
            "parameters": _make_params_from_variables(list(formula.variables)),
            "conditions": [_convert_condition(formula.condition)],
        }

    # --- everything else: flat string ---
    # Predicate, EqualTo, numeric comparisons, FunctionExpression, etc.
    return str(formula)


def _parse_effects(formula) -> ActionEffect:
    """
    Walk the effect formula tree and categorise each leaf into
    *add*, *delete*, *numeric*, or *conditional* lists.
    """
    add: List[str] = []
    delete: List[str] = []
    numeric: List[str] = []
    conditional: List[ConditionalEffect] = []

    def _walk(f):
        if isinstance(f, And):
            for op in f.operands:
                _walk(op)
            return

        if isinstance(f, PddlPredicate):
            add.append(str(f))
            return

        if isinstance(f, Not):
            # ``Not`` wrapping a Predicate → delete, otherwise treat nested
            arg = f.argument
            if isinstance(arg, PddlPredicate):
                delete.append(str(arg))
            elif isinstance(arg, And):
                # (not (and ...)) is unusual — flatten into delete list
                for op in arg.operands:
                    if isinstance(op, PddlPredicate):
                        delete.append(str(op))
                    else:
                        numeric.append(str(f))
            else:
                numeric.append(str(f))
            return

        if isinstance(f, When):
            cond = _convert_condition(f.condition)
            inner_eff = _parse_effects(f.effect)
            conditional.append(
                ConditionalEffect(
                    condition=[cond] if isinstance(cond, str) else [cond],
                    effect={
                        "add": inner_eff.add,
                        "delete": inner_eff.delete,
                        "numeric": inner_eff.numeric,
                    },
                )
            )
            return

        if isinstance(f, ForallEffect):
            add.append(str(f))
            return

        # numeric assignments / comparisons
        if isinstance(
            f,
            (
                Increase,
                Decrease,
                Assign,
                ScaleUp,
                ScaleDown,
                GreaterThan,
                LesserThan,
                GreaterEqualThan,
                LesserEqualThan,
            ),
        ):
            numeric.append(str(f))
            return

        # anything else goes to numeric as a flat string
        numeric.append(str(f))

    _walk(formula)

    return ActionEffect(
        add=add,
        delete=delete,
        numeric=numeric,
        conditional=conditional,
    )


def _make_params(terms) -> List[Parameter]:
    """Convert a tuple of pddl ``Term`` objects (from Predicate / Function)."""
    result: List[Parameter] = []
    for t in terms:
        type_str = next(iter(t.type_tags)) if t.type_tags else "object"
        result.append(Parameter(variable=f"?{t.name}", type=type_str))
    return result


def _make_params_from_variables(variables: List[Variable]) -> List[Dict[str, str]]:
    """Convert a list of pddl ``Variable`` objects (from quantifiers)."""
    result: List[Dict[str, str]] = []
    for v in variables:
        type_str = next(iter(v.type_tags)) if v.type_tags else "object"
        result.append({"variable": f"?{v.name}", "type": type_str})
    return result


# ---------------------------------------------------------------------------
# Problem component converters (from PDDL parser to L2P models)
# ---------------------------------------------------------------------------


def _convert_objects(objects: frozenset) -> List[PDDLObject]:
    """Converts pddl objects class into L2P class."""
    result: List[PDDLObject] = []
    for o in objects:
        type_str = next(iter(o.type_tags)) if o.type_tags else "object"
        result.append(PDDLObject(name=o.name, type=type_str))
    return result


def _convert_initial_state(init: frozenset) -> InitialState:
    """Converts pddl initial state class into L2P class."""
    facts: List[str] = []
    for item in init:
        facts.append(str(item))

    return InitialState(facts=facts, timed_facts=[])


def _convert_goal_state(goal) -> GoalState:
    """Converts pddl goal state class into L2P class."""
    if isinstance(goal, And):
        conditions = [_convert_condition(op) for op in goal.operands]
    else:
        conditions = [_convert_condition(goal)]
    return GoalState(conditions=conditions)


def _convert_metric(metric) -> Optional[Metric]:
    """Converts pddl metric class into L2P class."""
    if metric is None:
        return None
    return Metric(
        optimization=metric.optimization,
        expression=str(metric.expression),
    )
