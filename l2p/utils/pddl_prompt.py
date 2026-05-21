"""
This module contains collection of functions for formatting Python PDDL components into prompt strings.
"""

import importlib.resources
import json
import os
from pydantic import BaseModel
from types import SimpleNamespace
from typing import Any, Callable, Dict, Sequence, TypeVar


def safe_format(template: str, **kwargs) -> str:
    """
    Safely injects variables into a template string without breaking JSON curly braces.
    Replaces {key} with value for all provided kwargs.
    """
    result = template
    for key, value in kwargs.items():
        placeholder = f"{{{key}}}"
        result = result.replace(placeholder, str(value))
    return result


def load_custom_template(filepath: str) -> str:
    """
    Helper function for users to load their own custom .md prompt files.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] Custom template not found at: {filepath}")

    with open(file=filepath, mode="r", encoding="utf-8") as f:
        return f.read()


def load_default_template(folder: str, filename: str) -> str:
    """
    Helper function that default loads template dynamically from the `l2p.templates` package.
    """
    try:
        template_pth = importlib.resources.files("l2p.templates").joinpath(
            folder, filename
        )
        return template_pth.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"[ERROR] L2P template `{folder}/{filename}` not found."


T = TypeVar("T", bound=BaseModel)


def jsonify_components(components: Sequence[T]) -> str:
    """
    Serializes any sequence of Pydantic models (e.g., Type, Predicate, Action, etc.) into
    JSON string for LLM prompt injection. Excludes None values.
    """
    if not components:
        return ""

    return json.dumps([c.model_dump(exclude_none=True) for c in components], indent=4)


def build_ctx(**kwargs) -> str:
    """
    Dynamically builds an <existing-content> XML block based on provided PDDL kwargs.
    If no recognized kwargs are provided, it returns an empty string.

    Usage:
        ctx = build_prompt_ctx(types=[Type(...)]), predicates=[Predicate(...)]
    """
    injected_strings = []
    # iterate through valid mappings
    for kwarg_key, (header, serializer) in CTX_MAPPING.items():
        if kwarg_key in kwargs and kwargs[kwarg_key]:
            component_data = kwargs[kwarg_key]
            if not isinstance(component_data, list):
                component_data = [component_data]
            try:
                json_str = serializer(component_data)
                injected_strings.append(f"<{header}>\n{json_str}\n</{header}>")
            except Exception as e:
                print(f"[WARNING] Failed to serialize `{kwarg_key}` for context: {e}")

    if not injected_strings:
        return ""

    combined_ctx = "\n\n".join(injected_strings)
    return f"\n<existing_context>\n{combined_ctx}\n</existing_context>"


CTX_MAPPING: Dict[str, tuple[str, Callable[[Any], str]]] = {
    "requirements": ("requirements", jsonify_components),
    "types": ("types", jsonify_components),
    "constants": ("constants", jsonify_components),
    "predicates": ("predicates", jsonify_components),
    "functions": ("functions", jsonify_components),
    "constraints": ("constraints", jsonify_components),
    "derived_predicates": ("derived_predicates", jsonify_components),
    "parameters": ("parameters", jsonify_components),
    "preconditions": ("preconditions", jsonify_components),
    "effects": ("effects", jsonify_components),
    "nl_actions": ("nl_actions", jsonify_components),
    "actions": ("actions", jsonify_components),
    "durative_actions": ("durative_actions", jsonify_components),
    "durative_conditions": ("durative_conditions", jsonify_components),
    "durative_effects": ("durative_effects", jsonify_components),
    "events": ("events", jsonify_components),
    "processes": ("processes", jsonify_components),
}

# DEFAULT DOMAIN SYSTEM PROMPT TEMPLATES
DEF_DOMAIN_PROMPTS = SimpleNamespace(
    domain=load_default_template(folder="domain", filename="prompt_domain.md"),
    requirements=load_default_template(
        folder="domain", filename="prompt_requirements.md"
    ),
    types=load_default_template(folder="domain", filename="prompt_types.md"),
    constants=load_default_template(folder="domain", filename="prompt_constants.md"),
    predicates=load_default_template(folder="domain", filename="prompt_predicates.md"),
    functions=load_default_template(folder="domain", filename="prompt_functions.md"),
    constraints=load_default_template(
        folder="domain", filename="prompt_constraints.md"
    ),
    der_preds=load_default_template(
        folder="domain", filename="prompt_derived_predicates.md"
    ),
    actions=load_default_template(folder="domain", filename="prompt_actions.md"),
    parameters=load_default_template(folder="domain", filename="prompt_parameters.md"),
    preconds=load_default_template(folder="domain", filename="prompt_preconditions.md"),
    effects=load_default_template(folder="domain", filename="prompt_effects.md"),
    nl_actions=load_default_template(folder="domain", filename="prompt_nl_actions.md"),
    dur_actions=load_default_template(
        folder="domain", filename="prompt_durative_actions.md"
    ),
    dur_conds=load_default_template(
        folder="domain", filename="prompt_durative_conditions.md"
    ),
    dur_effects=load_default_template(
        folder="domain", filename="prompt_durative_effects.md"
    ),
    events=load_default_template(folder="domain", filename="prompt_events.md"),
    processes=load_default_template(folder="domain", filename="prompt_processes.md"),
)

# DEFAULT PROBLEM SYSTEM PROMPT TEMPLATES
DEF_PROBLEM_PROMPTS = SimpleNamespace(
    problem=load_default_template(folder="problem", filename="prompt_problem.md"),
    objects=load_default_template(folder="problem", filename="prompt_objects.md"),
    initial=load_default_template(
        folder="problem", filename="prompt_initial_states.md"
    ),
    goal=load_default_template(folder="problem", filename="prompt_goal_states.md"),
    constraints=load_default_template(
        folder="problem", filename="prompt_constraints.md"
    ),
    metric=load_default_template(folder="problem", filename="prompt_metric.md"),
)

# DEFAULT FEEDBACK SYSTEM PROMPT TEMPLATES
DEF_FB_PROMPTS = SimpleNamespace(
    diagnosis=load_default_template(folder="feedback", filename="prompt_diagnosis.md"),
    evaluate=load_default_template(folder="feedback", filename="prompt_evaluate.md"),
    reflection=load_default_template(
        folder="feedback", filename="prompt_reflection.md"
    ),
    revise=load_default_template(folder="feedback", filename="prompt_revise.md"),
    select=load_default_template(folder="feedback", filename="prompt_select.md"),
    plan_diagnosis=load_default_template(
        folder="feedback", filename="prompt_plan_diagnosis.md"
    ),
    plan_evaluate=load_default_template(
        folder="feedback", filename="prompt_plan_evaluate.md"
    ),
)
