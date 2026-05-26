"""MCP server exposing L2P operations as tools for AI agents.

Start via CLI:  l2p mcp
Or embed:      from l2p.mcp.server import serve; await serve()

Tools exposed to the MCP client (Claude Desktop, Claude Code, etc.):

  validate_component  — validate any PDDL component JSON against L2P rules
  format_component    — parse JSON and output PDDL-formatted string
  build_domain        — assemble a complete PDDL domain from DomainDetails JSON
  build_problem       — assemble a complete PDDL problem from ProblemDetails JSON
  run_planner         — execute FastDownward or Unified Planning
  get_schema          — get JSON Schema for any PDDM component

Resources:
  schema://{component}  — JSON Schema for a PDDL component
"""

import json
import sys
from typing import Any, Dict, List, Optional, Union

from pydantic import TypeAdapter

from l2p.domain_builder import DomainBuilder
from l2p.problem_builder import ProblemBuilder
from l2p.utils.pddl_types import *
from l2p.utils.pddl_format import *


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

COMPONENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "requirements": {"model": Requirement, "is_list": True, "format": format_requirements},
    "types": {"model": PDDLType, "is_list": True, "format": format_types},
    "constants": {"model": Constant, "is_list": True, "format": format_constants},
    "predicates": {"model": Predicate, "is_list": True, "format": format_predicates},
    "functions": {"model": Function, "is_list": True, "format": format_functions},
    "derived-predicates": {"model": DerivedPredicate, "is_list": True, "format": format_derived_predicates},
    "actions": {"model": Action, "is_list": True, "format": format_actions},
    "durative-actions": {"model": DurativeAction, "is_list": True, "format": format_durative_actions},
    "events": {"model": Event, "is_list": True, "format": format_events},
    "processes": {"model": Process, "is_list": True, "format": format_processes},
    "constraints": {"model": Constraint, "is_list": True, "format": format_constraints},
    "objects": {"model": PDDLObject, "is_list": True, "format": format_objects},
    "initial-state": {"model": InitialState, "is_list": False, "format": format_initial_state},
    "goal-state": {"model": GoalState, "is_list": False, "format": format_goal_states},
    "metric": {"model": Metric, "is_list": False, "format": format_metric},
    "parameters": {"model": Parameter, "is_list": True, "format": None},
}


def _parse_component(data: Union[str, list, dict], component: str) -> Any:
    """Parse and validate JSON into the Pydantic model for *component*."""
    info = COMPONENT_REGISTRY[component]
    model = info["model"]
    is_list = info["is_list"]

    if isinstance(data, str):
        raw = json.loads(data)
    else:
        raw = data

    if is_list:
        if not isinstance(raw, list):
            raw = [raw]
        adapter = TypeAdapter(List[model])
        return adapter.validate_python(raw)
    else:
        if isinstance(raw, list):
            raw = raw[0]
        return model.model_validate(raw)


def _format_component(parsed: Any, component: str) -> str:
    """Serialize a parsed model to PDDL string or JSON."""
    info = COMPONENT_REGISTRY.get(component)
    if not info or not info["format"]:
        items = parsed if isinstance(parsed, list) else [parsed]
        return json.dumps([i.model_dump(exclude_none=True) for i in items], indent=2)
    items = parsed if isinstance(parsed, list) else [parsed]
    return info["format"](items)


def _build_domain(data: Union[str, dict]) -> str:
    """Convert a DomainDetails JSON dict to a PDDL domain string."""
    if isinstance(data, str):
        data = json.loads(data)
    details = DomainDetails.model_validate(data)
    builder = DomainBuilder(domain_details=details)
    return builder.generate_domain(details)


def _build_problem(data: Union[str, dict]) -> str:
    """Convert a ProblemDetails JSON dict to a PDDL problem string."""
    if isinstance(data, str):
        data = json.loads(data)
    details = ProblemDetails.model_validate(data)
    builder = ProblemBuilder(problem_details=details)
    return builder.generate_problem(details)


def _run_planner(
    domain_pddl: str,
    problem_pddl: str,
    planner_type: str = "fast-downward",
    alias: str = "lama-first",
    engine: Optional[str] = None,
    timeout: int = 60,
    executable: Optional[str] = None,
) -> dict:
    """Run a planner and return result as a dict (serializable to JSON)."""
    import tempfile
    from pathlib import Path

    def _write_temp(suffix: str) -> str:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
        return tmp.name

    d_path = _write_temp(".pddl")
    p_path = _write_temp(".pddl")

    try:
        Path(d_path).write_text(domain_pddl)
        Path(p_path).write_text(problem_pddl)

        if planner_type == "fast-downward":
            from l2p.planner_builder import FastDownward

            planner = FastDownward(executable_path=executable or "downward/fast-downward.py")
            result = planner.run_planner(
                domain_path=d_path, problem_path=p_path, alias=alias, timeout=timeout
            )
        elif planner_type == "unified":
            from l2p.planner_builder import UnifiedPlanning

            planner = UnifiedPlanning()
            result = planner.run_planner(
                domain_path=d_path, problem_path=p_path,
                engine=engine or "aries", timeout=timeout
            )
        else:
            return {"error": f"Unknown planner: {planner_type}"}

        import dataclasses
        return dataclasses.asdict(result)
    finally:
        Path(d_path).unlink(missing_ok=True)
        Path(p_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

try:
    import mcp.server as mcp_server
    import mcp.server.stdio
    import mcp.types as types
    from mcp.server.models import InitializationOptions
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


async def serve():
    """Start the L2P MCP server on stdio transport.

    The server exposes tools that AI agents can use to build, validate,
    and plan with PDDL domains and problems.
    """
    if not HAS_MCP:
        print(
            "[ERROR] MCP SDK not installed. Run: pip install mcp",
            file=sys.stderr,
        )
        sys.exit(1)

    server = mcp_server.Server("l2p")

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @server.tool()
    async def validate_component(
        component: str,
        data: str,
    ) -> str:
        """Validate a PDDL component JSON string against L2P's semantic rules.

        Rules checked include: PDDL naming conventions (no reserved keywords,
        valid characters), type hierarchy (parents must exist, no cycles),
        parameter types (every ?var needs a declared type), variable scope,
        and symbol references.

        Args:
            component: The component type to validate. One of:
                types, constants, predicates, functions, derived-predicates,
                actions, durative-actions, events, processes, constraints,
                objects, initial-state, goal-state, metric
            data: JSON string of the component matching the Pydantic model
                for that component type. Use get_schema to see the expected shape.

        Returns:
            JSON string with "valid" (bool), "errors" (list of strings),
            and "warnings" (list of strings).
        """
        if component not in COMPONENT_REGISTRY:
            return json.dumps({"error": f"Unknown component: {component}"})

        try:
            parsed = _parse_component(data, component)
        except Exception as e:
            return json.dumps({"error": f"Parse error: {e}"})

        errors = []
        warnings = []

        items = parsed if isinstance(parsed, list) else [parsed]

        from l2p.validators.domain import DomainValidator as DomVal
        from l2p.validators.problem import ProblemValidator as ProbVal

        domain_components = {
            "requirements", "types", "constants", "predicates", "functions",
            "derived-predicates", "actions", "durative-actions", "events", "processes", "constraints"
        }

        if component in domain_components:
            validator = DomVal()
        else:
            validator = ProbVal()

        for item in items:
            result = validator.validate_component(item, {})
            if not result.valid:
                errors.extend(result.errors)
            warnings.extend(result.warnings)

        return json.dumps({
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }, indent=2)

    @server.tool()
    async def format_component(
        component: str,
        data: str,
    ) -> str:
        """Parse JSON and output the PDDL-formatted version of a component.

        Use this to see how a JSON component renders in PDDL syntax
        before assembling a full domain or problem.

        Args:
            component: The component type (types, predicates, actions, etc.).
            data: JSON string of the component.

        Returns:
            PDDL-formatted string of the component.
        """
        if component not in COMPONENT_REGISTRY:
            return json.dumps({"error": f"Unknown component: {component}"})
        try:
            parsed = _parse_component(data, component)
            return _format_component(parsed, component)
        except Exception as e:
            return json.dumps({"error": f"Error: {e}"})

    @server.tool()
    async def build_domain(
        data: str,
    ) -> str:
        """Build a complete PDDL domain from a DomainDetails JSON string.

        The JSON must match the DomainDetails Pydantic model.  Use
        get_schema("domain") or get_schema("domain", with_examples=True)
        to see the expected structure.

        Top-level fields:
            name (str, required),
            types (list of PDDLType),
            constants (list of Constant),
            predicates (list of Predicate),
            functions (list of Function),
            derived_predicates (list of DerivedPredicate),
            actions (list of Action),
            durative_actions (list of DurativeAction),
            events (list of Event),
            processes (list of Process),
            constraint (list of Constraint)

        Args:
            data: Full DomainDetails JSON string.

        Returns:
            PDDL domain string on success, or JSON error object.
        """
        try:
            return _build_domain(data)
        except Exception as e:
            return json.dumps({"error": f"Failed to build domain: {e}"})

    @server.tool()
    async def build_problem(
        data: str,
    ) -> str:
        """Build a complete PDDL problem from a ProblemDetails JSON string.

        The JSON must match the ProblemDetails Pydantic model.  Use
        get_schema("problem", with_examples=True) to see the expected shape.

        Top-level fields:
            name (str, required),
            domain_name (str, required),
            objects (list of PDDLObject),
            initial_state (InitialState: {facts: [...], timed_facts: [...]}),
            goal_state (GoalState: {conditions: [...]}),
            constraint (list of Constraint),
            metric (Metric: {optimization, expression})

        Args:
            data: Full ProblemDetails JSON string.

        Returns:
            PDDL problem string on success, or JSON error object.
        """
        try:
            return _build_problem(data)
        except Exception as e:
            return json.dumps({"error": f"Failed to build problem: {e}"})

    @server.tool()
    async def run_planner(
        domain_pddl: str,
        problem_pddl: str,
        planner_type: str = "fast-downward",
        alias: str = "lama-first",
        engine: Optional[str] = None,
        timeout: int = 60,
    ) -> str:
        """Run a planner on PDDL strings and return the result.

        Writes the PDDL strings to temporary files, executes the planner,
        and returns a serialised PlanningResult dict.

        Args:
            domain_pddl: The complete PDDL domain as a string.
            problem_pddl: The complete PDDL problem as a string.
            planner_type: Which planner to use — "fast-downward" (default)
                or "unified" (requires pip install unified-planning).
            alias: Fast Downward search alias (default: "lama-first").
                Common alternatives: "seq-opt-fdss-1", "seq-opt-bjolp".
            engine: Unified Planning engine name (default: "aries").
                Requires: pip install 'unified-planning[engines]'.
            timeout: Maximum execution time in seconds (default: 60).

        Returns:
            JSON string of a PlanningResult dict:
            {
              "is_successful": bool,
              "plan": [str, ...] or null,
              "error_message": str or null,
              "raw_output": str,
              "metrics": {}
            }
        """
        try:
            result = _run_planner(
                domain_pddl, problem_pddl, planner_type, alias, engine, timeout
            )
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @server.tool()
    async def get_schema(
        component: str,
        with_examples: bool = False,
    ) -> str:
        """Get the Pydantic JSON Schema for a PDDL component.

        Use this tool to see the exact JSON structure expected by
        build_domain, build_problem, format_component, and
        validate_component.

        Args:
            component: Component name. One of:
                types, constants, predicates, functions, derived-predicates,
                actions, durative-actions, events, processes, constraints,
                parameters, objects, initial-state, goal-state, metric,
                domain, problem, requirements
            with_examples: If True, includes a concrete JSON example
                alongside the schema (default: False).

        Returns:
            JSON string with "component", "schema" (JSON Schema dict),
            and optionally "example".
        """
        from l2p.cli.commands.schema import SCHEMAS, EXAMPLES_FULL, EXAMPLES
        model_cls = SCHEMAS.get(component)
        if not model_cls:
            return json.dumps({"error": f"Unknown component: {component}"})
        schema = model_cls.model_json_schema()
        output = {"component": component, "schema": schema}
        if with_examples:
            ex = EXAMPLES_FULL.get(component) or EXAMPLES.get(component)
            if ex:
                try:
                    output["example"] = json.loads(ex) if isinstance(ex, str) else ex
                except json.JSONDecodeError:
                    output["example"] = ex
        return json.dumps(output, indent=2)

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    @server.resource("schema://{component}")
    async def schema_resource(component: str) -> str:
        """JSON Schema for a PDDL component.

        Args:
            component: The component name (types, predicates, domain, etc.)
        """
        from l2p.cli.commands.schema import SCHEMAS
        model_cls = SCHEMAS.get(component)
        if not model_cls:
            return json.dumps({"error": f"Unknown component: {component}"})
        return json.dumps(model_cls.model_json_schema(), indent=2)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="l2p",
                server_version="0.3.3",
                capabilities=server.get_capabilities(
                    notifications=None,
                    experimental=None,
                ),
            ),
        )


def run():
    """Synchronous entry point for `l2p mcp`."""
    import asyncio
    asyncio.run(serve())
