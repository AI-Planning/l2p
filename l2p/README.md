# L2P: LLM-Powered PDDL Planning Library

> Generate PDDL domain and problem specifications from natural language using large language models.

**Documentation:** https://ai-planning.github.io/l2p/docs/

L2P is a tool that generates, validates, and offers feedback on PDDL domains and problems that is then run through an external planner. Users can customize their own pipelines, prompt templates, and large language models. L2P also supports tool usage by agents.

---

## Quickstart
Here is how you would generate PDDL predicates.

```python
import os
from l2p import UnifiedLLM
from l2p.domain_builder import DomainBuilder
from l2p.utils.pddl_types import PDDLType, Predicate
from l2p.utils.pddl_format import format_predicates

llm = UnifiedLLM(provider="openai", model="gpt-5-nano",
                 api_key=os.getenv("OPENAI_API_KEY"))

db = DomainBuilder()
results, _ = db.formalize_component(
    model=llm,
    component_class=Predicate,
    description="Blocksworld predicates.",
    types=[PDDLType(name="block", parent="object")]
)

print(format_predicates(results[Predicate]))
```

---

## Core Modules

### `domain_builder.py` - PDDL Domain Generation

Core class for constructing complete PDDL domains via LLM extraction.

**Supported features:**

| Component | PDDL Version |
|-----------|-------------|
| Types, Constants, Predicates | 1.2+ |
| Functions / Numeric Fluents | 2.1+ |
| Actions (params, preconditions, effects) | 1.2+ |
| Quantified Preconditions/Effects | 2.2+ |
| Conditional Effects | 2.2+ |
| Disjunctive Preconditions | 2.2+ |
| Action Costs | 2.1+ |
| Temporal Constraints | 2.2+ |
| Derived Predicates | 2.1+ |
| Durative Actions | 2.1+ |
| Events & Processes | PDDL+ |

```python
from l2p.domain_builder import DomainBuilder
from l2p.utils.pddl_types import Predicate

db = DomainBuilder()
results, raw = db.formalize_component(
    model=llm,
    component_class=Predicate,
    description="Model predicates for blocksworld.",
    types=[PDDLType(name="block", parent="object")]
)
predicates = results[Predicate]
```

> The library automatically infers PDDL requirements (`:strips`, `:typing`, `:numeric-fluents`, etc.) from generated components via `DomainBuilder.generate_requirements()`. Requirements are assembled from the structural features present in the model, therefore no manual annotation needed.

### `problem_builder.py` - PDDL Problem Instance Generation

Generates complete problem instances (objects, initial state, goals) from natural language.

```python
from l2p.problem_builder import ProblemBuilder

pb = ProblemBuilder()
results, _ = pb.formalize_component(
    model=llm,
    component_class=ProblemDetails,
    description="3 blocks stacked: b2 on b3, b3 on b1, b1 on table.",
    types=types,
    predicates=predicates
)
problem_pddl = pb.generate_problem(results[ProblemDetails][0])
```

---

### `feedback_builder.py` - LLM-Driven Quality Control

Provides a self-improvement loop using LLMs for different kinds of feedback strategies in the literature (e.g., diagnosis, revision, evaluation, and candidate selection).

| Method | Purpose |
|--------|---------|
| `llm_diagnose()` | Root-cause analysis of syntax/validation errors |
| `llm_revise()` | Fix broken components using a repair plan |
| `llm_evaluate()` | Semantic correctness against NL intent (LLM judge) |
| `llm_reflect()` | Extract durable lessons from failures |
| `llm_select()` | Choose the best candidate from multiple generations |
| `llm_evaluate_plan()` | Verify plan-level semantic soundness |
| `llm_diagnose_plan()` | Diagnose planner failures (unsolvable, timeout) |

```python
from l2p.feedback_builder import FeedbackBuilder

predicates = [
    Predicate(name="on-table", params=[{"variable": "?b", "type": "block"}]),
    Predicate(name="holding", params=[{"variable": "?b", "type": "block"}]),
    Predicate(name="clear", params=[]),
]

domain_desc = """
Standard PDDL blocksworld.
"""

fb = FeedbackBuilder()
diagnosis, _ = fb.llm_diagnose(
    model=llm, 
    artifact=predicates, # feed in model to fix
    errors="ValidationError: ...", # error derived from syntax validator
    description=domain_desc
)

print(diagnosis)
```

---

### `prompt_builder.py` - Structured Prompt Assembly

All default prompts (found in `l2p/templates`) used for `formalize_component()` in `DomainBuilder`, `ProblemBuilder`, and `FeedbackBuilder` correspond to a strict format template. User can use `PromptBuilder` to standardize LLM prompts with five configurable sections:

| Section | Purpose |
|---------|---------|
| **Role** | System persona (e.g., "You are a PDDL expert") |
| **Format** | Output schema / instructions |
| **Rules** | Numbered checklist of constraints |
| **Examples** | n-shot in-context demonstrations |
| **Task** | The specific NL input to solve |

```python
from l2p.prompt_builder import PromptBuilder

pb = (PromptBuilder()
    .set_role("You are a PDDL types generator.")
    .set_format("Your final answer must be outputted in the following JSON structure.")
    .set_format_example(component=PDDLType) # set extraction example block for component
    .add_rule("Use strict PDDL syntax.")
    .add_example("INPUT: ...\nOUTPUT: ...")
    .set_task("Generate types for a rover domain."))
prompt = pb.save_prompt(filename="my_prompt.md")
```

Refer to `l2p/templates` for a better idea how to format your prompts.

### Custom Multi-Component Templates

`l2p/templates/custom/` provides pre-built prompt templates that extract **multiple PDDL components in a single LLM call**. This improves cross-component consistency - the LLM sees the full state space and action models simultaneously, eliminating mismatches between components.

**Domain combinations (single LLM call):**

| Template | Components Extracted |
|----------|---------------------|
| `prompt_types_predicates` | Types + Predicates |
| `prompt_types_constants_predicates` | Types + Constants + Predicates |
| `prompt_types_predicates_functions` | Types + Predicates + Functions |
| `prompt_predicates_actions` | Predicates + Actions |
| `prompt_actions_constraints` | Actions + Constraints |
| `prompt_actions_durative_actions` | Actions + DurativeActions |
| `prompt_events_processes` | Events + Processes |
| `prompt_derived_predicates_predicates` | DerivedPredicates + Predicates |
| `prompt_types_predicates_functions_actions` | Types + Preds + Functions + Actions |

**Problem combinations (single LLM call):**

| Template | Components Extracted |
|----------|---------------------|
| `prompt_objects_initial_state` | Objects + InitialState |
| `prompt_objects_initial_goal` | Objects + Init + Goal |
| `prompt_initial_goal_metric` | Init + Goal + Metric |

Use them with `formalize_component()` by passing a **list of component classes** and the `prompt_template` argument:

```python
from l2p.utils.pddl_prompt import load_default_template

prompt = load_default_template("custom", "prompt_types_predicates_functions_actions.md")

results, raw = db.formalize_component(
    model=llm,
    component_class=[PDDLType, Predicate, Function, Action],
    prompt_template=prompt,
    description="A rover domain with battery management.",
)

types = results[PDDLType]
predicates = results[Predicate]
functions = results[Function]
actions = results[Action]
```

Each template follows the same structure (Role, Output Format with XML tags, Rules, Task) with added cross-reference constraints - e.g., every predicate used in actions must be defined in the predicates section.

### `planner_builder.py` - External Planner Integration

Abstract interface for running classical planners on generated PDDL. Ships with two backends:

**FastDownward** (submodule) - CLI-based:
```python
from l2p.planner_builder import FastDownward

planner = FastDownward(executable_path="downward/fast-downward.py")
result = planner.run_planner(domain_file="d.pddl", problem_file="p.pddl")
print(result.is_successful, result.plan)
```

**Unified Planning** - Python API:
```python
from l2p.planner_builder import UnifiedPlanning

planner = UnifiedPlanning()
result = planner.run_planner(
    domain_path="d.pddl", problem_path="p.pddl", engine="aries"
)
```

Both return a `PlanningResult` dataclass:
```python
@dataclass
class PlanningResult:
    is_successful: bool
    plan: Optional[List[str]]
    error_message: Optional[str]
    raw_output: str
    metrics: Dict[str, Any]
```

---

## Subpackages

| Module | Description |
|--------|-------------|
| `l2p/llm/` | LLM backends (OpenAI SDK, simonw/llm, HuggingFace, vLLM) |
| `l2p/utils/` | PDDL types, formatting, parsing, and prompt templates |
| `l2p/validators/` | Symbolic validation rules for domain/problem syntax |
| `l2p/cli/` | Interactive CLI for configuration and generation |
| `l2p/templates/` | Default prompt templates for domain, problem, and feedback |
