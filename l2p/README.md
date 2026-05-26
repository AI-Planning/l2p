# L2P: LLM-Powered PDDL Planning Library

> Generate PDDL domain and problem specifications from natural language using large language models.

**Documentation:** https://ai-planning.github.io/l2p/docs/

---

## Core Modules

### `prompt_builder.py` — Structured Prompt Assembly

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
    .set_role("You are a PDDL generator.")
    .set_format("Your final answer must be outputted in the following JSON structure.")
    .add_rule("Use strict PDDL syntax.")
    .add_example("INPUT: ...\nOUTPUT: ...")
    .set_task("Generate types for a rover domain."))
prompt = pb.save_prompt(filename="my_prompt.md")
```

Refer to `l2p/templates` for a better idea how to format your prompts.

### `domain_builder.py` — PDDL Domain Generation

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
parsed, raw = db.formalize_component(
    model=llm,
    component_class=Predicate,
    description="Model predicates for blocksworld.",
    types=[PDDLType(name="block", parent="object")]
)
predicates = parsed[Predicate]
```

### `problem_builder.py` — PDDL Problem Instance Generation

Generates complete problem instances (objects, initial state, goals) from natural language.

```python
from l2p.problem_builder import ProblemBuilder

pb = ProblemBuilder()
parsed, _ = pb.formalize_component(
    model=llm,
    component_class=ProblemDetails,
    description="3 blocks stacked: b2 on b3, b3 on b1, b1 on table.",
    types=types,
    predicates=predicates
)
problem_pddl = pb.generate_problem(parsed[ProblemDetails][0])
```

### `feedback_builder.py` — LLM-Driven Quality Control

Provides a self-improvement loop using LLMs for different kinds of feedback strategies in the literature (e.g., diagnosis, revision, evaluation, and candidate selection).

| Method | Purpose |
|--------|---------|
| `llm_diagnose()` | Root-cause analysis of syntax/validation errors |
| `llm_evaluate()` | Semantic correctness against NL intent (LLM judge) |
| `llm_reflect()` | Extract durable lessons from failures |
| `llm_revise()` | Fix broken components using a repair plan |
| `llm_select()` | Choose the best candidate from multiple generations |
| `llm_evaluate_plan()` | Verify plan-level semantic soundness |
| `llm_diagnose_plan()` | Diagnose planner failures (unsolvable, timeout) |

```python
fb = FeedbackBuilder()
diagnosis, _ = fb.llm_diagnose(
    model=llm, artifact=predicates,
    errors="ValidationError: ...",
    description=domain_desc
)
```

### `planner_builder.py` — External Planner Integration

Abstract interface for running classical planners on generated PDDL. Ships with two backends:

**FastDownward** (submodule) — CLI-based:
```python
from l2p.planner_builder import FastDownward

planner = FastDownward(executable_path="downward/fast-downward.py")
result = planner.run_planner(domain_file="d.pddl", problem_file="p.pddl")
print(result.is_successful, result.plan)
```

**Unified Planning** — Python API:
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

## Quickstart

```python
import os
from l2p import UnifiedLLM
from l2p.domain_builder import DomainBuilder
from l2p.utils.pddl_types import PDDLType, Predicate
from l2p.utils.pddl_format import format_predicates

llm = UnifiedLLM(provider="openai", model="gpt-4o-mini",
                 api_key=os.getenv("OPENAI_API_KEY"))

db = DomainBuilder()
parsed, _ = db.formalize_component(
    model=llm,
    component_class=Predicate,
    description="Blocksworld predicates.",
    types=[PDDLType(name="block", parent="object")]
)

print(format_predicates(parsed[Predicate]))
```

---

## PDDL Support & Requirements

The library automatically infers PDDL requirements (`:strips`, `:typing`, `:numeric-fluents`, etc.) from generated components via `DomainBuilder.generate_requirements()`. Requirements are assembled from the structural features present in the model, therefore no manual annotation needed.

---

## Subpackages

| Module | Description |
|--------|-------------|
| `l2p/llm/` | LLM backends (OpenAI SDK, simonw/llm, HuggingFace, vLLM) |
| `l2p/utils/` | PDDL types, formatting, parsing, and prompt templates |
| `l2p/validators/` | Symbolic validation rules for domain/problem syntax |
| `l2p/cli/` | Interactive CLI for configuration and generation |
| `l2p/templates/` | Default prompt templates for domain, problem, and feedback |
