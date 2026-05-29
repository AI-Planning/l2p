## `templates`

All default prompts (found in `l2p/templates`) used for `formalize_component()` in `DomainBuilder`, `ProblemBuilder`, and `FeedbackBuilder` correspond to a strict format template. User can use `PromptBuilder` to standardize LLM prompts with five configurable sections:

| Section | Purpose |
|---------|---------|
| **Role** | System persona (e.g., "You are a PDDL expert") |
| **Format** | Output schema / instructions |
| **Rules** | Numbered checklist of constraints |
| **Examples** | n-shot in-context demonstrations |
| **Task** | The specific NL input to solve |

### Standardized format prompt.md:
```
## ROLE
[...]

## OUTPUT FORMAT
<xml_tag>
[...]
</xml_tag>

## RULES
[...]

## EXAMPLE(S)
# Example 1:
[...]
    .
    .
    .
# Example n:
[...]
--------------------------------------------------

## TASK
[...]

{description}

{context}
```

*NOTE:* `description` and `context` placeholders are essential for injecting context when you formalize components.

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