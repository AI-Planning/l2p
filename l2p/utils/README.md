# L2P Utils - PDDL Data Model, Formatting, Parsing & Prompts

> Foundational types and utilities that power every L2P builder.

```
l2p/utils/
├── pddl_types.py     # Pydantic models for all PDDL components
├── pddl_format.py    # Model-to-PDDL-string serialization
├── pddl_parser.py    # LLM output extraction & PDDL-string-to-model parsing
├── pddl_prompt.py    # Prompt template management & context injection
└── __init__.py       # Re-exports all public symbols
```

---

## `pddl_types.py` - PDDL Data Models

All PDDL concepts are represented as **Pydantic v2 BaseModel** classes, giving you type validation, JSON serialization, and clear error messages out of the box.

### Domain-Side Models

| Model | PDDL Concept | Example |
|-------|-------------|---------|
| `Requirement` | `:strips`, `:typing` | `Requirement(name=":typing")` |
| `PDDLType` | Type with inheritance | `PDDLType(name="rover", parent="vehicle")` |
| `Constant` | Domain-wide constant | `Constant(name="base", type="location")` |
| `Parameter` | `?var - type` | `Parameter(variable="?r", type="rover")` |
| `Predicate` | State predicate | `Predicate(name="at", params=[...])` |
| `Function` | Numeric fluent | `Function(name="battery-level", params=[...])` |
| `DerivedPredicate` | Axiom | `DerivedPredicate(name="can-move", ...)` |
| `ActionPrecondition` | Precondition block | `ActionPrecondition(conditions=[...])` |
| `ActionEffect` | Effect block (add/delete/numeric/conditional) | `ActionEffect(add=[...])` |
| `Action` | Standard action | `Action(name="drive", params=[...])` |
| `DurativeAction` | Temporal action | `DurativeAction(name="transmit", duration=[...])` |
| `Constraint` | PDDL 3.0 trajectory constraint | `Constraint(condition=...)` |
| `Event` | PDDL+ event | `Event(name="battery-depleted", ...)` |
| `Process` | PDDL+ process | `Process(name="solar-charging", ...)` |
| `DomainDetails` | Root model for a complete domain | Aggregates all above |

### Problem-Side Models

| Model | PDDL Concept | Example |
|-------|-------------|---------|
| `PDDLObject` | Typed object instance | `PDDLObject(name="rover1", type="rover")` |
| `TimedFact` | Timed Initial Literal | `TimedFact(time=15.5, fact="(comm-blackout)")` |
| `InitialState` | Init block | `InitialState(facts=[...], timed_facts=[...])` |
| `GoalState` | Goal block | `GoalState(conditions=[...])` |
| `Metric` | Plan optimization | `Metric(optimization="minimize", expression="total-time")` |
| `ProblemDetails` | Root model for a complete problem | Aggregates all above |

### `LogicalCondition` - Recursive Condition Representation

PDDL formulas (preconditions, effects, goals) are represented as `Union[str, Dict]`:

- **Simple predicates:** `"(at ?r ?l)"`
- **Logical operators:** `{"operator": "not", "condition": ...}` / `{"operator": "and", "conditions": [...]}`
- **Quantifiers:** `{"quantifier": "forall", "parameters": [...], "conditions": [...]}`
- **PDDL 3.0 constraints:** `{"operator": "always", "condition": ...}`
- **Preferences:** `{"preference": "pref_name", "condition": ...}`

### Tag System

Every model has a `tag` class variable that maps to XML tags used in LLM prompts. For example, `Predicate.tag = ("predicates", "predicate")` means the LLM should wrap its output in `<predicates>...</predicates>`.

---

## `pddl_format.py` - Model-to-PDDL Serialization

Converts Pydantic models into standard PDDL string syntax.

| Function | Converts | Output Example |
|----------|----------|----------------|
| `format_requirements()` | `List[Requirement]` | `:strips :typing` |
| `format_types()` | `List[PDDLType]` | `block - object` |
| `format_predicates()` | `List[Predicate]` | `(on ?x - block ?y - block)` |
| `format_functions()` | `List[Function]` | `(battery-level ?r - rover)` |
| `format_action()` | `Action` | Full `(:action ...)` block |
| `format_actions()` | `List[Action]` | Multiple action blocks |
| `format_durative_action()` | `DurativeAction` | Full `(:durative-action ...)` block |
| `format_derived_predicate()` | `DerivedPredicate` | Full `(:derived ...)` block |
| `format_event()` / `format_process()` | `Event` / `Process` | Full PDDL+ block |
| `format_objects()` | `List[PDDLObject]` | `rover1 - rover` |
| `format_initial_state()` | `InitialState` | Facts + timed facts |
| `format_goal_states()` | `GoalState` | Condition block |
| `format_metric()` | `Metric` | `(:metric minimize total-time)` |
| `format_plan()` | `List[str]` | Numbered plan steps |
| `format_constraints()` | `List[Constraint]` | Constraint expression |
| `format_logic()` | `LogicalCondition` | Recursive PDDL formula |
| `format_params()` | `List[Parameter]` | `?r - rover ?l - location` |

### Helpers

- **`indent(string, level)`** - Indent PDDL blocks with consistent spacing
- **`remove_comments(text)`** - Strip `;`, `#`, `//` comments
- **`natural_sort_key(s)`** - Natural sort (`?b2` before `?b12`)

---

## `pddl_parser.py` - LLM Output & PDDL String Parsing

### LLM Output Extraction

The LLM is prompted to return JSON wrapped in XML tags. These functions parse that output:

```python
parse_xml_tags(llm_output, "predicates")
# -> ['[{"name": "clear", "params": [...]}, ...]']

parse_component(raw_blocks, Predicate, "predicates")
# -> [Predicate(name="clear", ...), ...]

parse_element(raw_blocks, InitialState, "initial")
# -> InitialState(facts=[...])
```

Both `parse_component` and `parse_element` validate output against Pydantic models and provide detailed error messages showing the expected schema when validation fails - useful for LLM debugging.

### PDDL String-to-Model Conversion

Parse existing `.pddl` files into L2P models:

```python
parse_domain_pddl(domain_str)      # -> DomainDetails
parse_problem_pddl(pddl_str)       # -> ProblemDetails
```

Converts raw PDDL into the internal Pydantic representation using the `pddl` library, supporting requirements, types, constants, predicates, functions, derived predicates, actions, objects, initial states, goal states, and metrics.

---

## `pddl_prompt.py` - Prompt Templates & Context

### Template Management

Default prompt templates are bundled in `l2p/templates/{domain,problem,feedback}/` and loaded via:

```python
load_default_template("domain", "prompt_predicates.md")
load_custom_template("/path/to/custom.md")
```

Three namespaces group the built-in templates:

| Namespace | Category | Templates |
|-----------|----------|-----------|
| `DEF_DOMAIN_PROMPTS` | Domain | domain, requirements, types, constants, predicates, functions, actions, durative_actions, events, processes, constraints, der_preds, nl_actions, parameters, preconds, effects, dur_conds, dur_effects |
| `DEF_PROBLEM_PROMPTS` | Problem | problem, objects, initial, goal, constraints, metric |
| `DEF_FB_PROMPTS` | Feedback | diagnosis, evaluate, reflection, revise, select, plan_diagnosis, plan_evaluate |

### Context Injection

```python
# Serialize Pydantic models to JSON context for prompt injection
jsonify_components([Predicate(name="clear", params=[...])])

# Build full <existing_context> block with nested XML tags
build_ctx(types=types, predicates=predicates)
```

The `safe_format()` function replaces `{placeholders}` in templates without breaking JSON curly braces:

```python
safe_format(template, description=domain_desc, context=build_ctx(...))
```
