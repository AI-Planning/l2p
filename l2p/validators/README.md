# L2P Validators - Symbolic PDDL Syntax & Semantic Checking

> Rule-based validation framework for catching PDDL errors before they reach a planner.

Syntax and semantic validators are the backbone to symbolic planning. Users / agents can use these classes and functions to check for code correctness. **It is heavily recommended** that one would validate after LLM PDDL generation.

```
l2p/validators/
├── base.py          # Core framework: ValidationResult, ValidationRule, SyntaxValidator
├── domain.py        # Domain-specific rules (@domain_rule decorator)
├── problem.py       # Problem-specific rules (@problem_rule decorator)
├── plan.py          # Plan validation (stub)
└── __init__.py
```

---

## Architecture

```
ValidationRule (ABC)
    ├── target_models  → which Pydantic types this rule applies to
    └── validate(target, context) → ValidationResult

SyntaxValidator (orchestrator)
    ├── register_rule(rule)
    └── validate_component(target, context) → ValidationResult
```

**Rules** operate on individual Pydantic models with access to the full generation context (all other domain/problem components). This lets a predicate rule check naming conventions, while an action rule verifies that its precondition symbols are actually declared in the domain.

---

## ValidationResult

```python
result = ValidationResult()
result.valid       # bool - False if any errors were added
result.errors      # List[str] - fatal errors
result.warnings    # List[str] - non-fatal warnings
result.add_error("...")
result.add_warning("...")
```

---

## Domain Validation Rules

Loaded automatically via `@domain_rule` decorators. Create a `DomainValidator` to run them:

```python
from l2p.validators.domain import DomainValidator

validator = DomainValidator()  # loads all decorated rules
result = validator.validate_component(
    target=my_action,
    context={"types": types, "predicates": predicates, ...}
)
```

| Rule | Applies To | Checks |
|------|-----------|--------|
| `validate_pddl_naming` | Types, Constants, Predicates, Functions, DerivedPredicates, Actions, DurativeActions, Events, Processes | ✅ Valid characters<br>✅ No `?` prefix<br>✅ No PDDL reserved keywords<br>✅ No duplicate names<br>⚠ Uppercase name warnings |
| `check_type_inheritance` | `PDDLType` | Parent type exists in domain |
| `check_type_cycle` | `PDDLType` | No cyclic inheritance chains (`A→B→A`) |
| `check_constant_inheritance` | `Constant` | Declared type exists |
| `check_parameter_types` | Predicate, Function, DerivedPredicate, Action, DurativeAction, Event, Process | Parameter types declared + `?` prefix on variables |
| `check_derived_predicate` | `DerivedPredicate` | Condition uses declared symbols |
| `check_action_precondition` | `Action`, `ActionPrecondition` | Precondition uses declared predicates/functions |
| `check_action_effect` | `Action`, `ActionEffect` | Effects use declared symbols |
| `check_dur_action_conditions` | `DurativeAction`, `DurativeActionConditions` | Conditions use declared symbols |
| `check_dur_action_effect` | `DurativeAction`, `DurativeActionEffect` | Effects use declared symbols |
| `check_constraint` | `Constraint` | Constraint uses declared symbols |
| `check_component_variables` | Action, DurativeAction, Event, Process, DerivedPredicate | All `?variables` declared in params + **semantic type matching** against predicate/function signatures |

### Semantic Variable & Type Checking

`check_component_variables` goes beyond simple existence - it builds a `DomainSemantics` index of all predicate/function signatures and verifies:

1. **Arity** - correct argument count per predicate
2. **Type compatibility** - each argument's type (or declared variable type) is a subtype of the parameter's expected type (following the type hierarchy)
3. **Scope resolution** - `forall`/`exists` quantifiers introduce new local variables

---

## Problem Validation Rules

```python
from l2p.validators.problem import ProblemValidator

validator = ProblemValidator()
result = validator.validate_component(
    target=my_initial_state,
    context={"types": types, "predicates": predicates, "objects": objects}
)
```

| Rule | Applies To | Checks |
|------|-----------|--------|
| `validate_pddl_naming` | `PDDLObject` | Name character validity, no PDDL keywords, no duplicates |
| `check_obj_type_inheritance` | `PDDLObject` | Object type exists in domain |
| `check_initial_state` | `InitialState` | Facts use declared objects/constants with correct types; rejects `?variables` in init |
| `check_goal_state` | `GoalState` | Goal conditions use declared objects with correct type signatures |
| `check_metric_syntax` | `Metric` | Functions used in `expression` are declared; no variables allowed |

Problem validation uses `ProblemSemantics` (analogous to `DomainSemantics`) to check object types against predicate signatures, including support for constants and timed initial literals.

---

## Adding Custom Rules

Register rules programmatically:

```python
from l2p.validators.base import FunctionalRule, SyntaxValidator
from l2p.utils.pddl_types import Action

def my_rule(target, context) -> ValidationResult:
    result = ValidationResult()
    if len(target.params) > 5:
        result.add_warning(f"Action '{target.name}' has many parameters.")
    return result

validator = SyntaxValidator()
validator.register_rule(
    FunctionalRule(name="param_count", targets=[Action], func=my_rule)
)
```

Or use decorators for domain-specific registries:

```python
from l2p.validators.domain import domain_rule
from l2p.utils.pddl_types import Predicate

@domain_rule(targets=[Predicate])
def check_predicate_names(target, context):
    result = ValidationResult()
    if target.name.startswith("is_"):
        result.add_warning("Consider removing 'is_' prefix.")
    return result

# Automatically loaded when DomainValidator(use_defaults=True)
```

--- 

## Validating a Full Domain or Problem

For convenience, both validators have a method that accepts the root model directly and runs every applicable rule with cross-component context:

```python
from l2p.validators.domain import DomainValidator
from l2p.validators.problem import ProblemValidator
from l2p.utils.pddl_types import DomainDetails, ProblemDetails

domain = DomainDetails(name="blocksworld", types=[...], predicates=[...], ...)
result = DomainValidator().validate_domain(domain)
if not result.valid:
    for error in result.errors:
        print(f"  {error}")
```

`validate_domain()` checks all **11 component fields**: types, constants, predicates, functions, derived_predicates, actions, durative_actions, events, processes, constraint, and requirements.

### Cross-Domain Problem Validation

To validate a problem against its domain (checking that object types exist in the domain, predicate/function calls in init and goal match declared signatures, etc.), pass the domain as a second argument:

```python
# Parse both files
from l2p.utils.pddl_parser import parse_domain_pddl, parse_problem_pddl

domain = parse_domain_pddl(open("domain.pddl").read())
problem = parse_problem_pddl(open("problem.pddl").read())

result = ProblemValidator().validate_problem(problem, domain=domain)
if not result.valid:
    for error in result.errors:
        print(f"  {error}")
```

Without a domain, `validate_problem()` checks structural validity (object names, metric syntax). With a domain, it additionally verifies:
- Object types are declared in the domain's type hierarchy
- Initial state and goal predicates match domain predicate signatures (arity + argument types)
- Metric expressions use declared functions
- All predicate/function symbols referenced in init and goal are defined in the domain

From the CLI:
```bash
l2p validate problem problem.pddl --domain domain.pddl
```

---

## Integration with Builders

Validation fits naturally in the generation pipeline:

```
1. LLM generates JSON → Pydantic model
2. DomainValidator.validate_component(model, context)
   Or validate the whole container: validate_domain(domain_details)
3. If errors → FeedbackBuilder.llm_diagnose() → llm_revise()
4. If valid → format to PDDL string → planner
```

The `FunctionalRule` wrapper makes it easy to convert any Python function into a formal rule, and the `@domain_rule`/`@problem_rule` decorators automatically register rules into the default validation pipeline.
