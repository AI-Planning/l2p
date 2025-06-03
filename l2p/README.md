# L2P: LLM-Powered PDDL Planning

**Official Documentation:**  
For detailed function references, visit our website: [**L2P Documention**](https://marcustantakoun.github.io/l2p.github.io/)

The L2P classes can be divided as follows:

---

## `domain_builder.py`
This class is responsible for generating PDDL domain information via LLMs.  
Full API reference: [**L2P Documention**](https://marcustantakoun.github.io/l2p.github.io/l2p.html)

### Features Supported:
- [x] **Types** (PDDL 1.2+)
- [x] **Constants** (PDDL 1.2+)
- [x] **Predicates** (PDDL 1.2+)
- [x] **Functions / Numerical Fluents** (PDDL 2.1+)
- [x] **Basic Action Parameters** (PDDL 1.2+)
- [x] **Basic Action Preconditions** (PDDL 1.2+)
- [x] **Basic Action Effects** (PDDL 1.2+)
- [x] **Quantified Preconditions and Effects** (PDDL 2.2+)
- [x] **Conditional Effects** (PDDL 2.2+)
- [x] **Disjunctive Preconditions** (PDDL 2.2+)
- [ ] **Action Costs** (PDDL 2.1+)
- [ ] **Temporal Constraints** (PDDL 2.2+)
- [ ] **Derived Predicates** (PDDL 2.1+)
- [ ] **Non-deterministic Actions** (PDDL 2.2+)
- [ ] **Mutex Relations** (PDDL 2.2+)

---

## `task_builder.py`
Responsible for generating PDDL task information via LLMs. 

### Features Supported:
- [x] **Objects** (PDDL 1.2+): Defines objects involved in the problem.
- [x] **Initial State** (PDDL 1.2+): Specifies the initial configuration of the world.
- [x] **Goal State** (PDDL 1.2+): Defines the conditions to achieve the goal.
- [x] **Negative Goals** (PDDL 2.2+): Specifies goals where predicates must be false.
- [ ] **Temporal Goal Definition** (PDDL 2.2+): Defines time-sensitive goals.
- [ ] **Quantified Goals** (PDDL 2.2+): Defines goals with quantification.
- [ ] **Durative Goals** (PDDL 2.2+): Specifies goals over a specific time duration.
- [ ] **Conditional Goals** (PDDL 2.2+): Defines goals based on certain conditions.
- [ ] **Metric Optimization** (PDDL 2.1+): Optimizes a given metric, such as minimizing resources.
- [ ] **Resource Constraints** (PDDL 2.1+): Limits on resources like robots or fuel.
- [ ] **Timeline Constraints** (PDDL 2.2+): Specifies constraints governing the sequence of events.
- [ ] **Preferences** (PDDL 3.0+): Defines soft, non-mandatory goals.

---

## `feedback_builder.py`
Returns feedback information via LLMs.

### General Functions:
- **`get_feedback()`**: Retrieves feedback based on user choice ("human", "llm", or "hybrid").
- **`human_feedback()`**: Allows user-provided human-in-the-loop feedback.

### Domain Feedback Functions:
- **`type_feedback()`**: Feedback on revised types.
- **`predicate_feedback()`**: Feedback on predicates.
- **`nl_action_feedback()`**: Feedback on natural language actions.
- **`pddl_action_feedback()`**: Feedback on PDDL actions.
- **`parameter_feedback()`**: Feedback on action parameters.
- **`precondition_feedback()`**: Feedback on action preconditions.
- **`effect_feedback()`**: Feedback on action effects.

### Problem Feedback Functions:
- **`task_feedback()`**: Complete feedback on revised PDDL tasks.
- **`objects_feedback()`**: Feedback on objects.
- **`initial_state_feedback()`**: Feedback on initial states.
- **`goal_state_feedback()`**: Feedback on goal states.

---

## `prompt_builder.py`
Generates prompt templates for LLMs to assemble organized prompts and swap between them.

### Components:
- **Roles**: Overview task for the LLM.
- **Format**: Defines format method for LLM to follow as final output.
- **Example**: Provides in-context examples.
- **Task**: Placeholder definitions for proper information extraction.

---

## ./llm Folder
This class is responsible for loading models. Currently, we provide LLM interface support for compatible OPENAI SDK providers, as well as Huggingface API. Users can implement specific backend provider LLM interfaces using **BaseLLM**, found in **l2p/llm/base.py**, which contains an abstract class and method for implementing any model classes in the case of other third-party LLM uses. 

Users can refer to l2p/llm/utils/llm.yaml to better understand (and create their own) model configuration options, including tokenizer settings, generation parameters, and provider-specific settings.

## utils
This parent folder contains other tools necessary for L2P. They consist of:

### pddl_format.py
Contains tools to format L2P's python structured PDDL components into strings required for **DomainBuilder.generate_domain** and **TaskBuilder.generate_task**.

### pddl_parser.py
Contains tools to parse L2P information extraction.

### pddl_types.py
Contains PDDL types 'Action' and 'Predicate' as well as Domain, Problem, Plan details, etc. These can be utilized to help organize builder method calls easier.

### pddl_validator.py
Contains tools to validate PDDL specifications and returns error feedback. Visit [**L2P Documention**](https://marcustantakoun.github.io/l2p.github.io/) for more information how to use the validators.

### pddl_planner.py
For ease of use, our library contains submodule [FastDownward](https://github.com/aibasel/downward/tree/308812cf7315fe896dbcd319493277d82aa36bd2). Fast Downward is a domain-independent classical planning system that users can run their PDDL domain and problem files on. The motivation is that the majority of papers involving PDDL-LLM usage uses this library as their planner.

This planner can be run like:
```python
from l2p.utils.pddl_planner import FastDownward

# retrieve pddl files
domain_file = "tests/pddl/test_domain.pddl"
problem_file = "tests/pddl/test_problem.pddl"

# instantiate FastDownward class
planner = FastDownward(planner_path="<PATH_TO>/downward/fast-downward.py")

# run plan
success, plan_str = planner.run_fast_downward(
    domain_file=domain_file,
    problem_file=problem_file,
    search_alg="lama-first"
)

print(plan_str)
```