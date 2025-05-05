# L2P: LLM-Powered PDDL Planning

**Official Documentation:**  
For detailed function references, visit our website: [Official API Documentation]().  
The L2P classes can be divided as follows:

---

## `domain_builder.py`
This class is responsible for generating PDDL domain information via LLMs.  
For full API reference, see the [documentation](docs/build/html/modules.html).

### Features Supported:
- [x] **Types** (PDDL 1.2+): Defines types for objects to organize and restrict possible object assignments in actions.
- [x] **Predicates** (PDDL 1.2+): Defines the truth values of propositions (true/false).
- [x] **Natural Language Actions** (PDDL 1.2+): Describes actions in natural language and converts them into formal PDDL actions.
- [x] **Basic Full PDDL Actions** (PDDL 1.2+): Complete actions with parameters, preconditions, and effects.
- [x] **Basic Action Parameters** (PDDL 1.2+): Defines parameters for actions.
- [x] **Basic Action Preconditions** (PDDL 1.2+): Specifies the conditions for actions.
- [x] **Basic Action Effects** (PDDL 1.2+): Defines the effects of actions on the state.
- [x] **Full Set of Basic PDDL Actions** (PDDL 1.2+): A complete specification of actions.
- [ ] **Action Costs** (PDDL 2.1+): Allows cost specifications for actions (e.g., optimizing time/resources).
- [ ] **Temporal Constraints** (PDDL 2.2+): Specifies time-related constraints on actions.
- [ ] **Disjunctive Preconditions** (PDDL 2.2+): Alternative conditions for action preconditions.
- [ ] **Derived Predicates** (PDDL 2.1+): Inferred predicates based on other conditions.
- [ ] **Non-deterministic Actions** (PDDL 2.2+): Models actions with multiple outcomes.
- [ ] **Conditional Effects/Quantifications** (PDDL 2.2+): Conditional action effects and use of quantifiers.
- [ ] **Mutex Relations** (PDDL 2.2+): Defines mutual exclusions between actions or predicates.

---

## `task_builder.py`
Responsible for generating PDDL task information via LLMs.  
Full API reference: See [documentation](docs/build/html/modules.html).

### Features Supported:
- [x] **Objects** (PDDL 1.2+): Defines objects involved in the problem.
- [x] **Initial State** (PDDL 1.2+): Specifies the initial configuration of the world.
- [x] **Goal State** (PDDL 1.2+): Defines the conditions to achieve the goal.
- [ ] **Quantified Goals** (PDDL 2.2+): Defines goals with quantification.
- [ ] **Negative Goals** (PDDL 2.2+): Specifies goals where predicates must be false.
- [ ] **Metric Optimization** (PDDL 2.1+): Optimizes a given metric, such as minimizing resources.
- [ ] **Timeline Constraints** (PDDL 2.2+): Specifies constraints governing the sequence of events.
- [ ] **Temporal Goal Definition** (PDDL 2.2+): Defines time-sensitive goals.
- [ ] **Preferences** (PDDL 3.0+): Defines soft, non-mandatory goals.
- [ ] **Resource Constraints** (PDDL 2.1+): Limits on resources like robots or fuel.
- [ ] **Durative Goals** (PDDL 2.2+): Specifies goals over a specific time duration.
- [ ] **Conditional Goals** (PDDL 2.2+): Defines goals based on certain conditions.

---

## `feedback_builder.py`
Returns feedback information via LLMs.

### General Functions:
- **`get_feedback()`**: Retrieves feedback based on user choice ("human", "llm", or "hybrid").
- **`human_feedback()`**: Allows user-provided human-in-the-loop feedback.

### Domain Feedback Functions:
- **`type_feedback()`**: Feedback on revised types.
- **`type_hierarchy_feedback()`**: Feedback on revised type hierarchy.
- **`nl_action_feedback()`**: Feedback on natural language actions.
- **`pddl_action_feedback()`**: Feedback on PDDL actions.
- **`parameter_feedback()`**: Feedback on action parameters.
- **`precondition_feedback()`**: Feedback on action preconditions.
- **`effect_feedback()`**: Feedback on action effects.
- **`predicate_feedback()`**: Feedback on predicates.
- **`task_feedback()`**: Complete feedback on revised PDDL tasks.

### Problem Feedback Functions:
- **`objects_feedback()`**: Feedback on objects.
- **`initial_state_feedback()`**: Feedback on initial states.
- **`goal_state_feedback()`**: Feedback on goal states.

---

## `prompt_builder.py`
Generates prompt templates for LLMs to assemble organized prompts and swap between them.

### Components:
- **Roles**: Overview task for the LLM.
- **Technique**: Defines prompting methods (e.g., CoT, ZPD).
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
Contains PDDL types 'Action' and 'Predicate' as well as Domain, Problem, and Plan details. These can be utilized to help organize builder method calls easier.

### pddl_validator.py
Contains tools to validate PDDL specifications and returns error feedback.

### pddl_planner.py
For ease of use, our library contains submodule [FastDownward](https://github.com/aibasel/downward/tree/308812cf7315fe896dbcd319493277d82aa36bd2). Fast Downward is a domain-independent classical planning system that users can run their PDDL domain and problem files on. The motivation is that the majority of papers involving PDDL-LLM usage uses this library as their planner.

This planner can be run like:
```python
from l2p.utils.pddl_planner import FastDownward

planner = FastDownward()   
domain = "path/to/domain.pddl"
problem = "path/to/problem.pddl"

pass, plan = planner.run_fast_downward(domain, problem)
```
