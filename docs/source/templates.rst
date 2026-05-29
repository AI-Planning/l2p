Templates
================
It is **highly** recommended to use the base templates to properly extract LLM output into the designated Python formats from these methods.

Below are some examples of the base prompt structure that should be used in this library with your customized prompt using the `PromptBuilder` class. More details of each methods' prompt structure is found in **l2p/templates**.

There are four main folders found in ``/templates``:

.. raw:: html

   <details>
   <summary><strong>templates/domain</strong></summary>
   <ul style="margin-left: 30px; margin-top: 2px;">
    <li><i>/prompt_types.md</i>: for ``DomainBuilder.formalize_component(model, PDDLType, ...)``</li>
    <li><i>/prompt_constants.md</i>: for ``DomainBuilder.formalize_component(model, Constant, ...)``</li>
    <li><i>/prompt_predicates.md</i>: for ``DomainBuilder.formalize_component(model, Predicate, ...)``</li>
    <li><i>/prompt_functions.md</i>: for ``DomainBuilder.formalize_component(model, Function, ...)``</li>
    <li><i>/prompt_derived_predicates.md</i>: for ``DomainBuilder.formalize_component(model, DerivedPredicate, ...)``</li>
    <li><i>/prompt_requirements.md</i>: for ``DomainBuilder.formalize_component(model, Requirement, ...)``</li>
    <li><i>/prompt_parameters.md</i>: for ``DomainBuilder.formalize_component(model, Parameter, ...)``</li>
    <li><i>/prompt_preconditions.md</i>: for ``DomainBuilder.formalize_component(model, ActionPrecondition, ...)``</li>
    <li><i>/prompt_effects.md</i>: for ``DomainBuilder.formalize_component(model, ActionEffect, ...)``</li>
    <li><i>/prompt_actions.md</i>: for ``DomainBuilder.formalize_component(model, Action, ...)``</li>
    <li><i>/prompt_nl_actions.md</i>: for ``DomainBuilder.extract_nl(model, "nl_actions", ...)``</li>
    <li><i>/prompt_durative_actions.md</i>: for ``DomainBuilder.formalize_component(model, DurativeAction, ...)``</li>
    <li><i>/prompt_durative_conditions.md</i>: for ``DomainBuilder.formalize_component(model, DurativeActionConditions, ...)``</li>
    <li><i>/prompt_durative_effects.md</i>: for ``DomainBuilder.formalize_component(model, DurativeActionEffect, ...)``</li>
    <li><i>/prompt_nl_durative_actions.md</i>: for ``DomainBuilder.extract_nl(model, "nl_durative_actions", ...)``</li>
    <li><i>/prompt_events.md</i>: for ``DomainBuilder.formalize_component(model, Event, ...)``</li>
    <li><i>/prompt_processes.md</i>: for ``DomainBuilder.formalize_component(model, Process, ...)``</li>
    <li><i>/prompt_constraints.md</i>: for ``DomainBuilder.formalize_component(model, Constraint, ...)``</li>
   </ul>
   </details>

.. raw:: html

   <details>
   <summary><strong>templates/problem</strong></summary>
   <ul style="margin-left: 30px; margin-top: 2px;">
     <li><i>/prompt_problem.md</i>: for ``ProblemBuilder.formalize_component(model, ProblemDetails, ...)``</li>
     <li><i>/prompt_objects.md</i>: for ``ProblemBuilder.formalize_component(model, PDDLObject, ...)``</li>
     <li><i>/prompt_initial_states.md</i>: for ``ProblemBuilder.formalize_component(model, InitialState, ...)``</li>
     <li><i>/prompt_goal_states.md</i>: for ``ProblemBuilder.formalize_component(model, GoalState, ...)``</li>
     <li><i>/prompt_metric.md</i>: for ``ProblemBuilder.formalize_component(model, Metric, ...)``</li>
     <li><i>/prompt_constraints.md</i>: for ``ProblemBuilder.formalize_component(model, Constraint, ...)``</li>
   </ul>
   </details>

.. raw:: html

   <details>
   <summary><strong>templates/feedback</strong></summary>
   <ul style="margin-left: 30px; margin-top: 2px;">
     <li><i>/prompt_diagnosis.md</i>: for ``FeedbackBuilder.llm_diagnose()``</li>
     <li><i>/prompt_evaluate.md</i>: for ``FeedbackBuilder.llm_evaluate()``</li>
     <li><i>/prompt_reflection.md</i>: for ``FeedbackBuilder.llm_reflect()``</li>
     <li><i>/prompt_revise.md</i>: for ``FeedbackBuilder.llm_revise()``</li>
     <li><i>/prompt_select.md</i>: for ``FeedbackBuilder.llm_select()``</li>
     <li><i>/prompt_plan_evaluate.md</i>: for ``FeedbackBuilder.llm_evaluate_plan()``</li>
     <li><i>/prompt_plan_diagnosis.md</i>: for ``FeedbackBuilder.llm_diagnose_plan()``</li>
   </ul>
   </details>

.. raw:: html

   <details>
   <summary><strong>templates/custom</strong> (Multi-Component)</summary>
   <p style="margin-left: 30px; margin-top: 8px;">
   These templates extract <strong>multiple PDDL components in a single LLM call</strong>,
   improving cross-component consistency.
   Use them with <code>formalize_component(model, cls_list=[...], ...)</code>.
   </p>
   <ul style="margin-left: 30px; margin-top: 2px;">
     <li><i>/prompt_types_predicates.md</i>: Types + Predicates</li>
     <li><i>/prompt_types_constants_predicates.md</i>: Types + Constants + Predicates</li>
     <li><i>/prompt_types_predicates_functions.md</i>: Types + Predicates + Functions</li>
     <li><i>/prompt_types_predicates_functions_actions.md</i>: Types + Preds + Functions + Actions</li>
     <li><i>/prompt_predicates_actions.md</i>: Predicates + Actions</li>
     <li><i>/prompt_actions_constraints.md</i>: Actions + Constraints</li>
     <li><i>/prompt_actions_durative_actions.md</i>: Actions + DurativeActions</li>
     <li><i>/prompt_events_processes.md</i>: Events + Processes</li>
     <li><i>/prompt_derived_predicates_predicates.md</i>: DerivedPredicates + Predicates</li>
     <li><i>/prompt_objects_initial_state.md</i>: Objects + InitialState</li>
     <li><i>/prompt_objects_initial_goal.md</i>: Objects + Init + Goal</li>
     <li><i>/prompt_initial_goal_metric.md</i>: Init + Goal + Metric</li>
   </ul>
   </details>


Domain Extraction Prompts Example
-------------------------------------------------------
This is an example using the ``PromptBuilder`` class:

.. code-block:: python
    :linenos:

    from l2p import Action
    from l2p import PromptBuilder

    role_desc = "You are a PDDL action constructor. Your job is to take " \
    "the task given in natural language and convert it into PDDL actions."

    format_desc = "You must follow the strict JSON object defined below. " \
    "Enclose your final answer in <actions> ... </actions> tags."

    pb = (PromptBuilder()
        .set_role(role_str=role_desc)
        .set_format(format_str=format_desc)
        .set_format_example(component=Action) # set extraction example block for component
        .add_rule(rule_str="Every action parameter must reference a defined type.")
        .add_rule(rule_str="Preconditions and effects must only use defined predicates.")
        .add_example("INPUT: ...\nOUTPUT: ...")
        .add_example("INPUT: ...\nOUTPUT: ...")
        .set_task("Generate actions for a logistics domain."))

    print(pb.generate_prompt())
    # alternatively - pb.save_prompt(filename="my_prompt.md")


The following is the output: ::

    ## ROLE
    You are a PDDL action constructor. Your job is to take the task given in natural language and convert it into PDDL actions.

    ## OUTPUT FORMAT
    You must follow the strict JSON object defined below. Enclose your final answer in <actions> ... </actions> tags.

    <actions>
    [
        {
            "name": "move-rover",
            "params": [
                {
                    "variable": "?r",
                    "type": "rover"
                },
                {
                    "variable": "?from",
                    "type": "waypoint"
                },
                {
                    "variable": "?to",
                    "type": "waypoint"
                }
            ],
            "preconditions": {
                "conditions": [
                    "(at ?r ?from)",
                    {
                        "operator": "not",
                        "condition": "(= ?from ?to)"
                    }
                ],
                "desc": "Optional (str)"
            },
            "effects": {
                "add": [
                    "(at ?r ?to)"
                ],
                "delete": [
                    "(at ?r ?from)"
                ],
                "numeric": [
                    "(decrease (battery-level ?r) 10.0)"
                ],
                "conditional": [
                    {
                        "condition": [
                            "(has-rock-sample ?r)"
                        ],
                        "effect": {
                            "add": [
                                "(carrying-heavy-load ?r)"
                            ],
                            "delete": [],
                            "numeric": []
                        }
                    }
                ],
                "desc": "Optional (str)"
            },
            "desc": "Optional (str)"
        }
    ]
    </actions>

    ## RULES
    1. Every action parameter must reference a defined type.
    2. Preconditions and effects must only use defined predicates.

    ## EXAMPLE(S)
    # Example 1:
    INPUT: ...
    OUTPUT: ...

    # Example 2:
    INPUT: ...
    OUTPUT: ...
    --------------------------------------------------

    ## TASK
    Generate actions for a logistics domain.

    {description}

    {context}


Users have the flexibility to customize all aspects of their prompts, with the exception of the provided base template. While users can include few-shot examples to guide the LLM, the base template must remain intact during inference.