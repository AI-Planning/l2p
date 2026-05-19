Templates
================
It is **highly** recommended to use the base templates to properly extract LLM output into the designated Python formats from these methods.

Below are some examples of the base prompt structure that should be used in this library with your customized prompt using the `PromptBuilder` class. More details of each methods' prompt structure is found in **l2p/templates**.

There are three main folders found in `/templates`:

.. raw:: html

   <details>
   <summary><strong>templates/domain_templates</strong></summary>
   <ul style="margin-left: 30px; margin-top: 2px;">
    <li><i>/extract_nl_actions.txt</i>: for `DomainBuilder.extract_nl_actions()`</li>
    <li><i>/formalize_domain_spec.txt</i>: for `DomainBuilder.formalize_domain_specs()`</li>
    <li><i>/formalize_types.txt</i>: for `DomainBuilder.formalize_types()</li>
    <li><i>/formalize_type_hierarchy.txt</i>: for `DomainBuilder.formalize_type_hierarchy()</li>
    <li><i>/formalize_constants.txt</i>: for `DomainBuilder.formalize_constants()</li>
    <li><i>/formalize_predicates.txt</i>: for `DomainBuilder.formalize_predicates()</li>
    <li><i>/formalize_functions.txt</i>: for `DomainBuilder.formalize_functions()</li>
    <li><i>/formalize_pddl_action.txt</i>: for `DomainBuilder.formalize_pddl_action()</li>
    <li><i>/formalize_pddl_actions.txt</i>: for `DomainBuilder.formalize_pddl_actions()</li>
    <li><i>/formalize_parameters.txt</i>: for `DomainBuilder.formalize_parameters()</li>
    <li><i>/formalize_preconditions.txt</i>: for `DomainBuilder.formalize_preconditions()</li>
    <li><i>/formalize_effects.txt</i>: for `DomainBuilder.formalize_effects()</li>
   </ul>
   </details>

.. raw:: html

   <details>
   <summary><strong>templates/task_templates</strong></summary>
   <ul style="margin-left: 30px; margin-top: 2px;">
     <li><i>/formalize_task.txt</i>: for `TaskBuilder.formalize_task()</li>
     <li><i>/formalize_objects.txt</i>: for `TaskBuilder.formalize_objects()</li>
     <li><i>/formalize_initial.txt</i>: for `TaskBuilder.formalize_initial_state()</li>
     <li><i>/formalize_goal.txt</i>: for `TaskBuilder.formalize_goal_state()</li>
   </ul>
   </details>

.. raw:: html

   <details>
   <summary><strong>templates/feedback_templates</strong></summary>
   <ul style="margin-left: 30px; margin-top: 2px;">
     <li><i>/feedback.txt</i>: base template needed for LLM to provide feedback in any functions in `FeedbackBuilder`</li>
   </ul>
   </details>


Domain Extraction Prompts Example
-------------------------------------------------------
This is an example using `l2p/templates/domain_templates/extract_pddl_action.txt`

.. code-block:: python
    :linenos:

    from l2p.prompt_builder import PromptBuilder
    from l2p.utils import load_file

    # LOAD BASE FORMAT TEMPLATE
    template_path = "templates/domain_templates/formalize_pddl_action.txt"
    format_template = load_file(template_path)

    role_desc = "You are a PDDL action constructor. Your job is to take " \
        "the task given in natural language and convert it into PDDL."
                
    task_desc = """
    ## Domain
    {domain_desc}

    ## Available types
    {types}

    ## Future actions to be implemented later
    {action_list}

    ## Action name (to implement now)
    {action_name}

    ## Action description (corresponding action description)
    {action_desc}

    ## Available predicates
    {predicates}
    """
            
    # assemble prompt
    action_construction_prompt = PromptBuilder(
        role=role_desc, 
        format=format_template, 
        task=task_desc
        )

    print(action_construction_prompt.generate_prompt())


The following is the output: ::
    
    [ROLE]: You are a PDDL action constructor. Your job is to take the task given in natural language and convert it into PDDL.

    ------------------------------------------------
    [FORMAT]: End your final answers underneath the headers: '### Action Parameters,' '### Action Preconditions,' '### Action Effects,' and '### New Predicates' with ``` ``` comment blocks in PDDL as so:

    ### Action Parameters
    ```
    - ?t - type: 'parameter_description'
    ```

    ### Action Preconditions
    ```
    (and
        (predicate_name ?t1 ?t2) ; COMMENT DESCRIPTION
    )
    ```

    ### Action Effects
    ```
    (and
        (predicate_name ?t1 ?t2) ; COMMENT DESCRIPTION
    )
    ```

    ### New Predicates
    ```
    - (predicate_name ?t1 - type_1 ?t2 - type_2): 'predicate_description'
    ``` 

    If there are no new predicates created, keep an empty space enclosed ```  ``` with the '### New Predicates' header.

    ------------------------------------------------
    [TASK]:
    Here is the task to solve:

    ## Domain
    {domain_desc}

    ## Available types
    {types}

    ## Future actions to be implemented later
    {action_list}

    ## Action name (to implement now)
    {action_name}

    ## Action description (corresponding action description)
    {action_desc}

    ## Available predicates
    {predicates}


Users have the flexibility to customize all aspects of their prompts, with the exception of the provided base template. While users can include few-shot examples to guide the LLM, the base template must remain intact during inference.