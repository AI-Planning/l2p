Getting Started
================

Installing
----------
L2P can be installed with pip::

    pip install l2p

Using L2P
-------------

First things first, import the whole L2P library, or necessary modules (see :doc:`l2p`)::

    from l2p import *

L2P requires access to an LLM. Set up your LLM class and methods using the abstract ``LLM(ABC)`` class. In this case, we have set up OpenAI's GPT models to our library for quickstart. ::

    export OPENAI_API_KEY='YOUR-KEY' # e.g. OPENAI_API_KEY='sk-123456'
    engine = "gpt-4o-mini"
    api_key = os.environ.get('OPENAI_API_KEY')
    openai_llm = OPENAI(model=engine, api_key=api_key)

Create your customised prompts using the ``PromptBuilder`` class. These prompt templates can be used for any of the extraction methods. This is an example of creating a prompt for type extraction: ::

    prompt_builder = PromptBuilder()

    role_desc = "Your role is to..."
    tech_desc = "You must follow this strategy..."
    ex_desc = "Here is an example..."
    task_desc = "Here are the given information..."

    type_extraction_prompt = PromptBuilder(
        role = role_desc,
        technique = tech_desc,
        examples = [ex_desc]
        task = task_desc
    )

Build PDDL domain components using the ``DomainBuilder`` class. This is an example of extracting PDDL types: ::

    domain = "The Blocksworld domain is..."

    domain_builder = DomainBuilder()

    domain_builder.formalize_types(
        model = openai_llm,
        domain_desc = domain,
        prompt_template = type_extraction_prompt
    )

Build PDDL problem components using the ``TaskBuilder`` class. This is an example of extracting PDDL initial states: ::

    problem = "There are 5 blocks..."

    task_builder = TaskBuilder()

    task_builder.formalize_initial_states(
        model = openai_llm,
        problem_desc = problem,
        prompt_template = task_prompt
    )

Build LLM feedback components using the ``FeedbackBuilder`` class. This is an example of getting LLM feedback from types: ::

    feedback_builder = FeedbackBuilder()

    feedback_builder.type_feedback(
        model = openai_llm,
        domain_desc = domain,
        response = "Retry this..."
        feedback_template = type_feedback_template,
        feedback_type = "llm"
    )

Below are actual runnable usage examples. This is the general setup to build domain predicates:

.. code-block:: python
    :linenos:

    import os, json
    from openai import OpenAI
    from l2p.domain_builder import DomainBuilder
    from l2p.llm_builder import GPT_Chat
    from l2p.utils.pddl_parser import prune_predicates, format_types

    def load_file(file_path):
        _, ext = os.path.splitext(file_path)
        with open(file_path, 'r') as file:
            if ext == '.json': return json.load(file)
            else: return file.read().strip()

    domain_builder = DomainBuilder()

    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', None)) # REPLACE WITH YOUR OWN OPENAI API KEY
    model = GPT_Chat(client=client, engine="gpt-4o-mini")

    # load in assumptions
    domain_desc = load_file(r'tests/usage/prompts/domain/blocksworld_domain.txt')
    formalize_predicates_prompt = load_file(r'tests/usage/prompts/domain/extract_predicates.txt')
    types = load_file(r'tests/usage/prompts/domain/types.json')
    action = load_file(r'tests/usage/prompts/domain/action.json')

    # extract predicates via LLM
    predicates, llm_output = domain_builder.formalize_predicates(
        model=model,
        domain_desc=domain_desc,
        prompt_template=formalize_predicates_prompt,
        types=types,
        nl_actions={action['action_name']: action['action_desc']}
        )

    # format key info into PDDL strings
    predicate_str = "\n".join([pred["clean"].replace(":", " ; ") for pred in predicates])

    print(f"PDDL domain predicates:\n{predicate_str}")

The following output is: ::

    ### OUTPUT
    (holding ?a - arm ?b - block) ;  true if the arm ?a is holding the block ?b
    (on_top ?b1 - block ?b2 - block) ;  true if the block ?b1 is on top of the block ?b2
    (clear ?b - block) ;  true if the block ?b is clear (no block on top of it)
    (on_table ?b - block) ;  true if the block ?b is on the table
    (empty ?a - arm) ;  true if the arm ?a is empty (not holding any block)

Here is how you would setup a PDDL problem:

.. code-block:: python
    :linenos:

    from l2p.task_builder import TaskBuilder

    task_builder = TaskBuilder()

    # load in assumptions
    problem_desc= load_file(r'tests/usage/prompts/problem/blocksworld_problem.txt')
    formalize_task_prompt = load_file(r'tests/usage/prompts/problem/extract_task.txt')

    # extract PDDL task specifications via LLM
    objects, initial_states, goal_states, llm_response = task_builder.formalize_task(
        model=model,
        problem_desc=problem_desc,
        prompt_template=formalize_task_prompt,
        types=types,
        predicates=predicates
        )

    # format key info into PDDL strings
    objects_str = task_builder.format_objects(objects)
    initial_str = task_builder.format_initial(initial_states)
    goal_str = task_builder.format_goal(goal_states)

    # generate task file
    pddl_problem = task_builder.generate_task("blocksworld_problem", objects_str, initial_str, goal_str)

    print(f"PDDL problem: {pddl_problem}")

The following output is: ::

    ### OUTPUT
    (define
        (problem blocksworld_problem_problem)
        (:domain blocksworld_problem)

        (:objects
            blue_block - object
            red_block - object
            yellow_block - object
            green_block - object
        )

        (:init
            (on_top blue_block red_block)
            (on_top red_block yellow_block)
            (on_table yellow_block)
            (on_table green_block)
            (clear blue_block)
            (clear green_block)
            (empty arm)
        )

        (:goal
            (and
                (on_top red_block green_block)
            )
        )
    )

Here is how you would setup a Feedback Mechanism:

.. code-block:: python
    :linenos:

    from l2p.feedback_builder import FeedbackBuilder

    feedback_builder = FeedbackBuilder()

    feedback_template = load_file(r'tests/usage/prompts/problem/feedback.txt')

    objects, initial, goal, feedback_response = feedback_builder.task_feedback(
        model=model,
        problem_desc=problem_desc,
        feedback_template=feedback_template,
        feedback_type="llm",
        predicates=predicates,
        types=types,
        llm_response=llm_response)

    print("FEEDBACK:\n", feedback_response)

The following output is: ::

    ### OUTPUT
    My concrete suggestions are the following:
    - Add the arm as an object:
        - arm - object
    - Include the missing predicate for the yellow block in the initial state:
        - (clear yellow_block): yellow_block is clear

    The revised initial state should look like this:
    ```
    (on_top blue_block red_block)
    (on_top red_block yellow_block)
    (on_table yellow_block)
    (on_table green_block)
    (clear green_block)
    (clear yellow_block)
    (empty arm)
    ```

    Overall, the response is: Yes.

***IMPORTANT***
It is **highly** recommended to use the base template found in :doc:`templates` to properly extract LLM output into the designated Python formats from these methods.
