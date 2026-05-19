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
    # OR
    from l2p import DomainBuilder, TaskBuilder, PromptBuilder, FeedbackBuilder
    
    # util functions
    from l2p.utils import *


L2P requires access to an LLM. Set up your LLM class and methods using the abstract ``BaseLLM(ABC)`` class. In this case, we have set up OpenAI's models to our library for quickstart. ::

    export OPENAI_API_KEY='YOUR-KEY' # e.g. OPENAI_API_KEY='sk-123456'
    engine = "gpt-4o-mini"
    api_key = os.environ.get('OPENAI_API_KEY')
    openai_llm = OPENAI(model=engine, api_key=api_key)

Users can pass any prompt into their LLMs--as long as the structured output prompt follows the respective function (see :doc:`templates`). We recommend using the ``PromptBuilder`` class that help organize prompts. This is a simple example:

.. code-block:: python
    :linenos:

    from l2p.prompt_builder import PromptBuilder
    prompt_builder = PromptBuilder()

    role_desc = "Your role is to..."
    format_desc = "You must follow this format..."
    ex_desc = "Here is an example..."
    task_desc = "{PLACEHOLDER}"

    prompt = PromptBuilder(
        role = role_desc,
        format = format_desc,
        examples = [ex_desc],
        task = task_desc
    )

    print(prompt.generate_prompt())

Generated prompt: ::

    [ROLE]: Your role is to...

    ------------------------------------------------
    [FORMAT]: You must follow this format...

    ------------------------------------------------
    [EXAMPLE(S)]:
    Example 1:
    Here is an example...

    ------------------------------------------------
    [TASK]:
    Here is the task to solve:
    {PLACEHOLDER}

Build PDDL domain components using the ``DomainBuilder`` class. This is an example of extracting PDDL types using PromptBuilder:

.. code-block:: python
    :linenos:

    import os
    from l2p.domain_builder import DomainBuilder
    from l2p.prompt_builder import PromptBuilder
    from l2p.llm.openai import OPENAI
    from l2p.utils import format_types, load_file

    api_key = os.environ.get('OPENAI_API_KEY')
    gpt_4o_mini = OPENAI(model="gpt-4o-mini", api_key=api_key)

    domain_builder = DomainBuilder()
    types_prompt = PromptBuilder(
        role="You are a PDDL assistant that is helping me design :types.",
        format=load_file("templates/domain_templates/formalize_type.txt"),
        task="{domain_desc}"
    )

    domain_desc = "The AI agent here is a mechanical robot arm that can pick and " \
        "place the blocks. Only one block may be moved at a time: it may either " \
        "be placed on the table or placed atop another block. Because of this, " \
        "any blocks that are, at a given time, under another block cannot be moved."

    # extract types via LLM
    types, llm_output, validation_info = domain_builder.formalize_types(
        model=gpt_4o_mini,
        domain_desc=domain_desc,
        prompt_template=types_prompt.generate_prompt()
        )

    # print out types
    print(format_types(types=types))

Generated types output: ::

    {
        'block': '; A physical object that can be picked up and moved by the robot arm.', 
        'table': '; A flat surface where blocks can be placed.', 
        'robot_arm': '; The mechanical device capable of picking and placing blocks.'
    }

Build PDDL problem components using the ``TaskBuilder`` class. This is an example of extracting PDDL initial states:

.. code-block:: python
    :linenos:

    import os
    from l2p.task_builder import TaskBuilder
    from l2p.prompt_builder import PromptBuilder
    from l2p.llm.openai import OPENAI
    from l2p.utils import format_initial, load_file

    api_key = os.environ.get('OPENAI_API_KEY')
    gpt_4o_mini = OPENAI(model="gpt-4o-mini", api_key=api_key)

    task_builder = TaskBuilder()
    init_prompt = PromptBuilder(
        role="You are a PDDL assistant that is helping me design :init problems",
        format=load_file("templates/task_templates/formalize_initial.txt"),
        task="{problem_desc}"
    )

    problem_desc = "There are four blocks currently. The blue block is on the red " \
        "which is on the yellow. The yellow and the green are on the table. I want " \
        "the red on top of the green."

    initial_states, llm_output, validation_info = task_builder.formalize_initial_state(
        model = gpt_4o_mini,
        problem_desc = problem_desc,
        prompt_template = init_prompt.generate_prompt()
    )

    print(format_initial(initial_states=initial_states))

Generated initial states: ::

    (on blue red)
    (on red yellow)
    (on yellow table)
    (on green table)

Build LLM feedback components using the ``FeedbackBuilder`` class. This is an example of getting LLM feedback from types:

.. code-block:: python
    :linenos:

    import os
    from l2p.feedback_builder import FeedbackBuilder
    from l2p.prompt_builder import PromptBuilder
    from l2p.llm.openai import OPENAI
    from l2p.utils import load_file

    api_key = os.environ.get('OPENAI_API_KEY')
    gpt_4o_mini = OPENAI(model="gpt-4o-mini", api_key=api_key)

    domain_desc = "The AI agent here is a mechanical robot arm that can pick and " \
            "place the blocks. Only one block may be moved at a time: it may either " \
            "be placed on the table or placed atop another block. Because of this, " \
            "any blocks that are, at a given time, under another block cannot be moved."

    types = {
        'block': '; A physical object that can be picked up and moved by the robot arm.', 
        'table': '; A flat surface where blocks can be placed.', 
        'robot_arm': '; The mechanical device capable of picking and placing blocks.',
        'carpet': '; a carpet for a room.' # unnecessary type for domain
        }

    feedback_builder = FeedbackBuilder()

    feedback_prompt = PromptBuilder(
        role="You are a PDDL assistant that is providing feedback to :types.",
        format=load_file("templates/feedback_templates/feedback.txt"),
        task="{domain_desc} \n\n##Types\n{types}"
    )

    no_feedback, llm_output = feedback_builder.type_feedback(
        model = gpt_4o_mini,
        domain_desc = domain_desc,
        feedback_template = feedback_prompt.generate_prompt(),
        feedback_type = "llm",
        types=types
    )

    print(no_feedback, llm_output)

Generated feedback: ::

    [NO FEEDBACK]: False 
    
    [LLM OUTPUT]
    ### JUDGMENT
    ```
    The type "carpet" seems unnecessary in the context of the task, as it does not relate to the actions of picking and placing blocks. Consider removing it to maintain focus on relevant types. 
    ```

Below are actual runnable usage examples. This is the general setup to build domain predicates:

.. code-block:: python
    :linenos:

    import os
    from l2p.domain_builder import DomainBuilder
    from l2p.llm.openai import OPENAI
    from l2p.utils import format_expression, load_file

    domain_builder = DomainBuilder()

    api_key = os.environ.get('OPENAI_API_KEY')
    gpt_4o_mini = OPENAI(model="gpt-4o-mini", api_key=api_key)

    # retrieve prompt information
    base_path='tests/usage/prompts/domain/'
    domain_desc = load_file(f'{base_path}blocksworld_domain.txt')
    predicates_prompt = load_file(f'{base_path}formalize_predicates.txt')
    types = load_file(f'{base_path}types.json')
    action = load_file(f'{base_path}action.json')

    # extract predicates via LLM
    predicates, llm_output, validation_info = domain_builder.formalize_predicates(
        model=gpt_4o_mini,
        domain_desc=domain_desc,
        prompt_template=predicates_prompt,
        types=types
        )

    # format key info into PDDL strings
    predicate_str = "\n".join([pred["raw"].replace(":", " ; ") for pred in predicates])

    print(f"PDDL domain predicates:\n{predicate_str}")

The following output is: ::

    PDDL domain predicates:
    - (holding ?a - arm ?b - block) ;  true if the arm ?a is currently holding the block ?b
    - (on_table ?b - block) ;  true if the block ?b is on the table
    - (clear ?b - block) ;  true if the block ?b is clear (no block on top of it)
    - (on_top ?b1 - block ?b2 - block) ;  true if the block ?b1 is on top of the block ?b2

Here is how you would setup a PDDL problem:

.. code-block:: python
    :linenos:

    import os
    from l2p.llm.openai import OPENAI
    from l2p.utils.pddl_types import Predicate
    from l2p.task_builder import TaskBuilder
    from l2p.utils import load_file

    task_builder = TaskBuilder() # initialize task builder class

    api_key = os.environ.get('OPENAI_API_KEY')
    llm = OPENAI(model="gpt-4o-mini", api_key=api_key)

    # load in assumptions
    problem_desc = load_file(r'tests/usage/prompts/problem/blocksworld_problem.txt')
    task_prompt = load_file(r'tests/usage/prompts/problem/formalize_task.txt')
    types = load_file(r'tests/usage/prompts/domain/types.json')
    predicates_json = load_file(r'tests/usage/prompts/domain/predicates.json')
    predicates: list[Predicate] = [Predicate(**item) for item in predicates_json]

    # extract PDDL task specifications via LLM
    objects, init, goal, llm_response, validation_info = task_builder.formalize_task(
        model=llm,
        problem_desc=problem_desc,
        prompt_template=task_prompt,
        types=types,
        predicates=predicates
        )

    # generate task file
    pddl_problem = task_builder.generate_task(
        domain_name="blocksworld",
        problem_name="blocksworld_problem",
        objects=objects,
        initial=init,
        goal=goal)

    print(f"PDDL problem:\n{pddl_problem}")

The following output is: ::

    PDDL problem:
    (define
        (problem blocksworld_problem)
        (:domain blocksworld)

        (:objects 
            blue_block - block
            red_block - block
            yellow_block - block
            green_block - block
            table1 - table
        )

        (:init
            (on_top blue_block red_block)
            (on_top red_block yellow_block)
            (on_table yellow_block)
            (on_table green_block)
            (clear blue_block)
            (clear green_block)
        )

        (:goal
            (and 
                (on_top red_block green_block)
                (clear green_block)
            )
        )
    )

***IMPORTANT***
It is **highly** recommended to use the base template found in :doc:`templates` in your final prompt to properly extract LLM output into the designated Python formats from these methods.
