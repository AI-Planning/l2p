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
    from l2p import DomainBuilder, ProblemBuilder, PromptBuilder, FeedbackBuilder
    from l2p.validators.domain import DomainValidator
    from l2p.validators.problem import ProblemValidator
    
    # util functions
    from l2p.utils.pddl_types import *
    from l2p.utils.pddl_format import *
    from l2p.utils.pddl_parser import *
    from l2p.utils.pddl_prompt import *


L2P requires access to an LLM. Set up your LLM class and methods using the abstract ``BaseLLM(ABC)`` class. In this case, we have set up OpenAI's models to our library for quickstart. ::

    export OPENAI_API_KEY='YOUR-KEY' # e.g. OPENAI_API_KEY='sk-123456'
    engine = "gpt-4o-mini"
    api_key = os.environ.get('OPENAI_API_KEY')
    openai_llm = OPENAI(model=engine, api_key=api_key)

Build PDDL domain components using the ``DomainBuilder`` class. This is an example of extracting PDDL types:

.. code-block:: python
    :linenos:

    import os
    from l2p import UnifiedLLM
    from l2p import DomainBuilder
    from l2p import PDDLType

    llm = UnifiedLLM(
        provider="openai", 
        model="gpt-5-nano", 
        api_key=os.environ.get('OPENAI_API_KEY'))

    domain_builder = DomainBuilder() # instantiate DomainBuilder

    # context description
    domain_desc = "The AI agent here is a mechanical robot arm that can pick and " \
    "place the blocks. Only one block may be moved at a time: it may either " \
    "be placed on the table or placed atop another block. Because of this, " \
    "any blocks that are, at a given time, under another block cannot be moved."

    # extract types via LLM
    results, llm_output = domain_builder.formalize_component(
        model=llm,
        component_class=PDDLType,
        description=domain_desc
        # prompt_template=[...] by default loads pre-defined prompt template
    )

    print(results[PDDLType])

Generated types output: ::

    [
        PDDLType(name='block', parent='object', desc='A block to pickup'), 
        PDDLType(name='table', parent='object', desc='Table where blocks are set down'), 
        PDDLType(name='robot_arm', parent='object', desc='Robot arm that manipulates blocks')
    ]

Build PDDL problem components using the ``ProblemBuilder`` class. This is an example of extracting PDDL initial states:

.. code-block:: python
    :linenos:

    from l2p import ProblemBuilder
    from l2p import InitialState, Predicate, PDDLType
    from l2p import format_initial_state

    problem_builder = ProblemBuilder() # instantiate ProblemBuilder

    # context
    problem_desc = "There are four blocks currently. The blue block is on the red " \
    "which is on the yellow. The yellow and the green are on the table. I want " \
    "the red on top of the green."
    types = [PDDLType(name='block', parent='object', desc='A block to pickup')]
    predicates = [
        Predicate(name="on", params=[{"variable": "?b1", "type": "block"}, {"variable": "?b2", "type": "block"}], desc="Block ?b1 is on block ?b2."),
        Predicate(name="on-table", params=[{"variable": "?b", "type": "block"}], desc="Block ?b is on table."),
        Predicate(name="holding", params=[{"variable": "?b", "type": "block"}], desc="Agent is holding block ?b."),
        Predicate(name="clear", params=[{"variable": "?b", "type": "block"}], desc="Block ?b is clear."),
        Predicate(name="arm-empty", params=[], desc="Arm is empty.")
    ]

    # extract initial states via LLM
    results, llm_output = problem_builder.formalize_component(
        model=llm,
        component_class=InitialState,
        description=problem_desc,
        # prompt_template=[...] by default loads pre-defined prompt template
        types=types,            # context kwargs
        predicates=predicates   # context kwargs
    )

    initial_state = results[InitialState][0]

    print(format_initial_state(init=initial_state))

Generated initial states: ::

    (on blue red)
    (on red yellow)
    (on-table yellow)
    (on-table green)
    (clear blue)
    (clear green)

Build LLM feedback components using the ``FeedbackBuilder`` class. This is an example of
using LLM-driven diagnosis to validate generated types:

.. code-block:: python
    :linenos:

    from l2p import FeedbackBuilder

    feedback_builder = FeedbackBuilder() # instantiate FeedbackBuilder

    # context description
    domain_desc = "The AI agent here is a mechanical robot arm that can pick and " \
    "place the blocks. Only one block may be moved at a time: it may either " \
    "be placed on the table or placed atop another block. Because of this, " \
    "any blocks that are, at a given time, under another block cannot be moved."

    # failure artifact (simulated generated LLM output)
    types = [
        PDDLType(name='block', parent='object', desc='A block to pickup.'),
        PDDLType(name='carpet', parent='object', desc='Just some carpet.')
    ]

    # simulate a validation failure for diagnosis
    errors=["Unnecessary type 'carpet' does not relate to any action."]

    # extract diagnosis via LLM
    diagnosis, llm_output = feedback_builder.llm_diagnose(
        model=llm,
        errors=errors,
        artifact=types,
        description=domain_desc,
    )

    print(diagnosis)

Generated diagnosis: ::

    {
        "summary": "The validator failed because the domain includes an unused type 'carpet' that does not relate to any action, causing the generation to be invalid.",
        "identified_errors": [
            {
            "error_type": "UnnecessaryType",
            "location_in_json": "types[1].name",
            "validator_message": "Unnecessary type 'carpet' does not relate to any action.",
            "root_cause_analysis": "The types section defines 'carpet' but no action, predicate, or goal references it; it was likely included by mistake or left over from a template."
            }
        ],
        "repair_plan": [
            "Step 1: Remove the entire type entry for 'carpet' from the types array (types[1]).",
            "Step 2: If 'carpet' is intentionally part of the domain, create or reference at least one action or predicate that uses the carpet type so that it is related to an action.",
            "Step 3: Re-run the validator to ensure the error is resolved and that all remaining types are used by actions or constraints.",
            "Step 4: Ensure the JSON structure remains valid (proper commas, brackets) after removal or modification."
        ]
    }

Below are other runnable usage examples. This is the general setup to build domain predicates:

.. code-block:: python
    :linenos:

    from l2p.domain_builder import DomainBuilder
    from l2p.utils.pddl_types import Predicate

    db = DomainBuilder()
    results, raw = db.formalize_component(
        model=llm,
        component_class=Predicate,
        description="Model predicates for blocksworld.",
        types=[PDDLType(name="block", parent="object")]
    )
    predicates = results[Predicate]

The following output is: ::

    (clear ?x - block)
    (arm-empty )
    (holding ?x - block)
    (on ?x - block ?y - block)
    (on-table ?x - block)

Here is how you would setup a PDDL problem:

.. code-block:: python
    :linenos:

    from l2p.problem_builder import ProblemBuilder
    from l2p.utils.pddl_types import ProblemDetails, PDDLType, Predicate

    pb = ProblemBuilder() # instantiate ProblemBuilder class

    # context
    types = [PDDLType(name="block", parent="object")]
    predicates = [
        Predicate(name="on", params=[
            {"variable": "?x", "type": "block"},
            {"variable": "?y", "type": "block"}
            ]),
        Predicate(name="on-table", params=[{"variable": "?x", "type": "block"}]),
        Predicate(name="holding", params=[{"variable": "?x", "type": "block"}]),
        Predicate(name="clear", params=[{"variable": "?x", "type": "block"}]),
        Predicate(name="arm-empty", params=[])
    ]

    problem_desc = """
    You have 3 blocks. 
    b2 is on top of b3. 
    b3 is on top of b1. 
    b1 is on the table. 
    b2 is clear. 
    Your arm is empty. 
    Your goal is to move the blocks. 
    b2 should be on top of b3. 
    b3 should be on top of b1. 
    """

    # generate problem
    results, llm_output = pb.formalize_component(
        model=llm,
        component_class=ProblemDetails, # component to generate
        description=problem_desc,
        types=types,            # pass in kwargs context
        predicates=predicates   # pass in kwargs context
    )

    # parse out problem from dictionary
    problem = results[ProblemDetails]

    # format problem in PDDL format
    problem_str = pb.generate_problem(problem[0])

    print(problem_str)

The following output is: ::

    (define (problem blocks-problem)
        (:domain blocks-world)
        (:objects b1 b2 b3 - block)
        (:init 
            (on b2 b3)
            (on b3 b1)
            (on-table b1)
            (clear b2)
            (arm-empty)
        )
        (:goal 
            (and (on b2 b3) (on b3 b1))
        )
    )

***IMPORTANT***
It is **highly** recommended to use the base template found in :doc:`templates` in your final prompt to properly extract LLM output into the designated Python formats from these methods.

For terminal-based workflows, including interactive generation, validation, and
planning, see the :doc:`cli` documentation.
