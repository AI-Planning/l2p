L2P
================
Below are the in-depth usage of L2P. It is **highly** recommended to use the base template found in :doc:`templates` to properly extract LLM output into the designated Python formats from these methods.

**Jump to:**

→ :ref:`promptbuilder`

→ :ref:`domainbuilder`

→ :ref:`problembuilder`

→ :ref:`feedbackbuilder`

→ :ref:`llm_backends`

→ :ref:`planners`

→ :ref:`validators`

→ :ref:`utils`

→ :doc:`cli`

.. _promptbuilder:

PromptBuilder
-------------

The ``l2p.prompt_builder.PromptBuilder`` class constructs structured prompt templates for LLM-based PDDL generation. Prompts are assembled from five configurable sections - Role, Output Format, Rules, Examples, and Task - ensuring consistent interaction with the underlying model. This standardization is critical for reliable parsing of LLM output back into structured data.

.. tip::
   All default templates in ``l2p/templates/`` follow this structure. You can create custom prompts using ``PromptBuilder`` and load them via ``load_custom_template()`` from ``l2p.utils.pddl_prompt``, or use ``formalize_component()`` with a ``prompt_template`` argument to bypass the builder entirely.

.. autoclass:: l2p.PromptBuilder
   :members:
   :undoc-members:
   :inherited-members:

.. _domainbuilder:

DomainBuilder
-------------

The ``l2p.domain_builder.DomainBuilder`` class is the core entry point for constructing complete PDDL domains via LLM extraction. It supports generating types, constants, predicates, functions, derived predicates, actions (including durative actions), events, processes, and constraints from natural language descriptions. The class also provides ``generate_domain()`` to assemble all components into a valid PDDL domain string.

.. tip::
   The library automatically infers PDDL requirements (``:strips``, ``:typing``, ``:numeric-fluents``, etc.) from generated components via ``DomainBuilder.generate_requirements()``. Requirements are assembled from the structural features present in the model - no manual annotation needed. Use ``set_*`` methods (e.g., ``set_types()``, ``set_predicates()``) to programmatically compose or override individual components.

.. autoclass:: l2p.DomainBuilder
   :members:
   :undoc-members:
   :inherited-members:

.. _problembuilder:

ProblemBuilder
--------------

The ``l2p.problem_builder.ProblemBuilder`` class generates complete PDDL problem instances - objects, initial state, goal state, constraints, and metrics - from natural language descriptions. It mirrors the ``DomainBuilder`` API with ``formalize_component()`` for LLM extraction and ``generate_problem()`` for assembling the final PDDL problem string.

.. tip::
   Problem generation depends on domain context. Always pass the relevant domain's **types** and **predicates** as keyword arguments to ``formalize_component()`` so the LLM has the vocabulary it needs to produce valid objects, initial states, and goals.

.. autoclass:: l2p.ProblemBuilder
   :members:
   :undoc-members:
   :inherited-members:

.. _feedbackbuilder:

FeedbackBuilder
---------------

The ``l2p.feedback_builder.FeedbackBuilder`` class provides an LLM-driven self-improvement loop for refining generated PDDL models. It supports several feedback strategies: diagnosis of syntax or validation errors (``llm_diagnose``), automated revision (``llm_revise``), semantic evaluation against natural language intent (``llm_evaluate``), generalized reflection (``llm_reflect``), and best-candidate selection from multiple generations (``llm_select``).

.. tip::
   Combine ``llm_diagnose`` and ``llm_revise`` to create a complete repair loop: diagnose the root cause of a validation error, then revise the broken component accordingly. For plan-level validation, use ``llm_evaluate_plan`` and ``llm_diagnose_plan`` to catch planner-level issues such as unsolvable goals or semantic loopholes.

.. autoclass:: l2p.FeedbackBuilder
   :members:
   :undoc-members:
   :inherited-members:

.. _llm_backends:

LLM Backends
------------
The ``l2p.llm`` package provides several LLM backends for interacting with different model providers.

BaseLLM
~~~~~~~
.. autoclass:: l2p.BaseLLM
   :members:
   :undoc-members:
   :inherited-members:

UnifiedLLM
~~~~~~~~~~
.. autoclass:: l2p.UnifiedLLM
   :members:
   :undoc-members:
   :inherited-members:

OPENAI
~~~~~~
.. autoclass:: l2p.OPENAI
   :members:
   :undoc-members:
   :inherited-members:

HUGGING_FACE
~~~~~~~~~~~~
.. autoclass:: l2p.HUGGING_FACE
   :members:
   :undoc-members:
   :inherited-members:

.. _planners:

Planners
--------
The ``l2p.planner_builder`` module provides planner integrations for executing
external automated planners on generated PDDL domain and problem instances.

.. automodule:: l2p.planner_builder
   :members:
   :undoc-members:
   :inherited-members:

.. _validators:

Validators
----------
The ``l2p.validators`` package provides PDDL component validation for generated
domain and problem specifications.

.. automodule:: l2p.validators.base
   :members:
   :undoc-members:
   :inherited-members:

.. automodule:: l2p.validators.domain
   :members:
   :undoc-members:
   :inherited-members:

.. automodule:: l2p.validators.problem
   :members:
   :undoc-members:
   :inherited-members:

.. _utils:

Utils
-----
The ``l2p.utils`` package contains several helper modules for working with PDDL and L2P processes.

PDDL Parser
~~~~~~~~~~~
.. automodule:: l2p.utils.pddl_parser
   :members:
   :undoc-members:
   :inherited-members:

PDDL Formatter
~~~~~~~~~~~~~~
.. automodule:: l2p.utils.pddl_format
   :members:
   :undoc-members:
   :inherited-members:

PDDL Types
~~~~~~~~~~
.. automodule:: l2p.utils.pddl_types
   :members:
   :undoc-members:
   :inherited-members:

PDDL Prompts
~~~~~~~~~~~~
.. automodule:: l2p.utils.pddl_prompt
   :members:
   :undoc-members:
   :inherited-members:
