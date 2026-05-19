Welcome to Language-to-Plan (L2P)!
==================================

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Contents:

   getting_started
   l2p
   paper_recreations
   templates
   types
   paper_feed

What is L2P?
------------
This library is a collection of tools for PDDL models extracted from natural language driven by large language models. This library is an expansion from the survey paper `"LLMs as Planning Formalizers: A Survey for Leveraging Large Language Models to Construct Automated Planning Models" <https://arxiv.org/abs/2503.18971v1>`_.
L2P is an offline, NL to PDDL system that supports domain-agnostic planning. It does this via creating an intermediate PDDL representation of the domain and task, which can then be solved by a classical planner.

Our GitHub can be found `here <https://github.com/AI-Planning/l2p>`_. L2P PyPI can be found `here <https://pypi.org/project/l2p/>`_.

Installation
------------

Install ``l2p`` by running::

   pip install l2p

Usage
-----
:doc:`getting_started` is the place to go to hit the ground running on using l2p.

The :doc:`l2p` documentation provides in-depth codeu usage for the library.


Support
-------
If you are having issues, please let us know.
Reach out to us at 20mt1@queensu.ca or by creating a GitHub issue.

License
-------
The project is licensed under the MIT license for the Queen's Mu Lab.

About L2P
---------
With the proliferation of related techniques to convert NL to PDDL, we are seeing an ever-increasing set of related methods. To bring them together under a single computational umbrella, and beyond just relating the work together conceptually as we have done thus far in this survey, we created a unified framework that encompasses the vast majority of existing methods: Language-to-Plan (L2P). This Python library is open source and captures a generalised version of the proposed `"LLM-Modulo" framework <https://arxiv.org/abs/2402.01817>`_ , which emphasizes soundness guarantees through iterative plan refining via external verifiers. While L2P embodies the core principles of the LLM-Modulo framework, which advocates for LLMs autonomously generating plan candidates themselves, it shifts the focus by facilitating the creation of PDDL files using LLMs – aligning with this paper's paradigm. This approach allows for the integration of external verifiers and feedback critics, enabling users to modify and refine the extracted models.
