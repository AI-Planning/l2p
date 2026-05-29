# l2p : LLM-driven Planning Model library kit

[![GitHub repo](https://img.shields.io/badge/github-repo-green)](https://github.com/AI-Planning/l2p)
[![PyPI](https://img.shields.io/pypi/v/l2p.svg)](https://pypi.org/project/l2p/)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/AI-Planning/l2p/blob/main/LICENSE)
<!-- [![Tests](https://github.com/simonw/llm/workflows/Test/badge.svg)](httpsß://github.com/simonw/llm/actions?query=workflow%3ATest) -->
<!-- [![Changelog](https://img.shields.io/github/v/release/simonw/llm?include_prereleases&label=changelog)](https://llm.datasette.io/en/stable/changelog.html) -->
<!-- [![Discord](https://img.shields.io/discord/823971286308356157?label=discord)](https://datasette.io/discord-llm)
[![Homebrew](https://img.shields.io/homebrew/installs/dy/llm?color=yellow&label=homebrew&logo=homebrew)](https://formulae.brew.sh/formula/llm) -->

This library is a collection of tools for PDDL model generation extracted from natural language driven by large language models. This library is an expansion from the survey paper [**LLMs as Planning Formalizers: A Survey for Leveraging Large Language Models to Construct Automated Planning Specifications**](https://aclanthology.org/2025.findings-acl.1291.pdf).

L2P is an offline, natural language-to-planning model system that supports domain-agnostic planning. It does this via creating an intermediate [PDDL](https://planning.wiki/guide/whatis/pddl) representation of the domain and task, which can then be solved by a classical planner. To stay up to date with the most current papers, please visit [**here**](https://ai-planning.github.io/l2p/docs/paper_feed.html).

Full library documentation can be found: [**L2P Documention**](https://ai-planning.github.io/l2p/docs/)

<!-- ## Quickstart
```python
l2p init
    |__
l2p config
l2p models
l2p templates
l2p generate
``` -->

## Quickstart

This is the general setup to build domain predicates:
```python
import os
from l2p import UnifiedLLM
from l2p.domain_builder import DomainBuilder
from l2p.utils.pddl_types import Predicate
from l2p.utils.pddl_format import format_predicates

# set up LLM
api_key = os.getenv("OPENAI_API_KEY")
llm = UnifiedLLM(provider="openai", model="gpt-5-nano", api_key=api_key)

db = DomainBuilder() # instantiate DomainBuilder class

# context
types = [PDDLType(name="block", parent="object")]

desc =  "I want you to model predicates from a standard PDDL blocksworld domain."

# generate predicates
parsed_output, raw_output = db.formalize_component(
    model=llm,
    component_class=Predicate, # component to generate
    description=desc,
    types=types                # pass in kwargs context
)

# parse out predicates list from dictionary
predicates = parsed_output.get(Predicate, [])

# format the predicates nicely
predicates_str = format_predicates(predicates)

print(predicates_str)

# OUTPUT:
# (clear ?x - block)
# (handempty )
# (holding ?x - block)
# (on ?x - block ?y - block)
# (on-table ?x - block)
```

Here is how you would setup a PDDL problem:
```python
from l2p.problem_builder import ProblemBuilder
from l2p.utils.pddl_types import ProblemDetails, PDDLType, Predicate

# context
types = [PDDLType(name="block", parent="object")]
predicates = [
    Predicate(name="on", params=[
        {"variable": "?b1", "type": "block"},
        {"variable": "?b2", "type": "block"}
        ]),
    Predicate(name="on-table", params=[{"variable": "?b", "type": "block"}]),
    Predicate(name="holding", params=[{"variable": "?b", "type": "block"}]),
]

pb = ProblemBuilder() # instantiate ProblemBuilder class

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
parsed_output, llm_output = pb.formalize_component(
    model=llm,
    component_class=ProblemDetails, # component to generate
    description=problem_desc,
    types=types,            # pass in kwargs context
    predicates=predicates   # pass in kwargs context
)

# parse out problem from dictionary
problem = parsed_output.get(ProblemDetails, [])

# format problem in PDDL format
problem_str = pb.generate_problem(problem[0])

print(problem_str)

# OUTPUT:
# (define (problem blocks-stacking-same-initial-goal)
#    (:domain blocks-world)
#    (:objects 
#       b1 b2 b3 - block
#    )
#    (:init 
#       (on b2 b3)
#       (on b3 b1)
#       (on-table b1)
#    )
#    (:goal 
#       (and (on b2 b3) (on b3 b1))
#    )
# )
```


## Installation and Setup
Currently, this repo has been tested for Python 3.11.10 but should be fine to install newer versions.

You can set up a Python environment using either [Conda](https://conda.io) or [venv](https://docs.python.org/3/library/venv.html) and install the dependencies via the following steps.

**Conda**
```
conda create -n L2P python=3.11.10
conda activate L2P
pip install -r requirements.txt
```

**venv**
```
python3.11.10 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

These environments can then be exited with `conda deactivate` and `deactivate` respectively. The instructions below assume that a suitable environemnt is active.

**API keys**

L2P requires access to an LLM. L2P provides support for models compatible with OpenAI SDK or [LLM](https://github.com/simonw/llm). To configure these, provide the necessary API-key in an environment variable.

```
export OPENAI_API_KEY='YOUR-KEY' # e.g. OPENAI_API_KEY='sk-123456'
export CLAUDE_API_KEY='...'
export DEEPSEEK_API_KEY='...'
export OLLAMA_API_KEY='...'
```

We can then use the `OPENAI` class for OpenAI-SDK supported models. Refer to [here](https://platform.openai.com/docs/quickstart) for more information:
```python
from l2p.llm.openai import OPENAI

llm = OPENAI(
    provider="openai",
    model="gpt-5-nano",
    config_path="l2p/llm/utils/openaiSDK.yaml"
)

response = llm.query("Hello, world!")
print(response)
```

**Ollama**

Additionally, we have included support for using local [Ollama](https://ollama.com) models. One can set up their environment like so:
```python
from l2p.llm.unified import UnifiedLLM

llm = UnifiedLLM(
    provider="ollama",
    model="llama2:7b",
    config_path="l2p/llm/utils/llm.yaml"
)

response = llm.query("Hello, world!")
print(response)
```

Users can refer to `l2p/llm/utils/llm.yaml` (for `UnifiedLLM`) or `l2p/llm/utils/openaiSDK.yaml` (for `OPENAI`) to better understand (and create their own) model configuration options, including tokenizer settings, generation parameters, and provider-specific settings.

`l2p/llm/base.py` contains an abstract class and method for implementing any model classes in the case of other third-party LLM uses.

## Planner
L2P contains an abstract class `Planner` found in `l2p/planner_builder.py`. Users can use this class to run planners on top to solve for generated domain and problems. 

For ease of use, our library contains submodule [FastDownward](https://github.com/aibasel/downward/tree/308812cf7315fe896dbcd319493277d82aa36bd2). Fast Downward is a domain-independent classical planning system that users can run their PDDL domain and problem files on. The motivation is that the majority of papers involving PDDL-LLM usage uses this library as their planner.

**IMPORTANT** FastDownward is a submodule in L2P. To use the planner, you must clone the GitHub repo of [FastDownward](https://github.com/aibasel/downward/tree/308812cf7315fe896dbcd319493277d82aa36bd2) and run the `planner_path` to that directory.

Here is a quick test set up:
```python
from l2p.planner_builder import FastDownward

domain_file = "tests/pddl/test_domain.pddl"
problem_file = "tests/pddl/test_problem.pddl"

# instantiate FastDownward class
planner = FastDownward(executable_path="<PATH_TO>/downward/fast-downward.py")

# run plan
plan_result = planner.run_planner(
    domain_file=domain_file,
    problem_file=problem_file,
    alias="lama-first"
)

print(plan_result.is_successful)
print(plan_result.plan)
```

Additionally, L2P also supports [**Unified Planning**](https://github.com/aiplan4eu/unified-planning) backend. Users must first download: `pip install unified-planning`. After installing unified-planning library, you must install specific planner: `pip install 'unified-planning[engine]'` to pass into the solver function.
```python
from l2p.planner_builder import UnifiedPlanning

planner = UnifiedPlanning()

plan_result = planner.run_planner(
    domain_path="tests/pddl/test_domain.pddl",
    problem_path="tests/pddl/test_problem.pddl",
    engine="aries" # specific planning backend
)

print(plan_result.is_successful)
print(plan_result.plan)
```

### Agentic CLI (for LLM agents & automation)

The fastest way to build PDDL models is piping structured JSON between non-interactive commands:

```bash
# 1. Configure an LLM provider
l2p init --backend openai --provider openai --model gpt-4o-mini

# 2. Look up the JSON schema an LLM should follow
l2p schema domain --examples

# 3. Set individual components (validate + format in one step)
l2p set types --data '[{"name":"block","parent":"object"}]' --json
l2p set predicates --data '[
  {"name":"clear","params":[{"variable":"?x","type":"block"}]},
  {"name":"on","params":[{"variable":"?x","type":"block"},{"variable":"?y","type":"block"}]}
]' --pddl

# 4. Assemble and render the full PDDL domain
l2p build domain --data '{
  "name":"blocksworld",
  "types":[{"name":"block","parent":"object"}],
  "predicates":[
    {"name":"clear","params":[{"variable":"?x","type":"block"}]},
    {"name":"on","params":[{"variable":"?x","type":"block"},{"variable":"?y","type":"block"}]}
  ],
  "actions":[
    {"name":"stack","params":[{"variable":"?x","type":"block"},{"variable":"?y","type":"block"}],
     "preconditions":{"conditions":["(clear ?y)","(holding ?x)"]},
     "effects":{"add":["(on ?x ?y)"],"delete":["(clear ?y)","(holding ?x)"]}}
  ]
}' -o domain.pddl

# 5. Validate the generated file
l2p validate domain domain.pddl

# 6. Run a planner on it
l2p plan --domain @domain.pddl --problem @problem.pddl --planner fast-downward --json
```

Every command is **stateless** — pass full JSON via `--data` or compose from individual flags. LLM agents can chain them naturally:

```bash
# Pipe validated JSON between commands
l2p set types --data '[...]' --json | l2p set predicates --stdin --json
```

## Contact
Please contact `20mt1@queensu.ca` for questions, comments, or feedback about the L2P library.
