# l2p : LLM-driven Planning Model library kit

This library is a collection of tools for PDDL model generation extracted from natural language driven by large language models. This library is an expansion from the survey paper [**LLMs as Planning Formalizers: A Survey for Leveraging Large Language Models to Construct Automated Planning Specifications**](https://arxiv.org/abs/2503.18971v1).

L2P is an offline, natural language-to-planning model system that supports domain-agnostic planning. It does this via creating an intermediate [PDDL](https://planning.wiki/guide/whatis/pddl) representation of the domain and task, which can then be solved by a classical planner.

Full library documentation can be found: [**L2P Documention**](https://marcustantakoun.github.io/l2p.github.io/)

## Usage

This is the general setup to build domain predicates:
```python
import os
from l2p.llm.openai import OPENAI
from l2p.utils import load_file
from l2p.domain_builder import DomainBuilder

domain_builder = DomainBuilder()

api_key = os.environ.get('OPENAI_API_KEY')
llm = OPENAI(model="gpt-4o-mini", api_key=api_key)

# retrieve prompt information
base_path='tests/usage/prompts/domain/'
domain_desc = load_file(f'{base_path}blocksworld_domain.txt')
predicates_prompt = load_file(f'{base_path}formalize_predicates.txt')
types = load_file(f'{base_path}types.json')
action = load_file(f'{base_path}action.json')

# extract predicates via LLM
predicates, llm_output, validation_info = domain_builder.formalize_predicates(
    model=llm,
    domain_desc=domain_desc,
    prompt_template=predicates_prompt,
    types=types
    )

# format key info into PDDL strings
predicate_str = "\n".join([pred["raw"].replace(":", " ; ") for pred in predicates])

print(f"PDDL domain predicates:\n{predicate_str}")
```

Here is how you would setup a PDDL problem:
```python
from l2p.utils.pddl_types import Predicate
from l2p.task_builder import TaskBuilder

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

print(f"### LLM OUTPUT:\n {pddl_problem}")
```

Here is how you would setup a Feedback Mechanism:
```python
from l2p.feedback_builder import FeedbackBuilder

feedback_builder = FeedbackBuilder()

api_key = os.environ.get('OPENAI_API_KEY')
llm = OPENAI(model="gpt-4o-mini", api_key=api_key)

problem_desc = load_file(r'tests/usage/prompts/problem/blocksworld_problem.txt')
types = load_file(r'tests/usage/prompts/domain/types.json')
feedback_template = load_file(r'tests/usage/prompts/problem/feedback.txt')
predicates_json = load_file(r'tests/usage/prompts/domain/predicates.json')
predicates: list[Predicate] = [Predicate(**item) for item in predicates_json]
llm_response = load_file(r'tests/usage/prompts/domain/llm_output_task.txt')

fb_pass, feedback_response = feedback_builder.task_feedback(
    model=llm,
    problem_desc=problem_desc,
    llm_output=llm_response,
    feedback_template=feedback_template,
    feedback_type="llm",
    predicates=predicates,
    types=types)

print("[FEEDBACK]\n", feedback_response)
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

L2P requires access to an LLM. L2P provides support for OpenAI's models and other providers compatible with OpenAI SDK. To configure these, provide the necessary API-key in an environment variable.

**OpenAI**
```
export OPENAI_API_KEY='YOUR-KEY' # e.g. OPENAI_API_KEY='sk-123456'
```

Refer to [here](https://platform.openai.com/docs/quickstart) for more information.

**HuggingFace**

Additionally, we have included support for using Huggingface models. One can set up their environment like so:
```
parser = argparse.ArgumentParser(description="Testing HF usage")
parser.add_argument('-test_hf', action='store_true')
parser.add_argument("--model", type=float, required=True, help = "model name")
parser.add_argument("--model_path", type=str, required=True, help = "path to llm")
parser.add_argument("--config_path", type=str, default="l2p/llm/utils/llm.yaml", help = "path to yaml configuration")
parser.add_argument("--provider", type=str, default="huggingface", help = "backend provider")
args = parser.parse_args()

huggingface_model = HUGGING_FACE(model=args.model, model_path=args.model_path, config_path=args.config_path, provider=args.provider)
```

Users can refer to l2p/llm/utils/llm.yaml to better understand (and create their own) model configuration options, including tokenizer settings, generation parameters, and provider-specific settings.

**l2p/llm/base.py** contains an abstract class and method for implementing any model classes in the case of other third-party LLM uses.

## Planner
For ease of use, our library contains submodule [FastDownward](https://github.com/aibasel/downward/tree/308812cf7315fe896dbcd319493277d82aa36bd2). Fast Downward is a domain-independent classical planning system that users can run their PDDL domain and problem files on. The motivation is that the majority of papers involving PDDL-LLM usage uses this library as their planner.

**IMPORTANT** FastDownward is a submodule in L2P. To use the planner, you must clone the GitHub repo of [FastDownward](https://github.com/aibasel/downward/tree/308812cf7315fe896dbcd319493277d82aa36bd2) and run the `planner_path` to that directory.

Here is a quick test set up:
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

To stay up to date with the most current papers, please visit [**here**](https://marcustantakoun.github.io/l2p.github.io/paper_feed.html).

## Contact
Please contact `20mt1@queensu.ca` for questions, comments, or feedback about the L2P library.
