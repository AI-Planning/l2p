# l2p : LLM-driven Planning Model library kit

This library is a collection of tools for PDDL model generation extracted from natural language driven by large language models. This library is an expansion from the survey paper **Leveraging Large Language Models for Automated Planning and Model Construction: A Survey** (coming soon)

L2P is an offline, natural language -to- planning model system that supports domain-agnostic planning. It does this via creating an intermediate [PDDL](https://planning.wiki/guide/whatis/pddl) representation of the domain and task, which can then be solved by a classical planner.

## Usage

This is the general setup to build domain predicates:
```python
from l2p.llm_builder import OPENAI
from l2p.utils import load_file
from l2p.domain_builder import DomainBuilder

domain_builder = DomainBuilder()

api_key = os.environ.get('OPENAI_API_KEY')
llm = OPENAI(model="gpt-4o-mini", api_key=api_key)

# retrieve prompt information
base_path='tests/usage/prompts/domain/'
domain_desc = load_file(f'{base_path}blocksworld_domain.txt')
extract_predicates_prompt = load_file(f'{base_path}extract_predicates.txt')
types = load_file(f'{base_path}types.json')
action = load_file(f'{base_path}action.json')

# extract predicates via LLM
predicates, llm_output = domain_builder.extract_predicates(
    model=llm,
    domain_desc=domain_desc,
    prompt_template=extract_predicates_prompt,
    types=types,
    nl_actions={action['action_name']: action['action_desc']}
    )

# format key info into PDDL strings
predicate_str = "\n".join([pred["clean"].replace(":", " ; ") for pred in predicates])

print(f"PDDL domain predicates:\n{predicate_str}")
```

Here is how you would setup a PDDL problem:
```python
from l2p.task_builder import TaskBuilder

task_builder = TaskBuilder()

api_key = os.environ.get('OPENAI_API_KEY')
llm = OPENAI(model="gpt-4o-mini", api_key=api_key)

# load in assumptions
problem_desc = load_file(r'tests/usage/prompts/problem/blocksworld_problem.txt')
extract_task_prompt = load_file(r'tests/usage/prompts/problem/extract_task.txt')
types = load_file(r'tests/usage/prompts/domain/types.json')
predicates_json = load_file(r'tests/usage/prompts/domain/predicates.json')
predicates: List[Predicate] = [Predicate(**item) for item in predicates_json]

# extract PDDL task specifications via LLM
objects, initial_states, goal_states, llm_response = task_builder.extract_task(
    model=llm,
    problem_desc=problem_desc,
    prompt_template=extract_task_prompt,
    types=types,
    predicates=predicates
    )

# format key info into PDDL strings
objects_str = task_builder.format_objects(objects)
initial_str = task_builder.format_initial(initial_states)
goal_str = task_builder.format_goal(goal_states)

# generate task file
pddl_problem = task_builder.generate_task(
    domain="blocksworld",
    problem="blocksworld_problem",
    objects=objects_str,
    initial=initial_str,
    goal=goal_str)

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
predicates: List[Predicate] = [Predicate(**item) for item in predicates_json]
llm_response = load_file(r'tests/usage/prompts/domain/llm_output_task.txt')

objects, initial, goal, feedback_response = feedback_builder.task_feedback(
    model=llm,
    problem_desc=problem_desc,
    feedback_template=feedback_template,
    feedback_type="llm",
    predicates=predicates,
    types=types,
    llm_response=llm_response)

print("FEEDBACK:\n", feedback_response)
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
L2P requires access to an LLM. L2P provides support for OpenAI's GPT-series models. To configure these, provide the necessary API-key in an environment variable.

**OpenAI**
```
export OPENAI_API_KEY='YOUR-KEY' # e.g. OPENAI_API_KEY='sk-123456'
```

Refer to [here](https://platform.openai.com/docs/quickstart) for more information.

**HuggingFace**

Additionally, we have included support for using Huggingface models. One can set up their environment like so:
```
parser = argparse.ArgumentParser(description="Define Parameters")
parser.add_argument('-test_dataset', action='store_true')
parser.add_argument("--temp", type=float, default=0.01, help = "temperature for sampling")
parser.add_argument("--max_len", type=int, default=4e3, help = "max number of tokens in answer")
parser.add_argument("--num_sample", type=int, default=1, help = "number of answers to sample")
parser.add_argument("--model_path", type=str, default="/path/to/model", help = "path to llm")
args = parser.parse_args()

huggingface_model = HUGGING_FACE(model_path=args.model_path, max_tokens=args.max_len, temperature=args.temp)
```

**llm_builder.py** contains an abstract class and method for implementing any model classes in the case of other third-party LLM uses.

## Planner
For ease of use, our library contains submodule [FastDownward](https://github.com/aibasel/downward/tree/308812cf7315fe896dbcd319493277d82aa36bd2). Fast Downward is a domain-independent classical planning system that users can run their PDDL domain and problem files on. The motivation is that the majority of papers involving PDDL-LLM usage uses this library as their planner.

**IMPORTANT** FastDownward is a submodule in L2P. To use the planner, you must clone the GitHub repo of [FastDownward](https://github.com/aibasel/downward/tree/308812cf7315fe896dbcd319493277d82aa36bd2) and run the `planner_path` to that directory.

## Current Works Reconstructed Using L2P
The following are papers that have been reconstructed so far. *Checked* boxes are completed, *unchecked* are papers currently in queue to be reconstructed. This list will be updated in the future.

- [x] `P+S` [[paper]](https://arxiv.org/abs/2205.05718)
- [x] `LLM+P` [[paper]](https://arxiv.org/abs/2304.11477)
- [x] `PROC2PDDL` [[paper]](https://arxiv.org/abs/2403.00092)
- [x] `LLM+DM` [[paper]](https://arxiv.org/abs/2305.14909)
- [x] `NL2Plan` [[paper]](https://arxiv.org/abs/2405.04215)
- [ ] `LLM+DP` [[paper]](https://arxiv.org/abs/2308.06391)
- [ ] `LLM+EW` [[paper]](https://arxiv.org/abs/2407.12979)
- [ ] `LLM+Consistency` [[paper]](https://arxiv.org/abs/2404.07751)
- [ ] `LaMMa-P`[[paper]](https://arxiv.org/abs/2409.20560)

## Current Model Construction Works
This section presents a taxonomy of research within Model Construction, organized into three broad categories: *Model Generation*, *Model Editing*, and *Model Benchmarks*. Within each category, the most recent contributions are listed last.

### Model Generation
This category can be further divided into: *Task Modelling*, ; *Domain Modelling*, *Hybrid Modelling*

***Task Translation Frameworks***
- **"Structured, flexible, and robust: benchmarking and improving large language models towards more human-like behaviour in out-of-distribution reasoning tasks"** Collins et al. (2022) [[paper]](https://arxiv.org/abs/2205.05718) [[code]](https://github.com/collinskatie/structured_flexible_and_robust)
- **"Translating natural language to planning goals with large-language models"** Xie et al. (2023) [[paper]](https://arxiv.org/abs/2302.05128) [[code]](https://github.com/clear-nus/gpt-pddl)
- **"Faithful Chain-of-Thought Reasoning"** Lyu et al. (2023) [[paper]](https://arxiv.org/abs/2301.13379) [[code]](https://github.com/veronica320/faithful-cot)
- **"LLM+P: Empowering Large Language Models with Optimal Planning Proficiency"** Liu et al. (2023) [[paper]](https://arxiv.org/abs/2304.11477) [[code]](https://github.com/Cranial-XIX/llm-pddl)
- **"Dynamic Planning with a LLM"** Dagan et al. (2023) [[paper]](https://arxiv.org/abs/2308.06391) [[code]](https://github.com/itl-ed/llm-dp)
- **"AutoGPT+P: Affordance-based Task Planning with Large Language Models"** Birr et al. (2024) [[paper]](https://arxiv.org/abs/2402.10778) [[code]](https://git.h2t.iar.kit.edu/sw/autogpt-p)
- **"TIC: Translate-Infer-Compile for accurate 'text to plan' using LLMs and logical intermediate representations"** Agarwal and Sreepathy (2024) [[paper]](https://arxiv.org/abs/2402.06608) [[code N/A]]()
- **"PDDLEGO: Iterative Planning in Textual Environments"** Zhang et al. (2024) [[paper]](https://arxiv.org/abs/2405.19793) [[code]](https://github.com/zharry29/nl-to-pddl)
- **"TwoStep: Multi-agent Task Planning using Classical Planners and Large Language Models"** Singh et al. (2024) [[paper]](https://arxiv.org/abs/2403.17246) [[code]](https://glamor-usc.github.io/twostep/)
- **"Towards Human Awareness in Robot Task Planning with Large Language Models"** Liu et al. (2024) [[paper]](https://arxiv.org/abs/2404.11267) [[code N/A]]()
- **"Anticipate & Collab: Data-driven Task Anticipation and Knowledge-driven Planning for Human-robot Collaboration"** Singh et al. (2024) [[paper]](https://arxiv.org/abs/2404.03587) [[code]](https://github.com/dataplan-hrc/DaTAPlan)
- **"PlanCollabNL: Leveraging Large Language Models for Adaptive Plan Generation in Human-Robot Collaboration"** Izquierdo-Badiola et al. (2024) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10610055) [[code N/A]]()
- **"TRIP-PAL: Travel Planning with Guarantees by Combining Large Language Models and Automated Planners"** Rosa et al. (2024) [[paper]](https://arxiv.org/abs/2406.10196) [[code N/A]]()
- **"Bootstrapping Object-level Planning with Large Language Models"** Paulius et al. (2024) [[paper]](https://arxiv.org/abs/2409.12262) [[code N/A]]()
- **"LaMMA-P: Generalizable Multi-Agent Long-Horizon Task Allocation and Planning with LM-Driven PDDL Planner"** Zhang et al. (2024) [[paper]](https://arxiv.org/abs/2409.20560) [[code]](https://drive.google.com/drive/folders/1dFfwJve4isw8E92bEQCSxIcQLmax-mqM)

***Domain Translation Frameworks***
- **"Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning"** Guan et al. (2023) [[paper]](https://arxiv.org/abs/2305.14909) [[code]](https://github.com/GuanSuns/LLMs-World-Models-for-Planning)
- **"Learning Compositional Behaviors from Demonstration and Language"** Liu et al. (2024) [[paper]](https://blade-bot.github.io) [[code N/A]]()
- **"Learning adaptive planning representations with natural language guidance"** Wong et al. (2023) [[paper]](https://arxiv.org/abs/2312.08566) [[code N/A]]()
- **"Integrating Action Knowledge and LLMs for Task Planning and Situation Handling in Open Worlds"** Ding et al. (2023) [[paper]](https://arxiv.org/abs/2305.17590) [[code]](https://github.com/yding25/GPT-Planner)
- **"Using Large Language Models to Extract Planning Knowledge from Common Vulnerabilities and Exposures"** Oates et al. (2024) [[paper]](https://icaps24.icaps-conference.org/program/workshops/keps-papers/KEPS-24_paper_12.pdf) [[code]](https://github.com/ronwalf/CLLaMP)
- **"PROC2PDDL: Open-Domain Planning Representations from Texts"** Zhang et al. (2024) [[paper]](https://arxiv.org/abs/2403.00092) [[code]](https://github.com/zharry29/proc2pddl)
- **"Autonomously Learning World-Model Representations For Efficient Robot Planning"** Shah (2024) [[paper]](https://keep.lib.asu.edu/items/193613) [[code N/A]]()
- **"Language-Augmented Symbolic Planner for Open-World Task Planning"** Chen at al. (2024) [[paper]](https://arxiv.org/abs/2407.09792) [[code N/A]]()
- **"Making Large Language Models into World Models with Precondition and Effect Knowledge"** Xie at al. (2024) [[paper]](https://arxiv.org/abs/2409.12278) [[code N/A]]()
- **"Planning in the Dark: LLM-Symbolic Planning Pipeline without Experts"** Huang et al. (2024) [[paper]](https://arxiv.org/abs/2409.15915) [[code]](https://anonymous.4open.science/r/Official-LLM-Symbolic-Planning-without-Experts-7466/README.md)

***Hybrid Translation Frameworks***
- **"There and Back Again: Extracting Formal Domains for Controllable Neurosymbolic Story Authoring"** Kelly et al. (2023) [[paper]](https://ojs.aaai.org/index.php/AIIDE/article/view/27502/27275) [[code]](https://github.com/alex-calderwood/there-and-back)
- **"The Neuro-Symbolic Inverse Planning Engine (NIPE): Modeling Probabilistic Social Inferences from Linguistic Inputs"** Ying et al. (2023) [[paper]](https://arxiv.org/abs/2306.14325) [[code N/A]]()
- **"MORPHeus: a Multimodal One-armed Robot-assisted Peeling System with Human Users In-the-loop"** Ye et al. (2024) [[paper]](https://emprise.cs.cornell.edu/morpheus/) [[code N/A]]()
- **"InterPreT: Interactive Predicate Learning from Language Feedback for Generalizable Task Planning"** Han et al. (2024) [[paper]](https://interpret-robot.github.io) [[code]](https://github.com/hmz-15/interactive-predicate-learning)
- **"Toward a Method to Generate Capability Ontologies from Natural Language Descriptions"** Silva et al. (2024) [[paper]](https://arxiv.org/abs/2406.07962) [[code N/A]]()
- **"DELTA: Decomposed Efficient Long-Term Robot Task Planning using Large Language Models"** Liu et al. (2024) [[paper]](https://arxiv.org/abs/2404.03275) [[code N/A]]()
- **"ISR-LLM: Iterative Self-Refined Large Language Model for Long-Horizon Sequential Task Planning"** Zhou et al. (2023) [[paper]](https://arxiv.org/abs/2308.13724) [[code]](https://github.com/ma-labo/ISR-LLM)
- **"Consolidating Trees of Robotic Plans Generated Using Large Language Models to Improve Reliability"** Sakib and Sun (2024) [[paper]](https://arxiv.org/abs/2401.07868) [[code N/A]]()
- **"NL2Plan: Robust LLM-Driven Planning from Minimal Text Descriptions"** Gestrin et al. (2024) [[paper]](https://arxiv.org/abs/2405.04215) [[code]](https://github.com/mrlab-ai/NL2Plan)
- **"Leveraging Environment Interaction for Automated PDDL Generation and Planning with Large Language Models"** Mahdavi et al. (2024) [[paper]](https://arxiv.org/abs/2407.12979) [[code]]()
- **"Generating consistent PDDL domains with Large Language Models"** Smirnov et al. (2024) [[paper]](https://arxiv.org/abs/2404.07751) [[code N/A]]()


### Model Editing
- **"Exploring the limitations of using large language models to fix planning tasks"** Gragera and Pozanco (2023) [[paper]](https://icaps23.icaps-conference.org/program/workshops/keps/KEPS-23_paper_3645.pdf) [[code N/A]]()
- **"Can LLMs Fix Issues with Reasoning Models? Towards More Likely Models for AI Planning"** Caglar et al. (2024) [[paper]](https://arxiv.org/abs/2311.13720) [[code N/A]]()
- **"Large Language Models as Planning Domain Generators"** Oswald et al. (2024) [[paper]](https://arxiv.org/abs/2405.06650) [[code]](https://github.com/IBM/NL2PDDL)
- **"Traversing the Linguistic Divide: Aligning Semantically Equivalent Fluents Through Model Refinement"** Sikes et al. (2024) [[paper]](https://drive.google.com/file/d/1gd7DOHY6ztiTO1jllDOmP-V8V-q1N0uG/view) [[code N/A]]()


### Model Benchmarks
- **"Planetarium: A Rigorous Benchmark for Translating Text to Structured Planning Languages"** Zuo et al. (2024) [[paper]](https://arxiv.org/abs/2407.03321) [[code]](https://github.com/batsresearch/planetarium)

As highlighted in our survey paper, there is a notable lack of benchmarking focused on LLMs for extracting PDDL specifications. Below, we present benchmarks evaluating the performance of LLMs directly applied to planning tasks, rather than solely modeling planning specifications:

- **"ALFWorld: Aligning Text and Embodied Environments for Interactive Learning"** Shridhar et al. (2021) [[paper]](https://arxiv.org/abs/2402.01817) [[code]](https://github.com/alfworld/alfworld)
- **"PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change"** Valmeekam et al. (2023) [[paper]](https://proceedings.neurips.cc//paper_files/paper/2023/hash/7a92bcdede88c7afd108072faf5485c8-Abstract-Datasets_and_Benchmarks.html) [[code]](https://github.com/karthikv792/LLMs-Planning)
- **"Automating the Generation of Prompts for LLM-based Action Choice in PDDL Planning"** Stein et al. (2023) [[paper]](https://arxiv.org/abs/2311.09830) [[code]](https://github.com/minecraft-saar/autoplanbench)
- **"NATURAL PLAN: Benchmarking LLMs on Natural Language Planning"** Zheng et al. (2024) [[paper]](https://arxiv.org/abs/2406.04520) [[code]](https://github.com/google-deepmind/natural-plan)
- **"LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks"** Kambhampati et al. (2024) [[paper]](https://arxiv.org/abs/2402.01817) [[code N/A]]()


## Contact
Please contact `20mt1@queensu.ca` for questions, comments, or feedback about the L2P library.
