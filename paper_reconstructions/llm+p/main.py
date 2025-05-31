"""
Paper: "LLM+P: Empowering Large Language Models with Optimal Planning Proficiency" Liu et al (2023)
Source code: https://github.com/Cranial-XIX/llm-pddl
Run: python3 -m paper_reconstructions.llm+p.main

Assumes the following:
    1. NL task description
    2. Ground-truth PDDL domain

This library only focuses on model generation, so it is not concerned with the other phase of LLM+P: LLM translating PDDL plans to NL.
This module contains all domains from LLM+P code (except Manipulation domain as L2P currently does not support cost actions).
Experimentation recreation was only done on first four problems of each domain. Example results found in `./domains/blocksworld/results`
"""

import os, argparse
from .domain import Domain
from l2p import *

DOMAINS = [
    "barman",
    "blocksworld",
    "floortile",
    "grippers",
    "storage",
    "termes",
    "tyreworld",
    "manipulation",
]


def create_llm_ic_pddl_prompt(task_nl, domain_pddl, context):
    # (LM+P), create the problem PDDL given the context

    context_nl, context_pddl, _ = context

    prompt = (
        f"I want you to solve planning problems. "
        + f"An example planning problem is: \n{context_nl} \n"
        + f"The problem PDDL file to this problem is: \n{context_pddl} \n"
        + f"Now I have a new planning problem and its description is: \n{task_nl} \n"
        + f"Provide me with the problem PDDL file that describes "
        + f"the new planning problem directly without further explanations. Do not return anything else."
    )

    # add in L2P default format prompt
    prompt += "\n\n" + load_file("templates/task_templates/formalize_task.txt")
    return prompt


def llm_ic_pddl_planner(args, problem_name):

    # create necessary classes
    task_builder = TaskBuilder()  # L2P task builder
    domain = Domain(name=args.domain)
    planner = FastDownward(planner_path=args.planner)  # FastDownward planner

    # initialize OpenAI engine
    api_key = os.environ.get("OPENAI_API_KEY")
    model = OPENAI(model=args.model, api_key=api_key)

    # extract assumptions
    context = domain.get_context()
    domain_pddl = domain.get_domain_pddl()
    domain_pddl_file = domain.get_domain_pddl_file()

    # create the tmp / result folders
    result_folder = f"paper_reconstructions/llm+p/domains/{domain.name}/results/pddl"
    plan_results_folder = (
        f"paper_reconstructions/llm+p/domains/{domain.name}/results/plan"
    )

    task = args.task  # extract task arguments

    # A. generate problem pddl file
    task_nl = domain.get_task(task)
    prompt = create_llm_ic_pddl_prompt(task_nl, domain_pddl, context)

    # query LLM using L2P
    objects, initial, goal, _, _ = task_builder.formalize_task(
        model=model,
        problem_desc="",
        prompt_template=prompt,
    )

    # generate proper PDDL structure
    pddl_problem = task_builder.generate_task(
        domain_name=domain.name,
        problem_name=problem_name,
        objects=objects,
        initial=initial,
        goal=goal,
    )

    # write generated pddl into folder
    pddl_file_path = os.path.join(result_folder, domain.get_task_name(task))
    os.makedirs(result_folder, exist_ok=True)
    with open(pddl_file_path, "w") as file:
        file.write(pddl_problem)

    # B. run planner
    plan_name = domain.get_task_name(task).replace("pddl", "txt")
    _, output = planner.run_fast_downward(
        domain_file=domain_pddl_file, problem_file=pddl_file_path
    )

    # write generated plan into folder
    plan_file_path = os.path.join(plan_results_folder, plan_name)
    os.makedirs(plan_results_folder, exist_ok=True)
    with open(plan_file_path, "w") as file:
        file.write(output)


if __name__ == "__main__":

    # load in arguments to run program
    parser = argparse.ArgumentParser(description="LLM+P")
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini"
    )  # experiment originally ran on o3-mini
    parser.add_argument("--domain", type=str, choices=DOMAINS, default="termes")
    parser.add_argument("--task", type=int, default=3)  # task to run
    parser.add_argument("--planner", type=str, default="downward/fast-downward.py")
    args = parser.parse_args()

    # run LLM+P method
    llm_ic_pddl_planner(args=args, problem_name="termes_problem")
