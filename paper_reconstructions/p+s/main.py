"""
Paper: "Structured, flexible, and robust: benchmarking and improving large language models towards more human-like behavior in out-of-distribution reasoning tasks" Collins et al. (2022)
Source code: https://github.com/collinskatie/structured_flexible_and_robust
Run: python3 -m paper_reconstructions.p+s.main

This framework (Part II of paper) utilizes "Language-of-Thought" (LOT) prompting where thinking operates like formal language.
Mental representations are structured symbolically with rules for logical reasoning and problem-solving. L2P follows this paradigm.
In this paper, they are tasking the LLM to translate natural language initial and goal states to PDDL via few shot prompting.
"""

from l2p import *


def run_parse_and_solve(
    model: BaseLLM,
    prompt_initial: str,
    prompt_goal: str,
    problem_path: str,
    problem_name: str,
    objects: dict[str, str],
):
    """
    Main framework of P+S - translate initial and goal states to PDDL from NL

    Args:
        model (BaseLLM): LLM model to run inference
        prompt_initial (str): prompt for translating initial state
        prompt_goal (str): prompt for translating goal state
        problem_path (str): directory of specific problem
        problem_name (str): PDDL problem name
        objects (dict[str,str]): objects of task file
    """

    task_builder = TaskBuilder()

    # extract initial states
    initial_states, _, _ = task_builder.formalize_initial_state(
        model=model, problem_desc="", prompt_template=prompt_initial, objects=""
    )

    # extract goal states
    goal_states, _, _ = task_builder.formalize_goal_state(
        model=model, problem_desc="", prompt_template=prompt_goal, objects=""
    )

    # take components and generate PDDL task format
    pddl_problem = task_builder.generate_task(
        "simple-blocks",
        problem_name,
        objects=objects,
        initial=initial_states,
        goal=goal_states,
    )

    # write the problem file to respective directory
    problem_file = problem_path + f"/{problem_name}.pddl"
    with open(problem_file, "w") as f:
        f.write(pddl_problem)

    return problem_file


def run_problem_01(init_examples, goal_examples):
    # PROBLEM 1
    problem_path = "paper_reconstructions/p+s/results/problem_01"
    objects_01 = {"sketchbook": "", "sweatshirt": "", "keyboard": "", "novel": ""}

    # assemble prompt template for initial and goal state extraction
    prompt_initial_01 = PromptBuilder(
        role=ROLE_INITIAL,
        examples=init_examples,
        task=load_file("paper_reconstructions/p+s/problems/initial/000.txt"),
    )

    prompt_goal_01 = PromptBuilder(
        role=ROLE_GOAL,
        examples=goal_examples,
        task=load_file("paper_reconstructions/p+s/problems/goal/000.txt"),
    )

    # run framework
    problem_file = run_parse_and_solve(
        model=openai_llm,
        prompt_initial=prompt_initial_01.generate_prompt(),
        prompt_goal=prompt_goal_01.generate_prompt(),
        problem_path=problem_path,
        problem_name="problem-01",
        objects=objects_01,
    )

    # run FastDownward planner
    _, result = planner.run_fast_downward(
        domain_file="paper_reconstructions/p+s/results/domain.pddl",
        problem_file=problem_file,
    )

    # write result of plan
    with open(problem_path + "/plan.txt", "w") as f:
        f.write(result)


def run_problem_02(init_examples, goal_examples):
    # PROBLEM 2
    problem_path = "paper_reconstructions/p+s/results/problem_02"
    objects_02 = {"newspaper": "", "accordion": "", "saucepan": "", "peacoat": ""}

    prompt_initial_02 = PromptBuilder(
        role=ROLE_INITIAL,
        examples=init_examples,
        task=load_file("paper_reconstructions/p+s/problems/initial/001.txt"),
    )

    prompt_goal_02 = PromptBuilder(
        role=ROLE_GOAL,
        examples=goal_examples,
        task=load_file("paper_reconstructions/p+s/problems/goal/001.txt"),
    )

    problem_file = run_parse_and_solve(
        model=openai_llm,
        prompt_initial=prompt_initial_02.generate_prompt(),
        prompt_goal=prompt_goal_02.generate_prompt(),
        problem_path=problem_path,
        problem_name="problem-02",
        objects=objects_02,
    )

    _, result = planner.run_fast_downward(
        domain_file="paper_reconstructions/p+s/results/domain.pddl",
        problem_file=problem_file,
    )

    with open(problem_path + "/plan.txt", "w") as f:
        f.write(result)


def run_problem_03(init_examples, goal_examples):
    # PROBLEM 3
    problem_path = "paper_reconstructions/p+s/results/problem_03"
    objects_03 = {"mouse-pad": "", "hacksaw": "", "saucepan": "", "raincoat": ""}

    # assemble prompt template for initial and goal state extraction
    prompt_initial_03 = PromptBuilder(
        role=ROLE_INITIAL,
        examples=init_examples,
        task=load_file("paper_reconstructions/p+s/problems/initial/002.txt"),
    )

    prompt_goal_03 = PromptBuilder(
        role=ROLE_GOAL,
        examples=goal_examples,
        task=load_file("paper_reconstructions/p+s/problems/goal/002.txt"),
    )

    problem_file = run_parse_and_solve(
        model=openai_llm,
        prompt_initial=prompt_initial_03.generate_prompt(),
        prompt_goal=prompt_goal_03.generate_prompt(),
        problem_path=problem_path,
        problem_name="problem-03",
        objects=objects_03,
    )

    _, result = planner.run_fast_downward(
        domain_file="paper_reconstructions/p+s/results/domain.pddl",
        problem_file=problem_file,
    )

    with open(problem_path + "/plan.txt", "w") as f:
        f.write(result)


if __name__ == "__main__":

    # setup L2P requirements
    engine = "gpt-4o-mini"
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_llm = OPENAI(model=engine, api_key=api_key)
    planner = FastDownward(planner_path="downward/fast-downward.py")

    # load in few shot examples
    folder_path = "paper_reconstructions/p+s/prompts/examples/initial"
    init_examples = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            file_content = load_file(file_path)
            init_examples.append(file_content)

    folder_path = "paper_reconstructions/p+s/prompts/examples/goal"
    goal_examples = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            file_content = load_file(file_path)
            goal_examples.append(file_content)

    # load in base templates
    ROLE = "Your task is to convert the natural language states into PDDL initial state predicates.\n\n"
    ROLE_INITIAL = ROLE + load_file("templates/task_templates/formalize_initial.txt")
    ROLE_GOAL = ROLE + load_file("templates/task_templates/formalize_goal.txt")
    DOMAIN_DIR = "paper_reconstructions/p+s/domain.pddl"

    # run problem sets
    run_problem_01(init_examples, goal_examples)
    run_problem_02(init_examples, goal_examples)
    run_problem_03(init_examples, goal_examples)
