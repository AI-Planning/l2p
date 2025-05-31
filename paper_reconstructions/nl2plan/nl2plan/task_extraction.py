"""
Step 5 (Task Extraction) of NL2Plan

This class queries the LLM to construct the PDDL task components (objects, initial, goal states).
Specifically, it takes in the predicates (and types for validation) to construct the
problem file given task description.
"""

from l2p import *


class TaskExtraction:
    def __init__(self):
        self.prompt_template = PromptBuilder()
        self.task_builder = TaskBuilder()
        self.feedback_builder = FeedbackBuilder()
        self.syntax_validator = SyntaxValidator()
        self.syntax_validator.headers = ["OBJECTS", "INITIAL", "GOAL"]
        self.syntax_validator.error_types = [
            "validate_header",
            "validate_duplicate_headers",
            "validate_unsupported_keywords",
            "validate_task_states",
            "validate_task_objects",
            "validate_task_states",
        ]

    def task_extraction(
        self,
        model: BaseLLM,
        problem_desc: str,
        task_extraction_prompt: PromptBuilder,
        types: dict[str, str] | list[dict[str, str]],
        predicates: list[Predicate],
        feedback_prompt: str,
        max_feedback_retries: int = 1,
        max_syntax_retries: int = 3,
    ) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
        """
        Main function to run task extraction. Specifically, extracts objects,
        initial state, and goal state as components to get PDDL problem.

        Args:
            - model (BaseLLM): The LLM language model connection.
            - problem_desc (str): not domain, but just problem description
            - task_extraction_prompt (PromptBuilder): task prompt extraction
            - types (dict[str,str]): domain types
            - predicates (list[Predicates]): predicates to work off of
            - feedback_prompt (str): feedback checklist prompt
            - error_prompt (str): error checklist prompt (via syntax validator)
            - max_attempts (str): attempts to construct proper task extraction

        Returns:
            - objects (dict[str,str]): PDDL object types {'?name':'type'}
            - initial (list[dict[str,str]]): initial state of problem
            - goal (list[dict[str,str]]): desired state of problem
                ex: state_1 = {'name': 'on_top', 'params': ['blue_block', 'red_block'], 'neg': False}
        """

        i = 0
        no_feedback = False
        llm_input_prompt = task_extraction_prompt.generate_prompt()

        # store last valid results
        last_valid_obj = None
        last_valid_init = None
        last_valid_goal = None

        while not no_feedback and i <= max_feedback_retries:
            # inner loop: repeat until syntax validator passes
            max_validation_retries = max_syntax_retries
            valid = False
            while not valid and max_validation_retries > 0:
                objects, initial, goal, llm_output, validation_info = (
                    self.task_builder.formalize_task(
                        model=model,
                        problem_desc=problem_desc,
                        prompt_template=llm_input_prompt,
                        types=types,
                        predicates=predicates,
                        syntax_validator=self.syntax_validator,
                    )
                )

                valid = validation_info[0]
                if valid:
                    # store last valid results
                    last_valid_obj = objects
                    last_valid_init = initial
                    last_valid_goal = goal
                else:
                    llm_input_prompt = self.generate_validation_prompt(
                        problem_desc=problem_desc,
                        types=types,
                        predicates=predicates,
                        original_llm_output=llm_output,
                        validation_info=validation_info,
                    )
                    max_validation_retries -= 1

            if i < max_feedback_retries:
                # feedback mechanism: after valid generation
                no_feedback, fb_msg = self.feedback_builder.task_feedback(
                    model=model,
                    problem_desc=problem_desc,
                    llm_output=llm_output,
                    feedback_template=feedback_prompt,
                    feedback_type="llm",
                    objects=objects,
                    initial=initial,
                    goal=goal,
                    types=types,
                    predicates=predicates,
                )
                if not no_feedback:
                    llm_input_prompt = self.generate_feedback_revision_prompt(
                        fb_msg=fb_msg,
                        problem_desc=problem_desc,
                        types=types,
                        predicates=predicates,
                        objects=objects,
                        initial=initial,
                        goal=goal,
                    )
                    i += 1
            else:
                break

        return last_valid_obj, last_valid_init, last_valid_goal

    def generate_validation_prompt(
        self,
        problem_desc: str,
        types: dict[str, str] | list[dict[str, str]],
        predicates: list[Predicate],
        original_llm_output: str,
        validation_info: tuple[bool, str],
    ) -> str:

        types_str = pretty_print_dict(types) if types else "No types provided."
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )

        prompt = load_file(
            "paper_reconstructions/nl2plan/prompts/task_extraction/error.txt"
        )
        prompt = (
            prompt.replace("{error_msg}", validation_info[1])
            .replace("{llm_response}", original_llm_output)
            .replace("{problem_desc}", problem_desc)
            .replace("{types}", types_str)
            .replace("{predicates}", preds_str)
        )

        return prompt

    def generate_feedback_revision_prompt(
        self,
        fb_msg: str,
        problem_desc: str,
        types: dict[str, str] | list[dict[str, str]],
        predicates: list[Predicate],
        objects: dict[str, str],
        initial: list[dict[str, str]],
        goal: list[dict[str, str]],
    ) -> str:

        types_str = pretty_print_dict(types) if types else "No types provided."
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )

        prompt = load_file(
            "paper_reconstructions/nl2plan/prompts/task_extraction/feedback_revision.txt"
        )
        prompt = (
            prompt.replace("{fb_msg}", fb_msg)
            .replace("{problem_desc}", problem_desc)
            .replace("{types}", types_str)
            .replace("{predicates}", preds_str)
            .replace("{objects}", format_objects(objects))
            .replace("{initial_states}", format_initial(initial))
            .replace("{goal_states}", format_goal(goal))
        )

        return prompt
