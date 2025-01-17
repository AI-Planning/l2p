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

    def task_extraction(
        self,
        model: LLM,
        problem_desc: str,
        task_extraction_prompt: PromptBuilder,
        types: dict[str, str],
        predicates: list[Predicate],
        feedback_prompt: str,
        error_prompt: str,
        max_attempts: int = 8,
    ) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
        """
        Main function to run task extraction. Specifically, extracts objects,
        initial state, and goal state as components to get PDDL problem.

        Args:
            - model (LLM): The LLM language model connection.
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

        # initial extraction of task
        objects, initial, goal, llm_response = self.task_builder.extract_task(
            model=model,
            problem_desc=problem_desc,
            prompt_template=task_extraction_prompt.generate_prompt(),
            types=format_types(types),
            predicates=predicates,
        )

        # keep iterating until output passes all syntax validation checks
        all_valid = True
        for _ in range(max_attempts):

            feedback_msgs = []
            all_valid = True
            types_list = format_types(types)
            types_list = {k: types_list[k] for i, k in enumerate(types_list) if i != 0}

            # list of validation checks
            validation_checks = [
                self.syntax_validator.validate_task_objects(objects, types_list),
                self.syntax_validator.validate_task_states(
                    initial, objects, predicates, "initial"
                ),
                self.syntax_validator.validate_task_states(
                    goal, objects, predicates, "goal"
                ),
            ]

            # perform each validation check
            for validator, args in validation_checks:
                is_valid, feedback_msg = validator, args
                if not is_valid:
                    all_valid = False
                    feedback_msgs.append(feedback_msg)

            # if any check fails, append feedback messages
            if not all_valid:
                error_prompt = error_prompt.replace("{error_msg}", str(feedback_msgs))
                error_prompt = error_prompt.replace(
                    "{task}", "goal and state extraction"
                )
                objects, initial, goal, _ = self.feedback_builder.task_feedback(
                    model=model,
                    problem_desc=problem_desc,
                    llm_response=llm_response,
                    feedback_template=error_prompt,
                    feedback_type="llm",
                )

            # if valid, break attempt loop
            else:
                break

        if not all_valid:
            raise ValueError(f"Validation failed: {feedback_msgs}")

        # extract feedback
        objects, initial, goal, _ = self.feedback_builder.task_feedback(
            model=model,
            problem_desc=problem_desc,
            llm_response=llm_response,
            feedback_template=feedback_prompt,
            feedback_type="llm",
            predicates=predicates,
            types=types,
            objects=objects,
            initial=initial,
            goal=goal,
        )

        # format the inputs into proper Python structures
        objects = self.task_builder.format_objects(objects)
        initial = self.task_builder.format_initial(initial)
        goal = self.task_builder.format_goal(goal)

        return objects, initial, goal
