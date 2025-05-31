"""
Step 4 (Action Construction) of NL2Plan

This class queries the LLM to construct the actions (from NL to PDDL) for given domain in
class data type `Action`.

This is the main algorithm (inspired by LLM+DM) that follows an action-by-action generation that
builds a dynamic predicate list. Each action is generated one by one to create the actions and
predicate list, then the process loops again, recreating new actions with the current predicate list.

This consists of two main steps: (i) creating a single action (running through feedback + validation check)
and (ii) looping through each of these actions from the NL actions found in the previous step.
"""

from l2p import *


class ActionConstruction:
    def __init__(self):
        self.prompt_template = PromptBuilder()
        self.domain_builder = DomainBuilder()
        self.feedback_builder = FeedbackBuilder()
        self.syntax_validator = SyntaxValidator()
        self.syntax_validator.unsupported_keywords = []
        self.syntax_validator.error_types = [
            "validate_header",
            "validate_duplicate_headers",
            "validate_unsupported_keywords",
            "validate_params",
            "validate_duplicate_predicates",
            "validate_types_predicates",
            "validate_format_predicates",
            "validate_usage_action",
        ]

    def construct_action(
        self,
        model: BaseLLM,
        domain_desc: str,
        act_constr_prompt: PromptBuilder,
        action_list: list[str],
        action_name: str,
        action_desc: str,
        predicates: list[Predicate],
        type_hierarchy: list[dict[str, str]],
        feedback_prompt: str = None,
        max_feedback_retries: int = 1,
        max_syntax_retries: int = 3,
    ) -> tuple[Action, list[Predicate]]:
        """
        Construct a single action from a given action description using LLM language model.
        NL2Plan defaults to 8 attempts to construct the action; in this case, we did 2 to reduce
        token usage (but could drastically affect results).

        Args:
            model (BaseLLM): The LLM language model connection.
            act_constr_prompt (PromptBuilder): action construction prompt.
            action_name (str): The name of the action.
            action_desc (str): The action description.
            predicates (list[Predicate]): A list of predicates.
            syntax_validator (SyntaxValidator): The PDDL syntax validator.
            feedback_prompt (str): Feedback template.
            max_attempts (int): Maximum number of iterations to construct action. Defaults to 3.
            feedback (bool): Whether to request feedback from LLM. Defaults to True.

        Returns:
            action (Action): The constructed action.
            new_predicates (list[Predicate]): A list of new predicates
        """

        # fill in action construction prompt placeholders
        action_prompt = act_constr_prompt.generate_prompt()
        action_prompt = action_prompt.replace("{action_desc}", action_desc)
        action_prompt = action_prompt.replace("{action_name}", action_name)

        if len(predicates) == 0:
            predicate_str = "No predicate has been defined yet"
        else:
            predicate_str = ""
            predicate_str = "\n".join([f"- {pred['clean']}" for pred in predicates])
        action_prompt = action_prompt.replace("{predicates}", predicate_str)

        # replace specific feedback template
        if feedback_prompt is not None:
            feedback_prompt = feedback_prompt.replace("{action_desc}", action_desc)
            feedback_prompt = feedback_prompt.replace("{action_name}", action_name)
        elif feedback:
            raise ValueError("Feedback template is required when feedback is enabled.")

        i = 0
        no_feedback = False
        llm_input_prompt = action_prompt

        while not no_feedback and i <= max_feedback_retries:
            print(f"Generating PDDL of action: `{action_name}`")

            # inner loop: repeat until syntax validator passes
            max_validation_retries = max_syntax_retries
            valid = False
            while not valid and max_validation_retries > 0:
                action, new_predicates, llm_output, validation_info = (
                    self.domain_builder.formalize_pddl_action(
                        model=model,
                        domain_desc=domain_desc,
                        prompt_template=llm_input_prompt,
                        action_name=action_name,
                        action_desc=action_desc,
                        action_list=action_list,
                        types=type_hierarchy,
                        predicates=predicates,
                        extract_new_preds=True,
                        syntax_validator=self.syntax_validator,
                    )
                )

                valid = validation_info[0]
                if not valid:
                    llm_input_prompt = self.generate_validation_prompt(
                        action_name=action_name,
                        action_desc=action_desc,
                        types=type_hierarchy,
                        predicates=predicates,
                        original_llm_output=llm_output,
                        validation_info=validation_info,
                    )
                    max_validation_retries -= 1

            if i < max_feedback_retries:
                # feedback mechanism: after valid generation
                no_feedback, fb_msg = self.feedback_builder.pddl_action_feedback(
                    model=model,
                    domain_desc=domain_desc,
                    llm_output=llm_output,
                    feedback_template=feedback_prompt,
                    feedback_type="llm",
                    action=action,
                    types=type_hierarchy,
                    predicates=predicates,
                )
                if not no_feedback:
                    llm_input_prompt = self.generate_feedback_revision_prompt(
                        fb_msg=fb_msg,
                        domain_desc=domain_desc,
                        action=action,
                        action_desc=action_desc,
                        predicates=predicates,
                        types=type_hierarchy,
                    )
                    i += 1
            else:
                break

        return action, new_predicates

    def action_construction(
        self,
        model: BaseLLM,
        domain_desc: str,
        act_constr_prompt: PromptBuilder,
        nl_actions: dict[str, str],
        type_hierarchy: list[dict[str, str]],
        feedback_prompt: str | None = None,
        max_syntax_retries: int = 3,
        max_feedback_retries: int = 1,
        max_iters: int = 1,
    ) -> tuple[list[Action], list[Predicate]]:
        """
        This function loops through all the natural language actions found in the
        previous step. The number of loops determines how many overall cycles that all actions
        will be generated on the current predicate list.

        Args:
            - model (BaseLLM): The LLM language model connection.
            - domain_desc (str): description of domain
            - act_constr_prompt (PromptBuilder): action prompt extraction
            - nl_actions (dict[str,str]): list of actions in natural language
            - type_hierarchy (dict[str,str]): dictionary of organized types
            - feedback_prompt (str): feedback prompt to check LLM output
            - max_attempts (int): attempts of creating a single action
            - max_iters (int): # of overall loops to construct full list of actions

        Returns:
            - actions (list[Action]): list of actions
            - predicates (list[Predicates]): list of predicates
        """

        action_list = [f"{name}: {desc}" for name, desc in nl_actions.items()]

        # run through main loop over each action
        predicates = []
        for iter in range(max_iters):
            actions = []
            print(f"Starting iteration {iter + 1} of action construction")
            curr_preds = len(predicates)

            for action_name, action_desc in nl_actions.items():
                action, new_predicates = self.construct_action(
                    model=model,
                    domain_desc=domain_desc,
                    act_constr_prompt=act_constr_prompt,
                    action_list=action_list,
                    action_name=action_name,
                    action_desc=action_desc,
                    predicates=predicates,
                    type_hierarchy=type_hierarchy,
                    feedback_prompt=feedback_prompt,
                    max_feedback_retries=max_feedback_retries,
                    max_syntax_retries=max_syntax_retries,
                )
                actions.append(action)
                predicates.extend(new_predicates)
                predicates = prune_predicates(predicates, actions)

            if len(predicates) == curr_preds:
                print("No new predicates created. Stopping action construction.")
                break
        else:
            print("Reached maximum iterations. Stopping action construction.")

        return actions, predicates

    def generate_validation_prompt(
        self,
        action_name: str,
        action_desc: str,
        types: list[dict[str, str]],
        predicates: list[Predicate],
        original_llm_output: str,
        validation_info: tuple[bool, str],
    ) -> str:
        prompt = load_file(
            "paper_reconstructions/nl2plan/prompts/action_construction/error.txt"
        )

        types_str = pretty_print_dict(types) if types else "No types provided."
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )

        prompt = (
            prompt.replace("{error_msg}", validation_info[1])
            .replace("{llm_response}", original_llm_output)
            .replace("{action_name}", action_name)
            .replace("{action_desc}", action_desc)
            .replace("{types}", types_str)
            .replace("{predicates}", preds_str)
        )

        return prompt

    def generate_feedback_revision_prompt(
        self,
        fb_msg: str,
        domain_desc: str,
        action: Action,
        action_desc: str,
        predicates: list[Predicate],
        types: list[dict[str, str]],
    ) -> str:
        prompt = load_file(
            "paper_reconstructions/nl2plan/prompts/action_construction/feedback_revision.txt"
        )

        act_name_str = action["name"] if action else "No action name provided."
        params_str = (
            "\n".join([f"{name} - {type}" for name, type in action["params"].items()])
            if action
            else "No parameters provided"
        )
        prec_str = action["preconditions"] if action else "No preconditions provided."
        eff_str = action["effects"] if action else "No effects provided."

        types_str = pretty_print_dict(types) if types else "No types provided."
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )

        prompt = (
            prompt.replace("{fb_msg}", fb_msg)
            .replace("{domain_desc}", domain_desc)
            .replace("{action_name}", act_name_str)
            .replace("{action_desc}", action_desc)
            .replace("{action_params}", params_str)
            .replace("{action_preconditions}", prec_str)
            .replace("{action_effects}", eff_str)
            .replace("{predicates}", preds_str)
            .replace("{types}", types_str)
        )

        return prompt
