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

    def construct_action(
        self,
        model: LLM,
        domain_desc: str,
        act_constr_prompt: PromptBuilder,
        action_list: str,
        action_name: str,
        action_desc: str,
        predicates: list[Predicate],
        type_hierarchy: dict[str, str],
        syntax_validator: SyntaxValidator,
        feedback_prompt: str = None,
        max_attempts=2,
        feedback=True,
    ) -> tuple[Action, list[Predicate]]:
        """
        Construct a single action from a given action description using LLM language model.
        NL2Plan defaults to 8 attempts to construct the action; in this case, we did 2 to reduce
        token usage (but could drastically affect results).

        Args:
            model (LLM): The LLM language model connection.
            act_constr_prompt (PromptBuilder): action construction prompt.
            action_name (str): The name of the action.
            action_desc (str): The action description.
            predicates (list[Predicate]): A list of predicates.
            syntax_validator (SyntaxValidator): The PDDL syntax validator.
            feedback_prompt (str): Feedback template.
            max_iters (int): Maximum number of iterations to construct action. Defaults to 8.
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

        # run through single action creation + feedback and validation check
        for iter in range(1, max_attempts + 1 + (feedback is not None)):
            print(f"Generating PDDL of action: `{action_name}`")

            # syntax check runs directly inside L2P extraction function
            try:
                action, new_predicates, llm_response, validation_info = (
                    self.domain_builder.extract_pddl_action(
                        model=model,
                        domain_desc=domain_desc,
                        prompt_template=action_prompt,
                        action_name=action_name,
                        action_desc=action_desc,
                        action_list=action_list,
                        predicates=predicates,
                        types=format_types(type_hierarchy),
                        syntax_validator=syntax_validator,
                    )
                )

                no_error, error_msg = validation_info

            except Exception as e:
                no_error = False
                error_msg = str(e)

            # if validation passes, go through feedback checklist
            if no_error:
                if feedback is not None:

                    action, new_predicates, response_fb, _, no_fb = (
                        self.feedback_builder.pddl_action_feedback(
                            model=model,
                            domain_desc=domain_desc,
                            llm_response=llm_response,
                            feedback_template=feedback_prompt,
                            feedback_type="llm",
                            action=action,
                            predicates=predicates,
                            types=type_hierarchy,
                        )
                    )

                    # if no feedback, then action is complete
                    if no_fb == True:
                        break

            # if validation does not pass, then generate error messages to run through LLM
            else:
                with open(
                    "paper_reconstructions/nl2plan/prompts/action_construction/error.txt"
                ) as f:
                    error_template = f.read().strip()
                error_prompt = error_template.replace("{action_name}", action_name)
                error_prompt = error_prompt.replace("{action_desc}", action_desc)
                error_prompt = error_prompt.replace("{predicates}", predicate_str)
                error_prompt = error_prompt.replace("{error_msg}", error_msg)
                error_prompt = error_prompt.replace("{llm_response}", llm_response)

                action, new_predicates, llm_response, validation_info = (
                    self.domain_builder.extract_pddl_action(
                        model=model,
                        domain_desc=domain_desc,
                        prompt_template=error_prompt,
                        action_name=action_name,
                        action_desc=action_desc,
                        action_list=action_list,
                        predicates=predicates,
                        types=type_hierarchy,
                        syntax_validator=syntax_validator,
                    )
                )

        return action, new_predicates

    def action_construction(
        self,
        model: LLM,
        domain_desc: str,
        act_constr_prompt: PromptBuilder,
        nl_actions: dict[str, str],
        type_hierarchy: dict[str, str],
        feedback_prompt: str | None = None,
        max_attempts: int = 2,
        max_iters: int = 2,
    ) -> tuple[list[Action], list[Predicate]]:
        """
        This function loops through all the natural language actions found in the
        previous step. The number of loops determines how many overall cycles that all actions
        will be generated on the current predicate list.

        Args:
            - model (LLM): The LLM language model connection.
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

        action_list = "\n".join(
            [f"- {name}: {desc}" for name, desc in nl_actions.items()]
        )

        syntax_validator = SyntaxValidator()  # create syntax validator to run checks on

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
                    syntax_validator=syntax_validator,
                    feedback_prompt=feedback_prompt,
                    max_attempts=max_attempts,
                    feedback=True,
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
