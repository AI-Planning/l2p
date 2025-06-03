"""
PDDL Feedback Generation Utilities

This module provides tools for generating feedback on LLM-generated PDDL domains and tasks.
It supports multiple feedback strategies including human-written or LLM-generated feedback.

NOTE: is worth noting that the usefulness of LLM feedback is experimental. It is inspired by the NL2PLAN framework
    (Gestrin et al., 2024) and is designed to provide feedback to LLM output w/o human intervention.
"""

from collections import OrderedDict
from .utils import *
from .llm import BaseLLM, require_llm


class FeedbackBuilder:
    """
    NOTE: this class only returns feedback flag/messages from original LLM output. Users
        must provide their own implementation of using this feedback to revise outputs.
    """

    @require_llm
    def get_feedback(
        self,
        model: BaseLLM,
        feedback_template: str,
        feedback_type: str,
        llm_output: str,
    ) -> tuple[bool, str]:
        """
        This retrieves the type of feedback user requests and returns feedack message.
        feedback_type takes in either "human" or "llm"
        """

        if feedback_type.lower() == "human":
            feedback_msg = self.human_feedback(llm_output)
        elif feedback_type.lower() == "llm":
            model.reset_tokens()
            feedback_msg = model.query(prompt=feedback_template)
        else:
            raise ValueError("Invalid feedback_type. Expected 'human' or 'llm'")

        # retrieve feedback message
        no_feedback = self.feedback_state(info=feedback_msg)

        if no_feedback:
            return True, feedback_msg

        return False, feedback_msg

    def feedback_state(self, info: str):
        """
        Confirms if feedback is needed from output.

        It expects a `### JUDGMENT` header with enclosed (```) boxes with an answer
        of either 'no feedback' or content of suggestion.

        Refer to l2p/templates/feedback_templates for proper LLM output structuring.
        """
        try:
            judgement_head = parse_heading(info, "JUDGMENT")
            judgement_raw = combine_blocks(judgement_head)
            if "no feedback" in judgement_raw.lower():
                return True
            else:
                return False
        except ValueError as e:
            # if heading is not found no feedback is needed
            return True

    def human_feedback(self, info: str):
        """This enables human-in-the-loop feedback mechanism."""

        print("START OF INFO\n", info)
        print("\nEND OF INFO\n\n")
        contents = []
        print("Provide feedback (or type 'done' to finish):\n")
        while True:
            line = input()
            if line.strip().lower() == "done":
                break
            contents.append(line)
        response = "\n".join(contents)

        if response.lower() == "no feedback" or response == "":
            feedback = "### JUDGMENT\n```\nno feedback\n```"
        else:
            feedback = f"### JUDGMENT\n```\n{response}\n```"

        return feedback

    @require_llm
    def type_feedback(
        self,
        model: BaseLLM,
        domain_desc: str,
        feedback_template: str,
        feedback_type: str = "llm",
        llm_output: str = "",
        types: dict[str, str] | list[dict[str, str]] = None,
    ) -> tuple[bool, str]:
        """
        Provides feedback to initial LLM output for :types.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain information
            feedback_template (str): prompt template to guide LLM to provide feedback to initial output
            feedback_type (str): type of feedback assistant - 'llm', 'human'
            llm_output (str): original LLM output
            types (dict[str,str] | list[dict[str,str]]): PDDL types of current specification

        Returns:
            no_fb (bool): flag that deems if feedback is not needed
            fb_msg (str): feedback message from assistant
        """

        types_str = pretty_print_dict(types) if types else "No types provided."

        prompt = (
            feedback_template.replace("{types}", types_str)
            .replace("{domain_desc}", domain_desc)
            .replace("{llm_output}", llm_output)
        )

        # retrieve feedback for types
        no_fb, fb_msg = self.get_feedback(model, prompt, feedback_type, llm_output)

        return no_fb, fb_msg

    @require_llm
    def nl_action_feedback(
        self,
        model: BaseLLM,
        domain_desc: str,
        feedback_template: str,
        feedback_type: str = "llm",
        llm_output: str = "",
        types: dict[str, str] | list[dict[str, str]] = None,
        nl_actions: dict[str, str] = None,
    ) -> tuple[bool, str]:
        """
        Provides feedback to initial LLM output for list of natural language actions for the domain

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain information
            feedback_template (str): prompt template to guide LLM to provide feedback to initial output
            feedback_type (str): type of feedback assistant - 'llm', 'human'
            llm_output (str): original LLM output
            types (dict[str,str] | list[dict[str,str]]): PDDL types of current specification
            nl_actions (dict[str,str]): optional to supplement feedback prompt

        Returns:
            no_fb (bool): flag that deems if feedback is not needed
            fb_msg (str): feedback message from assistant
        """

        # format string info replacements
        types_str = pretty_print_dict(types) if types else "No types provided."
        nl_act_str = (
            pretty_print_dict(nl_actions) if nl_actions else "No actions provided."
        )

        prompt = (
            feedback_template.replace("{domain_desc}", domain_desc)
            .replace("{types}", types_str)
            .replace("{nl_actions}", nl_act_str)
            .replace("{llm_output}", llm_output)
        )

        no_fb, fb_msg = self.get_feedback(model, prompt, feedback_type, llm_output)

        return no_fb, fb_msg

    @require_llm
    def pddl_action_feedback(
        self,
        model: BaseLLM,
        domain_desc: str,
        feedback_template: str,
        feedback_type: str = "llm",
        llm_output: str = "",
        action: Action = None,
        types: dict[str, str] | list[dict[str, str]] = None,
        constants: dict[str, str] = None,
        predicates: list[Predicate] = None,
        functions: list[Function] = None,
    ) -> tuple[bool, str]:
        """
        Provides feedback to initial LLM output of a PDDL action.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain information
            feedback_template (str): prompt template to guide LLM to provide feedback to initial output
            feedback_type (str): type of feedback assistant - 'llm', 'human'
            llm_output (str): original LLM output
            action (Action): current action specifications
            types (dict[str,str] | list[dict[str,str]]): PDDL types of current specification
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): PDDL predicates of current specification
            functions (list[Function]): PDDL functions of current specification

        Returns:
            no_fb (bool): flag that deems if feedback is not needed
            fb_msg (str): feedback message from assistant
        """

        # format string info replacements
        act_name_str = action["name"] if action else "No action name provided."
        params_str = (
            "\n".join([f"{name} - {type}" for name, type in action["params"].items()])
            if action
            else "No parameters provided"
        )
        prec_str = action["preconditions"] if action else "No preconditions provided."
        eff_str = action["effects"] if action else "No effects provided."

        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )
        funcs_str = (
            "\n".join([f"{func['raw']}" for func in functions])
            if functions
            else "No functions provided."
        )

        prompt = (
            feedback_template.replace("{domain_desc}", domain_desc)
            .replace("{action_name}", act_name_str)
            .replace("{action_params}", params_str)
            .replace("{action_preconditions}", prec_str)
            .replace("{action_effects}", eff_str)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{functions}", funcs_str)
            .replace("{llm_output}", llm_output)
        )

        no_fb, fb_msg = self.get_feedback(model, prompt, feedback_type, llm_output)

        return no_fb, fb_msg

    @require_llm
    def parameter_feedback(
        self,
        model: BaseLLM,
        domain_desc: str,
        feedback_template: str,
        feedback_type: str = "llm",
        llm_output: str = "",
        parameter: OrderedDict = None,
        action_name: str = None,
        action_desc: str = None,
        types: dict[str, str] | list[dict[str, str]] = None,
        constants: dict[str, str] = None,
    ) -> tuple[bool, str]:
        """
        Provides feedback to initial LLM output of a PDDL action parameter.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain information
            feedback_template (str): prompt template to guide LLM to provide feedback to initial output
            feedback_type (str): type of feedback assistant - 'llm', 'human'
            llm_output (str): original LLM output
            parameter (OrderedDict): PDDL params of current action
            action_name (str): name of action
            action_desc (str): description of action
            types (dict[str,str] | list[dict[str,str]]): PDDL types of current specification
            constants (dict[str,str]): current constants in specification, defaults to None

        Returns:
            no_fb (bool): flag that deems if feedback is not needed
            fb_msg (str): feedback message from assistant
        """

        # format string info replacements
        act_name_str = action_name if action_name else "No action name provided"
        act_desc_str = action_desc if action_desc else "No action description provided"
        params_str = (
            "\n".join([f"{name} - {type}" for name, type in parameter.items()])
            if parameter
            else "No parameters provided"
        )

        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )

        prompt = (
            feedback_template.replace("{domain_desc}", domain_desc)
            .replace("{action_name}", act_name_str)
            .replace("{action_desc}", act_desc_str)
            .replace("{action_params}", params_str)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{llm_output}", llm_output)
        )

        no_fb, fb_msg = self.get_feedback(model, prompt, feedback_type, llm_output)

        return no_fb, fb_msg

    @require_llm
    def precondition_feedback(
        self,
        model: BaseLLM,
        domain_desc: str,
        feedback_template: str,
        feedback_type: str = "llm",
        llm_output: str = "",
        parameter: OrderedDict = None,
        preconditions: str = None,
        action_name: str = None,
        action_desc: str = None,
        types: dict[str, str] | list[dict[str, str]] = None,
        constants: dict[str, str] = None,
        predicates: list[Predicate] = None,
        functions: list[Function] = None,
    ) -> tuple[bool, str]:
        """
        Provides feedback to initial LLM output of a PDDL action precondition.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain information
            feedback_template (str): prompt template to guide LLM to provide feedback to initial output
            feedback_type (str): type of feedback assistant - 'llm', 'human'
            llm_output (str): original LLM output
            parameter (OrderedDict): PDDL params of current action
            preconditions (str): PDDL precondition of current action
            action_name (str): name of action
            action_desc (str): description of action
            types (dict[str,str] | list[dict[str,str]]): dictionary of types currently in specification
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of predicates currently in specification
            functions (list[Function]): list of functions currently in specification

        Returns:
            no_fb (bool): flag that deems if feedback is not needed
            fb_msg (str): feedback message from assistant
        """

        # format string info replacements
        act_name_str = action_name if action_name else "No action name provided"
        act_desc_str = action_desc if action_desc else "No action description provided"
        params_str = (
            "\n".join([f"{name} - {type}" for name, type in parameter.items()])
            if parameter
            else "No parameters provided"
        )
        prec_str = preconditions if preconditions else "No preconditions provided."

        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )
        funcs_str = (
            "\n".join([f"{func['raw']}" for func in functions])
            if functions
            else "No functions provided."
        )

        prompt = (
            feedback_template.replace("{domain_desc}", domain_desc)
            .replace("{action_name}", act_name_str)
            .replace("{action_desc}", act_desc_str)
            .replace("{action_params}", params_str)
            .replace("{action_preconditions}", prec_str)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{functions}", funcs_str)
            .replace("{llm_output}", llm_output)
        )

        no_fb, fb_msg = self.get_feedback(model, prompt, feedback_type, llm_output)

        return no_fb, fb_msg

    @require_llm
    def effect_feedback(
        self,
        model: BaseLLM,
        domain_desc: str,
        feedback_template: str,
        feedback_type: str = "llm",
        llm_output: str = "",
        parameter: OrderedDict = None,
        preconditions: str = None,
        effects: str = None,
        action_name: str = None,
        action_desc: str = None,
        types: dict[str, str] | list[dict[str, str]] = None,
        constants: dict[str, str] = None,
        predicates: list[Predicate] = None,
        functions: list[Function] = None,
    ) -> tuple[bool, str]:
        """
        Provides feedback to initial LLM output of a PDDL action effect.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain information
            feedback_template (str): prompt template to guide LLM to provide feedback to initial output
            feedback_type (str): type of feedback assistant - 'llm', 'human'
            llm_output (str): original LLM output
            parameter (OrderedDict): PDDL params of current action
            preconditions (str): PDDL precondition of current action
            effects (str): PDDL effect of current action
            action_name (str): name of action
            action_desc (str): description of action
            types (dict[str,str] | list[dict[str,str]]): dictionary of types currently in specification
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of predicates currently in specification
            functions (list[Function]): list of functions currently in specification

        Returns:
            no_fb (bool): flag that deems if feedback is not needed
            fb_msg (str): feedback message from assistant
        """

        # format string info replacements
        act_name_str = action_name if action_name else "No action name provided"
        act_desc_str = action_desc if action_desc else "No action description provided"
        params_str = (
            "\n".join([f"{name} - {type}" for name, type in parameter.items()])
            if parameter
            else "No parameters provided"
        )
        prec_str = preconditions if preconditions else "No preconditions provided."
        eff_str = effects if effects else "No effects provided."

        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )
        funcs_str = (
            "\n".join([f"{func['raw']}" for func in functions])
            if functions
            else "No functions provided."
        )

        prompt = (
            feedback_template.replace("{domain_desc}", domain_desc)
            .replace("{action_name}", act_name_str)
            .replace("{action_desc}", act_desc_str)
            .replace("{action_params}", params_str)
            .replace("{action_preconditions}", prec_str)
            .replace("{action_effects}", eff_str)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{functions}", funcs_str)
            .replace("{llm_output}", llm_output)
        )

        no_fb, fb_msg = self.get_feedback(model, prompt, feedback_type, llm_output)

        return no_fb, fb_msg

    @require_llm
    def predicate_feedback(
        self,
        model: BaseLLM,
        domain_desc: str,
        feedback_template: str,
        feedback_type: str = "llm",
        llm_output: str = "",
        types: dict[str, str] | list[dict[str, str]] = None,
        constants: dict[str, str] = None,
        predicates: list[Predicate] = None,
    ) -> tuple[bool, str]:
        """
        Provides feedback to initial LLM output of PDDL predicates.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain information
            feedback_template (str): prompt template to guide LLM to provide feedback to initial output
            feedback_type (str): type of feedback assistant - 'llm', 'human'
            llm_output (str): original LLM output
            types (dict[str,str] | list[dict[str,str]]): dictionary of types currently in specification
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of predicates currently in specification

        Returns:
            no_fb (bool): flag that deems if feedback is not needed
            fb_msg (str): feedback message from assistant
        """

        # format string info replacements
        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )

        prompt = (
            feedback_template.replace("{domain_desc}", domain_desc)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{llm_output}", llm_output)
        )

        no_fb, fb_msg = self.get_feedback(model, prompt, feedback_type, llm_output)

        return no_fb, fb_msg

    @require_llm
    def task_feedback(
        self,
        model: BaseLLM,
        problem_desc: str,
        feedback_template: str,
        feedback_type: str = "llm",
        llm_output: str = "",
        objects: dict[str, str] = None,
        initial: list[dict[str, str]] = None,
        goal: list[dict[str, str]] = None,
        types: dict[str, str] | list[dict[str, str]] = None,
        constants: dict[str, str] = None,
        predicates: list[Predicate] = None,
        functions: list[Function] = None,
    ) -> tuple[bool, str]:
        """
        Provides feedback to initial LLM output of a PDDL task.

        Args:
            model (BaseLLM): LLM to query
            problem_desc (str): general problem information
            feedback_template (str): prompt template to guide LLM to provide feedback to initial output
            feedback_type (str): type of feedback assistant - 'llm', 'human'
            llm_output (str): original LLM output
            objects (dict[str,str]): objects of current task specification
            initial (list[dict[str,str]]): initial states of current task specification
            goal (list[dict[str,str]]): goal states of current task specification
            types (dict[str,str] | list[dict[str,str]]): dictionary of types currently in specification
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of predicates currently in specification
            functions (list[Function]): list of functions currently in specification

        Returns:
            no_fb (bool): flag that deems if feedback is not needed
            fb_msg (str): feedback message from assistant
        """

        # format string info replacements
        obj_str = (
            "\n".join([f"{obj} - {type}" for obj, type in objects.items()])
            if objects
            else "No objects provided."
        )
        init_str = format_initial(initial) if initial else "No initial state provided."
        goal_str = format_goal(goal) if goal else "No goal state provided."

        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )
        funcs_str = (
            "\n".join([f"{func['raw']}" for func in functions])
            if functions
            else "No functions provided."
        )

        prompt = (
            feedback_template.replace("{problem_desc}", problem_desc)
            .replace("{objects}", obj_str)
            .replace("{initial_states}", init_str)
            .replace("{goal_states}", goal_str)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{functions}", funcs_str)
            .replace("{llm_output}", llm_output)
        )

        no_fb, fb_msg = self.get_feedback(model, prompt, feedback_type, llm_output)

        return no_fb, fb_msg

    @require_llm
    def objects_feedback(
        self,
        model: BaseLLM,
        problem_desc: str,
        feedback_template: str,
        feedback_type: str = "llm",
        llm_output: str = "",
        objects: dict[str, str] = None,
        types: dict[str, str] | list[dict[str, str]] = None,
        constants: dict[str, str] = None,
        predicates: list[Predicate] = None,
        functions: list[Function] = None,
    ) -> tuple[bool, str]:
        """
        Provides feedback to initial LLM output of PDDL task objects.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain information
            feedback_template (str): prompt template to guide LLM to provide feedback to initial output
            feedback_type (str): type of feedback assistant - 'llm', 'human'
            llm_output (str): original LLM output
            objects (dict[str,str]): objects of current task specification
            types (dict[str,str] | list[dict[str,str]]): dictionary of types currently in specification
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of predicates currently in specification
            functions (list[Function]): list of functions currently in specification

        Returns:
            no_fb (bool): flag that deems if feedback is not needed
            fb_msg (str): feedback message from assistant
        """

        # format string info replacements
        obj_str = (
            "\n".join([f"{obj} - {type}" for obj, type in objects.items()])
            if objects
            else "No objects provided."
        )

        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )
        funcs_str = (
            "\n".join([f"{func['raw']}" for func in functions])
            if functions
            else "No functions provided."
        )

        prompt = (
            feedback_template.replace("{problem_desc}", problem_desc)
            .replace("{objects}", obj_str)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{functions}", funcs_str)
            .replace("{llm_output}", llm_output)
        )

        no_fb, fb_msg = self.get_feedback(model, prompt, feedback_type, llm_output)

        return no_fb, fb_msg

    @require_llm
    def initial_state_feedback(
        self,
        model: BaseLLM,
        problem_desc: str,
        feedback_template: str,
        feedback_type: str = "llm",
        llm_output: str = "",
        objects: dict[str, str] = None,
        initial: list[dict[str, str]] = None,
        types: dict[str, str] | list[dict[str, str]] = None,
        constants: dict[str, str] = None,
        predicates: list[Predicate] = None,
        functions: list[Function] = None,
    ) -> tuple[bool, str]:
        """
        Provides feedback to initial LLM output of PDDL task initial states.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain information
            feedback_template (str): prompt template to guide LLM to provide feedback to initial output
            feedback_type (str): type of feedback assistant - 'llm', 'human'
            llm_output (str): original LLM output
            objects (dict[str,str]): objects of current task specification
            initial (list[dict[str,str]]): initial states of current task specification
            types (dict[str,str] | list[dict[str,str]]): dictionary of types currently in specification
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of predicates currently in specification
            functions (list[Function]): list of functions currently in specification

        Returns:
            no_fb (bool): flag that deems if feedback is not needed
            fb_msg (str): feedback message from assistant
        """

        # format string info replacements
        obj_str = (
            "\n".join([f"{obj} - {type}" for obj, type in objects.items()])
            if objects
            else "No objects provided."
        )
        init_str = format_initial(initial) if initial else "No initial state provided."

        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )
        funcs_str = (
            "\n".join([f"{func['raw']}" for func in functions])
            if functions
            else "No functions provided."
        )

        prompt = (
            feedback_template.replace("{problem_desc}", problem_desc)
            .replace("{objects}", obj_str)
            .replace("{initial_states}", init_str)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{functions}", funcs_str)
            .replace("{llm_output}", llm_output)
        )

        no_fb, fb_msg = self.get_feedback(model, prompt, feedback_type, llm_output)

        return no_fb, fb_msg

    @require_llm
    def goal_state_feedback(
        self,
        model: BaseLLM,
        problem_desc: str,
        feedback_template: str,
        feedback_type: str = "llm",
        llm_output: str = "",
        objects: dict[str, str] = None,
        initial: list[dict[str, str]] = None,
        goal: list[dict[str, str]] = None,
        types: dict[str, str] | list[dict[str, str]] = None,
        constants: dict[str, str] = None,
        predicates: list[Predicate] = None,
        functions: list[Function] = None,
    ) -> tuple[bool, str]:
        """
        Provides feedback to initial LLM output of PDDL task goal states.

        Args:
            model (BaseLLM): LLM to query
            domain_desc (str): general domain information
            feedback_template (str): prompt template to guide LLM to provide feedback to initial output
            feedback_type (str): type of feedback assistant - 'llm', 'human'
            llm_output (str): original LLM output
            objects (dict[str,str]): objects of current task specification
            initial (list[dict[str,str]]): initial states of current task specification
            goal (list[dict[str,str]]): goal states of current task specification
            types (dict[str,str] | list[dict[str,str]]): dictionary of types currently in specification
            constants (dict[str,str]): current constants in specification, defaults to None
            predicates (list[Predicate]): list of predicates currently in specification
            functions (list[Function]): list of functions currently in specification

        Returns:
            no_fb (bool): flag that deems if feedback is not needed
            fb_msg (str): feedback message from assistant
        """

        # format string info replacements
        obj_str = (
            "\n".join([f"{obj} - {type}" for obj, type in objects.items()])
            if objects
            else "No objects provided."
        )
        init_str = format_initial(initial) if initial else "No initial state provided."
        goal_str = format_goal(goal) if goal else "No goal state provided."

        types_str = pretty_print_dict(types) if types else "No types provided."
        const_str = (
            format_constants(constants) if constants else "No constants provided."
        )
        preds_str = (
            "\n".join([f"{pred['raw']}" for pred in predicates])
            if predicates
            else "No predicates provided."
        )
        funcs_str = (
            "\n".join([f"{func['raw']}" for func in functions])
            if functions
            else "No functions provided."
        )

        prompt = (
            feedback_template.replace("{problem_desc}", problem_desc)
            .replace("{objects}", obj_str)
            .replace("{initial_states}", init_str)
            .replace("{goal_states}", goal_str)
            .replace("{types}", types_str)
            .replace("{constants}", const_str)
            .replace("{predicates}", preds_str)
            .replace("{functions}", funcs_str)
            .replace("{llm_output}", llm_output)
        )

        no_fb, fb_msg = self.get_feedback(model, prompt, feedback_type, llm_output)

        return no_fb, fb_msg
