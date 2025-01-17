"""
This file contains collection of functions for PDDL feedback generation purposes
"""

from .utils import *
from .llm_builder import LLM, require_llm
from .domain_builder import DomainBuilder
from .task_builder import TaskBuilder
from collections import OrderedDict
import textwrap

domain_builder = DomainBuilder()
task_builder = TaskBuilder()


class FeedbackBuilder:

    @require_llm
    def get_feedback(
        self, model: LLM, feedback_template: str, feedback_type: str, llm_response: str
    ) -> tuple[bool, str]:
        """
        This retrieves the type of feedback user requests and returns feedack message.
        feedback_type takes in either "human" "llm" or "hybrid" which it both
        """

        model.reset_tokens()

        if feedback_type.lower() == "human":
            feedback_msg = self.human_feedback(llm_response)
        elif feedback_type.lower() == "llm":
            feedback_msg = model.query(prompt=feedback_template)
        elif feedback_type.lower() == "hybrid":
            feedback_msg = model.query(prompt=feedback_template)
            response = (
                "\nORIGINAL LLM OUTPUT:\n"
                + llm_response
                + "\nFEEDBACK:\n"
                + feedback_msg
            )
            feedback_msg.replace("no feedback".lower(), "")
            feedback_msg += self.human_feedback(response)
        else:
            raise ValueError(
                "Invalid feedback_type. Expected 'human', 'llm', or 'hybrid'."
            )

        if "no feedback" in feedback_msg.lower() or len(feedback_msg.strip()) == 0:
            return True, feedback_msg

        return False, feedback_msg

    @require_llm
    def type_feedback(
        self,
        model: LLM,
        domain_desc: str,
        llm_response: str,
        feedback_template: str,
        feedback_type: str = "llm",
        types: dict[str, str] = None,
    ) -> tuple[dict[str, str], str]:
        """Makes LLM call using feedback prompt, then parses it into type format"""

        model.reset_tokens()

        type_str = format_dict(types) if types else "No types provided."

        feedback_template = feedback_template.replace("{domain_desc}", domain_desc)
        feedback_template = feedback_template.replace("{types}", type_str)
        feedback_template = feedback_template.replace("{llm_response}", llm_response)

        no_fb, fb_msg = self.get_feedback(
            model, feedback_template, feedback_type, llm_response
        )

        if not no_fb:
            structure_prompt = textwrap.dedent(
                """
            ## OUTPUT
            {
                "type_1": "description",
                "type_2": "description",
                "type_3": "description",
            }                              
            """
            )

            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f'\nEnd your final answer starting with "## OUTPUT" and then re-iterate an updated version of the Python dictionary pair like so:\n{structure_prompt}'
                f"\n\nApply the suggestions to your original answer:\n{type_str}"
            )

            model.reset_tokens()

            types, llm_response = domain_builder.extract_type(
                model, domain_desc, prompt, type_str
            )

        return types, llm_response

    @require_llm
    def type_hierarchy_feedback(
        self,
        model: LLM,
        domain_desc: str,
        llm_response: str,
        feedback_template: str,
        feedback_type: str = "llm",
        type_hierarchy: dict[str, str] = None,
    ) -> tuple[dict[str, str], str]:
        """Makes LLM call using feedback prompt, then parses it into type hierarchy format"""

        model.reset_tokens()

        type_str = (
            format_dict(type_hierarchy) if type_hierarchy else "No types provided."
        )

        feedback_template = feedback_template.replace("{domain_desc}", domain_desc)
        feedback_template = feedback_template.replace("{types}", type_str)
        feedback_template = feedback_template.replace("{llm_response}", llm_response)

        no_fb, fb_msg = self.get_feedback(
            model, feedback_template, feedback_type, llm_response
        )

        if not no_fb:
            structure_prompt = textwrap.dedent(
                """
            ## OUTPUT
            {
                "parent_type_1": "description",
                "children": [
                    {
                        "child_type_1": "description",
                        "children": [
                            {"child_child_type_1": "description", "children": []},
                            {"child_child_type_2": "description", "children": []}
                        ]
                    }
                ]
            }
            """
            )

            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f'\nEnd your final answer starting with "## OUTPUT" and then re-iterate an updated version of the Python dictionary pair like so:\n{structure_prompt}'
                f"\n\nApply the suggestions to your original answer:\n{type_str}"
            )

            model.reset_tokens()

            type_hierarchy, llm_response = domain_builder.extract_type_hierarchy(
                model, domain_desc, prompt, type_str
            )

        return type_hierarchy, llm_response

    @require_llm
    def nl_action_feedback(
        self,
        model: LLM,
        domain_desc: str,
        llm_response: str,
        feedback_template: str,
        feedback_type: str = "llm",
        nl_actions: dict[str, str] = None,
        type_hierarchy: dict[str, str] = None,
    ) -> tuple[dict[str, str], str]:
        """Makes LLM call using feedback prompt, then parses it into nl_action format"""

        model.reset_tokens()

        type_str = (
            format_dict(type_hierarchy) if type_hierarchy else "No types provided."
        )
        nl_action_str = (
            format_dict(nl_actions) if nl_actions else "No actions provided."
        )

        feedback_template = feedback_template.replace("{domain_desc}", domain_desc)
        feedback_template = feedback_template.replace("{llm_response}", llm_response)
        feedback_template = feedback_template.replace("{types}", type_str)
        feedback_template = feedback_template.replace("{nl_actions}", nl_action_str)

        no_fb, fb_msg = self.get_feedback(
            model, feedback_template, feedback_type, llm_response
        )

        if not no_fb:
            structure_prompt = textwrap.dedent(
                """
            ## OUTPUT
            {
                "action_name_1": "action_description",
                "action_name_2": "action_description",
                "action_name_3": "action_description"
            }
            """
            )

            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f'\nEnd your final answer starting with "## OUTPUT" and then re-iterate an updated version of the Python dictionary pair like so:\n{structure_prompt}'
                f"\n\nApply the suggestions to your original answer:\n{nl_action_str}"
            )

            model.reset_tokens()

            nl_actions, llm_response = domain_builder.extract_type_hierarchy(
                model, domain_desc, prompt
            )

        return nl_actions, llm_response

    @require_llm
    def pddl_action_feedback(
        self,
        model: LLM,
        domain_desc: str,
        llm_response: str,
        feedback_template: str,
        feedback_type: str = "llm",
        action: Action = None,
        predicates: list[Predicate] = None,
        types: dict[str, str] = None,
    ) -> tuple[Action, list[Predicate], str, tuple[bool, str], bool]:
        """Makes LLM call using feedback prompt, then parses it into action format"""

        model.reset_tokens()

        type_str = format_dict(types) if types else "No types provided."
        predicate_str = (
            format_predicates(predicates) if predicates else "No predicates provided."
        )
        param_str = (
            "\n".join([f"{name} - {type}" for name, type in action["params"].items()])
            if action
            else "No parameters provided"
        )
        action_name = action["name"] if action else "No action name provided"
        preconditions_str = (
            action["preconditions"] if action else "No preconditions provided."
        )
        effects_str = action["effects"] if action else "No effects provided."

        feedback_template = feedback_template.replace("{domain_desc}", domain_desc)
        feedback_template = feedback_template.replace("{llm_response}", llm_response)
        feedback_template = feedback_template.replace("{types}", type_str)
        feedback_template = feedback_template.replace("{predicates}", predicate_str)
        feedback_template = feedback_template.replace("{action_name}", action_name)
        feedback_template = feedback_template.replace("{action_params}", param_str)
        feedback_template = feedback_template.replace(
            "{action_preconditions}", preconditions_str
        )
        feedback_template = feedback_template.replace("{action_effects}", effects_str)

        no_fb, fb_msg = self.get_feedback(
            model, feedback_template, feedback_type, llm_response
        )

        validation_info = [True, "All validations passed."]
        if not no_fb:

            structure_prompt = textwrap.dedent(
                """
            End your final answers underneath the headers: '### Action Parameters,' '### Action Preconditions,' '### Action Effects,' and '### New Predicates' with ''' ''' comment blocks in PDDL as so:

            ### Action Parameters
            ```
            - ?t - type: 'parameter_description'
            ```

            ### Action Preconditions
            ```
            (and
                (predicate_name ?t1 ?t2) ; COMMENT DESCRIPTION
            )
            ```

            ### Action Effects
            ```
            (and
                (predicate_name ?t1 ?t2) ; COMMENT DESCRIPTION
            )
            ```

            ### New Predicates
            ```
            - (predicate_name ?t1 - type_1 ?t2 - type_2): 'predicate_description'
            ``` 

            If there are no new predicates created, keep an empty space enclosed ```  ``` with the '### New Predicates' header.
            """
            )

            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\n{structure_prompt}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            model.reset_tokens()

            action, predicates, llm_response, validation_info = (
                domain_builder.extract_pddl_action(
                    model, domain_desc, prompt, action_name
                )
            )
        return action, predicates, llm_response, validation_info, no_fb

    @require_llm
    def parameter_feedback(
        self,
        model: LLM,
        domain_desc: str,
        llm_response: str,
        feedback_template: str,
        feedback_type: str = "llm",
        parameter: OrderedDict = None,
        action_name: str = None,
        action_desc: str = None,
        types: dict[str, str] = None,
    ) -> tuple[OrderedDict, OrderedDict, str]:
        """Makes LLM call using feedback prompt, then parses it into parameter format"""

        model.reset_tokens()

        type_str = format_dict(types) if types else "No types provided."
        param_str = (
            "\n".join([f"{name} - {type}" for name, type in parameter.items()])
            if parameter
            else "No parameters provided"
        )
        action_name = action_name if action_name else "No action name provided."
        action_desc = action_desc if action_desc else "No action description provided."

        feedback_template = feedback_template.replace("{domain_desc}", domain_desc)
        feedback_template = feedback_template.replace("{llm_response}", llm_response)
        feedback_template = feedback_template.replace("{types}", type_str)
        feedback_template = feedback_template.replace("{action_name}", action_name)
        feedback_template = feedback_template.replace("{action_desc}", action_desc)
        feedback_template = feedback_template.replace("{parameters}", param_str)

        no_fb, fb_msg = self.get_feedback(
            model, feedback_template, feedback_type, llm_response
        )

        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            model.reset_tokens()

            param, param_raw, llm_response = domain_builder.extract_parameters(
                model, domain_desc, prompt, action_name, action_desc, types
            )
        return param, param_raw, llm_response

    @require_llm
    def precondition_feedback(
        self,
        model: LLM,
        domain_desc: str,
        llm_response: str,
        feedback_template: str,
        feedback_type: str = "llm",
        parameter: OrderedDict = None,
        preconditions: str = None,
        action_name: str = None,
        action_desc: str = None,
        types: dict[str, str] = None,
        predicates: list[Predicate] = None,
    ) -> tuple[str, list[Predicate], str]:
        """Makes LLM call using feedback prompt, then parses it into precondition format"""

        model.reset_tokens()

        type_str = format_dict(types) if types else "No types provided."
        predicate_str = (
            format_predicates(predicates) if predicates else "No predicates provided."
        )
        param_str = (
            "\n".join([f"{name} - {type}" for name, type in parameter.items()])
            if parameter
            else "No parameters provided"
        )
        action_name = action_name if action_name else "No action name provided."
        action_desc = action_desc if action_desc else "No action description provided."
        precondition_str = (
            preconditions if preconditions else "No preconditions provided."
        )

        feedback_template = feedback_template.replace("{domain_desc}", domain_desc)
        feedback_template = feedback_template.replace("{llm_response}", llm_response)
        feedback_template = feedback_template.replace("{types}", type_str)
        feedback_template = feedback_template.replace("{predicates}", predicate_str)
        feedback_template = feedback_template.replace("{action_name}", action_name)
        feedback_template = feedback_template.replace("{action_desc}", action_desc)
        feedback_template = feedback_template.replace("{parameters}", param_str)
        feedback_template = feedback_template.replace(
            "{action_preconditions}", precondition_str
        )

        no_fb, fb_msg = self.get_feedback(
            model, feedback_template, feedback_type, llm_response
        )

        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            model.reset_tokens()

            preconditions, new_predicates, llm_response = (
                domain_builder.extract_preconditions(
                    model, domain_desc, prompt, action_name, action_desc
                )
            )
        return preconditions, new_predicates, llm_response

    @require_llm
    def effect_feedback(
        self,
        model: LLM,
        domain_desc: str,
        llm_response: str,
        feedback_template: str,
        feedback_type: str = "llm",
        parameter: OrderedDict = None,
        preconditions: str = None,
        effects: str = None,
        action_name: str = None,
        action_desc: str = None,
        types: dict[str, str] = None,
        predicates: list[Predicate] = None,
    ) -> tuple[str, list[Predicate], str]:
        """Makes LLM call using feedback prompt, then parses it into effects format"""

        model.reset_tokens()

        type_str = format_dict(types) if types else "No types provided."
        predicate_str = (
            format_predicates(predicates) if predicates else "No predicates provided."
        )
        param_str = (
            "\n".join([f"{name} - {type}" for name, type in parameter.items()])
            if parameter
            else "No parameters provided"
        )
        action_name = action_name if action_name else "No action name provided."
        action_desc = action_desc if action_desc else "No action description provided."
        precondition_str = (
            preconditions if preconditions else "No preconditions provided."
        )
        effect_str = effects if effects else "No effects provided."

        feedback_template = feedback_template.replace("{domain_desc}", domain_desc)
        feedback_template = feedback_template.replace("{llm_response}", llm_response)
        feedback_template = feedback_template.replace("{types}", type_str)
        feedback_template = feedback_template.replace("{predicates}", predicate_str)
        feedback_template = feedback_template.replace("{action_name}", action_name)
        feedback_template = feedback_template.replace("{action_desc}", action_desc)
        feedback_template = feedback_template.replace("{parameters}", param_str)
        feedback_template = feedback_template.replace(
            "{action_preconditions}", precondition_str
        )
        feedback_template = feedback_template.replace("{action_effects}", effect_str)

        no_fb, fb_msg = self.get_feedback(
            model, feedback_template, feedback_type, llm_response
        )

        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            model.reset_tokens()

            effects, new_predicates, llm_response = domain_builder.extract_effects(
                model, domain_desc, prompt, action_name, action_desc
            )
        return effects, new_predicates, llm_response

    @require_llm
    def predicate_feedback(
        self,
        model: LLM,
        domain_desc: str,
        llm_response: str,
        feedback_template: str,
        feedback_type: str = "llm",
        types: dict[str, str] = None,
        predicates: list[Predicate] = None,
        nl_actions: dict[str, str] = None,
    ) -> tuple[list[Predicate], str]:
        """Makes LLM call using feedback prompt, then parses it into predicates format"""

        model.reset_tokens()

        type_str = format_dict(types) if types else "No types provided."
        predicate_str = (
            format_predicates(predicates) if predicates else "No predicates provided."
        )
        nl_action_str = (
            format_dict(nl_actions) if nl_actions else "No actions provided."
        )

        feedback_template = feedback_template.replace("{domain_desc}", domain_desc)
        feedback_template = feedback_template.replace("{llm_response}", llm_response)
        feedback_template = feedback_template.replace("{types}", type_str)
        feedback_template = feedback_template.replace("{predicates}", predicate_str)
        feedback_template = feedback_template.replace("{nl_actions}", nl_action_str)

        no_fb, fb_msg = self.get_feedback(
            model, feedback_template, feedback_type, llm_response
        )

        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            model.reset_tokens()

            new_predicates, llm_response = domain_builder.extract_predicates(
                model, domain_desc, prompt
            )
        return new_predicates, llm_response

    @require_llm
    def task_feedback(
        self,
        model: LLM,
        problem_desc: str,
        llm_response: str,
        feedback_template: str,
        feedback_type: str = "llm",
        predicates: list[Predicate] = None,
        types: dict[str, str] = None,
        objects: dict[str, str] = None,
        initial: list[dict[str, str]] = None,
        goal: list[dict[str, str]] = None,
    ) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]], str]:
        """Makes LLM call using feedback prompt, then parses it into object, initial, and goal format"""

        model.reset_tokens()

        type_str = format_dict(types) if types else "No types provided."
        predicate_str = (
            format_predicates(predicates) if predicates else "No predicates provided."
        )
        objects_str = (
            "\n".join([f"{obj} - {type}" for obj, type in objects.items()])
            if objects
            else "No objects provided."
        )
        initial_state_str = (
            task_builder.format_initial(initial)
            if initial
            else "No initial state provided."
        )
        goal_state_str = (
            task_builder.format_goal(goal) if goal else "No goal state provided."
        )

        feedback_template = feedback_template.replace("{problem_desc}", problem_desc)
        feedback_template = feedback_template.replace("{llm_response}", llm_response)
        feedback_template = feedback_template.replace("{types}", type_str)
        feedback_template = feedback_template.replace("{predicates}", predicate_str)
        feedback_template = feedback_template.replace("{objects}", objects_str)
        feedback_template = feedback_template.replace(
            "{initial_states}", initial_state_str
        )
        feedback_template = feedback_template.replace("{goal_states}", goal_state_str)

        no_fb, fb_msg = self.get_feedback(
            model, feedback_template, feedback_type, llm_response
        )

        if not no_fb:

            structure_prompt = textwrap.dedent(
                """
            End your final answer starting with headers (in order) "### OBJECTS" (with no brackets) "### INITIAL" and "### GOAL" containg respective content with ''' ''' comment blocks in PDDL as so:

            ### OBJECTS
            ```
            object1 - type_1
            object2 - type_2
            object3 - type_1
            ```

            ### INITIAL
            ```
            (predicate_name object1 object2) ; comment for initial state predicate 1
            (predicate_name object3 object4) ; comment for initial state predicate 2
            (predicate_name object5) ; comment for initial state predicate 3
            ```

            ### GOAL
            ```
            (and
            (predicate_name object) ; comment
            )
            ```

            Even if there is one goal state, it must contain the PDDL 'and' syntax. Each object must be declared separately with their type and not grouped - even if objects share the same type.
            """
            )

            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\n{structure_prompt}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            model.reset_tokens()

            objects, initial, goal, _ = task_builder.extract_task(
                model, problem_desc, prompt
            )

        return objects, initial, goal, fb_msg

    @require_llm
    def objects_feedback(
        self,
        model: LLM,
        problem_desc: str,
        llm_response: str,
        feedback_template: str,
        feedback_type: str = "llm",
        type_hierarchy: dict[str, str] = None,
        predicates: list[Predicate] = None,
        objects: dict[str, str] = None,
    ) -> tuple[dict[str, str], str]:
        """Makes LLM call using feedback prompt, then parses it into objects format"""

        model.reset_tokens()

        type_str = (
            format_dict(type_hierarchy) if type_hierarchy else "No types provided."
        )
        predicate_str = (
            format_predicates(predicates) if predicates else "No predicates provided."
        )
        objects_str = (
            "\n".join([f"{obj} - {type}" for obj, type in objects.items()])
            if objects
            else "No objects provided."
        )

        feedback_template = feedback_template.replace("{problem_desc}", problem_desc)
        feedback_template = feedback_template.replace("{llm_response}", llm_response)
        feedback_template = feedback_template.replace("{types}", type_str)
        feedback_template = feedback_template.replace("{predicates}", predicate_str)
        feedback_template = feedback_template.replace("{objects}", objects_str)

        no_fb, fb_msg = self.get_feedback(
            model, feedback_template, feedback_type, llm_response
        )

        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            model.reset_tokens()

            objects, llm_response = task_builder.extract_objects(
                model, problem_desc, prompt
            )

        return objects, llm_response

    @require_llm
    def initial_state_feedback(
        self,
        model: LLM,
        problem_desc: str,
        llm_response: str,
        feedback_template: str,
        feedback_type: str = "llm",
        type_hierarchy: dict[str, str] = None,
        predicates: list[Predicate] = None,
        objects: dict[str, str] = None,
        initial: list[dict[str, str]] = None,
    ) -> tuple[list[dict[str, str]], str]:
        """Makes LLM call using feedback prompt, then parses it into initial states format"""

        model.reset_tokens()

        type_str = (
            format_dict(type_hierarchy) if type_hierarchy else "No types provided."
        )
        predicate_str = (
            format_predicates(predicates) if predicates else "No predicates provided."
        )
        objects_str = (
            "\n".join([f"{obj} - {type}" for obj, type in objects.items()])
            if objects
            else "No objects provided."
        )
        initial_state_str = (
            task_builder.format_initial(initial)
            if initial
            else "No initial state provided."
        )

        feedback_template = feedback_template.replace("{problem_desc}", problem_desc)
        feedback_template = feedback_template.replace("{llm_response}", llm_response)
        feedback_template = feedback_template.replace("{types}", type_str)
        feedback_template = feedback_template.replace("{predicates}", predicate_str)
        feedback_template = feedback_template.replace("{objects}", objects_str)
        feedback_template = feedback_template.replace(
            "{initial_state}", initial_state_str
        )

        no_fb, fb_msg = self.get_feedback(
            model, feedback_template, feedback_type, llm_response
        )

        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            model.reset_tokens()

            initial, llm_response = task_builder.extract_initial_state(
                model, problem_desc, prompt
            )

        return initial, llm_response

    @require_llm
    def goal_state_feedback(
        self,
        model: LLM,
        problem_desc: str,
        llm_response: str,
        feedback_template: str,
        feedback_type: str = "llm",
        type_hierarchy: dict[str, str] = None,
        predicates: list[Predicate] = None,
        objects: dict[str, str] = None,
        initial: list[dict[str, str]] = None,
        goal: list[dict[str, str]] = None,
    ) -> tuple[list[dict[str, str]], str]:
        """Makes LLM call using feedback prompt, then parses it into goal states format"""

        model.reset_tokens()

        type_str = (
            format_dict(type_hierarchy) if type_hierarchy else "No types provided."
        )
        predicate_str = (
            format_predicates(predicates) if predicates else "No predicates provided."
        )
        objects_str = (
            "\n".join([f"{obj} - {type}" for obj, type in objects.items()])
            if objects
            else "No objects provided."
        )
        initial_state_str = (
            task_builder.format_initial(initial)
            if initial
            else "No initial state provided."
        )
        goal_state_str = (
            task_builder.format_goal(goal) if goal else "No goal state provided."
        )

        feedback_template = feedback_template.replace("{problem_desc}", problem_desc)
        feedback_template = feedback_template.replace("{llm_response}", llm_response)
        feedback_template = feedback_template.replace("{types}", type_str)
        feedback_template = feedback_template.replace("{predicates}", predicate_str)
        feedback_template = feedback_template.replace("{objects}", objects_str)
        feedback_template = feedback_template.replace(
            "{initial_state}", initial_state_str
        )
        feedback_template = feedback_template.replace("{initial_state}", goal_state_str)

        no_fb, fb_msg = self.get_feedback(
            model, feedback_template, feedback_type, llm_response
        )

        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            model.reset_tokens()

            goal, llm_response = task_builder.extract_goal_state(
                model, problem_desc, prompt
            )

        return goal, llm_response

    def human_feedback(self, info: str):
        """This enables human-in-the-loop feedback mechanism"""

        print("START OF INFO\n", info)
        print("\nEND OF INFO\n\n")
        contents = []
        print("Provide feedback (or type 'done' to finish):\n")
        while True:
            line = input()
            if line.strip().lower() == "done":
                break
            contents.append(line)
        resp = "\n".join(contents)

        if resp.strip().lower() == "no feedback":
            return "no feedback"

        return resp
