"""
This file contains collection of functions for PDDL feedback generation purposes
"""

from .llm_builder import LLM_Chat
from .domain_builder import DomainBuilder
from .task_builder import TaskBuilder
from .utils.pddl_parser import convert_to_dict, parse_action, parse_new_predicates, parse_objects, parse_initial, parse_goal, parse_params, format_dict, format_predicates
from .utils.pddl_types import Action, Predicate
from collections import OrderedDict

domain_builder = DomainBuilder()
task_builder = TaskBuilder()

class FeedbackBuilder:

    def get_feedback(
            self, 
            model: LLM_Chat, 
            feedback_template: str, 
            feedback_type: str, 
            llm_response: str
            ) -> tuple[bool, str]:
        """
        This retrieves the type of feedback user requests and returns feedack message.
        feedback_type takes in either "human" "llm" or "hybrid" which it both 
        """

        model.reset_tokens()
        
        if feedback_type.lower() == "human":
            feedback_msg = self.human_feedback(llm_response)
        elif feedback_type.lower() == "llm":
            feedback_msg = model.get_output(prompt=feedback_template)
        elif feedback_type.lower() == "hybrid":
            feedback_msg = model.get_output(prompt=feedback_template)
            response = "\nORIGINAL LLM OUTPUT:\n" + llm_response + "\nFEEDBACK:\n" + feedback_msg
            feedback_msg.replace("no feedback".lower(), "")
            feedback_msg += self.human_feedback(response)
        else:
            raise ValueError("Invalid feedback_type. Expected 'human', 'llm', or 'hybrid'.")
        
        if 'no feedback' in feedback_msg.lower() or len(feedback_msg.strip()) == 0:
            return True, feedback_msg
        
        return False, feedback_msg

    def type_feedback(
            self, 
            model: LLM_Chat, 
            domain_desc: str,
            llm_response: str,
            feedback_template: str, 
            feedback_type: str="llm",
            types: dict[str,str]=None, 
            ) -> tuple[dict[str,str], str]:
        """Makes LLM call using feedback prompt, then parses it into type format"""

        model.reset_tokens()

        type_str = format_dict(types) if types else "No types provided."

        feedback_template = feedback_template.replace('{domain_desc}', domain_desc)
        feedback_template = feedback_template.replace('{types}', type_str)
        feedback_template = feedback_template.replace('{llm_response}', llm_response)

        no_fb, fb_msg = self.get_feedback(model, feedback_template, feedback_type, llm_response)
    
        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            types, llm_response = domain_builder.extract_type(model, domain_desc, prompt)

        return types, llm_response

    def type_hierarchy_feedback(
            self, 
            model: LLM_Chat, 
            domain_desc: str,
            llm_response: str,
            feedback_template: str, 
            feedback_type: str="llm",
            type_hierarchy: dict[str,str]=None, 
            ) -> tuple[dict[str,str], str]:
        """Makes LLM call using feedback prompt, then parses it into type hierarchy format"""

        model.reset_tokens()

        type_str = format_dict(type_hierarchy) if type_hierarchy else "No types provided."

        feedback_template = feedback_template.replace('{domain_desc}', domain_desc)
        feedback_template = feedback_template.replace('{types}', type_str)
        feedback_template = feedback_template.replace('{llm_response}', llm_response)
        
        no_fb, fb_msg = self.get_feedback(model, feedback_template, feedback_type, llm_response)
    
        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )
            type_hierarchy, llm_response = domain_builder.extract_type_hierarchy(model, domain_desc, prompt)

        return type_hierarchy, llm_response

    def nl_action_feedback(
            self, 
            model: LLM_Chat, 
            domain_desc: str, 
            llm_response: str,
            feedback_template: str,
            feedback_type: str="llm",
            nl_actions: dict[str,str]=None,
            type_hierarchy: dict[str,str]=None, 
            ) -> tuple[dict[str,str], str]:
        """Makes LLM call using feedback prompt, then parses it into nl_action format"""

        model.reset_tokens()

        type_str = format_dict(type_hierarchy) if type_hierarchy else "No types provided."
        nl_action_str = format_dict(nl_actions) if nl_actions else "No actions provided."

        feedback_template = feedback_template.replace('{domain_desc}', domain_desc)
        feedback_template = feedback_template.replace('{llm_response}', llm_response)
        feedback_template = feedback_template.replace('{types}', type_str)
        feedback_template = feedback_template.replace('{nl_actions}', nl_action_str)

        no_fb, fb_msg = self.get_feedback(model, feedback_template, feedback_type, llm_response)
    
        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )
            nl_actions, llm_response = domain_builder.extract_type_hierarchy(model, domain_desc, prompt)

        return nl_actions, llm_response

    def pddl_action_feedback(
            self, 
            model: LLM_Chat, 
            domain_desc: str, 
            llm_response: str,
            feedback_template: str, 
            feedback_type: str="llm",
            action: Action=None, 
            predicates: list[Predicate]=None, 
            types: dict[str,str]=None
            ) -> tuple[Action, list[Predicate], str]:
        """Makes LLM call using feedback prompt, then parses it into action format"""

        model.reset_tokens()

        type_str = format_dict(types) if types else "No types provided."
        predicate_str = format_predicates(predicates) if predicates else "No predicates provided."
        param_str = ", ".join([f"{name} - {type}" for name, type in action['parameters'].items()]) \
                if action else "No parameters provided"
        action_name = action['name'] if action else "No action name provided"
        preconditions_str = action['preconditions'] if action else "No preconditions provided."
        effects_str = action['effects'] if action else "No effects provided."
        
        feedback_template = feedback_template.replace('{domain_desc}', domain_desc)
        feedback_template = feedback_template.replace('{llm_response}', llm_response)
        feedback_template = feedback_template.replace('{types}', type_str)
        feedback_template = feedback_template.replace('{predicates}', predicate_str)
        feedback_template = feedback_template.replace('{action_name}', action_name)
        feedback_template = feedback_template.replace('{parameters}', param_str)
        feedback_template = feedback_template.replace('{action_preconditions}', preconditions_str)
        feedback_template = feedback_template.replace('{action_effects}', effects_str)

        no_fb, fb_msg = self.get_feedback(model, feedback_template, feedback_type, llm_response)
    
        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            action, predicates, llm_response = domain_builder.extract_pddl_action(model, domain_desc, prompt, action_name)
        return action, predicates, llm_response

    def parameter_feedback(
            self, 
            model: LLM_Chat, 
            domain_desc: str, 
            llm_response: str,
            feedback_template: str, 
            feedback_type: str="llm", 
            parameter: OrderedDict=None, 
            action_name: str=None, 
            action_desc: str=None,
            types: dict[str,str]=None
            ) -> tuple[OrderedDict, OrderedDict, str]:
        """Makes LLM call using feedback prompt, then parses it into parameter format"""

        model.reset_tokens()

        type_str = format_dict(types) if types else "No types provided."
        param_str = "\n".join([f"{name} - {type}" for name, type in parameter.items()]) \
            if parameter else "No parameters provided"
        action_name = action_name if action_name else "No action name provided."
        action_desc = action_desc if action_desc else "No action description provided."
        
        feedback_template = feedback_template.replace('{domain_desc}', domain_desc)
        feedback_template = feedback_template.replace('{llm_response}', llm_response)
        feedback_template = feedback_template.replace('{types}', type_str)
        feedback_template = feedback_template.replace('{action_name}', action_name)
        feedback_template = feedback_template.replace('{action_desc}', action_desc)
        feedback_template = feedback_template.replace('{parameters}', param_str)

        no_fb, fb_msg = self.get_feedback(model, feedback_template, feedback_type, llm_response)
    
        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            param, param_raw, llm_response = domain_builder.extract_parameters(model, domain_desc, prompt, action_name, action_desc, types)
        return param, param_raw, llm_response

    def precondition_feedback(
            self, 
            model: LLM_Chat, 
            domain_desc: str, 
            llm_response: str,
            feedback_template: str, 
            feedback_type: str="llm", 
            parameter: OrderedDict=None, 
            preconditions: str=None,
            action_name: str=None, 
            action_desc: str=None,
            types: dict[str,str]=None,
            predicates: list[Predicate]=None
            ) -> tuple[str, list[Predicate], str]:
        """Makes LLM call using feedback prompt, then parses it into precondition format"""
        
        model.reset_tokens()

        type_str = format_dict(types) if types else "No types provided."
        predicate_str = format_predicates(predicates) if predicates else "No predicates provided."
        param_str = "\n".join([f"{name} - {type}" for name, type in parameter.items()]) \
            if parameter else "No parameters provided"
        action_name = action_name if action_name else "No action name provided."
        action_desc = action_desc if action_desc else "No action description provided."
        precondition_str = preconditions if preconditions else "No preconditions provided."
        
        feedback_template = feedback_template.replace('{domain_desc}', domain_desc)
        feedback_template = feedback_template.replace('{llm_response}', llm_response)
        feedback_template = feedback_template.replace('{types}', type_str)
        feedback_template = feedback_template.replace('{predicates}', predicate_str)
        feedback_template = feedback_template.replace('{action_name}', action_name)
        feedback_template = feedback_template.replace('{action_desc}', action_desc)
        feedback_template = feedback_template.replace('{parameters}', param_str)
        feedback_template = feedback_template.replace('{action_preconditions}', precondition_str)

        no_fb, fb_msg = self.get_feedback(model, feedback_template, feedback_type, llm_response)
    
        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            preconditions, new_predicates, llm_response = domain_builder.extract_preconditions(model, 
                                                                    domain_desc, prompt, action_name, action_desc)
        return preconditions, new_predicates, llm_response

    def effect_feedback(
            self, 
            model: LLM_Chat, 
            domain_desc: str, 
            llm_response: str,
            feedback_template: str, 
            feedback_type: str="llm", 
            parameter: OrderedDict=None, 
            preconditions: str=None,
            effects: str=None,
            action_name: str=None, 
            action_desc: str=None,
            types: dict[str,str]=None,
            predicates: list[Predicate]=None
            ) -> tuple[str, list[Predicate], str]:
        """Makes LLM call using feedback prompt, then parses it into effects format"""

        model.reset_tokens()

        type_str = format_dict(types) if types else "No types provided."
        predicate_str = format_predicates(predicates) if predicates else "No predicates provided."
        param_str = "\n".join([f"{name} - {type}" for name, type in parameter.items()]) \
            if parameter else "No parameters provided"
        action_name = action_name if action_name else "No action name provided."
        action_desc = action_desc if action_desc else "No action description provided."
        precondition_str = preconditions if preconditions else "No preconditions provided."
        effect_str = effects if effects else "No effects provided."
        
        feedback_template = feedback_template.replace('{domain_desc}', domain_desc)
        feedback_template = feedback_template.replace('{llm_response}', llm_response)
        feedback_template = feedback_template.replace('{types}', type_str)
        feedback_template = feedback_template.replace('{predicates}', predicate_str)
        feedback_template = feedback_template.replace('{action_name}', action_name)
        feedback_template = feedback_template.replace('{action_desc}', action_desc)
        feedback_template = feedback_template.replace('{parameters}', param_str)
        feedback_template = feedback_template.replace('{action_preconditions}', precondition_str)
        feedback_template = feedback_template.replace('{action_effects}', effect_str)

        no_fb, fb_msg = self.get_feedback(model, feedback_template, feedback_type, llm_response)
    
        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            effects, new_predicates, llm_response = domain_builder.extract_effects(model, 
                                                                    domain_desc, prompt, action_name, action_desc)
        return effects, new_predicates, llm_response

    def predicate_feedback(
            self,
            model: LLM_Chat, 
            domain_desc: str, 
            llm_response: str,
            feedback_template: str, 
            feedback_type: str="llm", 
            types: dict[str,str]=None,
            predicates: list[Predicate]=None,
            nl_actions: dict[str,str]=None
            ) -> tuple[list[Predicate], str]:
        """Makes LLM call using feedback prompt, then parses it into predicates format"""
        
        model.reset_tokens()

        type_str = format_dict(types) if types else "No types provided."
        predicate_str = format_predicates(predicates) if predicates else "No predicates provided."
        nl_action_str = format_dict(nl_actions) if nl_actions else "No actions provided."

        feedback_template = feedback_template.replace('{domain_desc}', domain_desc)
        feedback_template = feedback_template.replace('{llm_response}', llm_response)
        feedback_template = feedback_template.replace('{types}', type_str)
        feedback_template = feedback_template.replace('{predicates}', predicate_str)
        feedback_template = feedback_template.replace('{nl_actions}', nl_action_str)

        no_fb, fb_msg = self.get_feedback(model, feedback_template, feedback_type, llm_response)
    
        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            new_predicates, llm_response = domain_builder.extract_predicates(model, domain_desc, prompt)
        return new_predicates, llm_response


    def task_feedback(
            self, 
            model: LLM_Chat, 
            problem_desc: str, 
            llm_response: str,
            feedback_template: str, 
            feedback_type: str="llm",
            predicates: list[Predicate]=None, 
            types: dict[str,str]=None, 
            objects: dict[str,str]=None, 
            initial: list[dict[str,str]]=None, 
            goal: list[dict[str,str]]=None, 
            ) -> tuple[dict[str,str], list[dict[str,str]], list[dict[str,str]], str]:
        """Makes LLM call using feedback prompt, then parses it into object, initial, and goal format"""
        
        model.reset_tokens()

        type_str = format_dict(types) if types else "No types provided."
        predicate_str = format_predicates(predicates) if predicates else "No predicates provided."
        objects_str = "\n".join([f"{obj} - {type}" for obj, type in objects.items()]) if objects else "No objects provided."
        initial_state_str = task_builder.format_initial(initial) if initial else "No initial state provided."
        goal_state_str = task_builder.format_goal(goal) if goal else "No goal state provided."
        
        feedback_template = feedback_template.replace('{problem_desc}', problem_desc)
        feedback_template = feedback_template.replace('{llm_response}', llm_response)
        feedback_template = feedback_template.replace('{types}', type_str)
        feedback_template = feedback_template.replace('{predicates}', predicate_str)
        feedback_template = feedback_template.replace('{objects}', objects_str)
        feedback_template = feedback_template.replace('{initial_state}', initial_state_str)
        feedback_template = feedback_template.replace('{goal_state}', goal_state_str)
    
        no_fb, fb_msg = self.get_feedback(model, feedback_template, feedback_type, llm_response)
    
        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            objects, initial, goal, _ = task_builder.extract_task(model, problem_desc, prompt)

        return objects, initial, goal, fb_msg

    def objects_feedback(
        self, 
        model: LLM_Chat, 
        problem_desc: str,
        llm_response: str,
        feedback_template: str, 
        feedback_type: str="llm", 
        type_hierarchy: dict[str,str]=None, 
        predicates: list[Predicate]=None,
        objects: dict[str,str]=None
        ) -> tuple[dict[str,str], str]:
        """Makes LLM call using feedback prompt, then parses it into objects format"""
        
        model.reset_tokens()

        type_str = format_dict(type_hierarchy) if type_hierarchy else "No types provided."
        predicate_str = format_predicates(predicates) if predicates else "No predicates provided."
        objects_str = "\n".join([f"{obj} - {type}" for obj, type in objects.items()]) if objects else "No objects provided."
        
        feedback_template = feedback_template.replace('{problem_desc}', problem_desc)
        feedback_template = feedback_template.replace('{llm_response}', llm_response)
        feedback_template = feedback_template.replace('{types}', type_str)
        feedback_template = feedback_template.replace('{predicates}', predicate_str)
        feedback_template = feedback_template.replace('{objects}', objects_str)

        no_fb, fb_msg = self.get_feedback(model, feedback_template, feedback_type, llm_response)
    
        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            objects, llm_response = task_builder.extract_objects(model, problem_desc, prompt)

        return objects, llm_response

    def initial_state_feedback(
        self, 
        model: LLM_Chat, 
        problem_desc: str,
        llm_response: str,
        feedback_template: str, 
        feedback_type: str="llm", 
        type_hierarchy: dict[str,str]=None, 
        predicates: list[Predicate]=None,
        objects: dict[str,str]=None,
        initial: list[dict[str,str]]=None
        ) -> tuple[list[dict[str,str]], str]:
        """Makes LLM call using feedback prompt, then parses it into initial states format"""
        
        model.reset_tokens()

        type_str = format_dict(type_hierarchy) if type_hierarchy else "No types provided."
        predicate_str = format_predicates(predicates) if predicates else "No predicates provided."
        objects_str = "\n".join([f"{obj} - {type}" for obj, type in objects.items()]) if objects else "No objects provided."
        initial_state_str = task_builder.format_initial(initial) if initial else "No initial state provided."
        
        feedback_template = feedback_template.replace('{problem_desc}', problem_desc)
        feedback_template = feedback_template.replace('{llm_response}', llm_response)
        feedback_template = feedback_template.replace('{types}', type_str)
        feedback_template = feedback_template.replace('{predicates}', predicate_str)
        feedback_template = feedback_template.replace('{objects}', objects_str)
        feedback_template = feedback_template.replace('{initial_state}', initial_state_str)

        no_fb, fb_msg = self.get_feedback(model, feedback_template, feedback_type, llm_response)
    
        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            initial, llm_response = task_builder.extract_initial_state(model, problem_desc, prompt)

        return initial, llm_response

    def goal_state_feedback(
        self, 
        model: LLM_Chat, 
        problem_desc: str,
        llm_response: str,
        feedback_template: str, 
        feedback_type: str="llm", 
        type_hierarchy: dict[str,str]=None, 
        predicates: list[Predicate]=None,
        objects: dict[str,str]=None,
        initial: list[dict[str,str]]=None,
        goal: list[dict[str,str]]=None
        ) -> tuple[list[dict[str,str]], str]:
        """Makes LLM call using feedback prompt, then parses it into goal states format"""
        
        model.reset_tokens()

        type_str = format_dict(type_hierarchy) if type_hierarchy else "No types provided."
        predicate_str = format_predicates(predicates) if predicates else "No predicates provided."
        objects_str = "\n".join([f"{obj} - {type}" for obj, type in objects.items()]) if objects else "No objects provided."
        initial_state_str = task_builder.format_initial(initial) if initial else "No initial state provided."
        goal_state_str = task_builder.format_goal(goal) if goal else "No goal state provided."
        
        feedback_template = feedback_template.replace('{problem_desc}', problem_desc)
        feedback_template = feedback_template.replace('{llm_response}', llm_response)
        feedback_template = feedback_template.replace('{types}', type_str)
        feedback_template = feedback_template.replace('{predicates}', predicate_str)
        feedback_template = feedback_template.replace('{objects}', objects_str)
        feedback_template = feedback_template.replace('{initial_state}', initial_state_str)
        feedback_template = feedback_template.replace('{initial_state}', goal_state_str)

        no_fb, fb_msg = self.get_feedback(model, feedback_template, feedback_type, llm_response)
    
        if not no_fb:
            prompt = (
                f"\n\nYou now are revising your answer using feedback. Here is the feedback you outputted:\n{fb_msg}"
                f"\n\nFollow the same syntax format as the original output in your answer:\n{llm_response}"
            )

            goal, llm_response = task_builder.extract_goal_state(model, problem_desc, prompt)

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
            return "no feedback"  # No feedback
        
        return resp

if __name__ == "__main__":
    pass