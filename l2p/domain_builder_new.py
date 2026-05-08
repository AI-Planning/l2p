"""
PDDL Domain Formalization/Generation Functions

This module defines the `DomainBuilder` class and related utilities for constructing
PDDL domain specifications programmatically using structured Pydantic models.
"""

import time
from typing import Optional, Union, Any

# Assuming your new Pydantic definitions and parsers are in these modules
from l2p.utils.pddl_types_new import (
    DomainDetails, PDDLType, Constant, Predicate, Function, DerivedPredicate,
    Action, DurativeAction, Constraint, Event, Process
)
from l2p.utils.pddl_parser_new import parse_xml_tags, parse_element

from .llm import BaseLLM, require_llm
from .utils import SyntaxValidator

def indent(string: str, level: int = 2):
    """Indent string helper function to format PDDL domain/task"""
    return "   " * level + string.replace("\n", f"\n{'   ' * level}")


class DomainBuilder:
    def __init__(self, domain_details: Optional[DomainDetails] = None) -> None:
        """
        Initializes an L2P domain builder object using the new Pydantic DomainDetails root model.

        Args:
            domain_details (DomainDetails | None): A fully validated Pydantic model 
            containing the entire PDDL domain structure.
        """
        # If no domain_details is provided, initialize a fresh, empty DomainDetails model
        self.domain_details = domain_details or DomainDetails(name="temp_domain")


    # @require_llm
    # def formalize_domain(
    #     self,
    #     model: BaseLLM,
    #     domain_desc: str,
    #     domain_name: str,
    #     prompt_template: str,
    #     domain_details: Optional[DomainDetails] = None,
    #     syntax_validator: Optional[SyntaxValidator] = None,
    #     max_retries: int = 3,
    # ) -> tuple[str, str, tuple[bool, str]]:
    #     """
    #     Formalizes a complete PDDL domain using the LLM.

    #     Sends structured Domain details to the LLM for refinement. The LLM output must 
    #     be in a fenced code block: ```pddl ... ```.

    #     Args:
    #         model (BaseLLM): LLM to query
    #         domain_desc (str): Natural language description of the domain
    #         domain_name (str): The name of the domain
    #         prompt_template (str): Structured prompt template for domain formalization
    #         domain_details (DomainDetails | None): The Pydantic model containing the structure to refine
    #         syntax_validator: Syntax checker for generated domain, defaults to None
    #         max_retries: max # of retries if failure occurs, defaults to 3

    #     Returns:
    #         pddl_domain (str): the refined PDDL domain string
    #         llm_output (str): the raw string BaseLLM response
    #         validation_info (tuple[bool, str]): validation info
    #     """
        
    #     # Use provided details or fallback to self.domain_details
    #     details_to_use = domain_details or self.domain_details
    #     details_to_use.name = domain_name

    #     # Convert the Pydantic DomainDetails model back into a raw PDDL string representation
    #     # (Assuming you have a function or method that converts DomainDetails -> raw PDDL string)
    #     has_domain_info = len(details_to_use.types) > 0 or len(details_to_use.predicates) > 0 or len(details_to_use.actions) > 0
        
    #     if has_domain_info:
    #         domain_str = self.generate_domain(details_to_use)
    #         domain_section = (
    #             f"Review and refine the following PDDL domain - fix any syntax errors, add missing requirements, improve definitions, and ensure it follows PDDL conventions.\n\n"
    #             f"Please refine the following PDDL domain:\n\n"
    #             f"<domain>\n```pddl\n{domain_str}\n```\n</domain>"
    #         )
    #     else:
    #         domain_str = ""
    #         domain_section = ""

    #     prompt = (
    #         prompt_template
    #         .replace("{domain_desc}", domain_desc)
    #         .replace("{domain_str}", domain_str)
    #         .replace("{domain_section}", domain_section)
    #     )

    #     for attempt in range(max_retries):
    #         try:
    #             model.reset_tokens()
    #             llm_output = model.query(prompt=prompt)

    #             # 1. Try to extract the raw PDDL string from the LLM output
    #             pddl_domain = parse_domain(llm_output=llm_output)

    #             validation_info = (True, "All validations passed.")
                
    #             # 2. Run your external syntax validator on the raw string
    #             if syntax_validator:
    #                 for error_type in syntax_validator.error_types:
    #                     validator = getattr(syntax_validator, f"{error_type}", None)
    #                     if not callable(validator):
    #                         continue
                        
    #                     validation_info = validator(pddl_domain)
    #                     if not validation_info[0]:
    #                         # If syntax validation fails, return early so the caller can handle it
    #                         return pddl_domain, llm_output, validation_info

    #             return pddl_domain, llm_output, validation_info

    #         except ValueError as e:
    #             # Catch the highly specific ValueErrors raised by your new parse_domain function
    #             print(
    #                 f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
    #                 f"Sending feedback to LLM..."
    #             )
                
    #             # FEEDBACK LOOP: Append the error message to the prompt for the next try
    #             prompt += f"\n\n[SYSTEM ERROR IN PREVIOUS ATTEMPT]:\n{str(e)}\nPlease try again."
    #             time.sleep(2)
                
    #         except Exception as e:
    #             print(f"Unexpected error during attempt {attempt + 1}/{max_retries}: {e}. Retrying...")
    #             time.sleep(2)

    #     raise RuntimeError("Max retries exceeded. Failed to formalize domain.")
    

    def generate_domain(self, domain_details: DomainDetails) -> str:
        """
        Generates PDDL domain string from a Pydantic DomainDetails object.

        Args:
            domain_details (DomainDetails): The strictly validated PDDL domain structure.

        Returns:
            desc (str): PDDL domain in string format
        """

        # -------------------------------------------------------------------
        # NESTED HELPER FUNCTIONS FOR UNWRAPPING LOGICAL DICTIONARIES
        # -------------------------------------------------------------------
        def _parse_logic(cond: Any) -> str:
            """Recursively unpacks a LogicalCondition (str or dict) into a PDDL string."""
            if isinstance(cond, str):
                return cond
            
            if "operator" in cond:
                op = cond["operator"]
                if op == "not":
                    return f"(not {_parse_logic(cond['condition'])})"
                elif op in ["and", "or"]:
                    return f"({op} {' '.join(_parse_logic(c) for c in cond['conditions'])})"
                elif op == "imply":
                    ant = cond["antecedent"]
                    cons = cond["consequent"]
                    ant_str = f"(and {' '.join(_parse_logic(c) for c in ant)})" if len(ant) > 1 else _parse_logic(ant[0])
                    cons_str = f"(and {' '.join(_parse_logic(c) for c in cons)})" if len(cons) > 1 else _parse_logic(cons[0])
                    return f"(imply {ant_str} {cons_str})"
                # Basic support for PDDL 3 constraints if they leak into conditions
                elif op in ["always", "sometime", "at-most-once"]:
                    return f"({op} {_parse_logic(cond['condition'])})"
                
            elif "quantifier" in cond:
                q = cond["quantifier"]
                params = " ".join([f"{p['variable']} - {p['type']}" for p in cond["parameters"]])
                conds = cond["conditions"]
                conds_str = f"(and {' '.join(_parse_logic(c) for c in conds)})" if len(conds) > 1 else _parse_logic(conds[0])
                return f"({q} ({params}) {conds_str})"
                
            return ""

        def _build_and(conds: list) -> str:
            """Wraps a list of conditions in an (and ...) block if there is more than one."""
            if not conds:
                return ""
            if len(conds) == 1:
                return _parse_logic(conds[0])
            return f"(and {' '.join(_parse_logic(c) for c in conds)})"

        def _build_effect(eff: Any) -> str:
            """Converts an ActionEffect (or conditional dictionary) into a PDDL effect string."""
            # Handle both Pydantic ActionEffect objects and raw dictionaries (from ConditionalEffect)
            add = eff.add if hasattr(eff, 'add') else eff.get('add', [])
            delete = eff.delete if hasattr(eff, 'delete') else eff.get('delete', [])
            numeric = eff.numeric if hasattr(eff, 'numeric') else eff.get('numeric', [])
            conditional = eff.conditional if hasattr(eff, 'conditional') else eff.get('conditional', [])
            
            parts = []
            parts.extend([_parse_logic(c) for c in add])
            # Auto-wrap deletions in (not ...)
            parts.extend([f"(not {_parse_logic(c)})" for c in delete])
            parts.extend([_parse_logic(c) for c in numeric])
            
            for c in conditional:
                cond_str = _build_and(c.condition if hasattr(c, 'condition') else c.get('condition', []))
                eff_str = _build_effect(c.effect if hasattr(c, 'effect') else c.get('effect', {}))
                parts.append(f"(when {cond_str} {eff_str})")
                
            if not parts:
                return ""
            if len(parts) == 1:
                return parts[0]
            return f"(and {' '.join(parts)})"

        # -------------------------------------------------------------------
        # MAIN DOMAIN GENERATION
        # -------------------------------------------------------------------

        requirements = domain_details.requirements
        if not requirements:
            requirements = self.generate_requirements(
                types=domain_details.types, 
                functions=domain_details.functions, 
                actions=domain_details.actions
            )

        desc = f"(define (domain {domain_details.name})\n"
        
        if requirements:
            desc += indent(string=f"(:requirements\n   {' '.join(requirements)})", level=1)
            
        if domain_details.types:
            types_str = "\n".join([f"{t.name} - {t.parent}" for t in domain_details.types])
            desc += f"\n\n   (:types \n{indent(string=types_str, level=2)}\n   )"

        if domain_details.constants:
            const_str = "\n".join([f"{c.name} - {c.type}" for c in domain_details.constants])
            desc += f"\n\n   (:constants \n{indent(string=const_str, level=2)}\n   )"

        if not domain_details.predicates:
            print("[WARNING]: Domain has no predicates. This may cause planners to reject the domain or behave unexpectedly.")
        else:
            pred_str = "\n".join([f"({p.name} {' '.join([f'{param.variable} - {param.type}' for param in p.params])})" for p in domain_details.predicates])
            desc += f"\n\n   (:predicates \n{indent(string=pred_str, level=2)}\n   )"

        if domain_details.functions:
            func_str = "\n".join([f"({f.name} {' '.join([f'{param.variable} - {param.type}' for param in f.params])})" for f in domain_details.functions])
            desc += f"\n\n   (:functions \n{indent(string=func_str, level=2)}\n   )"

        if not domain_details.actions and not domain_details.durative_actions:
            print("[WARNING]: Domain has no actions.")
        else:
            # Format Standard Actions
            for action in domain_details.actions:
                param_str = " ".join([f"{p.variable} - {p.type}" for p in action.params])
                pre_str = _build_and(action.preconditions.conditions)
                eff_str = _build_effect(action.effects)
                
                desc += f"\n\n   (:action {action.name}\n"
                desc += f"      :parameters ({param_str})\n"
                if pre_str:
                    desc += f"      :precondition {pre_str}\n"
                if eff_str:
                    desc += f"      :effect {eff_str}\n"
                desc += "   )"
                
            # Format Durative Actions
            for d_act in domain_details.durative_actions:
                param_str = " ".join([f"{p.variable} - {p.type}" for p in d_act.params])
                
                # Duration
                dur_str = ""
                if d_act.duration:
                    dur_str = d_act.duration[0] if len(d_act.duration) == 1 else f"(and {' '.join(d_act.duration)})"
                
                # Conditions
                cond_parts = []
                if d_act.conditions.at_start:
                    cond_parts.append(f"(at start {_build_and(d_act.conditions.at_start)})")
                if d_act.conditions.over_all:
                    cond_parts.append(f"(over all {_build_and(d_act.conditions.over_all)})")
                if d_act.conditions.at_end:
                    cond_parts.append(f"(at end {_build_and(d_act.conditions.at_end)})")
                cond_str = f"(and {' '.join(cond_parts)})" if len(cond_parts) > 1 else (cond_parts[0] if cond_parts else "")

                # Effects
                eff_parts = []
                if d_act.effects.at_start:
                    if start_eff := _build_effect(d_act.effects.at_start):
                        eff_parts.append(f"(at start {start_eff})")
                if d_act.effects.at_end:
                    if end_eff := _build_effect(d_act.effects.at_end):
                        eff_parts.append(f"(at end {end_eff})")
                if d_act.effects.continuous:
                    if cont_eff := _build_and(d_act.effects.continuous):
                        eff_parts.append(cont_eff)
                eff_str = f"(and {' '.join(eff_parts)})" if len(eff_parts) > 1 else (eff_parts[0] if eff_parts else "")

                desc += f"\n\n   (:durative-action {d_act.name}\n"
                desc += f"      :parameters ({param_str})\n"
                if dur_str:
                    desc += f"      :duration {dur_str}\n"
                if cond_str:
                    desc += f"      :condition {cond_str}\n"
                if eff_str:
                    desc += f"      :effect {eff_str}\n"
                desc += "   )"

        desc += "\n)"
        desc = desc.replace("AND", "and").replace("OR", "or")
        
        return desc