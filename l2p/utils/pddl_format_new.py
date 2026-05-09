import re

from l2p.utils.pddl_types_new import (
    LogicalCondition, PDDLType, Constant, Predicate, Function, 
    Parameter, Action, ActionPrecondition, ActionEffect,
    DurativeAction, DomainDetails, Requirement
)

from l2p.utils.pddl_format import indent

def format_logic(cond: LogicalCondition) -> str:
    """Recursively unpacks a LogicalCondition (str or dict) into a PDDL string."""
    if isinstance(cond, str):
        return cond
    
    if "operator" in cond:
        op = cond["operator"]
        if op == "not":
            return f"(not {format_logic(cond['condition'])})"
        elif op in ["and", "or"]:
            return f"({op} {' '.join(format_logic(c) for c in cond['conditions'])})"
        elif op == "imply":
            ant = cond["antecedent"]
            cons = cond["consequent"]
            ant_str = format_condition_block(ant)
            cons_str = format_condition_block(cons)
            return f"(imply {ant_str} {cons_str})"
        # Support for PDDL 3 constraints
        elif op in ["always", "sometime", "at-most-once"]:
            return f"({op} {format_logic(cond['condition'])})"
        
    elif "quantifier" in cond:
        q = cond["quantifier"]
        params = " ".join([f"{p['variable']} - {p['type']}" for p in cond["parameters"]])
        conds_str = format_condition_block(cond["conditions"])
        return f"({q} ({params}) {conds_str})"
        
    return ""

def format_condition_block(conds: list[LogicalCondition]) -> str:
    """Wraps a list of logical conditions in an (and ...) block securely."""
    if not conds:
        return ""
    if len(conds) == 1:
        return format_logic(conds[0])
    return f"(and {' '.join(format_logic(c) for c in conds)})"

def format_effect_block(eff: ActionEffect) -> str:
    """Converts an ActionEffect (or dict from ConditionalEffect) into a PDDL effect string."""
    add = eff.add if hasattr(eff, 'add') else eff.get('add', [])
    delete = eff.delete if hasattr(eff, 'delete') else eff.get('delete', [])
    numeric = eff.numeric if hasattr(eff, 'numeric') else eff.get('numeric', [])
    conditional = eff.conditional if hasattr(eff, 'conditional') else eff.get('conditional', [])
    
    parts = []
    parts.extend([format_logic(c) for c in add])
    parts.extend([f"(not {format_logic(c)})" for c in delete])
    parts.extend([format_logic(c) for c in numeric])
    
    for c in conditional:
        cond_list = c.condition if hasattr(c, 'condition') else c.get('condition', [])
        eff_obj = c.effect if hasattr(c, 'effect') else c.get('effect', {})
        cond_str = format_condition_block(cond_list)
        eff_str = format_effect_block(eff_obj)
        parts.append(f"(when {cond_str} {eff_str})")
        
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"(and {' '.join(parts)})"

def format_requirements(reqs: list[Requirement]) -> str:
    return " ".join([f"{r.name}" for r in reqs])


def format_types(types: list[PDDLType]) -> str:
    return "\n".join([f"{t.name} - {t.parent}" for t in types])

def format_constants(constants: list[Constant]) -> str:
    return "\n".join([f"{c.name} - {c.type}" for c in constants])

def format_predicates(predicates: list[Predicate]) -> str:
    return "\n".join([
        f"({p.name} {' '.join([f'{param.variable} - {param.type}' for param in p.params])})" 
        for p in predicates
    ])

def format_functions(functions: list[Function]) -> str:
    return "\n".join([
        f"({f.name} {' '.join([f'{param.variable} - {param.type}' for param in f.params])})" 
        for f in functions
    ])

def format_params(params: list[Parameter]) -> str:
    return " ".join([f"{p.variable} - {p.type}" for p in params])

def format_action(action: Action) -> str:
    """Formats a single standard Action into a PDDL string."""
    param_str = format_params(params=action.params)
    pre_str = format_condition_block(conds=action.preconditions.conditions)
    eff_str = format_effect_block(eff=action.effects)
    
    desc = f"(:action {action.name}\n"
    desc += f"  :parameters ({param_str})\n"
    if pre_str:
        desc += f"  :precondition {pre_str}\n"
    if eff_str:
        desc += f"  :effect {eff_str}\n"
    desc += ")"
    return desc

def format_actions(actions: list[Action]) -> str:
    return "\n\n".join([format_action(a) for a in actions])


def format_durative_action(d_act: DurativeAction) -> str:
    """Formats a single DurativeAction into a PDDL string."""
    param_str = format_params(params=d_act.params)
    
    dur_str = ""
    if d_act.duration:
        dur_str = d_act.duration[0] if len(d_act.duration) == 1 else f"(and {' '.join(d_act.duration)})"
    
    cond_parts = []
    if d_act.conditions.at_start:
        cond_parts.append(f"(at start {format_condition_block(d_act.conditions.at_start)})")
    if d_act.conditions.over_all:
        cond_parts.append(f"(over all {format_condition_block(d_act.conditions.over_all)})")
    if d_act.conditions.at_end:
        cond_parts.append(f"(at end {format_condition_block(d_act.conditions.at_end)})")
    cond_str = f"(and {' '.join(cond_parts)})" if len(cond_parts) > 1 else (cond_parts[0] if cond_parts else "")

    eff_parts = []
    if d_act.effects.at_start:
        if start_eff := format_effect_block(d_act.effects.at_start):
            eff_parts.append(f"(at start {start_eff})")
    if d_act.effects.at_end:
        if end_eff := format_effect_block(d_act.effects.at_end):
            eff_parts.append(f"(at end {end_eff})")
    if d_act.effects.continuous:
        if cont_eff := format_condition_block(d_act.effects.continuous):
            eff_parts.append(cont_eff)
    eff_str = f"(and {' '.join(eff_parts)})" if len(eff_parts) > 1 else (eff_parts[0] if eff_parts else "")

    desc = f"(:durative-action {d_act.name}\n"
    desc += f"  :parameters ({param_str})\n"
    if dur_str:
        desc += f"  :duration {dur_str}\n"
    if cond_str:
        desc += f"  :condition {cond_str}\n"
    if eff_str:
        desc += f"  :effect {eff_str}\n"
    desc += ")"
    return desc

def format_durative_actions(d_actions: list[DurativeAction]) -> str:
    return "\n\n".join([format_durative_action(d) for d in d_actions])

def generate_domain(domain_details: DomainDetails) -> str:
    """
    Assembles all formatted components into a complete PDDL domain string.
    """
    requirements = domain_details.requirements
    # if not requirements:
    #     requirements = generate_requirements(
    #         types=domain_details.types, 
    #         functions=domain_details.functions, 
    #         actions=domain_details.actions
    #     )

    desc = f"(define (domain {domain_details.name})\n"
    
    if requirements:
        reqs_str = format_requirements(reqs=requirements)
        desc += indent(string=f"(:requirements\n   {reqs_str})", level=1)
        
    if domain_details.types:
        types_str = format_types(domain_details.types)
        desc += f"\n\n   (:types \n{indent(string=types_str, level=2)}\n   )"

    if domain_details.constants:
        const_str = format_constants(domain_details.constants)
        desc += f"\n\n   (:constants \n{indent(string=const_str, level=2)}\n   )"

    if not domain_details.predicates:
        print("[WARNING]: Domain has no predicates.")
    else:
        pred_str = format_predicates(domain_details.predicates)
        desc += f"\n\n   (:predicates \n{indent(string=pred_str, level=2)}\n   )"

    if domain_details.functions:
        func_str = format_functions(domain_details.functions)
        desc += f"\n\n   (:functions \n{indent(string=func_str, level=2)}\n   )"

    if not domain_details.actions and not domain_details.durative_actions:
        print("[WARNING]: Domain has no actions.")
    else:
        # Join all standard actions
        if domain_details.actions:
            actions_str = "\n\n".join([format_action(a) for a in domain_details.actions])
            desc += f"\n\n{indent(string=actions_str, level=1)}"
            
        # Join all durative actions
        if domain_details.durative_actions:
            d_actions_str = "\n\n".join([format_durative_action(d) for d in domain_details.durative_actions])
            desc += f"\n\n{indent(string=d_actions_str, level=1)}"

    desc += "\n)"
    desc = desc.replace("AND", "and").replace("OR", "or")
    return desc



# ---- HELPER FUNCTIONS ----


def indent(string: str, level: int = 2):
    """Indent string helper function to format PDDL domain/task"""
    return "   " * level + string.replace("\n", f"\n{'   ' * level}")


def remove_comments(text: str, comment_prefixes=[";", "#", "//"]) -> str:
    """Remove comments from text using multiple prefix styles."""

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped_line = line
        for prefix in comment_prefixes:
            if prefix in stripped_line:
                # only remove comment if prefix is not inside quotes or code
                stripped_line = stripped_line.split(prefix, 1)[0]
        cleaned_lines.append(stripped_line.rstrip())

    # remove blank lines and normalize whitespace
    cleaned = "\n".join(line for line in cleaned_lines if line.strip())
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)  # collapse multiple newlines

    return cleaned

if __name__ == "__main__":

    raw_data = {
        "name": "type2",
        "parent": "type1",
        "desc": "string (optional)"
    }

    PDDLtype = PDDLType(**raw_data)

    # 3. Now pass the actual object to the function
    output = format_types(types=[PDDLtype])
    
    print(output)