"""
This module contains collection of functions for formatting Python PDDL components into PDDL format strings.
"""

import re
from typing import List
from l2p.utils.pddl_types import *

# ---- DOMAIN ----
def format_requirements(reqs: list[Requirement]) -> str:
    return " ".join([f"{r.name}" for r in reqs])

def format_types(types: list[PDDLType]) -> str:
    sorted_types = sorted(types, key=lambda t: natural_sort_key(t.name))
    return "\n".join([f"{t.name} - {t.parent}" for t in sorted_types])

def format_constants(constants: list[Constant]) -> str:
    sorted_constants = sorted(constants, key=lambda c: natural_sort_key(c.name))
    return "\n".join([f"{c.name} - {c.type}" for c in sorted_constants])

def format_predicates(predicates: list[Predicate]) -> str:
    sorted_predicates = sorted(predicates, key=lambda p: natural_sort_key(p.name))
    return "\n".join([
        f"({p.name} {' '.join([f'{param.variable} - {param.type}' for param in p.params])})" 
        for p in sorted_predicates
    ])

def format_functions(functions: list[Function]) -> str:
    sorted_functions = sorted(functions, key=lambda f: natural_sort_key(f.name))
    return "\n".join([
        f"({f.name} {' '.join([f'{param.variable} - {param.type}' for param in f.params])})" 
        for f in sorted_functions
    ])

def format_constraints(constraints: list[Constraint]) -> str:
    parts = [format_logic(c.condition) for c in constraints]
    if len(parts) == 1:
        return parts[0]
    return f"(and {' '.join(parts)})"


def format_params(params: List[Parameter]) -> str:
    grouped_params = {}
    for p in params:
        # normalize empty types to "object"
        param_type = p.type.lower() if p.type else "object"
        
        if param_type not in grouped_params:
            grouped_params[param_type] = []
        grouped_params[param_type].append(p.variable)
        
    parts = []
    regular_types = [t for t in grouped_params.keys() if t != "object"]
    
    # sort the type names alphanumerically (e.g., 'block' before 'type')
    regular_types.sort(key=natural_sort_key)
    for param_type in regular_types:
        # sort the variables within this type group
        sorted_vars = sorted(grouped_params[param_type], key=natural_sort_key)
        var_str = " ".join(sorted_vars)
        parts.append(f"{var_str} - {param_type}")
            
    # handle 'object' variables last
    if "object" in grouped_params:
        sorted_obj_vars = sorted(grouped_params["object"], key=natural_sort_key)
        object_vars = " ".join(sorted_obj_vars)
        parts.append(object_vars)
            
    return " ".join(parts)


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


def format_derived_predicate(dp: DerivedPredicate) -> str:
    """Formats a single DerivedPredicate into a PDDL string."""
    param_str = format_params(params=dp.params)
    cond_str = format_logic(cond=dp.condition)
    
    desc = f"(:derived ({dp.name} {param_str})\n"
    desc += f"  {cond_str}\n"
    desc += ")"
    return desc

def format_derived_predicates(d_preds: list[DerivedPredicate]) -> str:
    return "\n\n".join([format_derived_predicate(dp) for dp in d_preds])


def format_event(event: Event) -> str:
    """Formats a single PDDL+ Event into a PDDL string."""
    param_str = format_params(params=event.params)
    pre_str = format_condition_block(conds=event.preconditions.conditions)
    eff_str = format_effect_block(eff=event.effects)
    
    desc = f"(:event {event.name}\n"
    desc += f"  :parameters ({param_str})\n"
    if pre_str:
        desc += f"  :precondition {pre_str}\n"
    if eff_str:
        desc += f"  :effect {eff_str}\n"
    desc += ")"
    return desc

def format_events(events: list[Event]) -> str:
    return "\n\n".join([format_event(e) for e in events])


def format_process(process: Process) -> str:
    """Formats a single PDDL+ Process into a PDDL string."""
    param_str = format_params(params=process.params)
    pre_str = format_condition_block(conds=process.preconditions.conditions)
    eff_str = format_effect_block(eff=process.effects)
    
    desc = f"(:process {process.name}\n"
    desc += f"  :parameters ({param_str})\n"
    if pre_str:
        desc += f"  :precondition {pre_str}\n"
    if eff_str:
        desc += f"  :effect {eff_str}\n"
    desc += ")"
    return desc

def format_processes(processes: list[Process]) -> str:
    return "\n\n".join([format_process(p) for p in processes])


# ---- PROBLEM ----
def format_objects(objects: list[PDDLObject]) -> str:
    grouped_objs = {}
    for obj in objects:
        obj_type = obj.type.lower() if obj.type else "object"
        if obj_type not in grouped_objs:
            grouped_objs[obj_type] = []
        grouped_objs[obj_type].append(obj.name)
        
    parts = []
    regular_types = [t for t in grouped_objs.keys() if t != "object"]
    regular_types.sort(key=natural_sort_key)
    
    for obj_type in regular_types:
        sorted_vars = sorted(grouped_objs[obj_type], key=natural_sort_key)
        var_str = " ".join(sorted_vars)
        parts.append(f"{var_str} - {obj_type}")
            
    if "object" in grouped_objs:
        sorted_obj_vars = sorted(grouped_objs["object"], key=natural_sort_key)
        object_vars = " ".join(sorted_obj_vars)
        parts.append(object_vars)
            
    return "\n".join(parts)

def format_initial_state(init: InitialState) -> str:
    """Formats standard initial facts and Timed Initial Literals (TILs/TIFs)."""
    parts = []
    
    # format standard facts
    for fact in init.facts:
        parts.append(format_logic(fact))
        
    # format timed facts securely
    for tf in init.timed_facts:
        fact_str = format_logic(tf.fact)
        parts.append(f"(at {tf.time} {fact_str})")
        
    return "\n".join(parts)

def format_goal_states(goals: GoalState) -> str:
    """Formats goal conditions. Implicitly wraps multiple conditions in an (and ...) block."""
    return format_condition_block(goals.conditions)

def format_metric(metric: Metric) -> str:
    """Formats a PDDL Plan Metric (minimize / maximize)."""
    return f"(:metric {metric.optimization} {metric.expression})"

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

def natural_sort_key(s: str) -> list:
    """Splits strings into text and numbers to ensure ?b2 comes before ?b12"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]