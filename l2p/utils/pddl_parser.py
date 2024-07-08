from collections import OrderedDict
from copy import deepcopy
from .pddl_types import Action, Predicate
import re, ast

def combine_blocks(heading_str: str):
    """Combine the inside of blocks from the heading string into a single string."""

    possible_blocks = heading_str.split("```")
    blocks = [possible_blocks[i] for i in range(1, len(possible_blocks), 2)] # Get the text between the ```s, every other one
    combined = "\n".join(blocks) # Join the blocks together
    return combined.replace("\n\n", "\n").strip() # Remove leading/trailing whitespace and internal empty lines

def parse_params(llm_output, include_internal=False):
    params_info = OrderedDict()
    params_heading = llm_output.split('Parameters')[1].strip().split('##')[0]
    params_str = combine_blocks(params_heading)
    for line in params_str.split('\n'):
        if line.strip() == '' or ('.' not in line and not line.strip().startswith('-')):
            print(f"[WARNING] checking param object types - empty line or not a valid line: '{line}'")
            continue
        if not (line.split('.')[0].strip().isdigit() or line.startswith('-')):
            print(f"[WARNING] checking param object types - not a valid line: '{line}'")
            continue
        try:
            p_info = [e for e in line.split(':')[0].split(' ') if e != '']
            param_name, param_type = p_info[1].strip(" `"), p_info[3].strip(" `")
            params_info[param_name] = param_type
        except Exception:
            print(f'[WARNING] checking param object types - fail to parse: {line}')
            continue
    if include_internal:
        precondition_heading = llm_output.split('Preconditions')[1].strip().split('##')[0]
        preconditions_str = combine_blocks(precondition_heading) # Should just be one, but this extracts it easily
        if "forall" in preconditions_str:
            forall_matches = re.findall(r'forall\s*\((.*?)\)', preconditions_str)
            forall_contents = [match.strip() for match in forall_matches]
            for content in forall_contents:
                sub_params = re.findall(r'\?[a-zA-Z0-9]+\s*-\s*[a-zA-Z0-9]+', content)
                for sub_param in sub_params:
                    param_name, param_type = [e.strip() for e in sub_param.split('-')]
                    params_info[param_name] = param_type
        if "exists" in preconditions_str:
            exists_matches = re.findall(r'exists\s*\((.*?)\)', preconditions_str)
            exists_contents = [match.strip() for match in exists_matches]
            for content in exists_contents:
                sub_params = re.findall(r'\?[a-zA-Z0-9]+\s*-\s*[a-zA-Z0-9]+', content)
                for sub_param in sub_params:
                    param_name, param_type = [e.strip() for e in sub_param.split('-')]
                    params_info[param_name] = param_type

    return params_info

def parse_new_predicates(llm_output) -> list[Predicate]:
    new_predicates = list()
    try:
        predicate_heading = llm_output.split('New Predicates\n')[1].strip().split('##')[0]
    except:
        raise Exception("Could not find the 'New Predicates' section in the output. Provide the entire response, including all headings even if some are unchanged.")
    predicate_output = combine_blocks(predicate_heading)
    #print(f'Parsing new predicates from: \n---\n{predicate_output}\n---\n', )
    for p_line in predicate_output.split('\n'):
        if ('.' not in p_line or not p_line.split('.')[0].strip().isdigit()) and not (p_line.startswith('-') or p_line.startswith('(')):
            if len(p_line.strip()) > 0:
                print(f'[WARNING] unable to parse the line: "{p_line}"')
            continue
        predicate_info = p_line.split(': ')[0].strip(" 1234567890.(-)`").split(' ')
        predicate_name = predicate_info[0]
        predicate_desc = p_line.split(': ')[1].strip() if ": " in p_line else ''

        # get the predicate type info
        if len(predicate_info) > 1:
            predicate_type_info = predicate_info[1:]
            predicate_type_info = [l.strip(" ()`") for l in predicate_type_info if l.strip(" ()`")]
        else:
            predicate_type_info = []
        params = OrderedDict()
        next_is_type = False
        upcoming_params = []
        for p in predicate_type_info:
            if next_is_type:
                if p.startswith('?'):
                    print(f"[WARNING] `{p}` is not a valid type for a variable, but it is being treated as one. Should be checked by syntax check later.")
                for up in upcoming_params:
                    params[up] = p
                next_is_type = False
                upcoming_params = []
            elif p == '-':
                next_is_type = True
            elif p.startswith('?'):
                upcoming_params.append(p) # the next type will be for this variable
            else:
                print(f"[WARNING] `{p}` is not corrrectly formatted. Assuming it's a variable name.")
                upcoming_params.append(f"?{p}")
        if next_is_type:
            print(f"[WARNING] The last type is not specified for `{p_line}`. Undefined are discarded.")
        if len(upcoming_params) > 0:
            print(f"[WARNING] The last {len(upcoming_params)} is not followed by a type name for {upcoming_params}. These are discarded")

        # generate a clean version of the predicate
        clean = f"({predicate_name} {' '.join([f'{k} - {v}' for k, v in params.items()])}): {predicate_desc}"

        # drop the index/dot
        p_line = p_line.strip(" 1234567890.-`") 
        new_predicates.append({
            'name': predicate_name, 
            'desc': predicate_desc, 
            'raw': p_line,
            'params': params,
            'clean': clean,
        })
    #print(f"Parsed {len(new_predicates)} new predicates: {[p['name'] for p in new_predicates]}", )
    return new_predicates


def parse_predicates(all_predicates):
    """
    This function assumes the predicate definitions adhere to PDDL grammar
    """
    all_predicates = deepcopy(all_predicates)
    for i, pred in enumerate(all_predicates):
        if 'params' in pred:
            continue
        pred_def = pred['raw'].split(': ')[0]
        pred_def = pred_def.strip(" ()`")  # drop any leading/strange formatting
        split_predicate = pred_def.split(' ')[1:]   # discard the predicate name
        split_predicate = [e for e in split_predicate if e != '']

        pred['params'] = OrderedDict()
        for j, p in enumerate(split_predicate):
            if j % 3 == 0:
                assert '?' in p, f'invalid predicate definition: {pred_def}'
                assert split_predicate[j+1] == '-', f'invalid predicate definition: {pred_def}'
                param_name, param_obj_type = p, split_predicate[j+2]
                pred['params'][param_name] = param_obj_type
    return all_predicates

def parse_action(llm_response: str, action_name: str) -> Action:
    """
    Parse an action from a given LLM output.

    Args:
        llm_response (str): The LLM output.
        action_name (str): The name of the action.

    Returns:
        Action: The parsed action.
    """
    parameters = parse_params(llm_response)
    try:
        preconditions = llm_response.split("Preconditions\n")[1].split("##")[0].split("```")[1].strip(" `\n")
    except:
        raise Exception("Could not find the 'Preconditions' section in the output. Provide the entire response, including all headings even if some are unchanged.")
    try:
        effects = llm_response.split("Effects\n")[1].split("##")[0].split("```")[1].strip(" `\n")
    except:
        raise Exception("Could not find the 'Effects' section in the output. Provide the entire response, including all headings even if some are unchanged.")
    return {"name": action_name, "parameters": parameters, "preconditions": preconditions, "effects": effects}


def prune_types(types: dict[str,str], predicates: list[Predicate], actions: list[Action]) -> dict[str,str]:
        """
        Prune types that are not used in any predicate or action.

        Args:
            types (list[str]): A list of types.
            predicates (list[Predicate]): A list of predicates.
            actions (list[Action]): A list of actions.

        Returns:
            list[str]: The pruned list of types.
        """

        used_types = {}
        for type in types:
            for pred in predicates:
                if type.split(' ')[0] in pred['params'].values():
                    used_types[type] = types[type]
                    break
            else:
                for action in actions:
                    if type.split(' ')[0] in action['parameters'].values():
                        used_types[type] = types[type]
                        break
                    if type.split(' ')[0] in action['preconditions'] or type.split(' ')[0] in action['effects']: # If the type is included in a "forall" or "exists" statement
                        used_types[type] = types[type]
                        break
        return used_types

def prune_predicates(predicates: list[Predicate], actions: list[Action]) -> list[Predicate]:
    """
    Remove predicates that are not used in any action.

    Args:
        predicates (list[Predicate]): A list of predicates.
        actions (list[Action]): A list of actions.

    Returns:
        list[Predicate]: The pruned list of predicates.
    """
    used_predicates = []
    seen_predicate_names = set()

    for pred in predicates:
        for action in actions:
            # Add a space or a ")" to avoid partial matches 
            names = [f"{pred['name']} ", f"{pred['name']})"]
            for name in names:
                if name in action['preconditions'] or name in action['effects']:
                    if pred['name'] not in seen_predicate_names:
                        used_predicates.append(pred)
                        seen_predicate_names.add(pred['name'])
                    break

    return used_predicates


def extract_types(type_hierarchy: dict[str,str]) -> dict[str,str]:
    def process_node(node, parent_type=None):
        current_type = list(node.keys())[0]
        description = node[current_type]
        parent_type = parent_type if parent_type else current_type

        name = f"{current_type} - {parent_type}" if current_type != parent_type else f"{current_type}"
        desc = f"; {description}"
        
        result[name] = desc

        for child in node.get("children", []):
            process_node(child, current_type)

    result = {}
    process_node(type_hierarchy)
    return result

def convert_to_dict(llm_response: str) -> dict[str,str]:
    
    dict_pattern = re.compile(r'{.*}', re.DOTALL) # regular expression to find the JSON-like dictionary structure
    match = dict_pattern.search(llm_response) # search for the pattern in the llm_response

    # safely evaluate the string to convert it into a Python dictionary
    if match:
        dict_str = match.group(0)
        try:
            dict = ast.literal_eval(dict_str)
            return dict
        except Exception as e:
            print(f"Error parsing dictionary: {e}")
            return None
    else:
        print("No dictionary found in the llm_response.")
        return None
    

def parse_objects(llm_response: str) -> dict[str, str]:
    """
    Extract objects from LLM response and returns dictionary string pairs object(name, type)
    Args:
        - llm_response (str):
        - types (dict[str,str]): WILL BE USED FOR CHECK ERROR RAISES
        - predicates (list[Predicate]): WILL BE USED FOR CHECK ERROR RAISES
    Returns:
        - dict[str,str]: objects
    """
    
    objects_head = extract_heading(llm_response, "Object Instances")
    objects_raw = combine_blocks(objects_head)
    objects_clean = clear_comments(objects_raw, comments=[':','//','#',';','(']) # Remove comments
    objects = {obj.split(" - ")[0].strip(" `"): obj.split(" - ")[1].strip(" `").lower() for obj in objects_clean.split("\n") if obj.strip()}

    # IMPLEMENT CHECKS (if objects are/are not in types/predicates)
    # objects_str = "\n".join([f"{obj} - {type}" for obj, type in objects.items()])

    return objects


def parse_initial(llm_response: str) -> str:
    """Extracts state (PDDL-init) from LLM response and returns it as a string"""
    state_head = extract_heading(llm_response, "State")
    state_raw = combine_blocks(state_head)
    state_clean = clear_comments(state_raw)

    states = []
    for line in state_clean.split("\n"):
        line = line.strip("- `()")
        if not line: # Skip empty lines
            continue
        name = line.split(" ")[0]
        if name == "not":
            neg = True
            name = line.split(" ")[1].strip("()") # Remove the `not` and the parentheses
            params = line.split(" ")[2:]
        else:
            neg = False
            params = line.split(" ")[1:] if len(line.split(" ")) > 1 else []
        states.append({"name": name, "params": params, "neg": neg})

    inner_str = [f"({state['name']} {' '.join(state['params'])})" for state in states] # The main part of each predicate
    full_str = [f"(not {inner})" if state["neg"] else inner for state, inner in zip(states, inner_str)] # Add the `not` if needed
    state_str = "\n".join(full_str) # Combine the states into a single string
    return state_str


def parse_goal(llm_response: str) -> str:
    """Extracts goal (PDDL-goal) from LLM response and returns it as a string"""
    goal_head = extract_heading(llm_response, "Goal")
    if goal_head.count("```") != 2:
        raise ValueError("Could not find exactly one block in the goal section of the LLM output. The goal has to be specified in a single block and as valid PDDL using the `and` and `not` operators. Likely this is caused by a too long response and limited context length. If so, try to shorten the message and exclude objects which aren't needed for the task.")
    goal_raw = goal_head.split("```")[1].strip() # Only a single block in the goal
    goal_clean = clear_comments(goal_raw)

    goal_pure = goal_clean.replace("and", "").replace("AND", "").replace("not", "").replace("NOT", "")
    goals = []
    for line in goal_pure.split("\n"):
        line = line.strip(" ()")
        if not line: # Skip empty lines
            continue
        name = line.split(" ")[0]
        params = line.split(" ")[1:] if len(line.split(" ")) > 1 else []
        goals.append({"name": name, "params": params})

    return goal_clean # Since the goal uses `and` and `not` recombining it is difficult 


def clear_comments(text: str, comments = [':','//','#',';']) -> str:
    """Remove comments from the text."""
    for comment in comments:
        text = "\n".join([line.split(comment)[0] for line in text.split("\n")])
    return text

def extract_heading(llm_output: str, heading: str):
    """Extract the text between the heading and the next second level heading in the LLM output."""
    if heading not in llm_output:
        print("#"*10, "LLM Output", "#"*10)
        print(llm_output)
        print("#"*30)
        raise ValueError(f"Could not find heading {heading} in the LLM output. Likely this is caused by a too long response and limited context length. If so, try to shorten the message and exclude objects which aren't needed for the task.")
    heading_str = llm_output.split(heading)[1].split("\n## ")[0].strip() # Get the text between the heading and the next heading
    return heading_str