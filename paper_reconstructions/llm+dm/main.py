

"""
Paper: "Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning" Guan et al. (2023)
Source code: https://github.com/GuanSuns/LLMs-World-Models-for-Planning
Run: python3 -m paper_reconstructions.llm+dm.main

Assumes the following:
    1. NL descriptions of all the actions
    2. A description of the domain
    3. Information of the object types and hierarchy - fixed set of object types specified in prompt
    
This paper contains 2 main module components (Step 1+2 in LLM+DM):
    1. `construct_action_models.py` - in this case we are only focusing on this!
    2. `correct_action_models.py` - excluded in this program
    
Changes made:
    > structure output from LLM to enable L2P type extractions
    > using L2P `extract_pddl_action` function in ABA algorithm
    > syntax validator + error message changes
"""

import os
from copy import deepcopy
from l2p import *

DOMAINS = [
    "household",
    "logistics",
    "tyreworld",
]

UNSUPPORTED_KEYWORDS = [
    "forall", 
    "when", 
    "exists", 
    "implies"
]

def get_action_prompt(prompt_template: str, action_desc: str):
    """Creates prompt for specific action."""
    
    action_desc_prompt = action_desc['desc']
    for i in action_desc['extra_info']:
        action_desc_prompt += ' ' + i
    
    full_prompt = str(prompt_template) + ' ' + action_desc_prompt
    
    return full_prompt, action_desc_prompt


def get_predicate_prompt(predicates):
    """Creates prompt for list of available predicates generated so far."""
    
    predicate_prompt = 'You can create and define new predicates, but you may also reuse the following predicates:'
    if len(predicates) == 0:
        predicate_prompt += '\nNo predicate has been defined yet'
    else:
        predicate_prompt += "\n".join([f"- {pred['clean']}" for pred in predicates])
    return predicate_prompt


def get_types(hierarchy_requirements):
    """Creates proper dictionary types (for L2P) from JSON format."""
    
    types = {
        name: description
        for name, description in hierarchy_requirements["hierarchy"].items()
        if name
    }
    return types


def construct_action(
    model: LLM,
    act_pred_prompt: str,
    action_name: str,
    predicates: list[Predicate],
    types: dict[str,str],
    max_iter: int = 8,
    syntax_validator: bool = True
    ):
    """
    This is the inner function of the overall `Action-by-action` algorithm. Specifically,
    this function generates a single action from the list of actions and new predicates created.
    Process looping until it abides the custom syntax validation check.
    
    Args:
        - model (LLM): the large language model to be inferenced
        - act_pred_prompt (str): contains information of action and format creation passed to LLM
        - action_name (str): current action to be generated
        - predicates (list[Predicate]): current list of predicates generated
        - types (dict[str,str]): domain types - used for validation checks
        - max_iter (int): max attempts of generating action (defaults at 8)
        - syntax_validator (bool): flag if syntax checker is on (defaults to True)
        
    Returns:
        - action (Action): action data type containing parameters, preconditions, and effects
        - predicates (list[Predicate]): list of updated predicates
        - llm_response (str): original LLM output
    """

    # better format for LLM to interpret
    predicate_str = "\n".join([f"- {pred['clean']}" for pred in predicates])
    
    # syntax validator check
    if syntax_validator: validator = SyntaxValidator()
    else: validator = None

    no_syntax_error = False
    i_iter = 0
    
    # generate single action in loop, loops until no syntax error or max iters reach
    while not no_syntax_error and i_iter < max_iter:
        i_iter += 1
        print(f'[INFO] generating PDDL of action: `{action_name}`')
        try:
                # L2P usage for extracting actions and predicates
                action, new_predicates, llm_response, validation_info = (
                    domain_builder.extract_pddl_action(
                        model=model,
                        domain_desc="",
                        prompt_template=act_pred_prompt,
                        action_name=action_name,
                        predicates=predicates,
                        types=types,
                        syntax_validator=validator,
                    )
                )

                # retrieve validation check and error message
                no_error, error_msg = validation_info

        except Exception as e:
            no_error = False
            error_msg = str(e)
            
        # if error exists, swap templates and return feedback message
        if not no_error:
            with open("paper_reconstructions/llm+dm/domains/error_prompt.txt") as f:
                error_template = f.read().strip()
            error_prompt = error_template.replace("{action_name}", action_name)
            error_prompt = error_prompt.replace("{predicates}", predicate_str)
            error_prompt = error_prompt.replace("{error_msg}", error_msg)
            error_prompt = error_prompt.replace("{llm_response}", llm_response)
            
            act_pred_prompt = error_prompt
        
        # break the loop if no syntax error was made
        else:
            no_syntax_error = True
            
    # if max attempts reached and there are still errors, print out error on action.
    if not no_syntax_error:
        print(f'[WARNING] syntax error remaining in the action model: {action_name}')
        
    predicates.extend(new_predicates) # extend the predicate list
    
    return action, predicates, llm_response



def run_llm_dm(
    model: LLM,
    domain: str = "household",
    max_iter: int = 2,
    max_attempts: int = 8
    ):
    """
    This is the main function for `construct_action_models.py` component of LLM+DM paper. Specifically, it generates
    actions (params, preconditions, effects) and predicates to create an overall PDDL domain file.
    
    Args:
        - model (LLM): the large language model to be inferenced
        - domain (str): choice of domain to task (defaults to `household`)
        - max_iter: outer loop iteration; # of overall action list resets (defaults to 2)
        - max_attempts: # of attempts to generate a single actions properly (defaults to 8)
    """
    
    # load in assumptions
    prompt_template = load_file("paper_reconstructions/llm+dm/domains/pddl_prompt.txt")
    domain_desc_str = load_file(f"paper_reconstructions/llm+dm/domains/{domain}/domain_desc.txt")
    
    if '{domain_desc}' in prompt_template:
        prompt_template = prompt_template.replace('{domain_desc}', domain_desc_str)
    
    action_model = load_file(f"paper_reconstructions/llm+dm/domains/{domain}/action_model.json")
    hierarchy_reqs = load_file(f"paper_reconstructions/llm+dm/domains/{domain}/hierarchy_requirements.json")
    
    reqs = [":" + r for r in hierarchy_reqs['requirements']]
    types = format_types(get_types(hierarchy_reqs))
    
    actions = list(action_model.keys())
    action_list = list()
    predicates = list()

    # initialize result folder
    result_log_dir = f"paper_reconstructions/llm+dm/results/{domain}"
    os.makedirs(result_log_dir, exist_ok=True)
    
    """
    Action-by-action algorithm: iteratively generates an action model (parameters, precondition, effects) one at a time. At the same time,
        it is generating new predicates if needed and is added to a dynamic list. At the end of the iterations, it is ran again once more to
        create the action models agains, but with using the new predicate list. This algorithm can iterative as many times as needed until no
        new predicates are added to the list. This is an action model refinement algorithm, that refines itself by a growing predicate list.
    """
    
    # outer loop that resets all action creation to be conditioned on updated predicate list
    for i_iter in range(max_iter):
        readable_results = '' # for logging purposes
        prev_predicate_list = deepcopy(predicates) # copy previous predicate list
        action_list = []
        
        # inner loop that generates a single action along with its predicates
        for _, action in enumerate(actions):
            
            # retrieve prompt for specific action
            action_prompt, _ = get_action_prompt(prompt_template, action_model[action])
            readable_results += '\n' * 2 + '#' * 20 + '\n' + f'Action: {action}\n' + '#' * 20 + '\n'
            
            # retrieve prompt for current predicate list
            predicate_prompt = get_predicate_prompt(predicates)
            readable_results += '-' * 20
            readable_results += f'\n{predicate_prompt}\n' + '-' * 20

            # assemble template
            action_predicate_prompt = f'{action_prompt}\n\n{predicate_prompt}'
            action_predicate_prompt += '\n\nParameters:'
            
            # create single action + corresponding predicates
            action, predicates, llm_response = construct_action(
                model, action_predicate_prompt, action, predicates, types, max_attempts, True)
            
            # add action to current list + remove any redundant predicates
            action_list.append(action)
            predicates = prune_predicates(predicates, action_list)
            
            readable_results += '\n' + '-' * 10 + '-' * 10 + '\n'
            readable_results += llm_response + '\n'

        # record log results into separate file of current iteration
        readable_results += '\n' + '-' * 10 + '-' * 10 + '\n'
        readable_results += 'Extracted predicates:\n'
        for i, p in enumerate(predicates):
            readable_results += f'\n{i + 1}. {p["raw"]}'
            
        with open(os.path.join(result_log_dir, f'{engine}_0_{i_iter}.txt'), 'w') as f:
            f.write(readable_results)
            
        gen_done = False
        if len(prev_predicate_list) == len(predicates):
            print(f'[INFO] iter {i_iter} | no new predicate has been defined, will terminate the process')
            gen_done = True

        if gen_done:
            break
        
        
    # format components for PDDL generation
    predicate_str = "\n".join(
            [pred["clean"].replace(":", " ; ") for pred in predicates]
        )
        
    pruned_types = {
        name: description
        for name, description in types.items()
        if name not in UNSUPPORTED_KEYWORDS
    }  # remove unsupported words
    types_str = "\n".join(pruned_types)
        
    # generate PDDL format
    pddl_domain = domain_builder.generate_domain(
            domain=domain,
            requirements=reqs,
            types=types_str,
            predicates=predicate_str,
            actions=action_list,
        )
    
    # write in PDDL domain file into results dir
    domain_file = f"{result_log_dir}/domain.pddl"
    with open(domain_file, "w") as f:
        f.write(pddl_domain)
        
    

if __name__ == "__main__":
    
    engine = "gpt-4o-mini"
    api_key = os.environ.get("OPENAI_API_KEY")
    gpt_model = OPENAI(model=engine, api_key=api_key)
    domain_builder = DomainBuilder()

    # run LLM+DM method on all domains
    run_llm_dm(model=gpt_model, domain="logistics")
    run_llm_dm(model=gpt_model, domain="household")
    run_llm_dm(model=gpt_model, domain="tyreworld")