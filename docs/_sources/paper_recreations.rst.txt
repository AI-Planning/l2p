Paper Recreations
=================
This library is a collection of tools for PDDL model generation extracted from natural language driven by large language models. This library is an expansion from the survey paper `"LLMs as Planning Formalizers: A Survey for Leveraging Large Language Models to Construct Automated Planning Models" <https://arxiv.org/abs/2503.18971v1>`_. 

The following are papers that have been reconstructed so far. This list will be updated in the future.

    + [x] `P+S` `[paper] <https://arxiv.org/abs/2205.05718>`__
    + [x] `LLM+P` `[paper] <https://arxiv.org/abs/2304.11477>`__
    + [x] `PROC2PDDL` `[paper] <https://arxiv.org/abs/2403.00092>`__
    + [x] `LLM+DM` `[paper] <https://arxiv.org/abs/2305.14909>`__
    + [x] `NL2PLAN` `[paper] <https://arxiv.org/abs/2405.04215>`__
    + [x] `TEXT2WORLD` `[paper] <https://arxiv.org/abs/2502.13092>`__
    + [x] `JS-PDDL` `[paper] <https://openreview.net/forum?id=VyTxXSPmbE&referrer=%5Bthe%20profile%20of%20Kelsey%20Sikes%5D(%2Fprofile%3Fid%3D~Kelsey_Sikes1)>`__

Papers that have been reconstructed so far and can be found in the `Github <https://github.com/AI-Planning/l2p>`_ repo. To see full list of current up-to-date papers in literature, please visit (see :doc:`paper_feed`). This list will be continuously updated.

----

Below is L2P code reconstruction of "action-by-action algorithm" from `"Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning" <https://arxiv.org/abs/2305.14909>`_:

.. code-block:: python
    :linenos:

    DOMAINS = [
    "household",
    "logistics",
    "tyreworld",
    ]

    def construct_action(
        model: BaseLLM,
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
            - model (BaseLLM): the large language model to be inferenced
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
        if syntax_validator: 
            validator = SyntaxValidator()
            validator.unsupported_keywords = []

            validator.error_types = [
                'validate_header', 'validate_duplicate_headers', 'validate_unsupported_keywords',
                'validate_params', 'validate_duplicate_predicates', 'validate_types_predicates',
                'validate_format_predicates', 'validate_usage_action'
                ]
            
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
                        domain_builder.formalize_pddl_action(
                            model=model,
                            domain_desc="",
                            prompt_template=act_pred_prompt,
                            action_name=action_name,
                            predicates=predicates,
                            types=types,
                            extract_new_preds=True,
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
        model: BaseLLM,
        domain: str = "household",
        max_iter: int = 2,
        max_attempts: int = 8
        ):
        """
        This is the main function for `construct_action_models.py` component of LLM+DM paper. Specifically, it generates
        actions (params, preconditions, effects) and predicates to create an overall PDDL domain file.
        
        Args:
            - model (BaseLLM): the large language model to be inferenced
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
        
        """
        Action-by-action algorithm: iteratively generates an action model (parameters, precondition, effects) one at a time. At the same time,
            it is generating new predicates if needed and is added to a dynamic list. At the end of the iterations, it is ran again once more to
            create the action models agains, but with using the new predicate list. This algorithm can iterative as many times as needed until no
            new predicates are added to the list. This is an action model refinement algorithm, that refines itself by a growing predicate list.
        """
        
        # outer loop that resets all action creation to be conditioned on updated predicate list
        for i_iter in range(max_iter):
            prev_predicate_list = deepcopy(predicates) # copy previous predicate list
            action_list = []
            
            # inner loop that generates a single action along with its predicates
            for _, action in enumerate(actions):
                
                # retrieve prompt for specific action
                action_prompt, _ = get_action_prompt(prompt_template, action_model[action])
                
                # retrieve prompt for current predicate list
                predicate_prompt = get_predicate_prompt(predicates)

                # assemble template
                action_predicate_prompt = f'{action_prompt}\n\n{predicate_prompt}'
                action_predicate_prompt += '\n\nParameters:'
                
                # create single action + corresponding predicates
                action, predicates, llm_response = construct_action(
                    model, action_predicate_prompt, action, predicates, types, max_attempts, True)
                
                # add action to current list + remove any redundant predicates
                action_list.append(action)
                predicates = prune_predicates(predicates, action_list)
                
            gen_done = False
            if len(prev_predicate_list) == len(predicates):
                print(f'[INFO] iter {i_iter} | no new predicate has been defined, will terminate the process')
                gen_done = True

            if gen_done:
                break
            
        # generate PDDL format
        pddl_domain = domain_builder.generate_domain(
                domain_name=domain,
                requirements=reqs,
                types=types,
                predicates=predicates,
                actions=action_list,
            )


    # helper functions
    def get_action_prompt(prompt_template: str, action_desc: str):
        """Creates prompt for specific action."""
        
        action_desc_prompt = action_desc['desc']
        for i in action_desc['extra_info']:
            action_desc_prompt += ' ' + i
        
        full_prompt = str(prompt_template) + ' ' + action_desc_prompt
        
        return full_prompt, action_desc_prompt


    def get_predicate_prompt(predicates):
        """Creates prompt for list of available predicates generated so far."""
        
        predicate_prompt = 'You can create and define new predicates, but you may also reuse the following predicates:\n'
        if len(predicates) == 0:
            predicate_prompt += 'No predicate has been defined yet'
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