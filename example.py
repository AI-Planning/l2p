# pip3 install NLtoPDDL

from NLtoPDDL import NLtoPDDLModel

# initialize model by API key
model = NLtoPDDLModel(api_key='API_KEY')
# model = NLtoPDDLModel(model='gpt-4', api_key='API_KEY')

# prompt from: https://github.com/GuanSuns/LLMs-World-Models-for-Planning/blob/main/prompts/common/action_description_prompt.txt
prompt = """
You are defining the domain (i.e. preconditions and effects) represented in PDDL format of an AI agent's actions. 
Information about the AI agent will be provided in the domain description. Note that individual conditions in preconditions and 
effects should be listed separately. For example, "object_1 is washed and heated" should be considered as two separate 
conditions "object_1 is washed" and "object_1 is heated". Also, in PDDL, two predicates cannot have the same name even 
if they have different parameters. Each predicate in PDDL must have a unique name, and its parameters must be explicitly 
defined in the predicate definition. It is recommended to define predicate names in an intuitive and readable way.

Domain information:
BlocksWorld is a planning domain in artificial intelligence. The AI agent here is a mechanical robot arm that 
can pick and place the blocks. Only one block may be moved at a time: it may either be placed on the table or 
placed atop another block. Because of this, any blocks that are, at a given time, under another block cannot be moved. 
There is only one type of object in this domain, and that is the block.

Here is an example from the classical BlocksWorld domain for demonstrating the output format.
[EXAMPLE]

[INSERT OTHER PROMPT TECHNIQUES]
"""

prompts = ["Problem description:", "Domain description:", ["Example 1:", "Example 2:"], "Additional Instructions:"]

# convert prompt to PDDL
pddl_domain = model.convert(prompt) # single conversation

# output PDDL domain
print(f"PDDL:\n{pddl_domain}")

# other library functions
model.convert_batch(prompts) # multiple conversations

verified, feedback = model.verify(pddl_domain) # returns boolean and a string containing feedback generated by external verifier
model.refine(pddl_domain, feedback) # returns refined PDDL domain file (either human-in-loop or external verifier feedback)

model.get_predicates(pddl_domain) # extract predicates from generated PDDL domain file

actions = model.get_actions(pddl_domain) # extract actions in a list from generated PDDL domain file
model.get_preconditions(actions[0]) # extract preconditions from specific action in generated PDDL domain file
model.get_effects(actions[0]) # extract effects from generated PDDL domain file

model.save_pddl_file(pddl_domain, "path/blocksworld_experiment.pddl")
pddl_domain = model.load_pddl_file("path/blocksworld_experiment.pddl")
model.print_pddl_file(pddl_domain)
model.get_llm()






"""Practical example (BACKPROMPTING MECHANISM)"""
while True:
    # convert prompt to PDDL
    pddl_domain = model.convert(prompt)

    # output PDDL domain
    print(f"PDDL:\n{pddl_domain}")

    verified, feedback = model.verify(pddl_domain)

    if verified:
        print('PDDL seems valid')
    else:
        print('PDDL is invalid', feedback)

    # human-in-the-loop feedback
    user_input = input("Enter 'exit' to finish, or provide additional instructions for refinement: ").strip()
    if user_input.lower() == 'exit':
        break
    else:
        prompt += f'\n{user_input}' + feedback # model.refine(pddl_domain, feedback)
        

# save file
model.save_pddl_file(pddl_domain, "path/blocksworld_experiment.pddl")

