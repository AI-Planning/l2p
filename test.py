import os
from l2p import *

domain_builder = DomainBuilder()

model = "llama2-7b"
model_path = "/Users/marcustantakoun/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf"
api_key = os.environ.get("HF_API_KEY")

llm = HUGGING_FACE(model=model, model_path=model_path, api_key=api_key)

response = llm.query("Hello, world!")
print(response)

# # retrieve prompt information
# base_path='tests/usage/prompts/domain/'
# domain_desc = load_file(f'{base_path}blocksworld_domain.txt')
# extract_predicates_prompt = load_file(f'{base_path}extract_predicates.txt')
# types = load_file(f'{base_path}types.json')
# action = load_file(f'{base_path}action.json')

# # extract predicates via LLM
# predicates, llm_output = domain_builder.extract_predicates(
#     model=llm,
#     domain_desc=domain_desc,
#     prompt_template=extract_predicates_prompt,
#     types=types,
#     nl_actions={action['action_name']: action['action_desc']}
#     )

# # format key info into PDDL strings
# predicate_str = "\n".join([pred["clean"].replace(":", " ; ") for pred in predicates])

# print(f"PDDL domain predicates:\n{predicate_str}")