from config import HF_API_KEY
from l2p import *

domain_builder = DomainBuilder()

model = "llama3.1-8b"
model_path = "/home/tant2002/scratch/llama3_1_8b_instruct"
api_key = HF_API_KEY

llm = HUGGING_FACE(model=model, model_path=model_path, api_key=api_key)

response = llm.query("Hello, world!")
print(response)

llm.reset_query_log()
llm.reset_tokens()

# retrieve prompt information
base_path='tests/usage/prompts/domain/'
domain_desc = load_file(f'{base_path}blocksworld_domain.txt')
extract_predicates_prompt = load_file(f'{base_path}extract_predicates.txt')
types = load_file(f'{base_path}types.json')
action = load_file(f'{base_path}action.json')

# extract predicates via LLM
predicates, llm_output = domain_builder.extract_predicates(
    model=llm,
    domain_desc=domain_desc,
    prompt_template=extract_predicates_prompt,
    types=types,
    nl_actions={action['action_name']: action['action_desc']}
    )

# format key info into PDDL strings
predicate_str = "\n".join([pred["clean"].replace(":", " ; ") for pred in predicates])

print(f"PDDL domain predicates:\n{predicate_str}")

print(llm.get_query_log())