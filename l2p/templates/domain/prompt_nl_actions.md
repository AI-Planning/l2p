## ROLE
Based on the natural language description (found under `## TASK`), your role is to model PDDL actions in natural language in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the dictionary inside specific XML tag `<nl_actions> ... </nl_actions>` using the JSON format shown below. Do not include Markdown backticks.

<nl_actions>
{
    "action_name_1": "action_description",
    "action_name_2": "action_description",
    "action_name_3": "action_description"
}
</nl_actions>

## RULES

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>

{context_injection}