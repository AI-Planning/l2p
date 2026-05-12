## ROLE
Based off of the natural language description (found under `## TASK`), your role is to model PDDL durative-actions in natural language in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the dictionary inside specific XML tag `<nl_durative_actions> ... </nl_durative_actions>` with the specified JSON object as shown below. Do not include Markdown backticks.

<nl_durative_actions>
{
    "durative_action_name_1": "action_description",
    "durative_action_name_2": "action_description",
    "durative_action_name_3": "action_description"
}
</nl_durative_actions>

## RULES

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>

{context_injection}