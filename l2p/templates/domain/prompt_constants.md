## ROLE
Based on the natural language description (found under `## TASK`), your role is to model PDDL domain constants (:constants) in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the constant definitions inside specific XML tag `<constants> ... </constants>` using the JSON format shown below. Do not include Markdown backticks.

<constants>
[
    {
        "name": "base_station",
        "type": "waypoint",
        "desc": "Optional (str)"
    },
    {
        "name": "constant_n",
        "type": "type_n",
        "desc": null
    }
]
</constants>

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "base_station" or "waypoint" unless they are explicitly defined in the domain description. You must extract the actual constant names and types from the text.

2. Provide ONLY a valid JSON list wrapped in `<constants>` tags.

3. Every constant MUST have a "name" (string), a "type" (string), and an optional description "desc" (string). 

4. The "type" MUST be exactly one of the types defined in the domain. If no specific type applies, use "object".

5. Constants are globally available objects instantiated in the domain file. Do NOT prefix constant names with a question mark (`?`).

6. If there are no constants explicitly described in the domain, output an empty list `[]`.

7. Ensure the final JSON is perfectly formatted with no trailing commas.

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>

{context}