## ROLE
Based off of the natural language description (found under `## TASK`), your role is to model PDDL domain functions in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the PDDL components inside specific XML tag `<functions> ... </functions>` with the specified JSON object as shown below. Do not include Markdown backticks.

<functions>
[
    {
        "name": "battery-level",
        "params": [
            {
                "variable": "?r",
                "type": "rover"
            }
        ],
        "desc": "Optional (str)"
    },
    {
        "name": "total-cost",
        "params": [],
        "desc": ""
    },
    {
        "name": "function_n",
        "params": [],
        "desc": ""
    }
]
</functions>

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "battery-level", "total-cost", or "rover" 
  unless they are explicitly defined in the domain description. You must extract the actual numeric functions and parameters from the text.
2. Provide ONLY a valid JSON list wrapped in `<functions>` tags.
3. Every function MUST have a "name" (string), a "params" list, and an optional description "desc" (string).
4. The "params" list must contain objects with "variable" (string) and "type" (string). 
5. Parameter variables MUST ALWAYS be prefixed with a question mark (e.g., `?r`, not `r`).
6. If a function is global and does not apply to a specific object (for example, tracking the total fuel used across the entire mission, or tracking action costs), leave its "params" list completely empty: `[]`.
7. If there are no numeric functions described in the domain, output an empty list `[]`.
8. Ensure the final JSON is perfectly formatted with no trailing commas.

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>