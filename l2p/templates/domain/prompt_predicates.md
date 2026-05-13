## ROLE
Based on the natural language description (found under `## TASK`), your role is to model PDDL domain predicates (:predicates) in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the predicate definitions inside specific XML tag `<predicates> ... </predicates>` using the JSON format shown below. Do not include Markdown backticks.

<predicates>
[
    {
        "name": "at",
        "params": [
            {
                "variable": "?r",
                "type": "rover"
            },
            {
                "variable": "?w",
                "type": "waypoint"
            }
        ],
        "desc": "Optional (str)"
    },
    {
        "name": "predicate_n",
        "params": [],
        "desc": ""
    }
]
</predicates>

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "at", "rover", or "calibrated" unless they are explicitly defined in the domain description. You must extract the actual predicates and parameters from the text.

2. Provide ONLY a valid JSON list wrapped in `<predicates>` tags.

3. Every predicate MUST have a "name" (string), a "params" list, and an optional description "desc" (string).

4. The "params" list must contain objects with "variable" (string) and "type" (string).

5. Parameter variables MUST ALWAYS be prefixed with a question mark (e.g., `?r`, not `r`).

6. If a predicate is a global boolean state that does not apply to a specific object (e.g., `(daylight)` or `(mission-started)`), leave its "params" list completely empty: `[]`.

7. If there are no standard predicates described in the domain, output an empty list `[]`.

8. Ensure the final JSON is perfectly formatted with no trailing commas.

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>

{context_injection}