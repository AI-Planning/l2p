## ROLE
Based on the natural language description (found under `## TASK`), your role is to model a PDDL domain's action preconditions (:precondition) in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the precondition definitions inside specific XML tag `<preconditions> ... </preconditions>` using the JSON format shown below. Do not include Markdown backticks.

<preconditions>
{
    "conditions": [
        "(at ?r ?l)",
        "(>= (battery-level ?r) 20.0)",
        {
            "operator": "not",
            "condition": "(busy ?r)"
        },
        {
            "quantifier": "forall",
            "parameters": [
                {
                    "variable": "?w",
                    "type": "waypoint"
                }
            ],
            "conditions": [
                "(visited ?w)"
            ]
        }
    ],
    "desc": "Optional (str)"
}
</preconditions>

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "?r", "battery-level", or "waypoint" unless explicitly defined in the domain description.

2. Provide ONLY a valid JSON object wrapped in `<preconditions>` tags.

3. The JSON object must always contain the keys: "conditions" and an optional "desc".

4. The "conditions" key must always be a JSON array. If there are no preconditions, use an empty list: `{"conditions": [], "desc": null}`.

5. Each item inside the "conditions" array must be a valid `LogicalCondition`. A `LogicalCondition` can be:
   - A simple PDDL string: `"(at ?r ?l)"` or `"(>= (battery-level ?r) 20)"`
   - A logical NOT dictionary: `{"operator": "not", "condition": "..."}`
   - A logical AND/OR dictionary: `{"operator": "and", "conditions": [...]}`
   - A logical IMPLY dictionary: `{"operator": "imply", "antecedent": [...], "consequent": [...]}`
   - A quantified FORALL/EXISTS dictionary: `{"quantifier": "forall", "parameters": [{"variable": "?p", "type": "packet"}], "conditions": [...]}`

6. Do NOT use an outer `and` operator to wrap all preconditions. List each top-level condition as a separate item in the "conditions" array (the system handles the root `and` automatically).

7. All parameter variables must be prefixed with a question mark (e.g., `?r`).

8. Ensure the final JSON is perfectly formatted with no trailing commas.

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>

{context}