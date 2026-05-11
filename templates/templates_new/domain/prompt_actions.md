## ROLE
Based off of the natural language description (found under `## TASK`), your role is to model PDDL domain actions in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the PDDL components inside specific XML tag `<actions> ... </actions>` with the specified JSON object as shown below. Do not include Markdown backticks.

<actions>
[
    {
        "name": "move-rover",
        "params": [
            {"variable": "?r", "type": "rover"},
            {"variable": "?from", "type": "waypoint"},
            {"variable": "?to", "type": "waypoint"}
        ],
        "preconditions": {
            "conditions": [
                "(at ?r ?from)",
                {
                    "operator": "not",
                    "condition": "(= ?from ?to)"
                }
            ],
            "desc": "Optional (str)"
        },
        "effects": {
            "add": ["(at ?r ?to)"],
            "delete": ["(at ?r ?from)"],
            "numeric": ["(decrease (battery-level ?r) 10.0)"],
            "conditional": [
                {
                    "condition": ["(has-rock-sample ?r)"],
                    "effect": {
                        "add": ["(carrying-heavy-load ?r)"],
                        "delete": [],
                        "numeric": []
                    },
                }
            ],
            "desc": "Optional (str)"
        },
        "desc": "Optional (str)"
    },
    {
        "name": "action_n",
        "params": [...],
        "preconditions": {...},
        "effects": {...},
        "desc": "Optional (str)"
    },
]
</actions>

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "navigate", "rover", or "battery-level" 
    unless explicitly defined in the domain description. You must extract actual actions, variables, conditions, and effects from the text.
2. Provide ONLY valid JSON list wrapped in `<actions>` tags.
3. Every action MUST have a "name" (string), "params" (list), "preconditions" (object), and "effects" (object).
4. The "params" list must contain objects with a "variable" (string) and "type" (string). Parameter variables MUST ALWAYS be prefixed with a question mark (e.g., `?r`).
5. The "preconditions" object contains a "conditions" list. All conditions in this list are implicitly joined by an "AND" operator. Use nested dictionaries for operators like "or", "not", "forall", or "exists".
6. The "effects" object contains "add", "delete", and "numeric" lists. 
   - Put positive boolean facts in "add".
   - Put negative boolean facts in "delete" (do not use the "not" operator here, just list the fact).
   - Put math operations like `(increase ...)` or `(decrease ...)` in "numeric".
7. The "effects" object can optionally include a "conditional" list for PDDL `when` effects. The "effect" field inside a conditional effect must be a dictionary with "add", "delete", and "numeric" lists.
8. Empty lists should be represented as `[]`. If an action has no preconditions, use `{"conditions": []}`.
9. Ensure the final JSON is perfectly formatted with no trailing commas.

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>