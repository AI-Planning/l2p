## ROLE
Based off of the natural language description (found under `## TASK`), your role is to model PDDL domain durative-actions in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the PDDL components inside specific XML tag `<durative_actions> ... </durative_actions>` with the specified JSON object as shown below. Do not include Markdown backticks.

<durative_actions>
[
    {
        "name": "transmit_data",
        "params": [
            {"variable": "?r", "type": "rover"}
        ],
        "duration": [
            "(>= ?duration 5.0)"
        ],
        "conditions": {
            "at_start": [
                "(at ?r base_station)"
            ],
            "over_all": [
                {
                    "operator": "not",
                    "condition": "(safe-mode ?r)"
                }
            ],
            "at_end": [],
            "desc": "Optional (str)"
        },
        "effects": {
            "at_start": {
                "add": ["(transmitting ?r)"],
                "delete": [],
                "numeric": [],
                "conditional": []
            },
            "at_end": {
                "add": ["(data-transmitted)"],
                "delete": ["(transmitting ?r)"],
                "numeric": [],
                "conditional": []
            },
            "continuous": [
                "(decrease (battery-level ?r) (* #t 2.0))"
            ],
            "desc": "Optional (str)"
        },
        "desc": "Optional (str)"
    },
    {
        "name": "action_n",
        "params": [],
        "duration": [],
        "conditions": {
            "at_start": [],
            "over_all": [],
            "at_end": [],
            "desc": ""
        },
        "effects": {
            "at_start": {
                "add": [],
                "delete": [],
                "numeric": [],
                "conditional": []
            },
            "at_end": {
                "add": [],
                "delete": [],
                "numeric": [],
                "conditional": []
            },
            "continuous": [],
            "desc": ""
        },
        "desc": ""
    }
]
</durative_actions>

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "transmit_data" or "rover" unless explicitly defined in the domain description. You must extract actual durative actions, variables, conditions, and effects from the text.

2. Provide ONLY a valid JSON list wrapped in `<durative_actions>` tags.

3. Every durative action MUST have "name", "params", "duration", "conditions", and "effects".

4. **Params List:** Must contain objects with "variable" and "type". Variables MUST be prefixed with a question mark (e.g., `?r`).

5. **Duration Array:** If the duration is an exact time, provide a single string like `["(= ?duration 5.0)"]`. If it is bounded, use multiple strings like `["(>= ?duration 5.0)", "(<= ?duration 10.0)"]`.

6. **Conditions Object:** Contains "at_start", "over_all", and "at_end" lists. You can use plain strings for simple predicates, or dictionaries for complex logic. Valid logic dictionaries include:
   - {"operator": "not", "condition": "(pred ?t)"}
   - {"operator": "and", "conditions": ["(pred1 ?t)", "(pred2 ?t)"]}
   - {"operator": "or", "conditions": ["(pred1 ?t)", "(pred2 ?t)"]}
   - {"operator": "imply", "antecedent": ["(pred1 ?t)"], "consequent": ["(pred2 ?t)"]}
   - {"quantifier": "forall", "parameters": [{"variable": "?t", "type": "type"}], "conditions": ["(pred ?t)"]}

7. **Effects Object:** Contains "at_start", "at_end", and "continuous". 
   - "at_start" and "at_end" contain "add", "delete", "numeric", and "conditional" lists. Do NOT use "add" or "delete" in conditions; they belong strictly here.
   - If an effect removes a state, put the plain string in the "delete" array (do not wrap it in a "not" operator).
   - "continuous" is a list of strings representing continuous numeric changes over time (e.g., using `#t` like `(decrease (battery ?r) (* #t 1.5))`).

8. If a temporal block (like `at_end`) has no conditions or effects, leave the array/object completely empty.

9. Ensure the final JSON is perfectly formatted with no trailing commas.

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>