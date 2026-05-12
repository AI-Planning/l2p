## ROLE
Based off of the natural language description (found under `## TASK`), your role is to model PDDL domain processes in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the PDDL components inside specific XML tag `<processes> ... </processes>` with the specified JSON object as shown below. Do not include Markdown backticks.

<processes>
[
    {
        "name": "solar-charging",
        "params": [
            {"variable": "?r", "type": "rover"}
        ],
        "preconditions": {
            "conditions": [
                "(solar-panel-deployed ?r)",
                "(in-sun ?r)",
                "(< (battery-level ?r) 100.0)"
            ],
            "desc": "Optional (str)"
        },
        "effects": {
            "add": [],
            "delete": [],
            "numeric": [
                "(increase (battery-level ?r) (* #t 2.5))"
            ],
            "conditional": [],
            "desc": "Optional (str)"
        },
        "desc": "Optional (str)"
    }
]
</processes>

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "solar-charging" or "rover" unless they are explicitly defined in the domain description. You must extract actual processes, variables, preconditions, and effects from the text.

2. Provide ONLY a valid JSON list wrapped in `<processes>` tags.

3. Every process MUST have "name", "params", "preconditions", and "effects".

4. **Params List:** Must contain objects with "variable" and "type". Variables MUST be prefixed with a question mark (e.g., `?r`).

5. **Preconditions Object:** Must be a dictionary containing a "conditions" array and an optional "desc". The "conditions" array represents the active state that must hold for the process to run. You can use plain strings, or logic dictionaries (e.g., `{"operator": "and", "conditions": [...]}`).

6. **Effects Object:** Must contain "add", "delete", "numeric", and "conditional" arrays. 
   - Since this is a PDDL+ process, continuous changes over time belong in the "numeric" array, typically using `#t` (e.g., `(increase (battery-level ?r) (* #t 2.5))`).
   - If there are no discrete effects, leave "add" and "delete" empty `[]`. Do NOT use the `not` operator inside "delete"; just include the plain predicate string.

7. Do NOT use an outer `and` operator in preconditions or effects. List each item as a separate string in its appropriate array.

8. Ensure the final JSON is perfectly formatted with no trailing commas.

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>