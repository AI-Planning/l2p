## ROLE
Based off of the natural language description (found under `## TASK`), your role is to model a PDDL domain's action effects in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the PDDL components inside specific XML tag `<effects> ... </effects>` with the specified JSON object as shown below. Do not include Markdown backticks.

<effects>
{
    "add": [
        "(at ?r ?to)"
    ],
    "delete": [
        "(at ?r ?from)"
    ],
    "numeric": [
        "(decrease (battery-level ?r) 5.0)",
        "(increase (total-cost) 1.0)"
    ],
    "conditional": [
        {
            "condition": [
                "(has-payload ?r)"
            ],
            "effect": {
                "add": ["(payload-delivered ?r)"],
                "delete": [],
                "numeric": []
            },
            "desc": "Optional (str)"
        }
    ],
    "desc": "Optional (str)"
}
</effects>

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "battery-level", "total-cost", or "?r" 
  unless they are explicitly defined in the domain description.

2. Provide ONLY a valid JSON object wrapped in `<effects>` tags.

3. The JSON object must always contain the keys: "add", "delete", "numeric", "conditional", and an optional description "desc".

4. "add", "delete", "numeric", and "conditional" must always be JSON arrays. If a category has no effects, use an empty list `[]`.

5. Put positive predicate effects in "add".

6. Put removed predicate effects in "delete". Do NOT use the `not` operator inside "delete"; just include the plain predicate string.

7. Put numeric updates such as `increase`, `decrease`, `assign`, `scale-up`, or `scale-down` in "numeric".

8. Use "conditional" only for effects that happen under an extra condition. Each item in "conditional" must contain:
   - "condition": a list of LogicalCondition items
   - "effect": an object with "add", "delete", and "numeric" arrays
   - optional "desc"

9. Do NOT use an outer `and` operator. List each effect as a separate item in its appropriate array.

10. Each item inside "add", "delete", "numeric", and "condition" must be a valid LogicalCondition.

11. Parameter variables must remain consistent with the domain and must use question-mark prefixes (for example, `?r`).

12. Ensure the final JSON is perfectly formatted with no trailing commas.

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>