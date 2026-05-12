## ROLE
Based off of the natural language description (found under `## TASK`), your role is to model a PDDL domain derived predicate (axiom) in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the PDDL components inside specific XML tag `<derived_predicates> ... </derived_predicates>` with the specified JSON object as shown below. Do not include Markdown backticks.

<derived_predicates>
[
    {
        "name": "can-transmit",
        "params": [
            {"variable": "?r", "type": "rover"}
        ],
        "condition": {
            "operator": "and",
            "conditions": [
                "(at ?r base_station)",
                "(>= (battery-level ?r) 50.0)"
            ]
        },
        "desc": "Optional (str)"
    },
    {
        "name": "is-busy",
        "params": [
            {"variable": "?r", "type": "rover"}
        ],
        "condition": {
            "operator": "or",
            "conditions": [
                "(moving ?r)",
                "(transmitting ?r)"
            ]
        },
        "desc": "Optional (str)"
    },
    {
        "name": "is-ready",
        "params": [{"variable": "?t1", "type": "type_1"}],
        "condition": {
            "quantifier": "exists",
            "parameters": [{"variable": "?p", "type": "packet"}],
            "conditions": ["(at ?t1 constant_1)"]
        },
        "desc": "Optional (str)"
    }
]
</derived_predicates>

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "can-transmit", "rover", or "?r" unless they are explicitly defined in the domain description. You must extract the actual derived predicates, variables, and conditions from the text.

2. Provide ONLY a valid JSON list wrapped in `<derived_predicates>` tags.

3. Every derived predicate MUST have a "name" (string), a "params" list, a "condition" object/string, and an optional description "desc" (string).

4. The "params" list must contain objects with "variable" (string) and "type" (string). Variables MUST ALWAYS be prefixed with a question mark (e.g., `?r`).

5. The "condition" field represents a PDDL LogicalCondition. It is the logical formula that, when evaluated to true, makes the derived predicate true. Conditions are the following:

5.1. Simple Predicates & Numeric Checks (str): 
    "(at ?r ?l)"
    "(>= (battery-level ?r) 20)"

5.2. Basic Logical Operators (Dict): 
    # NOT
    {"operator": "not", "condition": "(busy ?r)"}
    # AND / OR (Can contain nested conditions)
    {
        "operator": "and", 
        "conditions": [
            "(has-power ?r)",
            {"operator": "not", "condition": "(busy ?r)"}
        ]
    }
    # IMPLY (Requires antecedent and consequent)
    {
        "operator": "imply",
        "antecedent": ["(at ?r ?l)"],
        "consequent": ["(can-transmit ?r)"]
    }

5.3. Quantifiers (Dict):
    # FORALL / EXISTS
    {
        "quantifier": "forall",
        "parameters": [{"variable": "?p", "type": "packet"}],
        "conditions": ["(transmitted ?p)"]
    }

6. Derived predicates are evaluated dynamically based on the current state. They should NEVER be used in the `add` or `delete` effects of actions.

7. If there are no derived predicates described in the domain, output an empty list `[]`.

8. Ensure the final JSON is perfectly formatted with no trailing commas.

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>