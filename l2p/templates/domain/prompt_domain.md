## ROLE
Based on the natural language description (found under `## TASK`), your role is to model an entire PDDL domain in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the PDDL domain definitions inside specific XML tag `<domain> ... </domain>` using the JSON format shown below. Do not include Markdown backticks.

<domain>
{
    "name": "domain-name",
    "requirements": [
        {"name": ":strips", "desc": ""},
        {"name": ":typing", "desc": ""}
    ],
    "types": [
        {"name": "type_1", "parent": "object", "desc": ""},
        {"name": "type_2", "parent": "type_1", "desc": ""}
    ],
    "constants": [
        {"name": "constant_1", "type": "object", "desc": ""},
        {"name": "constant_2", "type": "type_1", "desc": ""}
    ],
    "predicates": [
        {
            "name": "predicate-name",
            "params": [
                {"variable": "?l1", "type": "type_1", "desc": ""},
                {"variable": "?l2", "type": "type_2", "desc": ""}
            ],
            "desc": ""
        }
    ],
    "functions": [
        {
            "name": "total-cost", 
            "params": [], 
            "desc": "Tracks global cost"
        },
        {
            "name": "function_n", 
            "params": [{"variable": "?t1", "type": "type_1"}], 
            "desc": ""
        }
    ],
    "derived_predicates": [
        {
            "name": "derived-name",
            "params": [{"variable": "?t1", "type": "type_1"}],
            "condition": {"operator": "and", "conditions": ["(predicate-name ?t1 constant_2)"]},
            "desc": ""
        }
    ],
    "actions": [
        {
            "name": "action_1",
            "params": [{"variable": "?t1", "type": "type_1"}],
            "preconditions": {
                "conditions": ["(predicate-name ?t1 constant_2)"],
                "desc": ""
            },
            "effects": {
                "add": [], "delete": [], "numeric": [], "conditional": [], "desc": ""
            },
            "desc": ""
        }
    ],
    "durative_actions": [
        {
            "name": "durative_action_1",
            "params": [{"variable": "?t1", "type": "type_1"}],
            "duration": ["(>= ?duration 5.0)"],
            "conditions": {
                "at_start": ["(predicate-name ?t1 constant_1)"],
                "over_all": [],
                "at_end": [],
                "desc": ""
            },
            "effects": {
                "at_start": {"add": [], "delete": [], "numeric": [], "conditional": [], "desc": ""},
                "at_end": {"add": ["(predicate-name ?t1 constant_2)"], "delete": [], "numeric": [], "conditional": [], "desc": ""},
                "continuous": ["(decrease (function_n ?t1) (* #t 1.0))"],
                "desc": ""
            },
            "desc": ""
        }
    ],
    "events": [
        {
            "name": "event_name_1",
            "params": [{"variable": "?t1", "type": "type_1"}],
            "preconditions": {
                "conditions": ["(<= (function_n ?t1) 0)"],
                "desc": "Triggers immediately when condition is met"
            },
            "effects": {
                "add": ["(predicate-name ?t1 constant_1)"],
                "delete": [],
                "numeric": [],
                "conditional": [],
                "desc": ""
            },
            "desc": ""
        }
    ],
    "processes": [
        {
            "name": "process_name_1",
            "params": [{"variable": "?t1", "type": "type_1"}],
            "preconditions": {
                "conditions": ["(predicate-name ?t1 constant_1)"],
                "desc": "Runs continuously while condition holds"
            },
            "effects": {
                "add": [],
                "delete": [],
                "numeric": ["(increase (function_n ?t1) (* #t 2.0))"],
                "conditional": [],
                "desc": "Continuous numeric change over time"
            },
            "desc": ""
        }
    ],
    "constraint": [
        {
            "condition": {
                {
                    "quantifier": "forall",
                    "parameters": [{"variable": "?t1", "type": "type_1"}],
                    "conditions": [
                        "operator": "always",
                        "condition": "(>= (function_n ?t1) 0)"
                    ]
                }
            },
            "desc": ""
        }
    ],
    "desc": "Optional description of the overall domain"
}
</domain>

## LOGICAL CONDITIONS FORMAT
When populating conditions, preconditions, or effects, you must output a mix of simple strings and recursive dictionaries based on the logic required:

1. Simple Predicates & Numeric Checks (str): 
    "(at ?r ?l)"
    "(>= (battery-level ?r) 20)"

2. Basic Logical Operators (Dict): 
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

3. Quantifiers (Dict):
    # FORALL / EXISTS
    {
        "quantifier": "forall",
        "parameters": [{"variable": "?p", "type": "packet"}],
        "conditions": ["(transmitted ?p)"]
    }

4. PDDL 3.0 Trajectory Constraints (Dict):
    # ALWAYS / SOMETIME / AT-MOST-ONCE (Basic modal operators)
    {"operator": "always", "condition": "(has-power ?r)"}
    
    # WITHIN / HOLD-AFTER (Time-bounded modal operators)
    {"operator": "within", "time": 10.5, "condition": "(transmitted ?p)"}
    
    # HOLD-DURING (Interval modal operator)
    {"operator": "hold-during", "time_start": 5.0, "time_end": 15.0, "condition": "(transmitting ?r)"}

    # SOMETIME-AFTER / SOMETIME-BEFORE / ALWAYS-WITHIN (Relational operators)
    {"operator": "always-within", "time": 5.0, "antecedent": "(error-detected ?r)", "consequent": "(safe-mode ?r)"}

5. PDDL 3.0 Preferences (Dict):
    # PREFERENCE (Assigns a name to a condition for metric tracking)
    {
        "preference": "pref_transmit_early",
        "condition": {"operator": "sometime", "condition": "(transmitted ?p)"}
    }

**STRICT USAGE RULES FOR LOGIC:**
- **Action Preconditions, Effects, and Derived Predicates:** You may ONLY use items 1, 2, and 3. You cannot use modal operators, trajectory constraints, or preferences to determine immediate state changes or preconditions.
- **Constraints (Global):** Global trajectory constraints (the `constraint` key) MUST use item 4 (PDDL 3.0 Trajectory Constraints), and are conjunction of various forall/exists statements.
- **Preferences:** Item 5 is generally reserved for problem files (`metric` optimization), but can be placed inside global constraints if the domain allows soft constraints.

## RULES
1. **Illustrative Example:** The JSON block in the Output Format is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "type_1" or "at" unless explicitly defined in the domain description. You must extract actual types, predicates, actions, etc.

2. **Valid JSON Only:** Provide ONLY a valid JSON object wrapped in `<domain>` tags. 

3. **Required Keys:** Your JSON object must include all the top-level keys shown in the example (`name`, `requirements`, `types`, `constants`, `predicates`, `functions`, `derived_predicates`, `actions`, `durative_actions`, `events`, `processes`, `constraint`).

4. **Empty Components:** If a specific component (like `durative_actions` or `events`) is not needed, leave its value as a completely empty array `[]`. Do not omit the key.

5. **Mandatory Components:** A functional domain requires states and transitions. Therefore, you MUST populate the "predicates" array with at least one item, AND you MUST populate at least one of the behavior arrays ("actions", "durative_actions", "events", or "processes"). Do not leave all of these empty. If the provided description does not explicitly define these, you must logically infer and create them to ensure the domain is fully functional.

6. **Parameter Prefixing:** Inside any nested objects, ensure that all parameter variables strictly start with a question mark (e.g., `?t1`).

7. **Syntax:** Ensure the final JSON is perfectly formatted with no trailing commas. Ensure all logical states follow the `LOGICAL CONDITIONS FORMAT` exactly.

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>

{context}