## ROLE
Based on the natural language description (found under `## TASK`), your role is to model an entire PDDL problem instance in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the PDDL problem definitions inside specific XML tag `<problem> ... </problem>` using the JSON format shown below. Do not include Markdown backticks.

<problem>
{
    "name": "problem-name",
    "domain_name": "domain-name",
    "objects": [
        {"name": "entity1", "type": "object", "desc": "Optional (str)"},
        {"name": "block1", "type": "block", "desc": "Optional (str)"},
        {"name": "table1", "type": "table", "desc": "Optional (str)"}
    ],
    "initial_state": {
        "facts": [
            "(on block1 table1)",
            "(= (weight block1) 5.0)"
        ],
        "timed_facts": [
            {
                "time": 10.5,
                "fact": "(communications-blackout)",
                "desc": "Optional (str)"
            }
        ],
        "desc": "Optional (str)"
    },
    "goal_state": {
        "conditions": [
            "(on block1 entity1)",
            {
                "operator": "not",
                "condition": "(on block1 table1)"
            }
        ],
        "desc": "Optional (str)"
    },
    "constraint": [
        {
            "condition": {
                "operator": "always",
                "condition": "(>= (battery-level rover1) 0)"
            },
            "desc": "Optional (str)"
        }
    ],
    "metric": {
        "optimization": "minimize",
        "expression": "(+ (total-cost) (* 10.0 (is-violated pref_name)) (* 5.0 (is-violated pref_name2)))",
        "desc": "Optional (str)"
    },
    "desc": "Optional description of the problem instance"
}
</problem>

## LOGICAL CONDITIONS FORMAT
When populating the initial state facts, timed facts, goal conditions, and constraints, you must output a mix of simple strings and dictionaries based on the logic required:

1. Simple Predicates & Numeric Checks (str): 
    "(at rover1 waypoint1)"
    "(= (battery-level rover1) 100)"

2. Basic Logical Operators (Dict): 
    # NOT
    {"operator": "not", "condition": "(busy rover1)"}
    
    # AND / OR
    {
        "operator": "and", 
        "conditions": [
            "(has-power rover1)",
            {"operator": "not", "condition": "(busy rover1)"}
        ]
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
    {"operator": "always", "condition": "(has-power rover1)"}
    
    # WITHIN / HOLD-AFTER (Time-bounded modal operators)
    {"operator": "within", "time": 10.5, "condition": "(transmitted packet1)"}
    
    # HOLD-DURING (Interval modal operator)
    {"operator": "hold-during", "time_start": 5.0, "time_end": 15.0, "condition": "(transmitting rover1)"}

5. PDDL 3.0 Preferences (Dict):
    # PREFERENCE (Assigns a name to a condition for metric tracking)
    {
        "preference": "pref_transmit_early",
        "condition": {"operator": "sometime", "condition": "(transmitted packet1)"}
    }

**STRICT USAGE RULES FOR PROBLEM LOGIC:**
- **Initial State:** `facts` can ONLY contain simple positive predicates or numeric assignments (Item 1). You cannot use logical operators (`not`, `and`, `or`), constraints, or preferences in the initial state. Negative facts are simply omitted.
- **Goals:** You may use items 1, 2, 3, and 5 (Preferences). Goals cannot contain trajectory constraints (Item 4).
- **Constraints (Problem-Specific):** Problem-specific trajectory constraints (the `constraint` array) MUST use item 4 (PDDL 3.0 Trajectory Constraints). Unlike domain constraints, variables here do not need to be quantified if they refer to specific grounded objects (e.g., `rover1`).

## RULES
1. **Illustrative Example:** The JSON block in the Output Format is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "block1" or "total-time" unless explicitly defined in the problem description. 

2. **Valid JSON Only:** Provide ONLY a valid JSON object wrapped in `<problem>` tags. 

3. **Required Keys:** Your JSON object must include all the top-level keys shown in the example (`name`, `domain_name`, `objects`, `initial_state`, `goal_state`, `constraint`, `metric`).

4. **Empty Components:** If a specific component (like `timed_facts`, `constraint`, or `metric`) is not needed, leave its value as a completely empty array `[]` or set it to `null`. Do not omit the key.

5. **Grounded Objects:** Unlike the Domain file, a Problem file deals with specific instances. Variables in goals, facts, and constraints should NOT have a question mark prefix `?` unless they are bound by a local quantifier (like `forall`). They should reference specific object names defined in the `objects` list (e.g., `rover1`).

6. **Metric Block:** The `optimization` field inside `metric` must strictly be either `"minimize"` or `"maximize"`. If no optimization is requested, leave the `metric` key as `null`.

## TASK
Please process the following problem:
<problem_description>
{problem_desc}
</problem_description>

{context_injection}