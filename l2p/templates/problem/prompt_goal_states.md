## ROLE
Based on the natural language description (found under `## TASK`), your role is to model PDDL problem goal states (:goal) in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the goal state definitions inside specific XML tag `<goal_states> ... </goal_states>` using the JSON format shown below. Do not include Markdown backticks.

<goal_states>
{
    "conditions": [
        "(at rover1 waypoint3)",
        "(data-transmitted)",
        {
            "preference": "pref_visit_wp4",
            "condition": "(at rover1 waypoint4)"
        },
        {
            "operator": "or",
            "conditions": [
                "(has-rock-sample rover1)",
                "(has-soil-sample rover1)"
            ]
        }
    ],
    "desc": "Optional (str)"
}
</goal_states>

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "rover1", "waypoint3", or "pref_visit_wp4" unless they are explicitly defined in the problem description. You must extract the actual predicates and objects from the text.

2. Provide ONLY a valid JSON object wrapped in `<goal_states>` tags.

3. The JSON must be a single object containing a "conditions" list and an optional description "desc" (string).

4. Every item in the "conditions" list represents a required goal state. All items in this list will automatically be treated as if they are joined by an "and" operator, so you do not need to wrap the entire list in an "and" block.

5. Items in the "conditions" list can be simple strings (for basic facts) or dictionaries (for preferences or logical operators like "or", "not", etc.).

6. If a goal is soft (optional but preferred), wrap it in a "preference" dictionary.

7. Problem-level goals must apply to specific instantiated objects (e.g., `rover1`). Do NOT prefix object names with a question mark (`?`).

8. If there are no goals described, output an empty list for conditions: `{"conditions": []}`.

9. Ensure the final JSON is perfectly formatted with no trailing commas.

## TASK
Please process the following problem:
<problem_description>
{problem_desc}
</problem_description>

{context_injection}