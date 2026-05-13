## ROLE
Based on the natural language description (found under `## TASK`), your role is to model PDDL problem initial states (:init) in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the initial state definitions inside specific XML tag `<initial_states> ... </initial_states>` using the JSON format shown below. Do not include Markdown backticks.

<initial_states>
{
    "facts": [
        "(connected waypoint1 waypoint2)",
        "(= (battery-level rover1) 100.0)"
    ],
    "timed_facts": [
        {
            "time": 10.5,
            "fact": "(communications-blackout)"
        },
        {
            "time": 25.0,
            "fact": "(= (solar-radiation) 50.0)"
        }
    ],
    "desc": "Optional (str)"
}
</initial_states>

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "rover1", "waypoint1", or "battery-level" unless they are explicitly defined in the problem description. You must extract the actual predicates, functions, and objects from the text.

2. Provide ONLY a valid JSON object wrapped in `<initial_states>` tags.

3. The JSON must contain a "facts" list, a "timed_facts" list, and an optional root-level description "desc" (string).

4. The "facts" list must contain standard PDDL facts and numeric assignments as strings. Use appropriate predicates defined in the domain.

5. The "timed_facts" list is strictly for PDDL 2.2 Timed Initial Literals or Fluents. If an event is scheduled to happen at a specific time during the plan execution, place it here.

6. Every object in "timed_facts" MUST have a "time" (float) and a "fact" (string).

7. Problem-level initial states must apply to specific instantiated objects (e.g., `rover1`). Do NOT prefix object names with a question mark (`?`).

8. If there are no standard facts, leave "facts" as an empty list `[]`. If there are no timed facts, leave "timed_facts" as an empty list `[]`.

9. Ensure the final JSON is perfectly formatted with no trailing commas.

## TASK
Please process the following problem:
<problem_description>
{problem_desc}
</problem_description>

{context_injection}