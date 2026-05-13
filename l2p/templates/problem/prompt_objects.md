## ROLE
Based on the natural language description (found under `## TASK`), your role is to model PDDL problem objects (:objects) in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the object definitions inside specific XML tag `<objects> ... </objects>` using the JSON format shown below. Do not include Markdown backticks.

<objects>
[
    {
        "name": "rover1", 
        "type": "rover", 
        "desc": "Optional (str)"
    },
    {
        "name": "wp1", 
        "type": "waypoint", 
        "desc": "Optional (str)"
    },
    {
        "name": "camera_obj", 
        "type": "object", 
        "desc": "Optional (str)"
    }
]
</objects>

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "rover1" or "waypoint" unless they are explicitly defined in the problem description. You must extract the actual objects and types from the text.

2. Provide ONLY a valid JSON list wrapped in `<objects>` tags.

3. Every object MUST have a "name" (string), a "type" (string), and an optional description "desc" (string).
4. The "type" MUST be exactly one of the types defined in the domain description. If no types were defined in the domain, you are permitted to create them or default to "object".

5. The "desc" field is an optional brief string explaining what the object represents.

6. Do not prefix names with question marks (e.g., use "rover1", not "?rover1").

7. Ensure the final JSON is perfectly formatted with no trailing commas.

## TASK
Please process the following problem:
<problem_description>
{problem_desc}
</problem_description>

{context_injection}