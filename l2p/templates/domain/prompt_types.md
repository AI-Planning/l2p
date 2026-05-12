## ROLE
Based off of the natural language description (found under `## TASK`), your role is to model PDDL domain types in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the PDDL components inside specific XML tag `<types> ... </types>` with the specified JSON object as shown below. Do not include Markdown backticks.

<types>
[
    {
        "name": "vehicle",
        "parent": "object",
        "desc": "Optional (str)"
    },
    {
        "name": "rover",
        "parent": "vehicle",
        "desc": "Optional (str)"
    }
]
</types>

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "vehicle" or "rover" unless they are explicitly defined in the domain description. You must extract the actual types and type hierarchies from the text.

2. Provide ONLY a valid JSON list wrapped in `<types>` tags.

3. Every type MUST have a "name" (string), a "parent" (string), and an optional description "desc" (string).

4. If a type does not have a specific parent type mentioned, its "parent" MUST be set to "object" (which is the root type in PDDL).

5. If type hierarchies exist (e.g., a car is a vehicle), ensure the child type's "parent" field accurately references the parent type's "name".

6. If there are no custom types described, output an empty list `[]`.

7. Ensure the final JSON is perfectly formatted with no trailing commas.

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>

{context_injection}