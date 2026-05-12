## ROLE
Based off of the natural language description (found under `## TASK`), your role is to model a PDDL domain's action parameters in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the PDDL components inside specific XML tag `<parameters> ... </parameters>` with the specified JSON object as shown below. Do not include Markdown backticks.

<parameters>
[
  {
    "variable": "?r",
    "type": "rover",
    "desc": "Optional (str)"
  },
  {
    "variable": "?from",
    "type": "waypoint",
    "desc": "Optional (str)"
  }
]
</parameters>

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "?r", "?from", "rover", or "waypoint" unless they are explicitly defined in the domain description.

2. Provide ONLY a valid JSON array wrapped in `<parameters>` tags.

3. Each item in the array must be a JSON object with:
   - "variable": a string
   - "type": a string
   - optional "desc": a string or null

4. The "variable" field is REQUIRED and must always start with a question mark `?`.

5. The "type" field is REQUIRED and must be a valid PDDL type name taken from the domain description.

6. Use "desc" only for a short optional explanation of the parameter's role. If no description is needed, use `null`.

7. Do not output any extra keys beyond "variable", "type", and "desc".

8. If there are no parameters, return an empty array: `<parameters>[]</parameters>`

9. Ensure the final JSON is perfectly formatted with no trailing commas.

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>

{context_injection}