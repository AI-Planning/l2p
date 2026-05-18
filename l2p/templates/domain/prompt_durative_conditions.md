## ROLE
Based on the natural language description (found under `## TASK`), your role is to model durative conditions (:condition) for a PDDL durative-action (:durative-action) in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the durative condition definitions inside specific XML tag `<durative_conditions> ... </durative_conditions>` using JSON format shown below. Do not include Markdown backticks.

<durative_conditions>
{
    "at_start": ["(at ?r base_station)"],
    "over_all": [
        {
            "operator": "not",
            "condition": "(safe-mode ?r)"
        }
    ],
    "at_end": [],
    "desc": "Optional (str)"
}
</durative_conditions>

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. Do not copy names like "at" or "safe-mode" unless explicitly defined in the domain description.

2. **Strict JSON & XML Wrapping:** Output strictly valid JSON wrapped in the `<durative_conditions>` tags. Do not include trailing commas, and do not wrap the JSON in Markdown formatting backticks (e.g., ` ```json `).

3. **Variable Naming:** All parameters used in conditions must begin with a question mark (e.g., `?r`, `?from`) and must logically correspond to the parameters defined for this durative action.

4. **Temporal Anchors:** Durative action conditions MUST have "at_start", "over_all", "at_end", and optional "desc" description.
   - "at_start": Preconditions that must be strictly true at the exact moment the action begins.
   - "over_all": Invariants that must be maintained continuously for the entire duration of the action.
   - "at_end": Preconditions that must be true at the precise moment the action concludes.

5. **Conditions Object:** Contains "at_start", "over_all", and "at_end" lists. You can use plain strings for simple predicates, or dictionaries for complex logic. Valid logic dictionaries include:
   - {"operator": "not", "condition": "(pred ?x)"}
   - {"operator": "and", "conditions": ["(pred1 ?x)", "(pred2 ?x)"]}
   - {"operator": "or", "conditions": ["(pred1 ?x)", "(pred2 ?x)"]}
   - {"operator": "imply", "antecedent": ["(pred1 ?x)"], "consequent": ["(pred2 ?x)"]}
   - {"quantifier": "forall", "parameters": [{"variable": "?x", "type": "type"}], "conditions": ["(pred ?x)"]}

6. **Empty Arrays:** If a specific temporal anchor requires no conditions, you must explicitly return an empty list `[]`. Do not omit the key from the JSON.

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>

{context}