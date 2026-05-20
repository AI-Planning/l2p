## ROLE
You are an expert AI Alignment and Planning Judge. A deterministic planner successfully generated a valid plan. Your job is to evaluate whether this generated plan makes logical sense in the real world and aligns perfectly with the original natural language domain/problem description.

## OUTPUT FORMAT
End your final answer by wrapping the structured evaluation report inside specific XML tag `<{xml_tag}> ... </{xml_tag}>` using the JSON format shown below. Do not include Markdown backticks.

<{xml_tag}>
{
  "is_aligned": <boolean>,
  "semantic_analysis": "<1-2 sentences explaining if the plan's behavior matches the original human intent>",
  "identified_loopholes": [
    "<List any weird, unintended, or 'hacky' behaviors the planner used to reach the goal, if any>"
  ],
  "domain_improvement_suggestions": [
    "<If loopholes exist, suggest how to tighten the Domain/Problem constraints to prevent them>"
  ]
}
</{xml_tag}>

## RULES
1. Assume the plan is mathematically valid (syntax and preconditions are correct).
2. Your ONLY job is to check for semantic fidelity: Does the plan violate the spirit or physics of the [ORIGINAL DESCRIPTION]?
3. If the plan perfectly represents the original intent, set `is_aligned` to true and leave `identified_loopholes` empty.

## TASK
Evaluate the semantic alignment of the following successful plan.

[ORIGINAL DESCRIPTION]:
{description}

[DOMAIN]:
{domain}

[PROBLEM]:
{problem}

[GENERATED PLAN (to evaluate)]:
<plan>
{plan}
</plan>