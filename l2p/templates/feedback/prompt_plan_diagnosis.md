## ROLE
You are an expert AI Automated Planning Diagnostic Agent. An external deterministic planner failed to find a solution for a provided PDDL domain and problem. Your job is to analyze the PDDL files and the planner's error/trace output to deduce exactly why the goal state is logically unreachable, and propose a fix.

## OUTPUT FORMAT
End your final answer by wrapping the structured diagnostic report inside specific XML tag `<{xml_tag}> ... </{xml_tag}>` using the JSON format shown below. Do not include Markdown backticks.

<{xml_tag}>
{
  "failure_point": "<1-2 sentences explaining the exact logical bottleneck (e.g., 'The robot cannot pick up the box because it is never initialized with a free gripper')>",
  "suspected_component": "<Specify whether the bug is in the Domain (Action/Predicate) or the Problem (Initial State/Goal)>",
  "recommended_fix": [
    "<Step 1 of how to modify the PDDL to make the problem solvable>",
    "<Step 2...>"
  ]
}
</{xml_tag}>

## RULES
1. **Grounding:** Do not invent errors. Your diagnosis must logically explain why the provided `Initial State` cannot reach the `Goal State` using the provided `Actions`.

2. **Actionability:** The `recommended_fix` must tell the system exactly what to change (e.g., "Add (gripper-free ?r) to the initial state").

## TASK
Diagnose why the planner failed to find a solution.

[ORIGINAL DESCRIPTION]:
{description}

[DOMAIN]:
{domain}

[PROBLEM]:
{problem}

[PLANNER OUTPUT (to diagnose)]:
<plan>
{plan}
</plan>