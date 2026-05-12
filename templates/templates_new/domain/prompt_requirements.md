## ROLE
Based off of the natural language description (found under `## TASK`), your role is to model PDDL domain :requirements in the following format.

## OUTPUT FORMAT
End your final answer by wrapping the PDDL components inside specific XML tag `<requirements> ... </requirements>` with the specified JSON object as shown below. Do not include Markdown backticks.

<requirements>
[
  {
    "name": ":strips",
    "desc": "Optional (str)"
  },
  {
    "name": ":typing",
    "desc": "Optional (str)"
  }
]
</requirements>

Here are all the :requirements available (you MUST strictly only use these):
PDDL Core (1.2):
  - :strips
  - :typing
  - :disjunctive-preconditions
  - :equality
  - :existential-preconditions
  - :universal-preconditions
  - :quantified-preconditions
  - :conditional-effects
  - :adl

PDDL Extended (1.2):
  - :action-expansions
  - :foreach-expansions
  - :dag-expansions
  - :domain-axioms
  - :subgoals-through-axioms
  - :safety-constraints
  - :expression-evaluation
  - :open-world
  - :true-negation
  - :ucpop

PDDL 2.1:
  - :fluents
  - :numeric-fluents
  - :durative-actions
  - :duration-inequalities
  - :durative-inequalities
  - :continuous-effects
  - :negative-preconditions
  - :timed-effects
  - :action-costs

PDDL 2.2:
  - :derived-predicates
  - :derived-functions
  - :timed-initial-literals
  - :timed-initial-fluents

PDDL 3.0
  - :constraints
  - :preferences

PDDL 3.1/+:
  - :object-fluents
  - :time

## RULES
1. The JSON block above is strictly an ILLUSTRATIVE EXAMPLE. You must extract the actual requirements needed based on the capabilities described in the text (e.g., if the text mentions time limits, you need `:durative-actions`; if it mentions soft goals, you need `:preferences`).

2. Provide ONLY a valid JSON list wrapped in `<requirements>` tags.

3. Every requirement MUST have a "name" (string) and an optional description "desc" (string).

4. The "name" MUST be exactly one of the valid strings from the permitted list provided above. It must include the leading colon (`:`).

5. Ensure `:strips` is always included as a baseline.

6. Ensure the final JSON is perfectly formatted with no trailing commas.

## TASK
Please process the following domain:
<domain_description>
{domain_desc}
</domain_description>