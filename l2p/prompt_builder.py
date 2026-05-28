"""
PromptBuilder prompt template generation class.

This module defines the `PromptBuilder` class for constructing prompt templates for
PDDL generation and feedback messages.

Refer to l2p/templates in: https://github.com/AI-Planning/l2p for how to structurally 
prompt LLMs so they are compatible with class function parsing.

This file uses inputted NL descriptions to generate Markdown prompt templates for the LLM.
The user does not have to use this class, but it is generally advisable for standardization.

Standardized format prompt.md:
```
## ROLE
[...]

## OUTPUT FORMAT
[...]

## RULES
[...]

## EXAMPLE(S)
# Example 1:
[...]
    .
    .
    .
# Example n:
[...]
--------------------------------------------------

## TASK
[...]

{context}
```
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from l2p.utils.pddl_prompt import safe_format
from l2p.utils.pddl_format import BaseModel

T = TypeVar("T", bound=BaseModel)


class PromptBuilder:
    def __init__(
        self,
        role: Optional[str] = None,
        format: Optional[str] = None,
        rules: Optional[Union[str, List[str]]] = None,
        examples: Optional[List[str]] = None,
        task: Optional[str] = None,
    ):

        self.role = role
        self.format = format
        self.task = task

        self.rules: List[str] = self._parse_to_list(rules)
        self.examples: List[str] = examples or []

    def _parse_to_list(self, item: Optional[Union[str, List[str]]]) -> List[str]:
        """Helper function to allow users to pass either a single string or a list."""
        if not item:
            return []
        return [item] if isinstance(item, str) else item

    def set_role(self, role_str: str) -> "PromptBuilder":
        """
        Sets the ``## ROLE`` section of the prompt.
        Defines the persona or system role the LLM should adopt.

        Args:
            role_str (str): The role description (e.g. 'You are an expert PDDL generator').
        """
        self.role = role_str
        return self

    def set_format(self, format_str: str) -> "PromptBuilder":
        """
        Sets the ``## OUTPUT FORMAT`` section of the prompt.
        Provides instructions or schemas defining exactly how the LLM should structure its answer.

        Args:
            format_str (str): The formatting instructions or JSON schema.
        """
        self.format = format_str
        return self

    def set_format_example(self, component: Type[T], is_list: bool=True) -> "PromptBuilder":
        """
        Appends a concrete JSON example for *component* (wrapped in the
        model's XML tag) to the ``## OUTPUT FORMAT`` section.
        Args:
            component (Type[BaseModel]): A Pydantic model class
                (e.g. ``PDDLType``, ``Action``, ``InitialState``)
            is_list (bool): sets flag to format list of components (from single extraction)
        """
        primary_tag = (
            component.tag if isinstance(component.tag, str) else component.tag[0]
        )
        example_data = _FORMAT_EXAMPLES.get(primary_tag)
        if example_data is None:
            raise KeyError(
                f"[ERROR] No predefined format example for '{component.__name__}' "
                f"(tag='{primary_tag}'). Available tags: "
                f"{', '.join(sorted(_FORMAT_EXAMPLES))}"
            )
        
        if is_list:
            example_data = [example_data]

        json_str = json.dumps(example_data, indent=4)
        example_block = f"<{primary_tag}>\n{json_str}\n</{primary_tag}>"

        if self.format:
            self.format += f"\n\n{example_block}"
        else:
            self.format = example_block

        return self

    def add_rule(self, rule_str: str) -> "PromptBuilder":
        """
        Appends a single rule to the `## RULES` section of the prompt.
        Rules are automatically numbered in the final output.

        Args:
            rule_str (str): A specific instruction or constraint for the LLM to follow.
        """
        self.rules.append(rule_str)
        return self

    def add_example(self, example: str) -> "PromptBuilder":
        """
        Appends a single n-shot example to the `## EXAMPLE(S)` section of the prompt.
        Examples are automatically numbered and formatted in the final output.

        Args:
            example (str): A complete input/output example for the LLM to learn from.
        """
        self.examples.append(example)
        return self

    def set_task(self, task_str: str) -> "PromptBuilder":
        """
        Sets the `## TASK` section of the prompt.
        Provides the specific natural language input or objective the LLM needs to solve.

        Args:
            task_str (str): The exact task description or natural language problem statement.
        """
        self.task = task_str
        return self

    def generate_prompt(self, **kwargs) -> str:
        """
        Generates the whole prompt in standard L2P format.
        If kwargs are provided, it dynamically replaces placeholders (e.g. {domain_desc}).
        """
        sections = []

        if self.role:
            sections.append(f"## ROLE\n{self.role}")

        if self.format:
            sections.append(f"## OUTPUT FORMAT\n{self.format}")

        if self.rules:
            rules_block = "## RULES\n" + "\n".join(
                [f"{i}. {rule}" for i, rule in enumerate(self.rules, 1)]
            )
            sections.append(rules_block)

        if self.examples:
            examples_list = [
                f"# Example {i}:\n{example}"
                for i, example in enumerate(self.examples, 1)
            ]
            examples_block = (
                "## EXAMPLE(S)\n" + "\n\n".join(examples_list) + f"\n{50*'-'}"
            )
            sections.append(examples_block)

        if self.task:
            sections.append(f"## TASK\n{self.task}")

        # inject context placeholders
        sections.append(f"{{description}}")
        sections.append(f"{{context}}")

        raw_prompt = "\n\n".join(sections).strip()

        if kwargs:
            return safe_format(raw_prompt, **kwargs)

        return raw_prompt

    def save_prompt(self, filename: str = "prompt.md", **kwargs):
        """
        Generates the prompt and saves it to a Markdown (.md) or Text (.txt) file.
        If no path is provided in the filename, it saves to the current working directory.

        Args:
            filename (str): The name/path of the file to save (e.g. 'prompt.md' or 'out/prompt.md').
            **kwargs: Dynamic placeholders passed to generate_prompt()
        """
        prompt_template = self.generate_prompt(**kwargs)
        save_path = Path(filename).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(prompt_template)

        print(f"[SUCCESS] Prompt saved to: {save_path}")


# ---------------------------------------------------------------------------
# Predefined concrete JSON examples for every PDDL model class.
# Each entry is keyed by the first element of the class's ``tag`` tuple
# (the primary XML tag).  The values are realistic PDDL examples that
# users will see in their generated ``## OUTPUT FORMAT`` section.
# ---------------------------------------------------------------------------

_FORMAT_EXAMPLES: Dict[str, Any] = {
    # ---- Domain components ----
    "requirements": {
        "name": ":strips",
        "desc": "Optional (str)"
    },
    "types": {
        "name": "vehicle",
        "parent": "object",
        "desc": "Optional (str)"
    },
    "constants": {
        "name": "base_station",
        "type": "waypoint",
        "desc": "Optional (str)"
    },
    "parameters": {
        "variable": "?r",
        "type": "rover",
        "desc": "Optional (str)"
    },
    "predicates": {
        "name": "at",
        "params": [
            {
                "variable": "?r",
                "type": "rover"
            },
            {
                "variable": "?w",
                "type": "waypoint"
            }
        ],
        "desc": "Optional (str)"
    },
    "functions": {
        "name": "battery-level",
        "params": [
            {
                "variable": "?r",
                "type": "rover"
            }
        ],
        "desc": "Optional (str)"
    },
    "der_preds": {
        "name": "can-transmit",
        "params": [
            {"variable": "?r", "type": "rover"}
        ],
        "condition": {
            "operator": "and",
            "conditions": [
                "(at ?r base_station)",
                "(>= (battery-level ?r) 50.0)"
            ]
        },
        "desc": "Optional (str)"
    },
    "preconditions": {
        "conditions": [
            "(at ?r ?l)",
            "(>= (battery-level ?r) 20.0)",
            {
                "operator": "not",
                "condition": "(busy ?r)"
            },
            {
                "quantifier": "forall",
                "parameters": [
                    {
                        "variable": "?w",
                        "type": "waypoint"
                    }
                ],
                "conditions": [
                    "(visited ?w)"
                ]
            }
        ],
        "desc": "Optional (str)"
    },
    "conditional_effects": {
        "condition": ["(has-rock-sample ?r)"],
        "effect": {
            "add": ["(carrying-heavy-load ?r)"],
            "delete": [],
            "numeric": ["(decrease (battery-level ?r) 10.0)"],
        },
        "desc": "Triggers when rover carries a rock sample",
    },
    "effects": {
        "add": [
            "(at ?r ?to)"
        ],
        "delete": [
            "(at ?r ?from)"
        ],
        "numeric": [
            "(decrease (battery-level ?r) 5.0)",
            "(increase (total-cost) 1.0)"
        ],
        "conditional": [
            {
                "condition": [
                    "(has-payload ?r)"
                ],
                "effect": {
                    "add": ["(payload-delivered ?r)"],
                    "delete": [],
                    "numeric": []
                },
                "desc": "Optional (str)"
            }
        ],
        "desc": "Optional (str)"
    },
    "actions": {
        "name": "move-rover",
        "params": [
            {"variable": "?r", "type": "rover"},
            {"variable": "?from", "type": "waypoint"},
            {"variable": "?to", "type": "waypoint"}
        ],
        "preconditions": {
            "conditions": [
                "(at ?r ?from)",
                {
                    "operator": "not",
                    "condition": "(= ?from ?to)"
                }
            ],
            "desc": "Optional (str)"
        },
        "effects": {
            "add": ["(at ?r ?to)"],
            "delete": ["(at ?r ?from)"],
            "numeric": ["(decrease (battery-level ?r) 10.0)"],
            "conditional": [
                {
                    "condition": ["(has-rock-sample ?r)"],
                    "effect": {
                        "add": ["(carrying-heavy-load ?r)"],
                        "delete": [],
                        "numeric": []
                    },
                }
            ],
            "desc": "Optional (str)"
        },
        "desc": "Optional (str)"
    },
    "dur_conds": {
        "at_start": ["(at ?r base_station)"],
        "over_all": [
            {
                "operator": "not",
                "condition": "(safe-mode ?r)"
            }
        ],
        "at_end": [],
        "desc": "Optional (str)"
    },
    "dur_effects": {
        "at_start": {
            "add": ["(moving ?r)"],
            "delete": ["(idle ?r)"],
            "numeric": [],
            "conditional": [],
        },
        "at_end": {
            "add": ["(at ?r ?to)"],
            "delete": ["(at ?r ?from)", "(moving ?r)"],
            "numeric": ["(assign (battery-level ?r) 0.0)"],
            "conditional": [],
        },
        "continuous": ["(decrease (battery-level ?r) (* #t 1.0))"],
        "desc": "Effects split across start and end of durative action",
    },
    "dur_actions": {
        "name": "navigate",
        "params": [
            {"variable": "?r", "type": "rover"},
            {"variable": "?from", "type": "waypoint"},
            {"variable": "?to", "type": "waypoint"},
        ],
        "duration": ["(= ?duration 10.0)"],
        "conditions": {
            "at_start": ["(at ?r ?from)"],
            "over_all": ["(has-power ?r)"],
            "at_end": ["(at ?r ?to)"],
        },
        "effects": {
            "at_start": {"add": ["(busy ?r)"], "delete": [], "numeric": [], "conditional": []},
            "at_end": {
                "add": ["(at ?r ?to)"],
                "delete": ["(at ?r ?from)", "(busy ?r)"],
                "numeric": [],
                "conditional": [],
            },
        },
        "desc": "Rover navigates between waypoints over a fixed duration",
    },
    "constraints": {
        "condition": {
            "operator": "always",
            "condition": "(<= (battery-level ?r) 100.0)",
        },
        "desc": "Battery level must never exceed 100",
    },
    "events": {
        "name": "battery-depleted",
        "params": [{"variable": "?r", "type": "rover"}],
        "preconditions": {
            "conditions": [
                {"operator": "and", "conditions": ["(at ?r ?l)", "(<= (battery-level ?r) 0.0)"]}
            ]
        },
        "effects": {
            "add": ["(dead ?r)"],
            "delete": ["(has-power ?r)"],
            "numeric": [],
            "conditional": [],
        },
        "desc": "Triggered when a rover runs out of battery",
    },
    "processes": {
        "name": "drain-battery",
        "params": [{"variable": "?r", "type": "rover"}],
        "preconditions": {
            "conditions": ["(moving ?r)"],
        },
        "effects": {
            "add": [],
            "delete": [],
            "numeric": ["(decrease (battery-level ?r) (* #t 0.5))"],
            "conditional": [],
        },
        "desc": "Continuously drains battery while the rover moves",
    },
    # ---- Problem components ----
    "objects": {"name": "rover1", "type": "rover", "desc": "Instance of a rover"},
    "timed_facts": {
        "time": 15.5,
        "fact": "(communications-blackout)",
        "desc": "Event triggers at t=15.5",
    },
    "initial": {
        "facts": [
            "(at rover1 waypoint0)",
            "(= (battery-level rover1) 100.0)",
            "(has-power rover1)",
            "(calibrated camera1)",
            "(connected waypoint0 waypoint1)",
        ],
        "timed_facts": [
            {"time": 50.0, "fact": "(= (solar-flare-level) 80.0)"}
        ],
        "desc": "Initial deployment state of the rover",
    },
    "goal": {
        "conditions": [
            "(at rover1 waypoint3)",
            "(data-transmitted)",
            {
                "operator": "or",
                "conditions": [
                    "(has-rock-sample rover1)",
                    "(has-soil-sample rover1)",
                ],
            },
        ],
        "desc": "Rover must reach waypoint3 and transmit data",
    },
    "metric": {
        "optimization": "minimize",
        "expression": "total-time",
        "desc": "Minimize makespan",
    },
}