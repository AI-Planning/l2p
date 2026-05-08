"""
This module contains a collection of helper functions that parses information from LLM output.
"""

import json
import re
from typing import List, TypeVar, Type as PyType
from pydantic import TypeAdapter, ValidationError

T = TypeVar('T')


def parse_xml_tags(llm_output: str, tag_name: str) -> List[str]:
    """
    Finds all instances of a specific XML tag pair and returns the text inside them.
    Returns a list of raw strings. Raises ValueError if no tags are found.
    """
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, llm_output, flags=re.DOTALL)

    if not matches:
        raise ValueError(
            f"Error: Could not find the expected XML tags <{tag_name}>...</{tag_name}> in your response. "
            f"Please ensure you wrap your JSON output strictly inside <{tag_name}> tags."
        )
        
    return [match.strip() for match in matches]


def parse_list(raw_blocks: List[str], model_class: PyType[T], tag_name: str) -> List[T]:
    """
    Iterates through raw text blocks and attempts to parse them into a List of Pydantic models.
    Returns the first successfully parsed list.
    """
    adapter = TypeAdapter(List[model_class])
    collected_errors = []
    
    for block in raw_blocks:
        try:
            return adapter.validate_json(block)
        except (ValidationError, ValueError) as e:
            collected_errors.append(str(e))
            continue
            
    expected_schema = json.dumps(model_class.model_json_schema(), indent=2)
    error_message = (
        f"Error: The JSON provided inside <{tag_name}> failed schema validation.\n\n"
        f"--- YOUR ERRORS ---\n"
        f"{collected_errors[0]}\n\n"
        f"--- EXPECTED JSON SCHEMA FOR A SINGLE ITEM IN THE LIST ---\n"
        f"Please ensure your output is a JSON array `[...]` containing objects that match this schema:\n"
        f"{expected_schema}"
    )
    raise ValueError(error_message)


def parse_element(raw_blocks: List[str], model_class: PyType[T], tag_name: str) -> T:
    """
    Iterates through raw text blocks and attempts to parse them into a SINGLE Pydantic model.
    Used for singular components like Problem Details, Metric, or Goal.
    """
    collected_errors = []
    
    for block in raw_blocks:
        try:
            return model_class.model_validate_json(block)
        except (ValidationError, ValueError) as e:
            collected_errors.append(str(e))
            continue
            
    expected_schema = json.dumps(model_class.model_json_schema(), indent=2)
    error_message = (
        f"Error: The JSON provided inside <{tag_name}> failed schema validation.\n\n"
        f"--- YOUR ERRORS ---\n"
        f"{collected_errors[0]}\n\n"
        f"--- EXPECTED JSON SCHEMA ---\n"
        f"Please ensure your output is a single JSON object `{{...}}` matching this schema:\n"
        f"{expected_schema}"
    )
    raise ValueError(error_message)