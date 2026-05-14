"""
This module contains a collection of helper functions that parses information from LLM output.
"""

import re
import inspect
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
        
    return [match.strip() for match in matches]


def parse_component(raw_blocks: List[str], model_class: PyType[T], tag_name: str) -> List[T]:
    """
    Iterates through raw text blocks and attempts to parse them into a List of Pydantic models.
    Returns the first successfully parsed list.
    """
    list_adapter = TypeAdapter(List[model_class])
    collected_errors = []
    
    for block in raw_blocks:
        try:
            return list_adapter.validate_json(block)
        except (ValidationError, ValueError) as list_err:
            try:
                single_obj = model_class.model_validate_json(block)
                return [single_obj]
            except (ValidationError, ValueError) as single_err:
                collected_errors.append(str(list_err))
                continue
            
    class_source = inspect.getsource(model_class)
    
    error_message = (
        f"Error: The JSON provided inside <{tag_name}> failed validation.\n\n"
        f"--- YOUR ERRORS ---\n"
        f"{collected_errors[0] if collected_errors else 'No parsable blocks found.'}\n\n"
        f"--- EXPECTED CLASS DEFINITION FOR THE LIST ---\n"
        f"Please ensure your output is a JSON array `[...]` containing objects that match this exact structure:\n\n"
        f"```python\n{class_source}```"
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
            
    # extract the exact Python source code of the class
    class_source = inspect.getsource(model_class)
    
    error_message = (
        f"Error: The JSON provided inside <{tag_name}> failed validation.\n\n"
        f"--- YOUR ERRORS ---\n"
        f"{collected_errors[0]}\n\n"
        f"--- EXPECTED CLASS DEFINITION ---\n"
        f"Please ensure your output is a single JSON object `{{...}}` matching this exact Pydantic model structure:\n\n"
        f"```python\n{class_source}```"
    )
    raise ValueError(error_message)