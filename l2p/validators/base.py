from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

class ValidationResult:
    def __init__(self):
        self.valid: bool = True
        self.errors: List[str] = []

    def add_error(self, msg: str):
        self.valid = False
        self.errors.append(msg)


class ValidationRule(ABC):
    """Abstract base class for all L2P validation rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of rule (debugging purposes)."""
        pass

    @abstractmethod
    def validate(self, target: BaseModel, context: dict) -> ValidationResult:
        """
        Validates target model.
        Args:
            target (BaseModel)): Pydantic model being verified (e.g., Action).
            context (dict): A dictionary of existing domain components (e.g., {"types":[...]})
        """
        pass