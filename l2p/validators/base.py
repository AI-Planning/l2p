from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type
from pydantic import BaseModel


class ValidationResult:
    """Stores the outcome of a validation run, errors, and warnings."""
    def __init__(self):
        self.valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, msg: str):
        """Adds a fatal error that fails validation."""
        self.valid = False
        self.errors.append(msg)
        
    def add_warning(self, msg: str):
        """Adds a non-fatal warning for the user."""
        self.warnings.append(msg)


class ValidationRule(ABC):
    """Abstract base class for all L2P validation rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the rule (for debugging purposes)."""
        pass

    @property
    @abstractmethod
    def target_models(self) -> List[Type[BaseModel]]:
        """List of Pydantic model classes this rule applies to."""
        pass

    @abstractmethod
    def validate(self, target: BaseModel, context: Dict[str, Any]) -> ValidationResult:
        """
        Validates target model.
        Args:
            target (BaseModel): Pydantic model being verified.
            context (dict): Dictionary of existing domain/problem components.
        """
        pass


class SyntaxValidator(ABC):
    """Base orchestrator for running validation rules."""
    
    def __init__(self):
        self.rules: List[ValidationRule] = []

    def register_rule(self, rule: ValidationRule):
        """Adds a new rule to the validation pipeline."""
        self.rules.append(rule)

    def validate_component(self, target: BaseModel, context: Dict[str, Any]) -> ValidationResult:
        """Runs all applicable rules against the target component."""
        final_result = ValidationResult()
        
        for rule in self.rules:
            if type(target) in rule.target_models:
                result = rule.validate(target, context)
                
                if not result.valid:
                    final_result.valid = False
                    final_result.errors.extend(result.errors)
                
                final_result.warnings.extend(result.warnings)
                
        return final_result