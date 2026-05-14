from typing import Any, Dict, List, Type
from pydantic import BaseModel

from l2p.validators.base import SyntaxValidator, ValidationRule, ValidationResult

class UndeclaredTypeRule(ValidationRule):
    """Checks if an action uses types that haven't been declared in the domain."""
    
    @property
    def name(self) -> str:
        return "UndeclaredTypeRule"
        
    @property
    def target_models(self) -> List[Type[BaseModel]]:
        pass

    def validate(self, target: BaseModel, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult()
                
        return result


class EmptyActionsWarningRule(ValidationRule):
    """Warns the user if the domain contains no actions."""
    
    @property
    def name(self) -> str:
        return "EmptyActionsWarningRule"
        
    @property
    def target_models(self) -> List[Type[BaseModel]]:
        pass

    def validate(self, target: BaseModel, context: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult()
            
        return result

class DomainValidator(SyntaxValidator):
    """Validator specifically for PDDL Domain components."""
    
    def __init__(self, use_defaults: bool = True, custom_rules: List[ValidationRule] = None):
        """
        Args:
            use_defaults: If True, loads the standard L2P PDDL validation rules.
            custom_rules: An optional list of user-defined ValidationRules to add.
        """
        super().__init__()
        
        # load L2P validation
        if use_defaults:
            self.register_rule(UndeclaredTypeRule())
            self.register_rule(EmptyActionsWarningRule())
            # ...
            
        # load user-provided custom rules
        if custom_rules:
            for rule in custom_rules:
                self.register_rule(rule)