"""
This module contains a collection of basic PDDL syntax validation functions. Users MUST pass in their validation
class instance into each formalization function in `DomainBuilder` or `ProblemBuilder`

For instance:

"""

from typing import List, Optional, Set

class SyntaxValidator:
    def __init__(self):    
        """
        Initializes an L2P custom syntax validator checker object.
        """
        self.errors: List[str] = []

    # DOMAIN VALIDATORS

    def validate_domain(self):
        pass

    def validate_requirements(self):
        pass

    def validate_types(self):
        pass

    def validate_constants(self):
        pass

    def validate_predicates(self):
        pass

    def validate_functions(self):
        pass

    def validate_constraints(self):
        pass

    def validate_der_predicates(self):
        pass

    def validate_parameters(self):
        pass

    def validate_preconditions(self):
        pass

    def validate_effects(self):
        pass

    def validate_action(self):
        pass

    def validate_actions(self):
        pass

    def validate_dur_conditions(self):
        pass

    def validate_dur_effects(self):
        pass

    def validate_dur_action(self):
        pass

    def validate_dur_actions(self):
        pass

    def validate_events(self):
        pass

    def validate_processes(self):
        pass


    # PROBLEM VALIDATORS'
    def validate_problem(self):
        pass

    def validate_objects(self):
        pass

    def validate_initial(self):
        pass

    def validate_goal(self):
        pass

    def validate_constraints(self):
        pass

    def validate_metric(self):
        pass