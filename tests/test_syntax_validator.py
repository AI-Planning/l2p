from l2p import *
import unittest, textwrap

class TestSyntaxValidator(unittest.TestCase):
    def setUp(self):
        self.syntax_validator = SyntaxValidator()

    def test_validate_params(self):
        
        types = {
            'arm': 'arm for a robot',
            'block': 'block that can be stacked and unstacked'
        }

        params = OrderedDict([('?a', 'arm'), ('?top', 'block'), ('?bottom', 'block')])
        
        # case 1: parameters are typed and types available
        flag, _ = self.syntax_validator.validate_params(parameters=params, types=types)
        self.assertEqual(flag, True)

        # case 2: parameters are untyped and types available
        untyped_params = OrderedDict([('?a', 'arm'), ('?top', ''), ('?bottom', '')])
        flag, _ = self.syntax_validator.validate_params(parameters=untyped_params, types=types)
        self.assertEqual(flag, True)

        # case 3: no types available
        flag, _ = self.syntax_validator.validate_params(parameters=params)
        self.assertEqual(flag, False)
        
        # case 3: parameter types not found in types
        incorrect_params = OrderedDict([('?a', 'table'), ('?top', 'block'), ('?bottom', 'block')])
        flag, _ = self.syntax_validator.validate_params(parameters=incorrect_params, types=types)
        self.assertEqual(flag, False)

        # case 4: parameter does not contain `?` in front of it
        missing_char_params = OrderedDict([('a', 'arm'), ('?top', 'block'), ('bottom', 'block')])
        flag, _ = self.syntax_validator.validate_params(parameters=missing_char_params, types=types)
        self.assertEqual(flag, False)

    
    def test_validate_types_predicates(self):

        types = {
            'arm': 'arm for a robot',
            'block': 'block that can be stacked and unstacked',
            'table': 'table that blocks sits on'
        }

        predicates = [Predicate({'name': 'on_top', 
                       'desc': 'true if the block ?b1 is on top of the block ?b2', 
                       'raw': '(on_top ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2', 
                       'params': OrderedDict([('?b1', 'block'), ('?b2', 'block')]), 
                       'clean': '(on_top ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2'})]

        # case 1: correct predicates and types
        flag, _ = self.syntax_validator.validate_types_predicates(predicates=predicates, types=types)
        self.assertEqual(flag, True)
        
        # case 2: no types
        flag, _ = self.syntax_validator.validate_types_predicates(predicates=predicates)
        self.assertEqual(flag, True)
        
        # case 3: predicate name is the same as a type
        incorrect_predicates = [
            Predicate({'name': 'on_top', 
                       'desc': 'true if the block ?b1 is on top of the block ?b2', 
                       'raw': '(on_top ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2', 
                       'params': OrderedDict([('?b1', 'block'), ('?b2', 'block')]), 
                       'clean': '(on_top ?b1 - block ?b2 - block)'}),
            Predicate({'name': 'under', 
                       'desc': 'true if the block ?b1 is under block ?b2', 
                       'raw': '(under ?b1 - block ?b2 - block): true if the block ?b1 is under block ?b2', 
                       'params': OrderedDict([('?b1', 'block'), ('?b2', 'block')]), 
                       'clean': '(under ?b1 - block ?b2 - block)'})
            ]
        
        types = {
            'on_top': 'true if the block ?b1 is on top of the block ?b2',
            'under': 'true if the block ?b1 is under block ?b2', 
            'table': 'table that blocks sits on'
            }
        
        flag, _ = self.syntax_validator.validate_types_predicates(predicates=incorrect_predicates, types=types)
        self.assertEqual(flag, False)
        
    
    def test_validate_duplicate_predicates(self):

        # case 1: duplicate predicate names but same parameters (identical predicates)
        predicates_1 = [
            Predicate({'name': 'on_top', 
                        'desc': 'true if the block ?b1 is on top of the block ?b2', 
                        'raw': '(on_top ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2', 
                        'params': OrderedDict([('?b1', 'block'), ('?b2', 'block')]), 
                        'clean': '(on_top ?b1 - block ?b2 - block)'}),
            Predicate({'name': 'under', 
                       'desc': 'true if the block ?b1 is under block ?b2', 
                       'raw': '(under ?b1 - block ?b2 - block): true if the block ?b1 is under block ?b2', 
                       'params': OrderedDict([('?b1', 'block'), ('?b2', 'block')]), 
                       'clean': '(under ?b1 - block ?b2 - block)'})
        ]
    
        predicates_2 = [
            Predicate({'name': 'on_top', 
                        'desc': 'true if the block ?b1 is on top of the block ?b2', 
                        'raw': '(on_top ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2', 
                        'params': OrderedDict([('?b1', 'block'), ('?b2', 'block')]), 
                        'clean': '(on_top ?b1 - block ?b2 - block)'})
        ]

        flag, _ = self.syntax_validator.validate_duplicate_predicates(curr_predicates=predicates_1, new_predicates=predicates_2)
        self.assertEqual(flag, True)

        # case 2: duplicate predicate names but different parameters
        predicates_3 = [
            Predicate({'name': 'on_top', 
                        'desc': 'true if the block ?b1 is on top of the block ?b2', 
                        'raw': '(on_top ?b2 - block ?b1 - block): true if the block ?b1 is on top of the block ?b2', 
                        'params': OrderedDict([('?b2', 'block'), ('?b1', 'block')]), 
                        'clean': '(on_top ?b2 - block ?b1 - block)'})
        ]

        flag, _ = self.syntax_validator.validate_duplicate_predicates(curr_predicates=predicates_1, new_predicates=predicates_3)
        self.assertEqual(flag, False)

        # case 3: different predicates
        predicates_4 = [
            Predicate({'name': 'next_to', 
                       'desc': 'true if the block ?b1 is next to block ?b2', 
                       'raw': '(next_to ?b1 - block ?b2 - block): true if the block ?b1 is next_to block ?b2', 
                       'params': OrderedDict([('?b1', 'block'), ('?b2', 'block')]), 
                       'clean': '(next_to ?b1 - block ?b2 - block)'})
        ]
        flag, _ = self.syntax_validator.validate_duplicate_predicates(curr_predicates=predicates_1, new_predicates=predicates_4)
        self.assertEqual(flag, True)


    def test_validate_format_predicate(self):
        
        # case 1: predicates are formatted correctly w/ correct predicate name, parameter string syntax
        predicates = [
            Predicate({'name': 'on_top', 
                        'desc': 'true if the block ?b1 is on top of the block ?b2', 
                        'raw': '(on_top ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2', 
                        'params': OrderedDict([('?b1', 'block'), ('?b2', 'block')]), 
                        'clean': '(on_top ?b1 - block ?b2 - block)'})
        ]

        types = {
            'arm': 'arm for a robot',
            'block': 'block that can be stacked and unstacked',
            'table': 'table that blocks sits on'
        }

        flag, _ = self.syntax_validator.validate_format_predicates(predicates=predicates, types=types)
        self.assertEqual(flag, True)

        # case 2: predicate name is missing / starts with incorrect `?`
        pred_missing_name = [
            Predicate({'name': '', 
                        'desc': 'true if the block ?b1 is on top of the block ?b2', 
                        'raw': '(?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2', 
                        'params': OrderedDict([('?b1', 'block'), ('?b2', 'block')]), 
                        'clean': '(?b1 - block ?b2 - block)'})
        ]

        flag, _ = self.syntax_validator.validate_format_predicates(predicates=pred_missing_name, types=types)
        self.assertEqual(flag, False)

        # case 3: parameter variables do not start with `?` syntax
        pred_incorrect_var = [
            Predicate({'name': 'on_top', 
                        'desc': 'true if the block ?b1 is on top of the block ?b2', 
                        'raw': '(on_top b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2', 
                        'params': OrderedDict([('b1', 'block'), ('?b2', 'block')]), 
                        'clean': '(on_top b1 - block ?b2 - block)'})
        ]

        flag, _ = self.syntax_validator.validate_format_predicates(predicates=pred_incorrect_var, types=types)
        self.assertEqual(flag, False)

        # other placement
        pred_incorrect_var = [
            Predicate({'name': 'on_top', 
                        'desc': 'true if the block ?b1 is on top of the block ?b2', 
                        'raw': '(on_top ?b1 - block b2 - block): true if the block ?b1 is on top of the block ?b2', 
                        'params': OrderedDict([('?b1', 'block'), ('b2', 'block')]), 
                        'clean': '(on_top ?b1 - block b2 - block)'})
        ]

        flag, _ = self.syntax_validator.validate_format_predicates(predicates=pred_incorrect_var, types=types)
        self.assertEqual(flag, False)

        # case 4: missing `-` syntax or using some other random character
        pred_incorrect_char = [
            Predicate({'name': 'on_top', 
                        'desc': 'true if the block ?b1 is on top of the block ?b2', 
                        'raw': '(on_top ?b1 - block ?b2 - ): true if the block ?b1 is on top of the block ?b2', 
                        'params': OrderedDict([('?b1', 'block'), ('?b2', '')]), 
                        'clean': '(on_top ?b1 - block ?b2 - )'})
        ]

        flag, _ = self.syntax_validator.validate_format_predicates(predicates=pred_incorrect_char, types=types)
        self.assertEqual(flag, False)

        pred_incorrect_char = [
            Predicate({'name': 'on_top', 
                        'desc': 'true if the block ?b1 is on top of the block ?b2', 
                        'raw': '(on_top ?b1 | block ?b2 | block): true if the block ?b1 is on top of the block ?b2', 
                        'params': OrderedDict([('?b1', 'block'), ('?b2', 'block')]), 
                        'clean': '(on_top ?b1 | block ?b2 | block)'})
        ]

        flag, _ = self.syntax_validator.validate_format_predicates(predicates=pred_incorrect_char, types=types)
        self.assertEqual(flag, False)

        # case 5: variable is untyped (allowed in PDDL)
        pred_untyped = [
            Predicate({'name': 'on_top', 
                        'desc': 'true if the block ?b1 is on top of ?x', 
                        'raw': '(on_top ?b1 - block ?x): true if the block ?b1 is on top of ?x', 
                        'params': OrderedDict([('?b1', 'block'), ('?x', '')]), 
                        'clean': '(on_top ?b1 - block ?x)'}),
            Predicate({'name': 'bottom', 
                        'desc': '', 
                        'raw': '(bottom ?b1 - block ?x ?y ?z): ', 
                        'params': OrderedDict([('?b1', 'block'), ('?x', ''), ('?y', ''), ('?z', '')]), 
                        'clean': '(bottom ?b1 - block ?x ?y ?z)'})
        ]

        flag, _ = self.syntax_validator.validate_format_predicates(predicates=pred_untyped, types=types)
        self.assertEqual(flag, True)

        # case 6: parameter types are not found in types list
        pred_incorrect_param_type = [
            Predicate({'name': 'on_top', 
                        'desc': 'true if the block ?b1 is on top of the block ?b2', 
                        'raw': '(on_top ?b1 - box ?b2 - triangle): true if the box ?b1 is on top of the triangle ?b2', 
                        'params': OrderedDict([('?b1', 'box'), ('?b2', 'triangle')]), 
                        'clean': '(on_top ?b1 - box ?b2 - triangle)'})
        ]

        flag, _ = self.syntax_validator.validate_format_predicates(predicates=pred_incorrect_param_type, types=types)
        self.assertEqual(flag, False)

        # no types exist
        flag, _ = self.syntax_validator.validate_format_predicates(predicates=pred_incorrect_param_type)
        self.assertEqual(flag, False)


    def test_validate_pddl_usage_predicates(self):

        # case 1: correct implementation – predicate aligns with action parameters and types
        types = {
            'arm': 'arm for a robot',
            'block': 'block that can be stacked and unstacked',
            'table': 'table that blocks sits on'
        }

        predicates = [
            Predicate({'name': 'on', 
                    'desc': 'true if the block ?b1 is on top of the block ?b2', 
                    'raw': '(on ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2', 
                    'params': OrderedDict([('?b1', 'block'), ('?b2', 'block')]), 
                    'clean': '(on ?b1 - block ?b2 - block)'}),
            Predicate({'name': 'holding', 
                    'desc': 'true if arm is holding a block', 
                    'raw': '(holding ?a - arm ?b - block): true if arm is holding a block', 
                    'params': OrderedDict([('?a', 'arm'), ('?b', 'block')]), 
                    'clean': '(holding ?a - arm ?b - block)'}),
            Predicate({'name': 'clear', 
                    'desc': 'true if a block does not have anything on top of it', 
                    'raw': '(clear ?b - block): true if a block does not have anything on top of it', 
                    'params': OrderedDict([('?b', 'block')]), 
                    'clean': '(clear ?b - block)'}),
            ]
        
        precond_str = "( and      ( holding ?a ?b1 )  ; The arm is holding the top block      (clear ?b2 )  ; The bottom block is clear  )"
        params_info = OrderedDict({'?b1': 'block', '?b2': 'block', '?a': 'arm'}), ['- ?b1 - block: The block being stacked on top', '- ?b2 - block: The block being stacked upon', '- ?a - arm: The arm performing the stacking action']

        flag, msg = self.syntax_validator.validate_pddl_usage_predicates(
            pddl=precond_str,
            predicates=predicates,
            action_params=params_info,
            types=types,
            part="preconditions"
        )
        self.assertEqual(flag, True)

        # case 2: incorrect number of predicate parameters in pddl component
        precond_str = "( and      ( holding ?b1 )  ; The arm is holding the top block      (clear ?b2 )  ; The bottom block is clear  )"
        flag, msg = self.syntax_validator.validate_pddl_usage_predicates(
            pddl=precond_str,
            predicates=predicates,
            action_params=params_info,
            types=types,
            part="preconditions"
        )
        self.assertEqual(flag, False)

        # case 3: predicate parameters include object types
        precond_str = "( and      ( holding ?a - arm ?b1 - block)  ; The arm is holding the top block      (clear ?b2 )  ; The bottom block is clear  )"
        flag, msg = self.syntax_validator.validate_pddl_usage_predicates(
            pddl=precond_str,
            predicates=predicates,
            action_params=params_info,
            types=types,
            part="preconditions"
        )
        self.assertEqual(flag, False)

        # case 3: parameters declared in predicate not found in action parameter
        precond_str = "( and      ( holding ?b ?b1)  ; The arm is holding the top block      (clear ?b2 )  ; The bottom block is clear  )"

        flag, msg = self.syntax_validator.validate_pddl_usage_predicates(
            pddl=precond_str,
            predicates=predicates,
            action_params=params_info,
            types=types,
            part="preconditions"
        )
        self.assertEqual(flag, False)

        # check if declared predicate object types align with original predicate types



    def test_validate_usage_predicates(self):
        pass
    def test_validate_overflow_predicates(self):
        pass
    def test_validate_task_states(self):
        pass
    def test_validate_header(self):
        pass
    def test_unsupported_keywords(self):
        pass
    def test_validate_keyword_usage(self):
        pass
    def test_validate_new_saction_creation(self):
        pass
    def test_validate_type(self):

        target_type = 'arm'

        claimed_type = 'arm_pit'

        types = {
            'arm': 'arm for a robot',
            'block': 'block that can be stacked and unstacked',
            'table': 'table that blocks sits on'
        }
        
        
        flag, msg = self.syntax_validator.validate_type(
            target_type=target_type, 
            claimed_type=claimed_type, 
            types=types
            )
        
        print(flag)
        print(msg)



if __name__ == "__main__":
    unittest.main()

    # llm_response = textwrap.dedent(
    #     """
    #     ### Action Parameters
    #     ```
    #     - ?b1 - block: The block being stacked on top
    #     - ?b2 - block: The block being stacked upon
    #     - ?a - arm: The arm performing the stacking action
    #     ```

    #     ### Action Preconditions
    #     ```
    #     (and
    #         (holding ?a ?b1) ; The arm is holding the top block
    #         (clear ?b2) ; The bottom block is clear
    #     )
    #     ```

    #     ### Action Effects
    #     ```
    #     (and
    #         (not (holding ?a ?b1)) ; The arm is no longer holding the top block
    #         (on ?b1 ?b2) ; The top block is now on the bottom block
    #         (not (clear ?b2)) ; The bottom block is no longer clear
    #     )
    #     ```

    #     ### New Predicates
    #     ```
    #     - (holding ?a - arm ?b - block): true if the arm ?a is holding the block ?b
    #     - (clear ?b - block): true if the block ?b is clear and can be stacked upon
    #     - (on ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2
    #     ```
    #     """
    # )

    # params_info = parse_params(llm_response)

    # print(params_info[0])
    # print(params_info[1])