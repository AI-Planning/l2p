from l2p import *
import unittest, textwrap

class TestSyntaxValidator(unittest.TestCase):
    def setUp(self):
        self.syntax_validator = SyntaxValidator()

    def test_validate_params(self):
        
        types = {
            'arm': 'arm for a robot',
            'block': 'block that can be stacked and unstacked',
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
        }

        predicates = [Predicate({'name': 'on_top', 
                       'desc': 'true if the block ?b1 is on top of the block ?b2', 
                       'raw': '(on_top ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2', 
                       'params': OrderedDict([('?b1', 'block'), ('?b2', 'block')]), 
                       'clean': '(on_top ?b1 - block ?b2 - block): true if the block ?b1 is on top of the block ?b2'})]

        flag, message = self.syntax_validator.validate_types_predicates(predicates=predicates, types=types)
        print(flag)
        print(message)


if __name__ == "__main__":
    unittest.main()