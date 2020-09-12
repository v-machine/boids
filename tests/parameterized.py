import json
from typing import Dict, Callable, Iterable

"""Enhances parameterization of python's default unittest module

Utility module designed to ease parameterization in unit testing with
python's default module unittest. Included functionalities are data
loading and decorator for subtesting.
"""

def load_json(path: str) -> Dict:
    """Loads data from a json file and returns a dict
    
    Param:
        path: str
            The file path to the json data
    Returns:
        data as dictionary
    """ 
    with open(path, 'r') as f:
        data = json.load(f)
    return data

class Subtest():
    """Decorator class that creates parameterized subtests
    
    Params:
        test_cases: Iterable[str, Iterable]
            An iterable of test case marker and the test cases

    Returns:
        decorated test: Callable
            The decorated test will run on each test case as a separate test
    Example:
        test_cases = (
            ('integer_case', 17, 59, 76),
            ('complex_case', complex(5, 3), complex(7, 4), complex(12, 7)))
        
        class TestSum(unittest.testCase):
        ...

            @Subtest(test_cases)
            def test_sum(self, case[1:]):
                args, expected = case[:-1], case[-1]
                self.assertEqual(sum(*args), expected)

    Note:
        Designed to work with python unittest module only.
    """
    def __init__(self, test_cases: Iterable):
        self.test_cases = test_cases
    
    def __call__(self, fn):
        def _test(fn_self):
            for maker, case in self.test_cases.items():
                with fn_self.subTest(maker):
                    fn(fn_self, case)
        return _test

def main():
    pass

if __name__ == "__main__":
    main()

