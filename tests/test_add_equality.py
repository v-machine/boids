import numpy as np
import numpy.testing as npt
import unittest

def ndarray_compare(arr1: np.ndarray, arr2: np.ndarray,
                    msg="numpy array not equal") -> bool:
    """numpy array comparison function to be added to unittest.Testcase"""
    return npt.assert_almost_equal(arr1, arr2)

class TestAddEquality(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, ndarray_compare)
    
    def test_nparray_equality(self):
        self.assertEqual(np.zeros(4), np.asarray([0, 0, 0, 0.00000000000001]))
    
    def test_decoy(self):
        self.assertEqual(0.25, 0.25)

if __name__ == "__main__":
    unittest.main()
