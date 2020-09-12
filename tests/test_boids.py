#!/usr/bin/env python3

import numpy as np
import numpy.testing as npt
import unittest
from collections import namedtuple
from typing import Dict, Callable
from boids.boids import Boids
from tests.parameterized import load_json, Subtest

"""Unit testing for class Boids.

author: Vincent Mai
version: 0.1.0
"""

PATH = "tests/test_data.json" 
CASES = load_json(PATH)
CASES_INIT           = CASES["test_init"]
CASES_PERCEIVE       = CASES["test_perceive"]
CASES_ALIGN          = CASES["test_align"]
# CASES_SEPARATE       = CASES["test_separate"]
# CASES_COHERE         = CASES["test_cohere"]
CASES_UPDATE_VEL     = CASES["test_update_velocity"]
CASES_UPDATE_LOC     = CASES["test_update_location"]
CASES_UPDATE_ACC     = CASES["test_update_acceleration"]
# CASES_UPDATE_BY_RULE = CASES["test_update_acc_by_rule"]
# CASES_APPEND_RULE    = CASES["test_append_rules"]
# CASES_UPDATE_COEFF   = CASES["test_update_coeff"]
# CASES_SWARM          = CASES["test_swarm"]

INIT="init"
INIT_STATE="init_state"
ARGS="args"
INPUT="input"
EXPECTED="expected"

def ndarray_compare(arr1: np.ndarray, arr2: np.ndarray,
                    msg=None) -> bool:
    """numpy array comparison function to be added to unittest.Testcase"""
    return npt.assert_almost_equal(arr1, arr2)

class TestBoids(unittest.TestCase):
    """Unit test suite for boids
       
       Testing all methods in the Boids' class. Test cases are stored in
       test_data.json. 
    """

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, ndarray_compare)
    
    @Subtest(CASES_INIT)
    def test_init(self, case):
        """Verify that all args are passed in correctly."""
        ARGS_TO_ATTRS = {
            "dims": "dims",
            "num_boids": "num_boids",
            "environ_bounds": "env_bounds",
            "max_velocity": "max_vel",
            "max_acceleration": "max_acc",
            "perceptual_range": "p_range"
        }
        args = case[ARGS]
        b = Boids(**args)
        for arg, val in args.items():
            self.assertEqual(b.__getattribute__(ARGS_TO_ATTRS[arg]), val)
        self.assertEqual(sum(b.rules.values()), 1)
    
    @Subtest(CASES_INIT)
    def test_init_boids_state(self, case):
        """Verifies state shape and state values are within bounds."""
        args = case[ARGS]
        b = Boids(**args)
        self.assertEqual(b.state.shape, tuple(case[EXPECTED]))
        for idx, limit in zip((Boids.Attr.VEL, Boids.Attr.ACC),
                              (b.max_vel, b.max_acc)):
            self.assertLessEqual(np.max(b.state[:, :, idx]), limit)
        for dim in range(b.dims):
            self.assertLessEqual(np.max(b.state[dim, :, Boids.Attr.LOC]),
                                 b.env_bounds[dim])
    
    @Subtest(CASES_PERCEIVE)
    def test_perceive(self, case):
        """
        Tests the correctness of perceptual matrix when neighbors are
        within or outside of perceptual range
        """
        b = Boids(**case[INIT])
        b.state = np.asarray(case[ARGS])
        self.assertEqual(np.asarray(case[EXPECTED]),
                         b._perceive())
        
    @Subtest(CASES_ALIGN)
    def test_align(self, case):
        b = Boids(**case[INIT])
        b.state = np.asarray(case[INIT_STATE])
        self.assertEqual(np.asarray(case[EXPECTED]),
                         b.align(np.asarray(case[ARGS]))) 
    
    @Subtest(CASES_UPDATE_ACC)
    def test_update_acc(self, case):
        b = Boids(**case[INIT])
        b.state = np.asarray(case[INIT_STATE], dtype=float)
        b._update_acc(np.asarray(case[ARGS]["acc_delta"]), case[ARGS]["coeff"])
        self.assertEqual(np.asarray(case[EXPECTED]), b.state)
    
    @Subtest(CASES)
    def test_update_acc_by_rules(self, case):
        pass

    @Subtest(CASES_UPDATE_VEL)
    def test_update_vel(self, case):
        b = Boids(**case[INIT])
        b.state = np.asarray(case[ARGS], dtype=float)
        b._update_vel()
        self.assertEqual(np.asarray(case[EXPECTED]), b.state)

    @Subtest(CASES_UPDATE_LOC)
    def test_update_loc(self, case):
        b = Boids(**case[INIT])
        b.state = np.asarray(case[ARGS], dtype=float)
        b._update_loc()
        self.assertEqual(np.asarray(case[EXPECTED]), b.state)
        
    @Subtest(CASES)
    def test_append_rules(self, case):
        pass

    @Subtest(CASES)
    def test_update_rule_coeffs(self, case):
        pass

    @Subtest(CASES)
    def test_swarm(self, case):
        pass

if __name__ == "__main__":
    unittest.main()
