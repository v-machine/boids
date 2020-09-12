#!/usr/bin/env python3

import json
import numpy as np
import unittest
from typing import Dict, Callable
from boids.boids import Boids

"""Unit testing for class Boids."""

class TestBoids(unittest.TestCase):
    """Unit test suite for boids
       
       Testing all methods in the Boids' class. Test cases are stored in
       test_data.json. 
    """
    ARGS_TO_ATTRS = {
        "dims": "dims",
        "num_boids": "num_boids",
        "environ_bounds": "env_bounds",
        "max_velocity": "max_vel",
        "max_acceleration": "max_acc",
        "perceptual_range": "p_range"
    }

    @classmethod
    def setUpClass(cls):
        """loads test cases from a json file"""
        with open("tests/test_data.json", 'r') as f:
            cls.cases = json.load(f)

    def test_init(self):
        for idx, case in TestBoids.cases.items():
            with self.subTest(idx):
                args = case["args"]
                b = Boids(**args)
                for arg, val in args.items():
                    self.assertEqual(b.__getattribute__(
                        TestBoids.ARGS_TO_ATTRS[arg]), val)
                self.assertEqual(sum(b.rules.values()), 1)

    def test_init_boids_state(self):
        for idx, case in TestBoids.cases.items():
            with self.subTest(idx):
                args = case["args"]
                b = Boids(**args)
                self.assertEqual(b.state.shape, (args["dims"], args["num_boids"],
                                                 len(Boids.Attr)))
                for idx, limit in zip((Boids.Attr.VEL, Boids.Attr.ACC),
                                      (b.max_vel, b.max_acc)):
                    self.assertLessEqual(np.max(b.state[:, :, idx]), limit)
                for dim in range(b.dims):
                    self.assertLessEqual(np.max(b.state[dim, :, Boids.Attr.LOC]),
                                         b.env_bounds[dim])

    def test_perceive(self):
        pass
    
    def test_align(self):
        pass 
    
    def test_update_acc(self):
        pass
    
    def test_update_acc_by_rules(self):
        pass

    def test_update_vel(self):
        pass

    def test_update_loc(self):
        pass

    def test_append_rules(self):
        pass

    def test_update_rule_coeffs(self):
        pass

    def test_swarm(self):
        pass

if __name__ == "__main__":
    unittest.main()
