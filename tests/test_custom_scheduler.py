import pytest
import unittest
import types
from doggmentator.trainer.custom_schedulers import custom_scheduler, get_custom_exp, get_custom_linear

class TestGenerators(unittest.TestCase):
    def test_exp_scheduler(self):
        exp_scheduler = get_custom_exp(
                max_steps = 3,
                start_val = 1,
                end_val = 2
            )
        assert isinstance(exp_scheduler, types.GeneratorType)
        results = list(exp_scheduler)
        assert len(results) == 3
        assert pytest.approx(results[0], 1.)
        assert pytest.approx(results[-1], 2.)

    def test_linear_scheduler(self):
        lin_scheduler = get_custom_linear(
                max_steps = 3,
                start_val = 1,
                end_val = 2
            )
        assert isinstance(lin_scheduler, types.GeneratorType)
        results = list(lin_scheduler)
        assert len(results) == 3
        assert pytest.approx(results[0], 1.)
        assert pytest.approx(results[-1], 2.)
