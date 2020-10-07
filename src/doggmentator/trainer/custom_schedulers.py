import numpy as np
from typing import Callable, Iterable


def custom_scheduler(
        max_steps: int,
        update_fn: Callable[[int], float]) -> float:
    """
    Create a custom generator for an input param
    """
    for step in range(max_steps):
        yield update_fn(step)


def get_custom_exp(
        max_steps: int,
        start_val: float,
        end_val: float) -> Iterable:
    """
    Create a custom exponential scheduler
    """
    assert isinstance(max_steps, int) and max_steps >= 1
    N0 = start_val
    N1 = np.log(start_val/end_val)/(max_steps-1)
    update_fn = lambda x: N0 * np.exp(N1 * x)
    return custom_scheduler(max_steps, update_fn)


def get_custom_linear(
        max_steps: int,
        start_val: float,
        end_val: float) -> Iterable:
    """
    Create a custom linear scheduler
    """
    assert isinstance(max_steps, int) and max_steps >= 1
    N1 = (end_val-start_val)/(max_steps-1)
    update_fn = lambda x: N1 * x + start_val
    return custom_scheduler(max_steps, update_fn)
