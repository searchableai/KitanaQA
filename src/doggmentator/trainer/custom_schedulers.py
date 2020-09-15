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
        max_val: float,
        min_val: float) -> Iterable:
    """
    Create a custom exponential scheduler
    """
    N0 = max_val
    N1 = np.log(min_val/max_val)/max_steps
    print(N0, N1)
    update_fn = lambda x: N0 * np.exp(N1 * x)
    return custom_scheduler(max_steps, update_fn)


def get_custom_linear(
        max_steps: int,
        start_val: float,
        end_val: float) -> Iterable:
    """
    Create a custom linear scheduler
    """
    N0 = min(start_val, end_val)
    N1 = max(start_val, end_val)
    N3 = (N1-N0)/max_steps
    if start_val > end_val:
        N3 *= -1
    update_fn = lambda x: N3 * x + start_val
    return custom_scheduler(max_steps, update_fn)
