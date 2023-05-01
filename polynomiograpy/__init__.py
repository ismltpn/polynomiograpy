from typing import Literal, Callable
from polynomiograpy.common.polynomial import Polynomial
from polynomiograpy.compute_screen import compute_screen
from polynomiograpy.methods import iter
import numpy as np

__version__ = "0.1.0"

__all__ = ["compute_screen_for_single_poly", "Polynomial"]


def compute_screen_for_single_poly(
    method: Literal["newton", "halley", "steffensen"],
    poly: Polynomial,
    delta: float,
    width: int,
    height: int,
    screen: np.ndarray,
    screen_buffer: np.ndarray,
    *,
    scale_x: float = 1,
    scale_y: float = 1,
    shift_x: float = 0,
    shift_y: float = 0,
    max_value: int = 16,
    reverse_color=False,
    channel: int = 0,
):
    assert method in {"newton", "halley", "steffensen"}, "Unknown method"
    func: Callable[[complex], int]
    if method == "newton":

        def func(val: complex) -> int:
            new_val, iter_count = iter.newton_method(
                poly,
                val,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    elif method == "halley":

        def func(val: complex) -> int:
            new_val, iter_count = iter.halley_method(
                poly,
                val,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    elif method == "steffensen":

        def func(val: complex) -> int:
            new_val, iter_count = iter.steffensen_method(
                poly,
                val,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    else:
        # cannot happen
        raise Exception("wtf")
    return compute_screen(
        func,
        width,
        height,
        screen,
        screen_buffer,
        scale_x=scale_x,
        scale_y=scale_y,
        shift_x=shift_x,
        shift_y=shift_y,
        max_value=max_value,
        reverse_color=reverse_color,
        channel=channel,
    )
