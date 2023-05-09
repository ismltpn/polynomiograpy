from typing import Literal, Callable
import numpy as np
from polynomiograpy.common.polynomial import Polynomial
from . import helpers
from . import methods
from .methods import available_methods

__all__ = [
    "compute_screen_for_single_poly",
    "available_methods",
]


def compute_screen_for_single_poly(
    method: Literal[
        "newton",
        "halley",
        "inverse_interpolation",
        "mullers",
        "secant",
        "steffensen",
    ],
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
    assert method in available_methods, "Unknown method"
    func: Callable[[complex], int]
    if method == "newton":

        def func(val: complex) -> int:
            new_val, iter_count = methods.newton_method(
                poly,
                val,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    elif method == "halley":

        def func(val: complex) -> int:
            new_val, iter_count = methods.halley_method(
                poly,
                val,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    elif method == "inverse_interpolation":

        def func(val: complex) -> int:
            new_val, iter_count = methods.inverse_interpolation_method(
                poly,
                val,
                None,
                None,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    elif method == "mullers":

        def func(val: complex) -> int:
            new_val, iter_count = methods.mullers_method(
                poly,
                val,
                None,
                None,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    elif method == "secant":

        def func(val: complex) -> int:
            new_val, iter_count = methods.secant_method(
                poly,
                val,
                None,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    elif method == "steffensen":

        def func(val: complex) -> int:
            new_val, iter_count = methods.steffensen_method(
                poly,
                val,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    else:
        # cannot happen
        raise Exception("wtf")
    return helpers.compute_np_screen(
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
