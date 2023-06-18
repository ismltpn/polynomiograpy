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
        "old_newton",
        "old_halley",
        "old_inverse_interpolation",
        "old_mullers",
        "old_secant",
        "old_steffensen",
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
    multithread: bool = False,
):
    """
    Computes a screen representation for a single polynomial by evaluating
    the specified method for each point in the complex plane defined by the
    screen dimensions.

    Args:
        method (Literal["newton", "halley", "inverse_interpolation", "mullers",
                "secant", "steffensen", "old_newton", "old_halley",
                "old_inverse_interpolation", "old_mullers", "old_secant",
                "old_steffensen"]):

            The method to use for computation. Must be one of the available methods.
        poly (Polynomial):
            The polynomial for which to compute the screen representation.
        delta (float):
            The tolerance value used by the method for convergence.
        width (int):
            Width of the screen.
        height (int):
            Height of the screen.
        screen (np.ndarray):
            Screen array to store the resulting representation.
        screen_buffer (np.ndarray):
            Temporary buffer array for intermediate calculations.
        scale_x (float, optional):
            Scaling factor for the x-axis. Defaults to 1.
        scale_y (float, optional):
            Scaling factor for the y-axis. Defaults to 1.
        shift_x (float, optional):
            Shift value for the x-axis. Defaults to 0.
        shift_y (float, optional):
            Shift value for the y-axis. Defaults to 0.
        max_value (int, optional):
            Maximum value used for color mapping. Defaults to 16.
        reverse_color (bool, optional):
            Flag to reverse the color mapping. Defaults to False.
        channel (int, optional):
            Color channel for color mapping. Defaults to 0.
        multithread (bool, optional):
            Flag indicating whether to use multithreading. Defaults to False.

    Returns:
        np.ndarray:
            The resulting screen representation for the single polynomial.

    Raises:
        AssertionError: If the specified method is not supported.

    Note:
        - The `method` argument specifies the numerical method to use for computation.
        - The `poly` argument should be an instance of the Polynomial class.
        - The `delta` argument defines the tolerance value used for convergence.
        - The resulting screen representation is stored in the `screen` array.
        - The color values are scaled to the range [0, 255].
        - By default, the function uses vectorized computation if available,
          otherwise it falls back to individual computation for each point.
        - Set the `multithread` flag to True to enable multithreading for not vectorized
          computation.

    """
    assert method in available_methods, "Unknown method"
    func: Callable[[complex], int]
    if method == "newton":

        def func(val: complex) -> int:
            iter_count = methods.newton_method_numpy(
                poly,
                val,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    elif method == "old_halley":

        def func(val: complex) -> int:
            new_val, iter_count = methods.halley_method(
                poly,
                val,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    elif method == "halley":

        def func(val: complex) -> int:
            iter_count = methods.halley_method_numpy(
                poly,
                val,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    elif method == "inverse_interpolation":

        def func(val: complex) -> int:
            iter_count = methods.inverse_interpolation_method_numpy(
                poly,
                val,
                None,
                None,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    elif method == "old_inverse_interpolation":

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
            iter_count = methods.mullers_method_numpy(
                poly,
                val,
                None,
                None,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    elif method == "old_mullers":

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
            iter_count = methods.secant_method_numpy(
                poly,
                val,
                None,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    elif method == "old_secant":

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
            iter_count = methods.steffensen_method_numpy(
                poly,
                val,
                delta,
                max_iter_count=max_value,
            )
            return iter_count

    elif method == "old_steffensen":

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
    if not method.startswith("old"):
        return helpers.compute_np_screen_vectorized(
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
    elif multithread:
        return helpers.compute_np_screen_multithread(
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
            thread_count=16,
        )
    else:
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
