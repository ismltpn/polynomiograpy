import numpy as np

from polynomiograpy.common.finite_field import FiniteField
from . import helpers


def compute_screen_for_finite_field_poly(
    ff: FiniteField,
    min_degree: int,
    max_degree: int,
    width: int,
    height: int,
    screen: np.ndarray,
    screen_buffer: np.ndarray,
    *,
    scale_x: float = 0.01,
    scale_y: float = 0.01,
    shift_x: float = 0,
    shift_y: float = 0,
    color_range: int = 8,
    channel: int = 0,
):
    roots: list[int] = []
    for deg in range(min_degree, max_degree + 1):
        for poly in ff.generate_polynomials(deg):
            roots.extend(poly.roots())
    return helpers.compute_screen_for_roots(
        roots,
        width,
        height,
        screen,
        screen_buffer,
        scale_x=scale_x,
        scale_y=scale_y,
        shift_x=shift_x,
        shift_y=shift_y,
        color_range=color_range,
        channel=channel,
    )


def compute_screen_for_finite_field_poly_multi_color(
    ff: FiniteField,
    min_degree: int,
    max_degree: int,
    width: int,
    height: int,
    screen: np.ndarray,
    screen_buffer: np.ndarray,
    *,
    scale_x: float = 0.01,
    scale_y: float = 0.01,
    shift_x: float = 0,
    shift_y: float = 0,
    color_range: int = 8,
):
    roots: list[list[int]] = []
    for deg in range(min_degree, max_degree + 1):
        roots_: list[int] = []
        for poly in ff.generate_polynomials(deg):
            roots_.extend(poly.roots())
        roots.append(roots_)
    return helpers.compute_screen_for_roots_multi_color(
        roots,
        width,
        height,
        screen,
        screen_buffer,
        scale_x=scale_x,
        scale_y=scale_y,
        shift_x=shift_x,
        shift_y=shift_y,
        color_range=color_range,
        colors=[
            np.array([255, 0, 0]),  # red
            np.array([204, 0, 51]),
            np.array([153, 0, 102]),
            np.array([102, 0, 153]),
            np.array([51, 0, 204]),
            np.array([0, 0, 255]),  # blue
            np.array([0, 51, 204]),
            np.array([0, 102, 153]),
            np.array([0, 153, 102]),
            np.array([0, 204, 51]),
        ],
    )
