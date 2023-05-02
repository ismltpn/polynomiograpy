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
