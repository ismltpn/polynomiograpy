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
    """
    Computes a screen representation of roots of polynomials over a finite field.

    Args:
        ff (:obj:`FiniteField`):
            FiniteField object representing the finite field to generate polynomials
            from.
        min_degree (int):
            Minimum degree of the polynomials to consider.
        max_degree (int):
            Maximum degree of the polynomials to consider.
        width (int):
            Width of the screen.
        height (int):
            Height of the screen.
        screen (:obj:`numpy.ndarray`):
            Screen array to store the resulting representation.
        screen_buffer (:obj:`numpy.ndarray`):
            Temporary buffer array for intermediate calculations.
        scale_x (float, optional):
            Scaling factor for the x-axis. Defaults to 0.01.
        scale_y (float, optional):
            Scaling factor for the y-axis. Defaults to 0.01.
        shift_x (float, optional):
            Shift value for the x-axis. Defaults to 0.
        shift_y (float, optional):
            Shift value for the y-axis. Defaults to 0.
        color_range (int, optional):
            Number of color shades to represent the roots. Defaults to 8.
        channel (int, optional):
            Color channel to store the computed representation. Defaults to 0.

    Returns:
        :obj:`numpy.ndarray`:
            The resulting screen representation with the roots of polynomials over the
            finite field plotted.

    Note:
        - The function internally generates polynomials over the given finite field
          with degrees ranging from `min_degree` to `max_degree`.
        - The function then computes the roots of these polynomials and uses the
          `compute_screen_for_roots` helper function to generate the screen
          representation.
        - The input screen and screen_buffer arrays are modified in-place.
        - The function assumes that the input screen and screen_buffer arrays
          have the correct shape and dtype.

    """
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
    colors: list[np.ndarray] = [
        np.array([255, 0, 0]),
        np.array([204, 0, 51]),
        np.array([153, 0, 102]),
        np.array([102, 0, 153]),
        np.array([51, 0, 204]),
        np.array([0, 0, 255]),
        np.array([0, 51, 204]),
        np.array([0, 102, 153]),
        np.array([0, 153, 102]),
        np.array([0, 204, 51]),
    ],
):
    """
    Computes a multi-color screen representation of roots of polynomials over a
    finite field.

    Args:
        ff (:obj:`FiniteField`):
            FiniteField object representing the finite field to generate
            polynomials from.
        min_degree (int):
            Minimum degree of the polynomials to consider.
        max_degree (int):
            Maximum degree of the polynomials to consider.
        width (int):
            Width of the screen.
        height (int):
            Height of the screen.
        screen (:obj:`numpy.ndarray`):
            Screen array to store the resulting representation.
        screen_buffer (:obj:`numpy.ndarray`):
            Temporary buffer array for intermediate calculations.
        scale_x (float, optional):
            Scaling factor for the x-axis. Defaults to 0.01.
        scale_y (float, optional):
            Scaling factor for the y-axis. Defaults to 0.01.
        shift_x (float, optional):
            Shift value for the x-axis. Defaults to 0.
        shift_y (float, optional):
            Shift value for the y-axis. Defaults to 0.
        color_range (int, optional):
            Number of color shades to represent the roots. Defaults to 8.
        colors (list[:obj:`numpy.ndarray`], optional):
            List of color arrays corresponding to each degree of roots.
            Defaults to a pre-defined list of colors.

    Returns:
        :obj:`numpy.ndarray`:
            The resulting screen representation with the roots of polynomials
            over the finite field plotted using multiple colors.

    Note:
        - The function internally generates polynomials over the given finite
          field with degrees ranging from `min_degree` to `max_degree`.
        - The function then computes the roots of these polynomials and uses
          the `compute_screen_for_roots_multi_color` helper function to generate
          the screen representation.
        - The input screen and screen_buffer arrays are modified in-place.
        - The function assumes that the input screen and screen_buffer arrays
          have the correct shape and dtype.
        - If the `colors` parameter is not provided, a pre-defined list of colors
          is used.

    """
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
        colors=colors,
    )
