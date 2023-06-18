import numpy as np


def compute_screen_for_roots(
    roots: list[int],
    width: int,
    height: int,
    screen: np.ndarray,
    screen_buffer: np.ndarray,
    *,
    scale_x: float = 1,
    scale_y: float = 1,
    shift_x: float = 0,
    shift_y: float = 0,
    color_range: int = 8,
    channel: int = 0,
):
    """
    Computes a screen representation of roots on a complex plane.

    Args:
        roots (list[int]):
            List of complex numbers (roots) to be plotted on the screen.
        width (int):
            Width of the screen.
        height (int):
            Height of the screen.
        screen (:obj:`numpy.ndarray`):
            Screen array to store the resulting representation.
        screen_buffer (:obj:`numpy.ndarray`):
            Temporary buffer array for intermediate calculations.
        scale_x (float, optional):
            Scaling factor for the x-axis. Defaults to 1.
        scale_y (float, optional):
            Scaling factor for the y-axis. Defaults to 1.
        shift_x (float, optional):
            Shift value for the x-axis. Defaults to 0.
        shift_y (float, optional):
            Shift value for the y-axis. Defaults to 0.
        color_range (int, optional):
            Number of color shades to represent the roots. Defaults to 8.
        channel (int, optional):
            Color channel index to store the computed representation. Defaults to 0.

    Returns:
        :obj:`numpy.ndarray`:
            The resulting screen representation with the roots plotted.

    Note:
        - The function modifies the input screen and screen_buffer arrays in-place.
        - The function assumes that the input screen and screen_buffer arrays have
          the correct shape and dtype.

    """
    screen_buffer.fill(0)
    origin_x = width / 2
    origin_y = height / 2
    color_intensity_per_root = 256 // color_range
    for root in roots:
        i = int((root.real - shift_x) / scale_x + origin_x)
        j = int(((root.imag - shift_y) / scale_y + origin_y))
        if i < 0 or j < 0 or i >= width or j >= height:
            pass
        else:
            screen_buffer[j, i, channel] += color_intensity_per_root
            if screen_buffer[j, i, channel] > 255:
                screen_buffer[j, i, channel] = 255
    screen[:, :, channel] = screen_buffer[:, :, channel]
    return np.flipud(screen)


def compute_screen_for_roots_multi_color(
    roots: list[list[int]],
    width: int,
    height: int,
    screen: np.ndarray,
    screen_buffer: np.ndarray,
    *,
    scale_x: float = 1,
    scale_y: float = 1,
    shift_x: float = 0,
    shift_y: float = 0,
    color_range: int = 8,
    colors: list[np.ndarray],
):
    """
    Computes a multi-color screen representation of roots on a complex plane.

    Args:
        roots (list[list[int]]):
            List of lists containing complex numbers (roots) to be plotted on the
            screen.
        width (int):
            Width of the screen.
        height (int):
            Height of the screen.
        screen (:obj:`numpy.ndarray`):
            Screen array to store the resulting representation.
        screen_buffer (:obj:`numpy.ndarray`):
            Temporary buffer array for intermediate calculations.
        scale_x (float, optional):
            Scaling factor for the x-axis. Defaults to 1.
        scale_y (float, optional):
            Scaling factor for the y-axis. Defaults to 1.
        shift_x (float, optional):
            Shift value for the x-axis. Defaults to 0.
        shift_y (float, optional):
            Shift value for the y-axis. Defaults to 0.
        color_range (int, optional):
            Number of color shades to represent the roots. Defaults to 8.
        colors (list[:obj:`numpy.ndarray`]):
            List of color arrays corresponding to each degree of roots.

    Returns:
        :obj:`numpy.ndarray`:
            The resulting screen representation with the roots plotted using multiple
            colors.

    Note:
        - The function modifies the input screen and screen_buffer arrays in-place.
        - The function assumes that the input screen and screen_buffer arrays have the
          correct shape and dtype.
        - Each sublist of `roots` corresponds to a different degree of roots,
          and the color for each degree is specified by the corresponding array in the
          `colors` list.

    """
    # screen_buffer.fill(255)
    screen_buffer.fill(0)
    origin_x = width / 2
    origin_y = height / 2
    # for deg, roots_ in reversed(list(enumerate(roots))):
    for deg, roots_ in enumerate(roots):
        color = colors[deg]
        color_resolved = color // color_range
        for root in roots_:
            i = int((root.real - shift_x) / scale_x + origin_x)
            j = int(((root.imag - shift_y) / scale_y + origin_y))
            radius = 1
            for i_delta in range(-(radius - 1), radius):
                for j_delta in range(-(radius - 1), radius):
                    if i_delta**2 + j_delta**2 < ((radius - 1) ** 2):
                        i_d = i + i_delta
                        j_d = j + j_delta
                        if i_d < 0 or j_d < 0 or i_d >= width or j_d >= height:
                            pass
                        else:
                            if (
                                screen_buffer[j_d, i_d, 0] == 255
                                and screen_buffer[j_d, i_d, 1] == 255
                                and screen_buffer[j_d, i_d, 2] == 255
                            ):
                                screen_buffer[j_d, i_d, :] = color_resolved
                                for c in range(3):
                                    if screen_buffer[j_d, i_d, c] > 255:
                                        screen_buffer[j_d, i_d, c] = 255
    screen[:, :, :] = screen_buffer[:, :, :]
    return np.flipud(screen)
