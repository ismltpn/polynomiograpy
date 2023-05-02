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
