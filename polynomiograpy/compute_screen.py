from typing import Callable
import numpy as np


def compute_screen(
    func: Callable[[complex], int],
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
    reverse_color: bool = False,
    channel: int = 0,
):
    assert len(screen.shape) >= 3, "Wrong shape for screen"
    assert len(screen_buffer.shape) >= 3, "Wrong shape for screen"
    assert screen.shape == screen_buffer.shape, "screen shape != screen buffer shape"
    origin_x = width / 2
    origin_y = height / 2
    for j in range(height):
        for i in range(width):
            x = (i - origin_x) * scale_x + shift_x
            y = -(j - origin_y) * scale_y + shift_y
            val = complex(x, y)
            res = func(val)
            screen_buffer[j, i, channel] = max_value - res if reverse_color else res
    screen[:, :, 0] = screen_buffer[:, :, 0] / max_value * 255
    screen[:, :, 1] = screen_buffer[:, :, 1] / max_value * 255
    screen[:, :, 2] = screen_buffer[:, :, 2] / max_value * 255
    if screen.shape[2] > 3:
        screen[:, :, 3] = 255 - np.max(screen[:, :, :3], 2)
    return np.flipud(screen)
