import numpy as np
from PIL import Image

import polynomiograpy


if __name__ == "__main__":
    polynomial = polynomiograpy.Polynomial([1, 0, 0, 1])
    print(polynomial)
    width = 1000
    height = 1000
    screen = np.zeros([height, width, 3], dtype=np.uint8)
    screen_buffer = np.zeros([height, width, 3], dtype=np.int64)
    delta = 0.1
    frame = 1
    scale_x = 6 / width
    scale_y = 6 / height
    inverse_res = 1
    polynomiograpy.compute_screen_for_single_poly(
        "steffensen",
        polynomial,
        delta,
        width,
        height,
        screen,
        screen_buffer,
        scale_x=scale_x,
        scale_y=scale_y,
        reverse_color=False,
    )
    filename = "example_output.png"
    im = Image.fromarray(screen, mode="RGB")
    im.save(filename, format="PNG")
    print(f"Saved to {filename}")
