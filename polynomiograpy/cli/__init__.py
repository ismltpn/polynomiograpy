import numpy as np
from PIL import Image
from polynomiograpy.common.polynomial import Polynomial
from polynomiograpy.methods.iter import available_methods

__all__ = ["run"]


def input_with_default(prompt, default):
    res = input(prompt)
    if len(res) < 1:
        return default
    return res


def run():
    from polynomiograpy import compute_screen_for_single_poly

    print("** Complex Plane Setup **")
    min_real = int(input_with_default("Min Real (-3): ", "-3"))
    max_real = int(input_with_default("Max Real (3): ", "3"))
    min_imag = int(input_with_default("Min Imag (-3): ", "-3"))
    max_imag = int(input_with_default("Max Imag (3): ", "3"))
    print("** Polynomial **")
    poly_degree = int(input("Degree of the polynomial: "))
    coeffs = []
    for i in range(poly_degree + 1):
        coeffs.append(
            int(input_with_default(f"Int coefficient of x^{poly_degree-i} (1): ", "1"))
        )
    coeffs.reverse()
    poly = Polynomial(coeffs=coeffs)
    print(f"Polynomial: {poly}")
    print("** Method **")
    print(f"Available methods: {available_methods}")
    method = input_with_default("Method (newton): ", "newton")
    delta = float(input_with_default("convergence threshold (0.1): ", "0.1"))
    max_iter = int(input_with_default("Max iteration (16): ", "16"))
    print("** Output Setup **")
    width = int(input_with_default("Width (1000): ", "1000"))
    height = int(input_with_default("Height (1000): ", "1000"))
    reverse_color = input_with_default("Reverse Colors (y/N): ", "n") == "y"
    output_filename = input_with_default("Output (out.png): ", "out.png")

    print(f"Generating the output for polynomial {poly} using {method} method")
    screen = np.zeros([height, width, 3], dtype=np.uint8)
    screen_buffer = np.zeros([height, width, 3], dtype=np.int64)

    scale_x = (max_real - min_real) / width
    scale_y = (max_imag - min_imag) / height
    shift_x = (max_real + min_real) / 2
    shift_y = (max_imag + min_imag) / 2
    compute_screen_for_single_poly(
        method,
        poly,
        delta,
        width,
        height,
        screen,
        screen_buffer,
        scale_x=scale_x,
        scale_y=scale_y,
        shift_x=shift_x,
        shift_y=shift_y,
        max_value=max_iter,
        reverse_color=reverse_color,
    )
    im = Image.fromarray(screen, mode="RGB")
    im.save(output_filename, format="PNG")
    print(f"Saved to {output_filename}")
