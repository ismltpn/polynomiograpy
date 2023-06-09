import numpy as np
from PIL import Image
from polynomiograpy.common.finite_field import FiniteField
from polynomiograpy.common.polynomial import Polynomial
from polynomiograpy.iterations.methods import available_methods
from polynomiograpy.iterations import compute_screen_for_single_poly
from polynomiograpy.roots import (
    compute_screen_for_finite_field_poly,
    compute_screen_for_finite_field_poly_multi_color,
)

__all__ = ["run", "run_iter", "run_root"]


def input_with_default(prompt, default):
    res = input(prompt)
    if len(res) < 1:
        return default
    return res


def run():
    """
    Runs the interactive program for generating visualizations using iterative
    methods or polynomials over a finite field.

    The function displays a menu with two options: "Iterative methods" and
    "Poly over finite field". The user is prompted to choose an option by
    entering the corresponding number.

    - If "Iterative methods" is selected (option 1), the function calls the
      :py:func:`run_iter()` function to generate visualizations for a single polynomial
      using iterative methods.
    - If "Poly over finite field" is selected (option 2), the function calls
      the :py:func:`run_root()` function to generate visualizations for polynomials over
      a finite field.

    The function continuously prompts the user to choose an option until a valid
    option is selected. If an invalid option is entered, an error message is
    displayed.

    Note: The :py:func:`run_iter()` and :py:func:`run_root()` functions handle the
    configuration and generation of visualizations for specific modes, and their
    respective docstrings provide further details about their functionalities.
    """
    while True:
        print("1. Iterative methods")
        print("2. Poly over finite field")
        opt = input_with_default("Which tool do you want to use (1): ", "1")
        if opt == "1":
            run_iter()
            return
        elif opt == "2":
            run_root()
            return
        else:
            print("Wrong usage")


def run_root():
    """
    Runs the program to generate visualizations for polynomials over a finite field.

    The function prompts the user to provide settings for the complex plane,
    finite field elements, polynomial degrees, and output parameters. It allows
    the user to choose between generating visualizations with a single color
    for all degrees or using different colors for each degree.

    The visualization is generated by computing the screen for all polynomials
    over the finite field within the specified degree range. The resulting image
    is saved to a file specified by the user.
    """
    print("** Complex Plane Setup **")
    min_real = float(input_with_default("Min Real (-3): ", "-3"))
    max_real = float(input_with_default("Max Real (3): ", "3"))
    min_imag = float(input_with_default("Min Imag (-3): ", "-3"))
    max_imag = float(input_with_default("Max Imag (3): ", "3"))
    print("** Finite Field **")
    finite_field_elements: str = input_with_default(
        "Finite field elements (1,0): ", "1,0"
    )
    parsed_elements = [int(e) for e in finite_field_elements.split(",")]
    finite_field = FiniteField(elements=parsed_elements)
    min_degree = int(input_with_default("Min degree (1): ", "1"))
    max_degree = int(input_with_default("Max degree (5): ", "5"))
    print("** Output Setup **")
    width = int(input_with_default("Width (1000): ", "1000"))
    height = int(input_with_default("Height (1000): ", "1000"))
    multi_color = (
        input_with_default(
            "Use different color for each degree? (Deg diff must be at most 7) (y/N): ",
            "n",
        )
        == "y"
    )
    color_range = int(input_with_default("Color range (8): ", "8"))
    output_filename = input_with_default("Output (out.png): ", "out.png")

    print(
        f"Generating the output for all polynomials over {finite_field} "
        f"from degree {min_degree} to degree {max_degree}"
    )
    screen = np.zeros([height, width, 3], dtype=np.uint8)
    screen_buffer = np.zeros([height, width, 3], dtype=np.int64)

    scale_x = (max_real - min_real) / width
    scale_y = (max_imag - min_imag) / height
    shift_x = (max_real + min_real) / 2
    shift_y = (max_imag + min_imag) / 2
    if multi_color:
        compute_screen_for_finite_field_poly_multi_color(
            finite_field,
            min_degree,
            max_degree,
            width,
            height,
            screen,
            screen_buffer,
            scale_x=scale_x,
            scale_y=scale_y,
            shift_x=shift_x,
            shift_y=shift_y,
            color_range=color_range,
        )
    else:
        compute_screen_for_finite_field_poly(
            finite_field,
            min_degree,
            max_degree,
            width,
            height,
            screen,
            screen_buffer,
            scale_x=scale_x,
            scale_y=scale_y,
            shift_x=shift_x,
            shift_y=shift_y,
            color_range=color_range,
        )
    im = Image.fromarray(screen, mode="RGB")
    im.save(output_filename, format="PNG")
    print(f"Saved to {output_filename}")


def run_iter():
    """
    Runs the program to generate visualizations using iterative methods.

    The function prompts the user to provide settings for the complex plane
    and the polynomial to visualize. It then allows the user to choose a
    method for visualization, along with the required parameters. Finally, it
    prompts for output settings such as image dimensions and color options.

    The visualization is generated by computing the screen for a single
    polynomial using the selected method and parameters. The resulting image
    is saved to a file specified by the user.
    """
    print("** Complex Plane Setup **")
    min_real = float(input_with_default("Min Real (-3): ", "-3"))
    max_real = float(input_with_default("Max Real (3): ", "3"))
    min_imag = float(input_with_default("Min Imag (-3): ", "-3"))
    max_imag = float(input_with_default("Max Imag (3): ", "3"))
    print("** Polynomial **")
    coeffs: str = input_with_default(
        "Coefficients from degree d to 0 (1,0,0,1): ", "1,0,0,1"
    )
    parsed_coeffs = [int(e) for e in coeffs.split(",")]
    parsed_coeffs.reverse()
    poly = Polynomial(coeffs=parsed_coeffs)
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
    screen_buffer = np.zeros([height, width, 3], dtype=np.complex128)

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
