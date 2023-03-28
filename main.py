from typing import List
import pyglet
import numpy as np
from PIL import Image


class Polynomial:
    def __init__(self, coeffs):
        # a[0] * x^0 + a[1] * x^1 + ... + a[n] + x^n
        self.coeffs = coeffs

    @classmethod
    def _eval_polynomial(cls, polynomial, x):
        res = 0
        for deg, coeff in enumerate(polynomial):
            res += coeff * (x**deg)
        return res

    def eval(self, x):
        return self._eval_polynomial(self.coeffs, x)

    def deriv(self):
        if len(self.coeffs) == 1:
            return Polynomial([0])
        deriv_coeffs = [0 for i in range(len(self.coeffs) - 1)]
        for deg, coeff in enumerate(self.coeffs):
            if deg == 0:
                pass
            new_coeff = coeff * deg
            deriv_coeffs[deg - 1] = new_coeff
        return Polynomial(deriv_coeffs)

    def eval_deriv(self, x):
        deriv_coeffs = [0 for i in range(len(self.coeffs) - 1)]
        for deg, coeff in enumerate(self.coeffs):
            if deg == 0:
                pass
            new_coeff = coeff * deg
            deriv_coeffs[deg - 1] = new_coeff
        return self._eval_polynomial(deriv_coeffs, x)

    def __str__(self):
        res = ""
        deg = len(self.coeffs) - 1
        for coef in reversed(self.coeffs):
            res += f"{coef}x^{deg}"
            if deg > 0:
                res += " + "
            deg -= 1
        return res


def sigmoid(x, delta):
    return 1 / (1 + np.exp(-(x ** (1 / delta))))


class Demo(pyglet.window.Window):
    def __init__(
        self,
        width: int,
        height: int,
        scale_x: float,
        scale_y: float,
        polynomials: List[Polynomial],
        one_over_res: int,
        delta: int,
    ):
        super().__init__(width * one_over_res, height * one_over_res, "Polynomiography")
        self.one_over_res = one_over_res
        self.screen_ = np.zeros([height, width, 3], dtype=np.uint8)
        self.screen_buffer = np.zeros([height, width, 3], dtype=np.int64)
        self.pitch = width * 3
        self.image_data = pyglet.image.ImageData(
            width * one_over_res,
            height * one_over_res,
            "RGB",
            expand_3d_array(
                self.screen_, self.one_over_res, self.one_over_res
            ).tobytes(),
            self.pitch,
        )
        self.polynomials = polynomials
        self.delta = delta
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.recompute_image_data()

    def recompute_image_data(self):
        width, height = self.width, self.height
        data = compute_screen_for_poly_newton(
            self.polynomials,
            width // self.one_over_res,
            height // self.one_over_res,
            self.scale_x,
            self.scale_y,
            self.screen_,
            self.screen_buffer,
            self.delta,
            -5,
        )
        rescaled_data = expand_3d_array(
            data, self.one_over_res, self.one_over_res
        ).tobytes()
        self.image_data.set_data("RGB", self.pitch, rescaled_data)

    def on_draw(self):
        self.clear()
        self.image_data.blit(0, 0)

    def update(self, delta_time):
        # print("Update", delta_time)
        # self.delta += 0.01
        self.scale_x /= self.delta
        self.scale_y /= self.delta
        self.recompute_image_data()


def expand_array(arr, row_factor, col_factor):
    return np.repeat(np.repeat(arr, col_factor, axis=1), row_factor, axis=0)


def expand_3d_array(arr, row_factor, col_factor):
    shape = arr.shape
    expanded_arr = np.empty(
        (shape[0] * row_factor, shape[1] * col_factor, shape[2]), dtype=arr.dtype
    )
    for i in range(shape[2]):
        slice_data = arr[:, :, i]
        expanded_slice = expand_array(slice_data, row_factor, col_factor)
        expanded_arr[:, :, i] = np.array(expanded_slice)
    return expanded_arr


def save3d(data, filename):
    with open(filename, "w") as outfile:
        outfile.write("# Array shape: {0}\n".format(data.shape))
        for data_slice in data:
            np.savetxt(outfile, data_slice, fmt="%-7.2f")
            outfile.write("# New slice\n")


MAX_ITER_COUNT = 40


def newton_method_iter_count(poly, x, delta, step=0):
    if step > MAX_ITER_COUNT:
        return x, 0
    res = poly.eval(x)
    deriv_res = poly.eval_deriv(x)
    if deriv_res == 0:
        return x, 0
    newton_res = x - res / deriv_res
    if abs(newton_res - x) < delta:
        return newton_res, 0
    else:
        new_res, count = newton_method_iter_count(poly, newton_res, delta, step + 1)
        return new_res, count + 1


def compute_screen_for_poly_newton(
    polynomials,
    width,
    height,
    scale_x,
    scale_y,
    screen,
    screen_buffer,
    delta,
    shift_x=0,
    shift_y=0,
):
    origin_x = width / 2
    origin_y = height / 2
    poly1 = polynomials[0]
    poly2 = polynomials[1]
    poly3 = polynomials[2]
    for j in range(height):
        for i in range(width):
            x = (i - origin_x) * scale_x + shift_x
            y = -(j - origin_y) * scale_y + shift_y
            val = complex(x, y)
            res, iter_count_1 = newton_method_iter_count(poly1, val, delta)
            res, iter_count_2 = newton_method_iter_count(poly2, val, delta)
            res, iter_count_3 = newton_method_iter_count(poly3, val, delta)
            screen_buffer[j, i, 0] = iter_count_1
            screen_buffer[j, i, 1] = iter_count_2
            screen_buffer[j, i, 2] = iter_count_3
            # eval_res_abs = abs(eval_res)
            # eval_deriv_res_abs = abs(eval_deriv_res)
            # pixel_val_1 = sigmoid(eval_res_abs, delta) * 255
            # pixel_val_2 = sigmoid(eval_deriv_res_abs, delta) * 255
            # screen_buffer[j, i, 0] = pixel_val_1
            # screen_buffer[j, i, 1] = pixel_val_2
    # max0 = np.max(screen_buffer[:, :, 0])
    # max1 = np.max(screen_buffer[:, :, 1])
    # min0 = np.min(screen_buffer[:, :, 0])
    # min1 = np.min(screen_buffer[:, :, 1])
    # print(min0, max0)
    # print(min1, max1)
    screen[:, :, 0] = screen_buffer[:, :, 0] / MAX_ITER_COUNT * 255
    screen[:, :, 1] = screen_buffer[:, :, 1] / MAX_ITER_COUNT * 255
    screen[:, :, 2] = screen_buffer[:, :, 2] / MAX_ITER_COUNT * 255
    return np.flipud(screen)


if __name__ == "__main__":
    polynomials = [
        Polynomial([31, 69, 22, 11, 13, 99, 102]),
        Polynomial([4, 29, 42, 49, 67, 18, 111]),
        Polynomial([482, 333, 16, 14, 31, 69, 62]),
    ]
    # p1 = Polynomial([-92500, 6000, 3100, -240, -1, 0, 1])
    # p1 = Polynomial([-2, 0, 1, 2])
    # p1 = Polynomial([0, 360, -48, 14, -2, 1])
    # p1 = Polynomial([-0.0036, 360, -48.0001, 14, -2.00001, 1])
    for poly in polynomials:
        print(poly)
    width = 300
    height = 300
    screen = np.zeros([height, width, 3], dtype=np.uint8)
    screen_buffer = np.zeros([height, width, 3], dtype=np.int64)
    delta = 0.1
    frame = 1
    scale_x = 0.1
    scale_y = 0.1
    inverse_res = 1
    """
    while scale_x > 0:
        compute_screen_for_poly(
            p1,
            width,
            height,
            scale_x,
            scale_y,
            screen,
            screen_buffer,
            delta,
        )
        im = Image.fromarray(screen, mode="RGB")
        im.save(f"frames/frame{frame:04d}.png", format="PNG")
        frame += 1
        if delta > 2:
            scale_x -= 0.001
            scale_y -= 0.001
        else:
            delta += 0.01
    """
    while scale_x >= 0.0001:
        compute_screen_for_poly_newton(
            polynomials,
            width,
            height,
            scale_x,
            scale_y,
            screen,
            screen_buffer,
            delta,
            -0.9,
        )
        print(f"Frame: {frame} scale: {scale_x}")
        im = Image.fromarray(screen, mode="RGB")
        im.save(f"frames/frame{frame:04d}.png", format="PNG")
        frame += 1
        scale_x /= 1.05
        scale_y /= 1.05

    # window = Demo(300, 300, scale_x, scale_y, polynomials, inverse_res, delta)
    # pyglet.clock.schedule_interval(window.update, 1 / 30.0)
    # pyglet.app.run()

# 1x^6 + -1x^4 + -240x^3 + 3100x^2 + 6000x^1 + -92500

"""
(x - 0.00001) (-4 j + x + 2) (-3 j + x - 3) (3 j + x - 3) (4 j + x + 2)
x^5 - 2.00001 x^4 + 14. x^3 - 48.0001 x^2 + 360. x - 0.0036

[- 0.0036, 360, -48.0001, 14, -2.00001, 1]
"""


"""
x (x^4 - 2 x^3 + 14 x^2 - 48 x + 360)

[0, 360, -48, 14, -2, 1]
"""
