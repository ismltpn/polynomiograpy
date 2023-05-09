from typing import Optional
from polynomiograpy import common

available_methods = {"newton", "halley", "secant", "steffensen"}


def newton_method(
    poly: common.polynomial.Polynomial,
    x: complex,
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
) -> tuple[complex, int]:
    if step > max_iter_count:
        return x, 0
    res = poly.eval(x)
    deriv_res = poly.eval_deriv(x)
    if deriv_res == 0:
        return x, max_iter_count - step
    newton_res = x - res / deriv_res
    if abs(newton_res - x) < delta:
        return newton_res, 0
    else:
        new_res, count = newton_method(
            poly,
            newton_res,
            delta,
            step=step + 1,
            max_iter_count=max_iter_count,
        )
        return new_res, count + 1


def halley_method(
    poly: common.polynomial.Polynomial,
    x: complex,
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
) -> tuple[complex, int]:
    if step > max_iter_count:
        return x, 0
    res = poly.eval(x)
    deriv_res = poly.eval_deriv(x)
    deriv_deriv_res = poly.deriv().eval_deriv(x)
    denom = -(res * deriv_deriv_res) + 2 * deriv_res * deriv_res
    if denom == 0:
        return x, max_iter_count - step
    halley_res = x - 2 * (-deriv_res * res) / denom
    if abs(halley_res - x) < delta:
        return halley_res, 0
    else:
        new_res, count = halley_method(
            poly,
            halley_res,
            delta,
            step=step + 1,
            max_iter_count=max_iter_count,
        )
        return new_res, count + 1


def secant_method(
    poly: common.polynomial.Polynomial,
    x_0: complex,
    x_1: Optional[complex],
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
) -> tuple[complex, int]:
    if step > max_iter_count:
        return x_0, 0
    if x_1 is None:
        x_1, _ = newton_method(poly, x_0, delta, step=0, max_iter_count=0)
    fx_0 = poly.eval(x_0)
    fx_1 = poly.eval(x_1)
    if fx_1 == fx_0:
        return x_1, max_iter_count - step
    res = x_1 - fx_1 * (x_1 - x_0) / (fx_1 - fx_0)
    if abs(res - x_1) < delta:
        return res, 0
    else:
        new_res, count = secant_method(
            poly,
            x_1,
            res,
            delta,
            step=step + 1,
            max_iter_count=max_iter_count,
        )
        return new_res, count + 1


def steffensen_method(
    poly: common.polynomial.Polynomial,
    x: complex,
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
) -> tuple[complex, int]:
    if step > max_iter_count:
        return x, 0
    res = poly.eval(x)
    if res == 0:
        return x, 0
    denom = poly.eval(x + res) / res - 1
    if denom == 0:
        return x, max_iter_count - step
    steffensen_res = x - res / denom
    if abs(steffensen_res - x) < delta:
        return steffensen_res, 0
    else:
        new_res, count = steffensen_method(
            poly,
            steffensen_res,
            delta,
            step=step + 1,
            max_iter_count=max_iter_count,
        )
        return new_res, count + 1
