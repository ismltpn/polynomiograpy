from typing import Optional
from polynomiograpy import common
import numpy as np

available_methods = [
    "newton",
    "halley",
    "inverse_interpolation",
    "mullers",
    "secant",
    "steffensen",
    "old_newton",
    "old_halley",
    "old_inverse_interpolation",
    "old_mullers",
    "old_secant",
    "old_steffensen",
]


def newton_method_loop(
    poly: common.polynomial.Polynomial,
    x: complex,
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
):
    count = 0
    while step < max_iter_count:
        res = poly.eval(x)
        deriv_res = poly.eval_deriv(x)
        if deriv_res == 0:
            return x, max_iter_count - step - 1
        newton_res = x - res / deriv_res
        if abs(newton_res - x) < delta:
            return newton_res, count
        x = newton_res
        step += 1
        count += 1
    return x, count


def newton_method(
    poly: common.polynomial.Polynomial,
    x: complex,
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
) -> tuple[complex, int]:
    if step >= max_iter_count:
        return x, 0
    res = poly.eval(x)
    deriv_res = poly.eval_deriv(x)
    if deriv_res == 0:
        return x, max_iter_count - step - 1
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


def newton_method_numpy(
    poly: common.polynomial.Polynomial,
    x: np.ndarray,
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
):
    iter_counts = np.zeros(x.shape, np.int64)
    deriv = poly.deriv()
    while step < max_iter_count:
        res = poly.eval(x)
        deriv_res = deriv.eval(x)
        deriv_res_non_zero_ix = (deriv_res != 0).nonzero()
        x[deriv_res_non_zero_ix] = (
            x[deriv_res_non_zero_ix]
            - res[deriv_res_non_zero_ix] / deriv_res[deriv_res_non_zero_ix]
        )
        iter_counts[deriv_res == 0] = max_iter_count - 1
        iter_counts[np.logical_and(deriv_res != 0, abs(-res / deriv_res) >= delta)] += 1
        step += 1
    return iter_counts


def halley_method(
    poly: common.polynomial.Polynomial,
    x: complex,
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
) -> tuple[complex, int]:
    if step >= max_iter_count:
        return x, 0
    res = poly.eval(x)
    deriv_res = poly.eval_deriv(x)
    deriv_deriv_res = poly.deriv().eval_deriv(x)
    denom = -(res * deriv_deriv_res) + 2 * deriv_res * deriv_res

    if denom == 0:
        return x, max_iter_count - step - 1
    halley_res = x - 2 * deriv_res * res / denom
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


def halley_method_numpy(
    poly: common.polynomial.Polynomial,
    x: np.ndarray,
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
):
    iter_counts = np.zeros(x.shape, np.int64)
    deriv = poly.deriv()
    deriv_deriv = deriv.deriv()
    while step < max_iter_count:
        res = poly.eval(x)
        deriv_res = deriv.eval(x)
        deriv_deriv_res = deriv_deriv.eval(x)
        denom = -(res * deriv_deriv_res) + 2 * deriv_res * deriv_res
        denom_non_zero_ix = (denom != 0).nonzero()
        x[denom_non_zero_ix] = (
            x[denom_non_zero_ix]
            - 2
            * deriv_res[denom_non_zero_ix]
            * res[denom_non_zero_ix]
            / denom[denom_non_zero_ix]
        )
        iter_counts[denom == 0] = max_iter_count - 1
        iter_counts[
            np.logical_and(denom != 0, abs(2 * (-deriv_res * res) / denom) >= delta)
        ] += 1
        step += 1
    return iter_counts


def inverse_interpolation_method(
    poly: common.polynomial.Polynomial,
    x_0: complex,
    x_1: Optional[complex],
    x_2: Optional[complex],
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
) -> tuple[complex, int]:
    if step >= max_iter_count:
        return x_0, 0
    if x_1 is None:
        x_1 = x_0 - 0.1
    if x_2 is None:
        x_2 = x_0 + 0.1
    fx_0 = poly.eval(x_0)
    fx_1 = poly.eval(x_1)
    fx_2 = poly.eval(x_2)
    if fx_2 == fx_1 or fx_2 == fx_0 or fx_1 == fx_0:
        return x_1, max_iter_count - step - 1
    term1 = x_0 * (fx_1 * fx_2 / ((fx_0 - fx_1) * (fx_0 - fx_2)))
    term2 = x_1 * (fx_0 * fx_2 / ((fx_1 - fx_0) * (fx_1 - fx_2)))
    term3 = x_2 * (fx_0 * fx_1 / ((fx_2 - fx_0) * (fx_2 - fx_1)))
    res = term1 + term2 + term3
    if abs(res - x_2) < delta:
        return res, 0
    else:
        new_res, count = inverse_interpolation_method(
            poly,
            x_1,
            x_2,
            res,
            delta,
            step=step + 1,
            max_iter_count=max_iter_count,
        )
        return new_res, count + 1


def inverse_interpolation_method_numpy(
    poly: common.polynomial.Polynomial,
    x_0: np.ndarray,
    x_1: Optional[np.ndarray],
    x_2: Optional[np.ndarray],
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
):
    iter_counts = np.zeros(x_0.shape, np.int64)
    if x_1 is None:
        x_1 = x_0 - 0.1
    if x_2 is None:
        x_2 = x_0 + 0.1
    while step < max_iter_count:
        fx_0 = poly.eval(x_0)
        fx_1 = poly.eval(x_1)
        fx_2 = poly.eval(x_2)
        denom_non_zero_ix = np.logical_and(
            np.logical_and(fx_2 != fx_1, fx_1 != fx_0), fx_2 != fx_0
        )
        denom_zero_ix = np.logical_or(
            np.logical_or(fx_2 == fx_1, fx_1 == fx_0), fx_2 == fx_0
        )
        term1 = np.divide(
            x_0 * fx_1 * fx_2, (fx_0 - fx_1) * (fx_0 - fx_2), where=denom_non_zero_ix
        )
        term2 = np.divide(
            x_1 * fx_0 * fx_2, (fx_1 - fx_0) * (fx_1 - fx_2), where=denom_non_zero_ix
        )
        term3 = np.divide(
            x_2 * fx_0 * fx_1, (fx_2 - fx_1) * (fx_2 - fx_0), where=denom_non_zero_ix
        )
        res = x_2.copy()
        res[denom_non_zero_ix] = (
            term1[denom_non_zero_ix]
            + term2[denom_non_zero_ix]
            + term3[denom_non_zero_ix]
        )
        iter_counts[denom_zero_ix] = max_iter_count - 1
        next_iter_mask = np.logical_and(denom_non_zero_ix, abs(res - x_2) >= delta)
        iter_counts[next_iter_mask] += 1
        x_0[next_iter_mask] = x_1[next_iter_mask]
        x_1[next_iter_mask] = x_2[next_iter_mask]
        x_2[next_iter_mask] = res[next_iter_mask]
        step += 1
    return iter_counts


def mullers_method(
    poly: common.polynomial.Polynomial,
    x_0: complex,
    x_1: Optional[complex],
    x_2: Optional[complex],
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
) -> tuple[complex, int]:
    if step >= max_iter_count:
        return x_0, 0
    if x_1 is None:
        x_1, _ = newton_method(poly, x_0, delta, step=0, max_iter_count=1)
    if x_2 is None:
        x_2, _ = newton_method(poly, x_1, delta, step=0, max_iter_count=1)
    fx_0 = poly.eval(x_0)
    fx_1 = poly.eval(x_1)
    fx_2 = poly.eval(x_2)
    if x_1 == x_0:
        return x_2, 0
    q = (x_2 - x_1) / (x_1 - x_0)
    a = q * fx_2 - q * (1 + q) * fx_1 + q**2 * fx_0
    b = (2 * q + 1) * fx_2 - (1 + q) ** 2 * fx_1 + q**2 * fx_0
    c = (1 + q) * fx_2
    d1 = b + np.sqrt(b * b - 4 * a * c)
    d2 = b - np.sqrt(b * b - 4 * a * c)
    denom = max(
        d1,
        d2,
    )
    if denom == 0:
        return x_2, max_iter_count - step - 1
    res = x_2 - (x_2 - x_1) * (2 * c) / denom
    if abs(res - x_2) < delta:
        return res, 0
    else:
        new_res, count = mullers_method(
            poly,
            x_1,
            x_2,
            res,
            delta,
            step=step + 1,
            max_iter_count=max_iter_count,
        )
        return new_res, count + 1


def mullers_method_numpy(
    poly: common.polynomial.Polynomial,
    x_0: np.ndarray,
    x_1: Optional[np.ndarray],
    x_2: Optional[np.ndarray],
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
):
    iter_counts = np.zeros(x_0.shape, np.int64)
    deriv = poly.deriv()
    if x_1 is None:
        x_1 = x_0 - poly.eval(x_0) / deriv.eval(x_0)
        # x_1 = x_0 - 0.1
    if x_2 is None:
        x_2 = x_1 - poly.eval(x_1) / deriv.eval(x_1)
        # x_2 = x_0 + 0.1
    while step < max_iter_count:
        fx_0 = poly.eval(x_0)
        fx_1 = poly.eval(x_1)
        fx_2 = poly.eval(x_2)
        q = (x_2 - x_1) / (x_1 - x_0)
        a = q * fx_2 - q * (1 + q) * fx_1 + q**2 * fx_0
        b = (2 * q + 1) * fx_2 - (1 + q) ** 2 * fx_1 + q**2 * fx_0
        c = (1 + q) * fx_2
        d_1 = b + np.sqrt(b * b - 4 * a * c)
        d_2 = b - np.sqrt(b * b - 4 * a * c)
        denom = np.maximum(d_1, d_2)
        res = x_2.copy()
        diff = np.divide((x_2 - x_1) * (2 * c), denom, where=denom != 0)
        res[denom != 0] = x_2[denom != 0] - diff[denom != 0]
        iter_counts[denom == 0] = max_iter_count - 1
        next_iter_mask = (
            np.logical_and(
                np.logical_and((x_1 - x_0) != 0, x_1 != x_2),
                np.logical_and(
                    denom != 0,
                    abs(res - x_2) >= delta,
                ),
            ),
        )
        iter_counts[next_iter_mask] += 1
        x_0[next_iter_mask] = x_1[next_iter_mask]
        x_1[next_iter_mask] = x_2[next_iter_mask]
        x_2[next_iter_mask] = res[next_iter_mask]
        step += 1
    return iter_counts


def secant_method(
    poly: common.polynomial.Polynomial,
    x_0: complex,
    x_1: Optional[complex],
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
) -> tuple[complex, int]:
    if step >= max_iter_count:
        return x_0, 0
    if x_1 is None:
        x_1 = x_0 - 0.1
    fx_0 = poly.eval(x_0)
    fx_1 = poly.eval(x_1)
    if fx_1 == fx_0:
        return x_1, max_iter_count - step - 1
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


def secant_method_numpy(
    poly: common.polynomial.Polynomial,
    x_0: np.ndarray,
    x_1: Optional[np.ndarray],
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
):
    iter_counts = np.zeros(x_0.shape, np.int64)
    if x_1 is None:
        x_1 = x_0 - 0.1
    while step < max_iter_count:
        fx_0 = poly.eval(x_0)
        fx_1 = poly.eval(x_1)
        denom = fx_1 - fx_0
        res = x_1.copy()
        diff = np.divide(fx_1 * (x_1 - x_0), denom, where=denom != 0)
        res[denom != 0] = x_1[denom != 0] - diff[denom != 0]
        iter_counts[denom == 0] = max_iter_count - 1
        next_iter_mask = (
            np.logical_and(
                denom != 0,
                abs(res - x_1) >= delta,
            ),
        )
        iter_counts[next_iter_mask] += 1
        x_0[next_iter_mask] = x_1[next_iter_mask]
        x_1[next_iter_mask] = res[next_iter_mask]
        step += 1
    return iter_counts


def steffensen_method(
    poly: common.polynomial.Polynomial,
    x: complex,
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
) -> tuple[complex, int]:
    if step >= max_iter_count:
        return x, 0
    res = poly.eval(x)
    if res == 0:
        return x, 0
    denom = poly.eval(x + res) / res - 1
    if denom == 0:
        return x, max_iter_count - step - 1
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


def steffensen_method_numpy(
    poly: common.polynomial.Polynomial,
    x: np.ndarray,
    delta: float,
    *,
    step: int = 0,
    max_iter_count: int = 16,
):
    iter_counts = np.zeros(x.shape, np.int64)
    while step < max_iter_count:
        res = poly.eval(x)
        denom = poly.eval(x + res) - res
        diff = np.zeros(x.shape, dtype=x.dtype)
        np.divide(res * res, denom, out=diff, where=denom != 0)
        steffensen_res = x.copy()
        steffensen_res[denom != 0] = x[denom != 0] - diff[denom != 0]
        iter_counts[np.logical_and(res != 0, denom == 0)] = max_iter_count - 1
        next_iter_mask = np.logical_and(
            np.logical_and(res != 0, denom != 0),
            abs(x - steffensen_res) >= delta,
        )
        iter_counts[next_iter_mask] += 1
        x[next_iter_mask] = steffensen_res[next_iter_mask]
        step += 1
    return iter_counts
