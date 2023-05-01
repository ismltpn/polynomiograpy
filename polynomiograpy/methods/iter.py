from polynomiograpy import common


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
        return x, 0
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
