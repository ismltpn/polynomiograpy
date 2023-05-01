__all__ = ["Polynomial"]


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
        start = True
        res = ""
        deg = len(self.coeffs) - 1
        for coef in reversed(self.coeffs):
            if coef != 0:
                if not start:
                    res += " + "
                res += f"{coef}x^{deg}"
                start = False
            deg -= 1
        return res
