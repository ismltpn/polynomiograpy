import numpy as np

__all__ = ["FiniteField"]


class FiniteField:
    def __init__(self, elements):
        self.elements = elements

    def generate_polynomials(self, degree) -> list[np.polynomial.Polynomial]:
        polynomials = []
        coefs = self._generate_coeffs(degree, False)
        for coef in coefs:
            polynomials.append(np.polynomial.Polynomial(coef=coef))
        return polynomials

    def _generate_coeffs(self, degree, allow_zero):
        leading_coeffs = (
            self.elements
            if allow_zero
            else list(filter(lambda x: x != 0, self.elements))
        )
        res = []
        for leading_coef in leading_coeffs:
            if degree == 0:
                res.append([leading_coef])
            else:
                for coefs in self._generate_coeffs(degree - 1, True):
                    res.append([*coefs, leading_coef])
        return res

    def __str__(self):
        return str(self.elements)
