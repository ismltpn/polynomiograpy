__all__ = ["Polynomial"]


class Polynomial:
    """
    Represents a polynomial and provides methods for evaluation and differentiation.
    """

    def __init__(self, coeffs):
        """
        Initialize the polynomial with the specified coefficients.

        Args:
            coeffs (list): The list of polynomial coefficients.
                The coefficients should be in the order [a0, a1, ..., an],
                where a0 is the coefficient of the zero-degree term, a1 is the
                coefficient of the first-degree term, and so on.
        """
        self.coeffs = coeffs

    @classmethod
    def _eval_polynomial(cls, polynomial, x):
        """
        Evaluate the polynomial at a specific value of x.

        Args:
            polynomial (list): The list of polynomial coefficients.
            x (float): The value of x at which to evaluate the polynomial.

        Returns:
            float: The result of evaluating the polynomial at x.
        """
        res = 0
        for deg, coeff in enumerate(polynomial):
            res += coeff * (x**deg)
        return res

    def eval(self, x):
        """
        Evaluate the polynomial at a specific value of x.

        Args:
            x (float|complex): The value of x at which to evaluate the polynomial.

        Returns:
            float|complex: The result of evaluating the polynomial at x.
        """
        return self._eval_polynomial(self.coeffs, x)

    def deriv(self):
        """
        Compute the derivative of the polynomial.

        Returns:
            Polynomial: The derivative of the polynomial as a new Polynomial object.
        """
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
        """
        Evaluate the derivative of the polynomial at a specific value of x.

        Args:
            x (float|complex): The value of x at which to evaluate the derivative.

        Returns:
            float|complex: The result of evaluating the derivative of the polynomial at
            x.
        """
        deriv_coeffs = [0 for i in range(len(self.coeffs) - 1)]
        for deg, coeff in enumerate(self.coeffs):
            if deg == 0:
                pass
            new_coeff = coeff * deg
            deriv_coeffs[deg - 1] = new_coeff
        return self._eval_polynomial(deriv_coeffs, x)

    def __str__(self):
        """
        Return a string representation of the polynomial.

        Returns:
            str: The string representation of the polynomial.
        """
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
