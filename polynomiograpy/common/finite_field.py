import numpy as np

__all__ = ["FiniteField"]


class FiniteField:
    """
    Represents a finite field and provides methods for generating polynomials over the
    field.
    """

    def __init__(self, elements):
        """
        Initialize the finite field with the specified elements.

        Args:
            elements (List[int]): The elements of the finite field.
        """
        self.elements = elements

    def generate_polynomials(self, degree) -> list[np.polynomial.Polynomial]:
        """
        Generates polynomials of the specified degree over the finite field.

        Args:
            degree (int): The degree of the polynomials to generate.

        Returns:
            List[poly.Polynomial]: A list of polynomials over the finite field.
        """
        polynomials = []
        coefs = self._generate_coeffs(degree, False)
        for coef in coefs:
            polynomials.append(np.polynomial.Polynomial(coef=coef))
        return polynomials

    def _generate_coeffs(self, degree, allow_zero):
        """
        Generates the coefficients for polynomials of the specified degree.

        Args:
            degree (int): The degree of the polynomials.
            allow_zero (bool): Flag indicating whether to allow zero coefficients.

        Returns:
            List[List[int]]: A list of coefficient lists for the polynomials.
        """
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
        """
        Returns a string representation of the finite field.

        Returns:
            str: A string representation of the finite field.
        """
        return str(self.elements)
