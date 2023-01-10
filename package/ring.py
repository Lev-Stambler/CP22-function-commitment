from nptyping import NDArray
import functools
from numpy.fft import fft, ifft
import math
import numpy as np
from c_intmm import intmm


class Ring():
    """
        Each polynomial is represented by a Numpy vector with coefficients for terms
        in ascending order. I.e. p[n-1] is the coefficient for x^(n - 1) and p[n-1] is the coefficient for x^(n-1) = 1
    """

    def __init__(self, n, nptype=np.int32):
        """
            :param n - the degree of the polynomial plus 1. I.e. we are modding over x^n + 1
        """
        self.n = n
        self.nptype = nptype

        self.default_n = lambda n: n if n is not None else self.n

    # From https://github.com/j2kun/negacyclic
    def primitive_nth_root(self, n):
        """Return a primitive nth root of unity."""
        return math.cos(2 * math.pi / n) + 1.0j * math.sin(2 * math.pi / n)

    def random(self, n=None, repeat=1):
        # TODO: MAKE SECURE!!
        return np.copy(np.random.randint(2 ** (4), size=(repeat, self.default_n(n)), dtype=self.nptype))

    # From https://github.com/j2kun/negacyclic
    def negacyclic_polymul_complex_twist(self, p1, p2):
        """Computes a poly multiplication mod (X^N + 1) where N = len(p1).
        Uses the idea on page 332 (pdf page 8) of "Fast multiplication and its
        applications" by Daniel Bernstein.
        http://cr.yp.to/lineartime/multapps-20080515.pdf
        """
        n = p2.shape[0]
        primitive_root = self.primitive_nth_root(2 * n)
        root_powers = primitive_root ** np.arange(n // 2)

        p1_preprocessed = (p1[: n // 2] + 1j * p1[n // 2:]) * root_powers
        p2_preprocessed = (p2[: n // 2] + 1j * p2[n // 2:]) * root_powers

        p1_ft = fft(p1_preprocessed)
        p2_ft = fft(p2_preprocessed)
        prod = p1_ft * p2_ft
        ifft_prod = ifft(prod)
        ifft_rotated = ifft_prod * \
            primitive_root ** np.arange(0, -n // 2, -1)

        return np.round(
            np.concatenate(
                [np.real(ifft_rotated), np.imag(ifft_rotated)])
        ).astype(p1.dtype)

        # p1_preprocessed = np.concatenate([p1, -p1])
        # p2_preprocessed = np.concatenate([p2, -p2])
        # product = fft(p1_preprocessed) * fft(p2_preprocessed)
        # inverted = ifft(product)
        # rounded = np.round(np.real(inverted)).astype(p1.dtype)
        # return (rounded[: p1.shape[0]] - rounded[p1.shape[0] :]) // 4

        # p1_preprocessed = np.concatenate([p1, -p1])
        # p2_preprocessed = np.concatenate([p2, -p2])
        # product = fft(p1_preprocessed) * fft(p2_preprocessed)
        # inverted = ifft(product)
        # rounded = np.round(0.5 * np.real(inverted)).astype(p1.dtype)
        # return rounded[: p1.shape[0]]

    def _np_polymul_mod(self, poly1, poly2, poly_mod):
        # Reversing the list order because numpy polymul interprets the polynomial
        # with higher-order coefficients first, whereas our code does the opposite
        np_mul = np.polymul(list(reversed(poly1)), list(reversed(poly2)))
        (_, np_poly_mod) = np.polydiv(np_mul, poly_mod)
        np_pad = np.pad(
            np_poly_mod,
            (len(poly1) - len(np_poly_mod), 0),
            "constant",
            constant_values=(0, 0),
        )
        return np.array(list(reversed(np_pad)), dtype=self.nptype)

    def _np_negacyclic_polymul(self, poly1, poly2):
        # a reference implementation for negacyclic polynomial multiplication
        # poly_mod represents the polynomial to divide by: x^N + 1, N = len(a)
        poly_mod = np.zeros(len(poly1) + 1, np.uint32)
        poly_mod[0] = 1
        poly_mod[len(poly1)] = 1
        return self._np_polymul_mod(poly1, poly2, poly_mod)

    def mul_mat(self, A, B):
        """Multiply matrix A by B"""
        assert(len(A.shape) == 3), "The shape of a matrix is 3"
        assert(len(B.shape) == 3), "The shape of a matrix is 3"
        assert(A.shape[1] == B.shape[0])
        assert(A.shape[2] == B.shape[2])

        B_swapped = B.swapaxes(0, 1)
        C = np.zeros((A.shape[0], B_swapped.shape[0], A.shape[2]), dtype=self.nptype)
        for i in range(A.shape[0]):
            for j in range(B_swapped.shape[0]):
                r = self.dot(A[i], B_swapped[j])
                C[i, j, :] = np.array([r])
        return C
    
    def mat_norm(self, A):
        """
            Get the L1 Norm as defined in CP22 of a ring matrix
        """
        assert(len(A.shape) == 3), "Expected a ring matrix"
        col_norms = [self.col_vector_norm(A[:, i, :]) for i in range(A.shape[1])]
        return max(col_norms)

    def col_vector_norm(self, v):
        """Get the L1 Norm as defined in CP22 of a ring vector"""
        assert(len(v.shape) == 2), "Expected a ring vector"
        # We can just **sum** over v as the norm of a cyclomatic matrix
        # equals to the sum of the polynomial's coefficients
        return v.sum()


    def mul_right_matrix(self, A, B):
        """Multiply vector A by a matrix B all over field R"""
        assert (len(A.shape) == 2), "The shape of a vector must be 2, the outer is for the vector, the inner represent's each polynomial"
        assert (len(B.shape) == 3), "The shape of a matrix must be 3, the outer two is for the matrix, the inner represent's each polynomial"
        assert (A.shape[0] == B.shape[0]
                ), "Expected the dimensions of A and B to match"
        assert (A.shape[1] == B.shape[2]
                ), "Expected the dimensions of A and B to match in the polynomial term"

        B_swapped = B.swapaxes(0, 1)
        C = np.zeros((B_swapped.shape[0], A.shape[1]), dtype=self.nptype)
        for i in range(B_swapped.shape[0]):
            r = self.dot(A, B_swapped[i])[0]
            C[i, :] = np.array([r])
        return C

    def dot(self, A: NDArray, B: NDArray):
        """Get the dot product for two vectors with dimension \\ell where each entry is a polynomial"""
        assert (len(A.shape) == 2 and len(B.shape) ==
                2), "The shapes of both vectors must be 2"
        assert (A.shape[0] == B.shape[0]
                ), "Expected the dimensions of A and B to match"
        cum = self.zeros(A.shape[1])
        for i in range(A.shape[0]):
            p = self.negacyclic_polymul_complex_twist(A[i], B[i])
            cum = self.sum(
                cum, p)

        return cum

    def sum(self, p1: NDArray, p2: NDArray):
        """Sum to polynomials together. Because they are both np coefficient arrays, we can just add them modulo our field"""
        return (p1 + p2) 

    def sub(self, p1: NDArray, p2: NDArray):
        """Subtract to polynomials together. Because they are both np coefficient arrays, we can just add them modulo our field"""
        return (p1 - p2)

    def mult_iden(self, n=None):
        """Get the multiplicative identity which is just a constant 1"""
        c = np.zeros((1, self.default_n(n)), dtype=self.nptype)
        c[0, 0] = 1
        return c

    def mult_iden_square_mat(self, mat_size, n=None):
        """Get the multiplicative identity which is just a constant 1"""
        elem = self.mult_iden(n)
        r = np.zeros((mat_size, mat_size, self.default_n(n)),
                     dtype=self.nptype)
        # TODO: faster np version
        for i in range(mat_size):
            r[i, i] = elem[:]
        return r

    def additive_iden(self, n=None):
        c = np.zeros((1, self.default_n(n)), dtype=self.nptype)
        return c

    def zeros(self, n=None):
        return np.zeros((1, self.default_n(n)), dtype=self.nptype)

    def ones(self, n=None):
        return np.ones((1, self.default_n(n)), dtype=self.nptype)

    def repeat(self, p, r):
        # To make this more efficient we could be using Numpy somewhere
        return np.repeat(p, [r], axis=0)

    def stack(self, ps):
        # To make this more efficient we could be using Numpy somewhere
        return np.stack(ps)

    def concat(self, ps, axis=-1):
        # To make this more efficient we could be using Numpy somewhere
        return np.concatenate(ps, axis=axis)

    def kron(self, A, B: NDArray):
        """
            Here we, somewhat counterintuitively reverse the order that we kron.
            B can be any np vector/array while A specifically represents a polynomial
            thus this way things still "makes sense" when we kron as polynomial structure is preserved
        """
        return np.kron(A, B)
