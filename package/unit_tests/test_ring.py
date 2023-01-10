from hypothesis import given, settings
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import sampled_from
import numpy as np
import pytest
from .. import ring as _ring

q = 2 ** 16
security_param = n = 10
ell = 16

ring = _ring.Ring(n, np.uint16)


@given(
    #  Get a vector of dim ell
    lists(
        lists(integers(min_value=0, max_value=q - 1),
              min_size=n, max_size=n),
        min_size=ell, max_size=ell
    ),
    # Get an ell x 2 ell matrix
    lists(
        lists(
            integers(min_value=0, max_value=q - 1), min_size=n, max_size=n
        ), min_size=2 * ell, max_size=2 * ell
    )
)
@settings(max_examples=20)
def test_matrix_mult_right(p1Vec, p2Mat):
    B = np.stack(ell * [np.array(p2Mat)])
    C = ring.mul_right_matrix(np.array(p1Vec), B)
    assert (C.shape == (2 * ell, n))

@given(
    #  Get a vector of dim ell
    lists(
        lists(integers(min_value=0, max_value=q - 1),
              min_size=ell, max_size=ell),
        min_size=10, max_size=30
    ),
    # Get an ell x 2 ell matrix
    lists(
        lists(
            integers(min_value=0, max_value=q - 1), min_size=10, max_size=30
        ), min_size=ell, max_size=ell
    )
)
@settings(max_examples=20)
def test_matrix_mult(AMat, BMat):
    AMat = np.array(AMat)
    BMat = np.array(BMat)
    ref = np.matmul(AMat, BMat)
    A = np.zeros((AMat.shape[0], AMat.shape[1], ell))
    B = np.zeros((BMat.shape[0], BMat.shape[1], ell))
    for i in range(AMat.shape[0]):
        for j in range(AMat.shape[1]):
            A[i, j, :] = AMat[i, j] * ring.mult_iden(n)
    for i in range(BMat.shape[0]):
        for j in range(BMat.shape[1]):
            B[i, j, :] = BMat[i, j] * ring.mult_iden(n)
    C = ring.mul_mat(A, B)
    for i in range(AMat.shape[0]):
        for j in range(BMat.shape[1]):
            assert(C[i, j, 0] == ref[i, j])

 