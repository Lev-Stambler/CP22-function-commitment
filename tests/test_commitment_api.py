from package import circuit, commitment_api as comm, utils
import numpy as np

security_param = 10  # TODO: make 128


def test_simple_add():
    """
            Test a simple add function
    """
    params = comm.CommitmentParams(
        security_param, d_stack_size=security_param, ell=32, nptype=np.int32)
    circ = circuit.Circuit(2, [circuit.Gate(params.ring, [0, 1], add=True)], 1)
    k = 2
    w = params.ell * params.n
    inp_C_gates = [params.ring.random(
        n=params.n, repeat=params.ell) for i in range(2)]
    xs = [
        np.array([params.n * [10]], dtype=params.nptype),
        np.array([params.n * [10]], dtype=params.nptype),
    ]

    commitment = comm.commit(circ, inp_C_gates, params)
    opening = comm.open(circ, inp_C_gates, xs, params)
    np.testing.assert_array_equal(opening.x, xs[0] * 2)

    verify_val, m = comm.verify(circ, inp_C_gates, commitment.C,
                                xs, opening.x, opening.S, params)
    assert (verify_val == True), m


def test_scalar():
    """
        Test a simple right multiply
    """
    params = comm.CommitmentParams(
        security_param, d_stack_size=security_param, ell=32, nptype=np.int32)
    ring = params.ring
    scalarMul = (10 * ring.mult_iden(params.n))

    gate_scale_params = {
        'scalarMul': scalarMul
    }
    circ = circuit.Circuit(
        1, [circuit.Gate(ring, [0], scalarMul=True, params=gate_scale_params)], depth=1)

    d = security_param
    w = params.ell * params.n
    inp_C_gates = [params.ring.random(
        n=params.n, repeat=params.ell) for i in range(1)]

    xs = [
        np.array([params.n * [10]], dtype=params.nptype),
    ]

    commitment = comm.commit(circ, inp_C_gates, params)

    opening = comm.open(circ, inp_C_gates, xs, params)
    # TODO: factor out pattern...
    assert (np.array_equal(opening.x, np.array(
        1 * [security_param * [100]])))


    verify_val, msg = comm.verify(circ, inp_C_gates, commitment.C,
                                  xs, opening.x, opening.S, params)
    assert (verify_val == True), msg


def test_simple_linear():
    """
            Test a simple scalar_mult function
    """
    # Add x0 + x1 = x3. Then take x3 and scalar mul by X2 to get x4. Then output x3 + x4
    params = comm.CommitmentParams(security_param, d_stack_size=security_param)
    ring = params.ring
    scalarMul = (13 * ring.mult_iden(params.n))
    circ = circuit.Circuit(2, [circuit.Gate(ring, [0, 1], add=True),
                               circuit.Gate(ring,
                                            [2], scalarMul=True,
                                            params={'scalarMul': scalarMul}
                                            )
                               ], depth=2)

    inp_C_gates = [params.ring.random(
        n=params.n, repeat=params.ell) for i in range(2)]
    xs = [
        np.array([params.n * [3]], dtype=params.nptype),
        np.array([params.n * [2]], dtype=params.nptype),
    ]

    commitment = comm.commit(circ, inp_C_gates, params)

    opening = comm.open(circ, inp_C_gates, xs, params)
    assert (np.array_equal(opening.x, np.array(
        [security_param * [65]])))


    verify_val, msg = comm.verify(circ, inp_C_gates, commitment.C,
                                  xs, opening.x, opening.S, params)
    assert (verify_val == True), msg


def test_random_circuit():
    """
            Test a simple random circuit, again I love random circuit testing...
    """
    pass
