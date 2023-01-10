from nptyping import NDArray
import numpy as np
import homomorphic as homomorphic
import circuit as circuit
from ring import Ring
import utils as utils
import math


class CommitmentParams:
    def __init__(self, lattice_dim_n: int, d_stack_size=1, ell=32, nptype=np.int32):
        self.ring = Ring(lattice_dim_n, nptype)
        self.n = lattice_dim_n
        self.d = d_stack_size
        self.ell = ell
        self.nptype = nptype

    def norm_kappa_S_bound(self, max_circuit_depth, ell):
        """
                We only need circuit S bound as it is most permissive.
                For performance gains, linear or branching program bounds help.
                See the top of page 11 of the paper
        """
        w = self.n * ell
        kappa = w ** max_circuit_depth
        beta = 2 * w * kappa + self.n

        # * 10 to ensure that any constants are taken care of
        # TODO: check if the above makes sense
        assert (2 ** (ell - 1) > beta * 10 *
                math.sqrt(self.n)), "q is too small"
        return kappa


def commit(f: circuit.Circuit, Cs: list[NDArray], params: CommitmentParams):
    """
            Create the commitment to f

            :param f: the function to commit to
            :param C: a random matrix which is publicly known
    """
    input_wires = [circuit.EvaledWire(params.ring,
                                      C, None, params.n, None, params.ell) for C in Cs]
    Cf = f.eval_with_proof(input_wires, only_get_commitment=True)
    return Cf


def open(f: circuit.Circuit, Cs: list[NDArray], xs: list[NDArray], params: CommitmentParams) -> circuit.EvaledWire:
    input_wires = [circuit.EvaledWire(params.ring,
                                      C, None, params.n, xs[i], params.ell) for i, C in enumerate(Cs)]
    return f.eval_with_proof(input_wires, only_get_commitment=False)


def verify(f: circuit.Circuit, Cs: list[NDArray], Cf: NDArray, xs: list[NDArray],
           y: NDArray, Sf: NDArray, params: CommitmentParams) -> bool:
    g = utils.memo_calc_g(params.ell, params.ring.nptype)
    C = params.ring.concat(Cs, axis=0)
    x = params.ring.concat(xs, axis=0)

    #  TODO: check me
    CSub = params.ring.sub(C, params.ring.kron(x, g))
    LHS = params.ring.mul_right_matrix(CSub, Sf)
    RHS = params.ring.sub(Cf, params.ring.kron(y, g))

    S_norm = params.ring.mat_norm(Sf)
    kappa = params.norm_kappa_S_bound(f.depth, params.ell)
    if S_norm >= kappa:
        return False, f"Norm too large at {S_norm} with kappa {kappa}"
    eq_comp = np.array_equal(LHS, RHS)
    arrs_eq = eq_comp

    if arrs_eq == False:
        return False, "Matrix check does not work"
    return True, "Verified without problem"

# TODO: CHEC ME OUT https://eprint.iacr.org/2016/381.pdf
