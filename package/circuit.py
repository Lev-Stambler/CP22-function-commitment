import math
from nptyping import NDArray
from types import NoneType
import numpy as np
import homomorphic as homomorphic
from utils import *
from ring import Ring


class EvaledWire:
    def __init__(self, ring: Ring, C: NDArray, S: NDArray | None, n: int, x: NDArray | None, ell=32):
        """
                :param S can be `None`
                :param x can be `None`
        """
        self.ell = ell
        self.C = C
        # If S is None use the identity, this means to keep things the same
        self.S = S if S is not None else ring.mult_iden_square_mat(ell)
        self.n = n
        self.x = x


class Gate:
    def __init__(self, ring: Ring, input_wires: list[int], mul=False, add=False, scalarMul=False, eq=False, params={}):
        self.ring = ring
        self.input_wires = input_wires
        if mul:
            raise "Not yet implemented"
            self.gate_type = "MUL"
        elif add:
            self.gate_type = "ADD"
        elif scalarMul:
            self.gate_type = "ScMUL"
            self.scalarMul = params["scalarMul"]
        elif eq:
            self.gate_type = "Eq"
        else:
            raise "Please set choose a gate in the init parameters"

    def mul_eval(self, a: EvaledWire, b: EvaledWire, skip_S_and_x=False) -> EvaledWire:
        pass

    def add_eval(self, a: EvaledWire, b: EvaledWire, skip_S_and_x=False) -> EvaledWire:
        assert (a.n == b.n), "Expected dimensions for addition to equal"
        C = self.ring.concat((a.C, b.C), axis=0)
        n = C.shape[0]
        hom = homomorphic.Homomorphic(n, self.ring, a.ell)
        CPlus, SPlus = hom.homomorphic_add(C)
        n = a.n

        if skip_S_and_x:
            return EvaledWire(self.ring, CPlus, None, n, None, a.ell)

        S = self.ring.concat((a.S, b.S), axis=0)
        return EvaledWire(self.ring, CPlus, S, a.n, a.x + b.x, ell=a.ell)

    def scalar_mul_eval(self, a: EvaledWire, skip_S_and_x=False) -> EvaledWire:
        scale = self.scalarMul
        assert (
            a.n == scale.shape[-1]), "Expected dimension of scale matrix to match that of input wire"

        C = a.C
        n = C.shape[0]
        hom = homomorphic.Homomorphic(n, self.ring, a.ell)
        CScale, SSCale = hom.homomorphic_scalar_mult(C, scale)
        if skip_S_and_x:
            return EvaledWire(self.ring, CScale, None, a.n, None, a.ell)
        # Compose S
        S = self.ring.mul_mat(a.S, SSCale)
        return EvaledWire(self.ring, CScale, S, a.n, np.array(self.ring.dot(a.x, scale)), a.ell)

    def eval(self, input_wire_vals: list[EvaledWire], skip_S_and_x=False) -> EvaledWire:
        """
                Evaluate the gate and return the results

                :param input_wire_vals: a list of tuples, each item in the list representing the wire values.
                The first item in the tuple is the C matrix, the second the S matrix which may be none, and finally
                the x vector which may be null
        """
        if self.gate_type == "MUL":
            raise "Not yet implemented"
        elif self.gate_type == "ADD":
            return self.add_eval(input_wire_vals[0], input_wire_vals[1], skip_S_and_x=skip_S_and_x)
        elif self.gate_type == "ScMUL":
            return self.scalar_mul_eval(input_wire_vals[0], skip_S_and_x=skip_S_and_x)
        elif self.gate_type == "Eq":
            raise "Not yet implemented"
            pass
        else:
            raise f"Non-existent gate type: {self.gate_type}"


class Circuit:
    """
            A class representing the circuit which is evaluated homomorphically.
            The circuit is represented as a list of gates. When evaluating the circuit,
            we have an array of wire values which starts off as input values. After going
            through each gate, we append the gate's output to a list of wire values
    """

    def __init__(self, n_input_wires: int, gates: list[Gate], depth: int):
        self.gates = gates
        self.n_input_wires = n_input_wires
        self.depth = depth

    def eval_with_proof(self, inputs: list[EvaledWire], only_get_commitment=False) -> EvaledWire:
        """
                Evaluate the circuit by going through the list of gates one at a time and
                evaluating their input to produce an output

                :return the last evaled wire
        """
        wire_vals: list[EvaledWire] = inputs
        for gate in self.gates:
            gate_inputs = [wire_vals[i] for i in gate.input_wires]
            out = gate.eval(gate_inputs, skip_S_and_x=only_get_commitment)
            wire_vals.append(out)
        return wire_vals[-1]
