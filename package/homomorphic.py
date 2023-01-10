from nptyping import NDArray
import numpy as np
import math
from utils import *
from ring import Ring

class Homomorphic():
	def __init__(self, n, ring: Ring, ell=32) -> None:
		self.ell = ell
		self.n = n
		self.g = memo_calc_g(self.ell, ring.nptype)
		self.g_inv = gen_g_inv(self.ell)
		self.w = n * self.ell
		self.ring = ring

	def homomorphic_add(self, C) -> tuple[NDArray, NDArray]:
		"""
			:return CPlus and SPlus
		"""
		t = self.ring.mult_iden_square_mat(self.ell)
		SPlus = self.ring.concat((t, t), axis=0)
		CPlus = self.ring.mul_right_matrix(C, SPlus)
		return CPlus, SPlus

	def homomorphic_scalar_mult(self, C, Y):
		"""
			:param Y is a d by d' matrix

			:return CScale and Scale
		"""
		Ykroned = self.ring.kron(Y, self.g)
		SScaleMul = self.g_inv(Ykroned).swapaxes(0, 1)
		
		CScaleMul = self.ring.mul_right_matrix(C, SScaleMul)
		
		return CScaleMul, SScaleMul

	def homomorphic_mult(self):
		raise "Not yet implemented"

	def homomorphic_eq(self, C, XPrime):
		pass