from nptyping import NDArray
import numpy as np
import math
from c_intmm import intmm
from ring import Ring

_calced_gs = {}
def memo_calc_g(ell: int, npdtype: np.dtype) -> NDArray:
	if ell in _calced_gs:
		return _calced_gs[ell]
	g = np.zeros((1, ell), dtype=npdtype)
	for i in range(ell - 1):
		g[0, i] = 2 ** i
	# Because we are working over ints, not uints
	g[0, ell - 1] = -1 * 2 ** (ell - 1)
	r = g.transpose()	
	_calced_gs[ell] = r
	return r

def gen_g_inv(ell: NDArray):
	"""
		Either scale out 
	"""
	def f(z:  NDArray)-> NDArray:
		# Much thanks to https://stackoverflow.com/questions/21918267/convert-decimal-range-to-numpy-array-with-each-bit-being-an-array-element
		# Assume the array is 2D and get back a 3D array
		multi_dim = z[:, :, np.newaxis] >> np.arange(ell) & 1
		two_D = (multi_dim.swapaxes(1, 2))
		return two_D

	return f