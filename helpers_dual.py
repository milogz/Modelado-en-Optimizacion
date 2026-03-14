"""
helpers_dual.py -- Stub para el notebook 3-5 (Dualidad).
Las funciones se completaran en Fase 2.
"""
import numpy as np
from helpers_bases import mat_to_bmatrix, vec_to_bmatrix


def dual_info(A, b, c):
    """
    Placeholder: dado un LP en forma estandar (max c^T x, Ax <= b, x >= 0),
    retorna la formulacion dual (min b^T y, A^T y >= c, y >= 0).
    """
    return {
        "A_dual": A.T,
        "b_dual": c,
        "c_dual": b,
    }
