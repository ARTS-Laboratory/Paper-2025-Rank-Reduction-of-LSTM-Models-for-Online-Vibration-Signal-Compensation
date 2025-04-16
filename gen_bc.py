# Copyright (c) UofSC ARTS Lab, 2025

import numpy as np

# Computes the B and C matrices for the DoF decomposition given a
# reduced SVD as input.
def gen_bc(u, s, vt, r):
    u1 = u[:r, :r]
    u2 = u[r:, :r]

    B = u1 @ s @ vt
    C = u2 @ np.linalg.inv(u1)

    return B, C
