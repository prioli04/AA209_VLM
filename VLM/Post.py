from typing import NamedTuple
from .Parameters import Parameters
from .Wing import Wing

import numpy as np
import sys

class Post:
    class Result(NamedTuple):
        CL: float
        CD: float
        CL_CD: float
        efficiency: float

    def __init__(self, CL_tol: float, CD_tol: float):
        self._result: Post.Result | None = None
        self._cursor_up = False
        self._converged = False
        self._iter = 0

        self._CL_tol = CL_tol
        self._CD_tol = CD_tol

        self._CL_prev = 0.0
        self._CD_prev = 0.0

        self._CL_res = 0.0
        self._CD_res = 0.0

    def export_results(self):
        return self._result
    
    def print_results(self):
        fields = self._result._fields
        N = len(fields) + 1

        # '\x1b' -> Start escape sequence
        # '[' -> Control sequence introducer
        # '2K' ->  Clear entire line
        ERASE_LINE = "\x1b[2K"

        # '\x1b' -> Start escape sequence
        # '[' -> Control sequence introducer
        # 'NA' ->  Move cursor up N lines
        CURSOR_UP_N = f"\x1b[{N}A"

        if self._cursor_up:
            sys.stdout.write(CURSOR_UP_N)

        else:
            print()
            self._cursor_up = True

        sys.stdout.write(ERASE_LINE)
        sys.stdout.write(f"Iteration {self._iter}: CL_res = {self._CL_res:.5e} \t CD_res = {self._CD_res:.5e} \n")

        for f in fields:
            sys.stdout.write(ERASE_LINE)
            sys.stdout.write(f"{f}: {self._result.__getattribute__(f):.5f}\n")

    def is_converged(self):
        return self._converged

    def compute_coefficients(self, wing_mesh: Wing, params: Parameters, Gammas: np.ndarray, w_ind: np.ndarray):
        self._iter += 1
        ny = Gammas.shape[1]
        delta_L = np.zeros(ny)
        delta_D = np.zeros(ny)

        rho = params.rho
        V_inf = params.V_inf
        S = params.S
        AR = params.AR

        _, C14_y, _ = wing_mesh.get_C14()

        for j in range(ny):
            delta_y = np.abs(C14_y[-1, j + 1] - C14_y[-1, j])

            delta_L[j] = rho * V_inf * Gammas[-1, j] * delta_y
            delta_D[j] = - 0.5 * rho * w_ind[j] * Gammas[-1, j] * delta_y

        L = delta_L.sum()
        D = delta_D.sum()

        CL = L / (0.5 * rho * V_inf**2 * (0.5 * S))
        CD = D / (0.5 * rho * V_inf**2 * (0.5 * S))

        self._CL_res = CL - self._CL_prev
        self._CD_res = CD - self._CD_prev

        self._converged = self._CL_res < self._CL_tol and self._CD_res < self._CD_tol
        self._CL_prev = CL
        self._CD_prev = CD

        CL_CD = CL / CD
        efficiency = CL**2 / (CD * np.pi * AR)
        self._result = Post.Result(CL, CD, CL_CD, efficiency)