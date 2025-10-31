from typing import NamedTuple
import numpy as np
from .Parameters import Parameters
from .Wing import Wing

class Post:
    class Result(NamedTuple):
        CL: float
        CD: float
        CL_CD: float
        efficiency: float

    def __init__(self, CL_tol: float, CD_tol: float):
        self._result: Post.Result | None = None
        self._converged = False
        self._CL_tol = CL_tol
        self._CD_tol = CD_tol

        self._CL_prev = 0.0
        self._CD_prev = 0.0

    def export_results(self):
        return self._result
    
    def print_results(self):
        for f in self._result._fields:
            print(f"{f}: {self._result.__getattribute__(f)}")

    def is_converged(self):
        return self._converged

    def compute_coefficients(self, wing_mesh: Wing, params: Parameters, Gammas: np.ndarray, w_ind: np.ndarray):
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

        self._converged = CL - self._CL_prev < self._CL_tol and CD - self._CD_prev < self._CD_tol
        self._CL_prev = CL
        self._CD_prev = CD

        CL_CD = CL / CD
        efficiency = CL**2 / (CD * np.pi * AR)
        self._result = Post.Result(CL, CD, CL_CD, efficiency)