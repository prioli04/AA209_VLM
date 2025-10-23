from typing import NamedTuple
import numpy as np
import VLM

class Post:
    class Result(NamedTuple):
        CL: float
        CD: float
        CL_CD: float
        efficiency: float

    def __init__(self):
        self._result: Post.Result | None = None

    def export_results(self):
        return self._result
    
    def print_results(self):
        for f in self._result._fields:
            print(f"{f}: {self._result.__getattribute__(f)}")

    def compute_coefficients(self, mesh: VLM.Mesh, params: VLM.Parameters, Gammas: np.ndarray, w_ind: np.ndarray):
        delta_L = np.zeros_like(Gammas)
        delta_D = np.zeros_like(Gammas)
        nx, ny = Gammas.shape

        rho = params.rho
        V_inf = params.V_inf
        S = params.S
        AR = params.AR

        _, quarter_chords_y, _ = mesh.get_quarter_chords()

        for i in range(nx):
            for j in range(ny):
                delta_y = np.abs(quarter_chords_y[i, j + 1] - quarter_chords_y[i, j])
                delta_Gamma = Gammas[i, j] - Gammas[i - 1, j] if i != 0 else Gammas[i, j]

                delta_L[i, j] = rho * V_inf * delta_Gamma * delta_y
                delta_D[i, j] = -rho * w_ind[i, j] * delta_Gamma * delta_y

        L = delta_L.sum()
        D = delta_D.sum()

        CL = L / (0.5 * rho * V_inf**2 * (0.5 * S))
        CD = D / (0.5 * rho * V_inf**2 * (0.5 * S))

        CL_CD = CL / CD
        efficiency = CL**2 / (CD * np.pi * AR)
        self._result = Post.Result(CL, CD, CL_CD, efficiency)