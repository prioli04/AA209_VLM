from typing import NamedTuple
from .Parameters import Parameters
from .WingPanels import WingPanels

import matplotlib.pyplot as plt
import numpy as np
import sys

class Post:
    class Result(NamedTuple):
        Cl_sec: np.ndarray
        Cd_sec: np.ndarray
        CL: float
        CD: float
        CL_CD: float
        efficiency: float

    def __init__(self, wing_mesh: WingPanels, params: Parameters):
        self._result: Post.Result | None = None
        self._cursor_up = False
        self._converged = False
        self._iter = 0
        self._sym = params.sym
        self._wake_fixed = params.wake_fixed

        self._rho = params.rho
        self._V_inf = params.V_inf
        self._AR = params.AR
        self._S = params.S

        _, self._C14_y, _ = wing_mesh.get_C14()
        self._vortices_x, _, _ = wing_mesh.get_control_points()
        self._delta_y = np.abs(np.diff(self._C14_y[-1, :]))
        self._chords = wing_mesh.get_chords()

        self._CL_tol = np.inf if self._wake_fixed else params.CL_tol
        self._CD_tol = np.inf if self._wake_fixed else params.CD_tol

        self._CL_prev = 0.0
        self._CD_prev = 0.0

        self._CL_res = 0.0
        self._CD_res = 0.0

    def export_results(self):
        return self._result
    
    def get_sectional_results(self):
        return self._result.Cl_sec, self._result.Cd_sec
    
    def print_results(self):
        fields = list(self._result._fields)
        fields.remove("Cl_sec")
        fields.remove("Cd_sec")
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

        if not self._wake_fixed:
            sys.stdout.write(ERASE_LINE)
            sys.stdout.write(f"Iteration {self._iter}: CL_res = {self._CL_res:.5e} \t CD_res = {self._CD_res:.5e} \n")

        for f in fields:
            sys.stdout.write(ERASE_LINE)
            sys.stdout.write(f"{f}: {self._result.__getattribute__(f):.5f}\n")

    def is_converged(self):
        return self._converged

    def compute_coefficients(self, Gammas: np.ndarray, w_ind: np.ndarray, plot=False):
        self._iter += 1
        ny = Gammas.shape[1]
        delta_L, delta_D = np.zeros(ny), np.zeros(ny)
        Cl_sec, Cd_sec = np.zeros(ny), np.zeros(ny)

        S_ref = 0.5 * self._S if self._sym else self._S

        for j in range(ny):
            delta_L[j] = self._rho * self._V_inf * Gammas[-1, j] * self._delta_y[j]
            delta_D[j] = - 0.5 * self._rho * w_ind[j] * Gammas[-1, j] * self._delta_y[j]

            Cl_sec[j] = delta_L[j] / (0.5 * self._rho * self._V_inf**2 * self._delta_y[j] * self._chords[j])
            Cd_sec[j] = delta_D[j] / (0.5 * self._rho * self._V_inf**2 * self._delta_y[j] * self._chords[j])

        L = delta_L.sum()
        D = delta_D.sum()

        CL = L / (0.5 * self._rho * self._V_inf**2 * S_ref)
        CD = D / (0.5 * self._rho * self._V_inf**2 * S_ref)

        self._CL_res = CL - self._CL_prev
        self._CD_res = CD - self._CD_prev

        self._converged = self._CL_res < self._CL_tol and self._CD_res < self._CD_tol
        self._CL_prev = CL
        self._CD_prev = CD

        CL_CD = CL / CD
        efficiency = CL**2 / (CD * np.pi * self._AR)
        self._result = Post.Result(Cl_sec, Cd_sec, CL, CD, CL_CD, efficiency)

        if plot:
            self._plot_trefftz()

    def compute_coefficients_decambering(self, Gammas: np.ndarray):
        ny = Gammas.shape[1]
        delta_L = np.zeros(ny)
        Cl_sec = np.zeros(ny)
        Cm_sec = np.zeros(ny)

        for j in range(ny):
            delta_L[j] = self._rho * self._V_inf * Gammas[-1, j] * self._delta_y[j]
            Cl_sec[j] = delta_L[j] / (0.5 * self._rho * self._V_inf**2 * self._delta_y[j] * self._chords[j])

            for i in range(Gammas.shape[0]):
                g = Gammas[i, j] - Gammas[i - 1, j] if i > 0 else Gammas[0, j]
                Cm_sec[j] = -2.0 * np.cos(0.0) * np.sum(g * (self._vortices_x[:, j] - 0.25 * self._chords[j])) / (self._V_inf * self._chords[j]**2)

        return Cl_sec, Cm_sec

    # @staticmethod
    def _plot_trefftz(self):
        plt.figure()
        plt.plot((self._C14_y[-1, :-1] + self._C14_y[-1, 1:]) / 2.0, self._result.Cl_sec)
        plt.xlabel("Y [m]")
        plt.ylabel("Cl [-]")
        plt.show(block=False)

    def plot_delta1(self, delta1: np.ndarray):
        plt.figure()
        plt.plot((self._C14_y[-1, :-1] + self._C14_y[-1, 1:]) / 2.0, np.rad2deg(delta1))
        plt.xlabel("Y [m]")
        plt.ylabel("delta1 [°]")
        plt.ylim([-10.0, 10.0])
        plt.show()