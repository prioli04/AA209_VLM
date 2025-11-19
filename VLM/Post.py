from .Parameters import Parameters
from .Result import Result
from .WingPanels import WingPanels

import matplotlib.pyplot as plt
import numpy as np
import sys

class Post:
    def __init__(self, wing_mesh: WingPanels, params: Parameters):
        self._cursor_up = False
        self._converged = False
        self._iter = 0
        self._sym = params.sym
        self._wake_fixed = params.wake_fixed

        self._rho = params.rho
        self._V_inf = params.V_inf
        self._AR = params.AR
        self._S = params.S
        self._q_inf = 0.5 * self._rho * self._V_inf**2

        _, self._C14_y, self._C14_z = wing_mesh.get_C14()
        self._vortices_x, _, _ = wing_mesh.get_control_points()
        self._delta_y = np.abs(np.diff(self._C14_y[-1, :]))
        self._chords = wing_mesh.get_chords()

        self._result = Result()
        self._tolerances = Result.Coefs_3D(params.CL_tol, params.CD_tol, params.CY_tol, params.CMl_tol, params.CM_tol, params.CN_tol)

    def _check_convergence(self):
        self._converged = np.all((self._result.residuals.to_array() - self._tolerances.to_array()) < 0.0)

    def export_results(self):
        return self._result
    
    def get_sectional_results(self):
        return self._result.Cl_sec, self._result.Cd_sec
    
    def print_results(self):
        self._result.print(self._iter, self._wake_fixed)

    def is_converged(self):
        return self._converged

    def compute_coefficients(self, Gammas: np.ndarray, w_ind: np.ndarray, plot=False):
        dy, dz = np.diff(self._C14_y[-1, :]), np.diff(self._C14_z[-1, :])
        theta_i = np.atan2(dz, dy)
        delta_s_i = np.sqrt(dy**2 + dz**2)

        S_ref = 0.5 * self._S if self._sym else self._S

        delta_phi_i = Gammas[-1, :]
        delta_L = self._rho * self._V_inf * delta_phi_i * np.cos(theta_i) * delta_s_i
        delta_Y = -self._rho * self._V_inf * delta_phi_i * np.sin(theta_i) * delta_s_i
        delta_Di = - 0.5 * self._rho * w_ind * Gammas[-1, :] * delta_s_i

        Cl_sec = delta_L / (self._q_inf * self._delta_y * self._chords)
        Cd_sec = delta_Di / (self._q_inf * self._delta_y * self._chords)

        CL = np.sum(delta_L) / (self._q_inf * S_ref)
        CY = np.sum(delta_Y) / (self._q_inf * S_ref)
        CD = np.sum(delta_Di) / (self._q_inf * S_ref)

        coefs_3D = Result.Coefs_3D(CL, CD, CY)
        self._result.update(Cl_sec, Cd_sec, coefs_3D, self._AR)
        self._check_convergence()

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