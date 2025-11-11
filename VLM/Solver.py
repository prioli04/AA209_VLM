from .Flows import Flows
from .Panels import Panels
from .Parameters import Parameters
from .Post import Post

import numpy as np
import time

class Solver:
    def __init__(self, panels: Panels, params: Parameters):
        self._params = params       
        self._wake_panels = panels.get_wake_panels()

        self._wing_panels = panels.get_wing_panels()
        self._wing_C14X, self._wing_C14Y, self._wing_C14Z = self._wing_panels.C14_VORING()
        self._wing_nx, self._wing_ny = self._wing_panels.get_dimensions()
        self._n_wing_panels = self._wing_nx * self._wing_ny

        self._AIC: np.ndarray = np.zeros((self._n_wing_panels, self._n_wing_panels))
        self._B_trefftz: np.ndarray = np.zeros((self._wing_ny, self._wing_ny))
        self._RHS: np.ndarray = np.zeros((self._n_wing_panels, 1))

        self._compute_aerodynamic_influence()
        self._post = Post(params.CL_tol, params.CD_tol)

        panels.print_wing_geom()
        self._wake_panels.print_wake_params()
        self._params.print_run_params()

    def solve(self):
        t0 = time.time()

        inv_AIC = np.linalg.inv(self._AIC)
        d_wake = self._params.wake_dt * self._params.V_inf * np.array([1.0, 0.0, 0.0])

        while not self._post.is_converged():
            self._update_RHS()

            Gammas = inv_AIC @ self._RHS
            Gammas = Gammas.reshape(self._wing_nx, -1)

            w_ind = self._B_trefftz @ Gammas[-1, :].T

            self._wing_panels.update_Gammas(Gammas)
            self._wing_panels.update_w_ind_trefftz(w_ind)
            
            self._post.compute_coefficients(self._wing_panels, self._params, Gammas, w_ind)
            self._post.print_results()

            self._wake_panels.wake_rollup(self._wing_C14X, self._wing_C14Y, self._wing_C14Z, Gammas, d_wake)

        print(f"\nCompleted in {(time.time() - t0):.2f} s.")
        return self._post.export_results()
    
    def _compute_aerodynamic_influence(self):
        C14X, C14Y, C14Z = self._wing_panels.C14_VORING()
        control_points = self._wing_panels.control_points_VORING(self._n_wing_panels)
        normals = self._wing_panels.normal_VORING(self._n_wing_panels)

        V = Flows.VORING(C14X, C14Y, C14Z, control_points, np.ones((self._n_wing_panels, self._n_wing_panels, 1)), True, self._params.ground, np.deg2rad(self._params.alfa_deg))
        self._AIC[:] = np.sum(V * normals, axis=2)

        control_points_trefftz = self._wing_panels.control_points_TREFFTZ()
        C14_trefftz = self._wing_panels.C14_TREFFTZ()

        for i in range(self._wing_ny):
            CP = control_points_trefftz[i, :]

            for j in range(self._wing_ny):
                P1 = C14_trefftz[j, :]
                P2 = C14_trefftz[j + 1, :]

                self._B_trefftz[i, j] = Solver.bij_trefftz(CP, P1, P2, 1.0, True, self._params.ground, np.deg2rad(self._params.alfa_deg))

    def _update_RHS(self):
        V_inf_vec = Solver._V_inf_vec(self._params)
        normals = self._wing_panels.normal_RHS()

        wake_influence = self._wake_panels.compute_wake_influence(self._wing_panels, self._wing_ny)
        self._RHS[:] = -np.sum((V_inf_vec + wake_influence) * normals, axis=1).reshape(-1, 1)

    @staticmethod
    def bij_trefftz(P: np.ndarray, P1: np.ndarray, P2: np.ndarray, Gamma: float, sym: bool, ground: bool, alfa: float = 0.0):
        V_ind_trefftz1 = Flows.VOR2D(P1[1], P1[2], P[1], P[2], Gamma)
        V_ind_trefftz2 = Flows.VOR2D(P2[1], P2[2], P[1], P[2], Gamma)
        V_ind = V_ind_trefftz1 - V_ind_trefftz2

        if sym:
            P_sym = P * np.array([1.0, -1.0, 1.0])
            V_ind_trefftz1 = Flows.VOR2D(P1[1], P1[2], P_sym[1], P_sym[2], Gamma)
            V_ind_trefftz2 = Flows.VOR2D(P2[1], P2[2], P_sym[1], P_sym[2], Gamma)
            V_ind += (V_ind_trefftz1 - V_ind_trefftz2) * np.array([1.0, -1.0, 1.0])

            if ground:
                P_ground_sym = P_sym * np.array([1.0, 1.0, -(1.0 / np.cos(alfa))])

                V_ind_trefftz1 = Flows.VOR2D(P1[1], P1[2], P_ground_sym[1], P_ground_sym[2], Gamma)
                V_ind_trefftz2 = Flows.VOR2D(P2[1], P2[2], P_ground_sym[1], P_ground_sym[2], Gamma)
                V_ind += (V_ind_trefftz1 - V_ind_trefftz2) * np.array([1.0, -1.0, -1.0])

        if ground:
            P_ground = P * np.array([1.0, 1.0, -(1.0 / np.cos(alfa))])

            V_ind_trefftz1 = Flows.VOR2D(P1[1], P1[2], P_ground[1], P_ground[2], Gamma)
            V_ind_trefftz2 = Flows.VOR2D(P2[1], P2[2], P_ground[1], P_ground[2], Gamma)
            V_ind += (V_ind_trefftz1 - V_ind_trefftz2) * np.array([1.0, 1.0, -1.0])

        return V_ind[2]

    @staticmethod
    def _V_inf_vec(params: Parameters):
        alfa_rad = np.deg2rad(params.alfa_deg)
        return params.V_inf * np.array([np.cos(alfa_rad), 0.0, np.sin(alfa_rad)])