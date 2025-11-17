from .Decambering import Decambering
from .Flows import Flows
from .Panels import Panels
from .Parameters import Parameters
from .Post import Post

import numpy as np
import time

class Solver:
    def __init__(self, panels: Panels, params: Parameters):
        if params.sym and params.beta_deg != 0.0:
            raise ValueError("Sideslip angle is different than 0° and the symmetry flag is activated.")

        self._params = params       
        self._wake_panels = panels.get_wake_panels()

        self._wing_panels = panels.get_wing_panels()
        self._wing_C14X, self._wing_C14Y, self._wing_C14Z = self._wing_panels.C14_VORING()
        self._wing_nx, self._wing_ny = self._wing_panels.get_dimensions()
        self._n_wing_panels = self._wing_nx * self._wing_ny

        self._AIC: np.ndarray = np.zeros((self._n_wing_panels, self._n_wing_panels))
        self.inv_AIC: np.ndarray = np.zeros((self._n_wing_panels, self._n_wing_panels))
        self._B_trefftz: np.ndarray = np.zeros((self._wing_ny, self._wing_ny))
        self._RHS: np.ndarray = np.zeros((self._n_wing_panels, 1))

        self._compute_aerodynamic_influence()
        self._post = Post(self._wing_panels, params)
        self._decambering = Decambering(self._wing_panels, params)

        panels.print_wing_geom()

        if self._wake_panels is not None:
            self._wake_panels.print_wake_params()
            
        self._params.print_run_params()

    def solve(self):
        t0 = time.time()
        d_wake = self._params.wake_dt * self._params.V_inf_vec()

        while not self._post.is_converged():
            Gammas = self._solve_linear_system()

            if self._params.decambering:
                Gammas = self._decamber_wing(Gammas)

            w_ind = self._B_trefftz @ Gammas[-1, :].T

            self._wing_panels.update_Gammas(Gammas)
            self._wing_panels.update_w_ind_trefftz(w_ind)
            
            self._post.compute_coefficients(Gammas, w_ind)
            self._post.print_results()
                
            if self._wake_panels is not None:
                self._wake_panels.wake_rollup(self._wing_C14X, self._wing_C14Y, self._wing_C14Z, Gammas, d_wake)

        print(f"\nCompleted in {(time.time() - t0):.2f} s.")
        return self._post.export_results()
    
    def _compute_aerodynamic_influence(self):
        C14X, C14Y, C14Z = self._wing_panels.C14_VORING()
        control_points = self._wing_panels.control_points_VORING(self._n_wing_panels)
        normals = self._wing_panels.normal_VORING(self._n_wing_panels)

        V = Flows.VORING(C14X, C14Y, C14Z, control_points, np.ones((self._n_wing_panels, self._n_wing_panels, 1)), self._params.sym, self._params.ground)
        self._AIC[:] = np.sum(V * normals, axis=2)
        
        if self._params.wake_fixed:
            C14X_w = np.hstack([C14X[-self._wing_ny:, 2:4], 1e6 * np.ones((self._wing_ny, 2))]) 
            V_w = Flows.VORING(C14X_w, C14Y[-self._wing_ny:, :], C14Z[-self._wing_ny:, :], control_points[:, -self._wing_ny:, :], np.ones((self._n_wing_panels, self._wing_ny, 1)), self._params.sym, self._params.ground)
            self._AIC[:, -self._wing_ny:] += np.sum(V_w * normals[:, -self._wing_ny:], axis=2)

        self._inv_AIC = np.linalg.inv(self._AIC)
        control_points_trefftz = self._wing_panels.control_points_TREFFTZ()
        C14_trefftz = self._wing_panels.C14_TREFFTZ()

        for i in range(self._wing_ny):
            CP = control_points_trefftz[i, :]

            for j in range(self._wing_ny):
                P1 = C14_trefftz[j, :]
                P2 = C14_trefftz[j + 1, :]

                self._B_trefftz[i, j] = Solver.bij_trefftz(CP, P1, P2, 1.0, self._params.sym, self._params.ground)

    def _update_RHS(self, decambering: bool):
        V_inf_vec = self._params.V_inf_vec()
        normals = self._wing_panels.normal_RHS()

        if decambering:
            normals = self._decambering.decamber_normals(normals)

        if self._wake_panels is not None:
            wake_influence = self._wake_panels.compute_wake_influence(self._wing_panels, self._wing_ny)
            self._RHS[:] = -np.sum((V_inf_vec + wake_influence) * normals, axis=1).reshape(-1, 1)

        else:
            self._RHS[:] = -np.sum(V_inf_vec * normals, axis=1).reshape(-1, 1)

    def _solve_linear_system(self, decambering: bool = False):
        self._update_RHS(decambering)
        Gammas = self._inv_AIC @ self._RHS
        return Gammas.reshape(self._wing_nx, -1)

    def _decamber_wing(self, Gammas: np.ndarray):
        iter = 0
        F = np.inf * np.ones((self._wing_ny, 1))

        t0 = time.time()
        while np.linalg.norm(F) > 1e-2 and iter < 200:
            Cl_s = self._post.compute_coefficients_decambering(Gammas)
            alfa_s = self._decambering.compute_effective_alfa(Cl_s)
            # F = self._decambering.compute_residuals_scheme1(Cl_s, np.zeros_like(Cl_s), alfa_s)

            Jl1 = np.zeros((self._wing_ny, self._wing_ny))

            for j in range(self._wing_ny):
                self._decambering.perturb_delta1_at(j)
                Gammas = self._solve_linear_system(decambering=True)
                Cl_p = self._post.compute_coefficients_decambering(Gammas)
                Jl1[:, j] = (Cl_p - Cl_s) / Decambering.p()

                alfa_p = self._decambering.compute_effective_alfa(Cl_p)
                F[j] = self._decambering.compute_residuals_scheme2(alfa_s[j], Cl_s[j], alfa_p[j], Cl_p[j], j)
                self._decambering.unperturb_delta1()

            delta1 = 0.1 * np.linalg.solve(Jl1, -F)
            self._decambering.update_delta1(delta1)
            Gammas = self._solve_linear_system(decambering=True)
            print(np.linalg.norm(F))
            iter += 1

        if iter >= 200:
            Gammas = np.zeros_like(Gammas)
            
        print(f"\nDecambering took {(time.time() - t0):.2f} s.")
        return Gammas
    
    @staticmethod
    def bij_trefftz(P: np.ndarray, P1: np.ndarray, P2: np.ndarray, Gamma: float, sym: bool, ground: bool):
        V_ind_trefftz1 = Flows.VOR2D(P1[1], P1[2], P[1], P[2], Gamma)
        V_ind_trefftz2 = Flows.VOR2D(P2[1], P2[2], P[1], P[2], Gamma)
        V_ind = V_ind_trefftz1 - V_ind_trefftz2

        if sym:
            P_sym = P * np.array([1.0, -1.0, 1.0])
            V_ind_trefftz1 = Flows.VOR2D(P1[1], P1[2], P_sym[1], P_sym[2], Gamma)
            V_ind_trefftz2 = Flows.VOR2D(P2[1], P2[2], P_sym[1], P_sym[2], Gamma)
            V_ind += (V_ind_trefftz1 - V_ind_trefftz2) * np.array([-1.0, 1.0])

            if ground:
                P_ground_sym = P_sym * np.array([1.0, 1.0, -1.0])

                V_ind_trefftz1 = Flows.VOR2D(P1[1], P1[2], P_ground_sym[1], P_ground_sym[2], Gamma)
                V_ind_trefftz2 = Flows.VOR2D(P2[1], P2[2], P_ground_sym[1], P_ground_sym[2], Gamma)
                V_ind += (V_ind_trefftz1 - V_ind_trefftz2) * np.array([-1.0, -1.0])

        if ground:
            P_ground = P * np.array([1.0, 1.0, -1.0])

            V_ind_trefftz1 = Flows.VOR2D(P1[1], P1[2], P_ground[1], P_ground[2], Gamma)
            V_ind_trefftz2 = Flows.VOR2D(P2[1], P2[2], P_ground[1], P_ground[2], Gamma)
            V_ind += (V_ind_trefftz1 - V_ind_trefftz2) * np.array([1.0, -1.0])

        return V_ind[1]