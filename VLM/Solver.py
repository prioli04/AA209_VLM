import numpy as np
from .PanelGrid import PanelGrid
from .Panels import Panels
from .Parameters import Parameters
from .Post import Post

class Solver:
    def __init__(self, panels: Panels, params: Parameters):
        self._params = params        
        self._wake_panels = panels.get_wake_panels()

        self._wing_panels = panels.get_wing_panels()
        self._wing_nx, self._wing_ny = self._wing_panels.get_dimensions()
        self._n_wing_panels = self._wing_nx * self._wing_ny

        self._TE_points = self._wing_panels.extract_TE_points()
        self._wake_panels.add_TE(self._TE_points)

        self._AIC: np.ndarray = np.zeros((self._n_wing_panels, self._n_wing_panels))
        self._B: np.ndarray = np.zeros((self._n_wing_panels, self._n_wing_panels))
        self._RHS: np.ndarray = np.zeros((self._n_wing_panels, 1))

        self._compute_aerodynamic_influence()
        self._post = Post()

    def solve(self):
        inv_AIC = np.linalg.inv(self._AIC)
        d_wake = self._params.wake_dt * self._params.V_inf * np.array([1.0, 0.0, 0.0])

        for it in range(self._params.wake_steps):
            self._update_RHS(it)

            Gammas = inv_AIC @ self._RHS
            w_ind = self._B @ Gammas

            Gammas = Gammas.reshape(self._wing_nx, -1)
            w_ind = w_ind.reshape(self._wing_nx, -1)

            self._wing_panels.update_Gammas(Gammas)
            self._wing_panels.update_w_ind(w_ind)
            
            self._post.compute_coefficients(self._wing_panels, self._params, Gammas, w_ind)
            self.print_results()
            print()

            self._wake_rollup(Gammas[-1, :], it, d_wake)

        return self._post.export_results()
    
    def print_results(self):
        self._post.print_results()

    def _compute_aerodynamic_influence(self):
        C14X, C14Y, C14Z = self._wing_panels.C14_VORING()
        control_points = self._wing_panels.control_points_VORING(self._n_wing_panels)
        normals = self._wing_panels.normal_VORING(self._n_wing_panels)

        V, V_star = Solver._VORING(C14X, C14Y, C14Z, control_points, np.ones((self._n_wing_panels, self._n_wing_panels, 1)), True)
        self._AIC[:] = np.sum(V * normals, axis=2)
        self._B[:] = np.sum(V_star * normals, axis=2)

    def _wake_rollup(self, TE_Gammas: np.ndarray, it: int, d_wake: np.ndarray):
        self._wake_panels.step_wake(it, self._TE_points, d_wake)
        self._wake_panels.update_Gammas(TE_Gammas)

        if it != 0:
            offset_map = self._build_wake_offset_map(it)
            self._wake_panels.offset_wake(offset_map)

    def _build_wake_offset_map(self, it):
        dt = self._params.wake_dt
        wake_panels_nx, wake_panels_ny = self._wake_panels.get_dimensions()

        wing_C14X, wing_C14Y, wing_C14Z = self._wing_panels.C14_VORING()
        wake_C14X, wake_C14Y, wake_C14Z = self._wake_panels.C14_VORING(it)

        wake_C14_as_CP_wing = self._wake_panels.C14_as_control_points(it, self._n_wing_panels)
        wake_C14_as_CP_wake = self._wake_panels.C14_as_control_points(it, wake_C14X.shape[0])
        
        wing_Gammas = self._wing_panels.get_Gammas()
        wing_Gammas = np.tile(wing_Gammas.reshape(1, -1), [wake_C14_as_CP_wing.shape[0], 1])[:, :, np.newaxis]

        wake_Gammas = self._wake_panels.get_Gammas(it)
        wake_Gammas = np.tile(wake_Gammas.reshape(1, -1), [wake_C14_as_CP_wake.shape[0], 1])[:, :, np.newaxis]

        offset_map_X = np.zeros((wake_panels_nx + 1, wake_panels_ny + 1))
        offset_map_Y = np.zeros((wake_panels_nx + 1, wake_panels_ny + 1))
        offset_map_Z = np.zeros((wake_panels_nx + 1, wake_panels_ny + 1))

        V = np.zeros((wake_C14_as_CP_wake.shape[0], 3))

        dV_wing, _ = Solver._VORING(wing_C14X, wing_C14Y, wing_C14Z, wake_C14_as_CP_wing, wing_Gammas, True)
        V += np.sum(dV_wing, axis=1)

        dV_wake, _ = Solver._VORING(wake_C14X, wake_C14Y, wake_C14Z, wake_C14_as_CP_wake, wake_Gammas, True)
        V += np.sum(dV_wake, axis=1)
            
        offset_point = dt * V
        offset_map_X[1:it + 1, :] = offset_point[:, 0].reshape(-1, wake_panels_ny + 1)
        offset_map_Y[1:it + 1, :] = offset_point[:, 1].reshape(-1, wake_panels_ny + 1)
        offset_map_Z[1:it + 1, :] = offset_point[:, 2].reshape(-1, wake_panels_ny + 1)

        return PanelGrid.GridVector3(offset_map_X, offset_map_Y, offset_map_Z)

    def _update_RHS(self, it: int):
        V_inf_vec = Solver._V_inf_vec(self._params)
        normals = self._wing_panels.normal_RHS()
        self._RHS[:] = -np.sum((V_inf_vec + self._compute_wake_influence(it)) * normals, axis=1).reshape(-1, 1)

    def _compute_wake_influence(self, it: int):
        if it == 0:
            return 0.0

        else:
            wake_C14X, wake_C14Y, wake_C14Z = self._wake_panels.C14_VORING(it)
            control_points = self._wing_panels.control_points_VORING(wake_C14X.shape[0])

            wake_Gammas = self._wake_panels.get_Gammas(it)
            wake_Gammas = np.tile(wake_Gammas.reshape(1, -1), [control_points.shape[0], 1])[:, :, np.newaxis]

            dV_w, _ = Solver._VORING(wake_C14X, wake_C14Y, wake_C14Z, control_points, wake_Gammas, True)
            return np.sum(dV_w, axis=1)

    @staticmethod
    def _VORING(
            C14X: np.ndarray,
            C14Y: np.ndarray,
            C14Z: np.ndarray,
            P: np.ndarray, 
            Gamma: np.ndarray, 
            sym: bool):
        
        P1 = np.vstack((C14X[:, 0], C14Y[:, 0], C14Z[:, 0])).T
        P2 = np.vstack((C14X[:, 1], C14Y[:, 1], C14Z[:, 1])).T
        P3 = np.vstack((C14X[:, 2], C14Y[:, 2], C14Z[:, 2])).T
        P4 = np.vstack((C14X[:, 3], C14Y[:, 3], C14Z[:, 3])).T

        P1 = np.tile(P1, [P.shape[0], 1, 1])
        P2 = np.tile(P2, [P.shape[0], 1, 1])
        P3 = np.tile(P3, [P.shape[0], 1, 1])
        P4 = np.tile(P4, [P.shape[0], 1, 1])

        V1 = Solver._VORTXL(P, P1, P2, Gamma)
        V2 = Solver._VORTXL(P, P2, P3, Gamma)
        V3 = Solver._VORTXL(P, P3, P4, Gamma)
        V4 = Solver._VORTXL(P, P4, P1, Gamma)

        V = V1 + V2 + V3 + V4
        V_star = V2 + V4

        if sym:
            P_sym = P * np.array([1.0, -1.0, 1.0])
            V1 = Solver._VORTXL(P_sym, P1, P2, Gamma)
            V2 = Solver._VORTXL(P_sym, P2, P3, Gamma)
            V3 = Solver._VORTXL(P_sym, P3, P4, Gamma)
            V4 = Solver._VORTXL(P_sym, P4, P1, Gamma)

            V += (V1 + V2 + V3 + V4) * np.array([1.0, -1.0, 1.0])
            V_star += (V2 + V4) * np.array([1.0, -1.0, 1.0])

        return V, V_star

    @staticmethod
    def _VORTXL(
            P: np.ndarray, 
            P1: np.ndarray, 
            P2: np.ndarray, 
            Gamma: np.ndarray):
        
        eps = 1e-6

        r1_vec = P - P1
        r2_vec = P - P2

        r1 = np.linalg.norm(r1_vec, axis=2)[:, :, np.newaxis]
        r2 = np.linalg.norm(r2_vec, axis=2)[:, :, np.newaxis]

        r1_cross_r2 = np.cross(r1_vec, r2_vec, axis=2)
        norm_r1_cross_r2 = np.linalg.norm(r1_cross_r2, axis=2)[:, :, np.newaxis]

        r_cut = np.logical_and(r1 > eps, r2 > eps)
        r_cut = np.logical_and(r_cut, norm_r1_cross_r2 > eps).squeeze()

        r0_vec = P2 - P1
        r0_dot_r1 = np.sum(r0_vec * r1_vec, axis=2)[:, :, np.newaxis]
        r0_dot_r2 = np.sum(r0_vec * r2_vec, axis=2)[:, :, np.newaxis]

        V = (Gamma / (4.0 * np.pi * norm_r1_cross_r2**2)) * (r0_dot_r1 / r1 - r0_dot_r2 / r2) * r1_cross_r2
        V[~r_cut, :] = 0.0
        return V

    @staticmethod
    def _V_inf_vec(params: Parameters):
        alfa_rad = np.deg2rad(params.alfa_deg)
        return params.V_inf * np.array([np.cos(alfa_rad), 0.0, np.sin(alfa_rad)])