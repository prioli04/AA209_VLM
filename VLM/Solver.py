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
        d_wake = self._params.wake_dt * self._params.V_inf * np.array([1.0, 0.0, 0.0])#Solver._V_inf_vec(self._params)

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
        C14 = self._wing_panels.get_C14()
        control_points = self._wing_panels.control_points_as_vector()
        normals = self._wing_panels.normals_as_vector()

        for k in range(self._n_wing_panels):
            L = 0

            for i in range(self._wing_nx):
                for j in range(self._wing_ny):
                    V, V_star = Solver._VORING(C14, control_points[k], i, j, 1.0, True)
                    self._AIC[k, L] = np.dot(V, normals[k])
                    self._B[k, L] = np.dot(V_star, normals[k])
                    L += 1
    
    def _wake_rollup(self, TE_Gammas: np.ndarray, it: int, d_wake: np.ndarray):
        self._wake_panels.step_wake(it, self._TE_points, d_wake)
        self._wake_panels.update_Gammas(TE_Gammas)

        # if it != 0:
        offset_map = self._build_wake_offset_map(it)
        self._wake_panels.offset_wake(offset_map)

    def _build_wake_offset_map(self, it):
        dt = self._params.wake_dt

        wake_C14 = self._wake_panels.get_C14()
        wing_Gammas = self._wing_panels.get_Gammas()
        wake_Gammas = self._wake_panels.get_Gammas()

        wake_panels_nx, wake_panels_ny = self._wake_panels.get_dimensions()
        n_wake_panels = wake_panels_nx * wake_panels_ny

        offset_map_X = np.zeros_like(wake_C14.X)
        offset_map_Y = np.zeros_like(wake_C14.X)
        offset_map_Z = np.zeros_like(wake_C14.X)

        wing_C14 = self._wing_panels.get_C14()

        for i in range(1, it + 1):
            for j in range(wake_panels_ny + 1):
                wake_point = np.array([wake_C14.X[i, j], wake_C14.Y[i, j], wake_C14.Z[i, j]])
                V_point = np.zeros(3) #self._params.V_inf * np.ones(3)

                for K in range(self._n_wing_panels):
                    i_wing, j_wing = Solver._get_ij_indicies(K, self._wing_ny)
                    dV_point, _ = Solver._VORING(wing_C14, wake_point, i_wing, j_wing, wing_Gammas[i_wing, j_wing], True)
                    V_point += dV_point

                for k in range(n_wake_panels):
                    i_wake, j_wake = Solver._get_ij_indicies(k, wake_panels_ny)
                    dV_point, _ = Solver._VORING(wake_C14, wake_point, i_wake, j_wake, wake_Gammas[i_wake, j_wake], True)
                    V_point += dV_point
            
                offset_point = dt * V_point
                offset_map_X[i, j] = offset_point[0]
                offset_map_Y[i, j] = offset_point[1]
                offset_map_Z[i, j] = offset_point[2]

        return PanelGrid.GridVector3(offset_map_X, offset_map_Y, offset_map_Z)

    def _update_RHS(self, it: int):
        V_inf_vec = Solver._V_inf_vec(self._params)
        control_points = self._wing_panels.control_points_as_vector()
        normals = self._wing_panels.normals_as_vector()

        for k in range(self._n_wing_panels):
            self._RHS[k] = -np.dot(V_inf_vec + self._compute_wake_influence(control_points[k], it), normals[k])

    def _compute_wake_influence(self, CP: np.ndarray, it: int):
        V_w = 0.0

        if it != 0:
            wake_nx, wake_ny = self._wake_panels.get_dimensions()
            wake_C14 = self._wake_panels.get_C14()
            Gammas_wake = self._wake_panels.get_Gammas()

            for i in range(it):
                for j in range(wake_ny):
                    dV_w, _ = Solver._VORING(wake_C14, CP, i, j, Gammas_wake[i , j], True)
                    V_w += dV_w

        return V_w

    @staticmethod
    def _VORING(
            quarter_chords: PanelGrid.GridVector3, 
            P: np.ndarray, 
            i: int, 
            j: int, 
            Gamma: float, 
            sym: bool):
        
        C14_x, C14_y, C14_z = quarter_chords

        P1 = np.array([C14_x[i, j], C14_y[i, j], C14_z[i, j]])
        P2 = np.array([C14_x[i, j + 1], C14_y[i, j + 1], C14_z[i, j + 1]])
        P3 = np.array([C14_x[i + 1, j + 1], C14_y[i + 1, j + 1], C14_z[i + 1, j + 1]])
        P4 = np.array([C14_x[i + 1, j], C14_y[i + 1, j], C14_z[i + 1, j]])

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
            Gamma: float):
        
        eps = 1e-6
        V = np.array([0.0, 0.0, 0.0])

        r1_vec = P - P1
        r2_vec = P - P2

        r1 = np.linalg.norm(r1_vec)
        r2 = np.linalg.norm(r2_vec)

        r1_cross_r2 = np.cross(r1_vec, r2_vec)
        norm_r1_cross_r2 = np.linalg.norm(r1_cross_r2)
        
        if r1 > eps and r2 > eps and norm_r1_cross_r2 > eps:
            r0_vec = P2 - P1
            r0_dot_r1 = np.dot(r0_vec, r1_vec)
            r0_dot_r2 = np.dot(r0_vec, r2_vec)

            V = (Gamma / (4.0 * np.pi * norm_r1_cross_r2**2)) * (r0_dot_r1 / r1 - r0_dot_r2 / r2) * r1_cross_r2

        return V
    
    @staticmethod
    def _V_inf_vec(params: Parameters):
        alfa_rad = np.deg2rad(params.alfa_deg)
        return params.V_inf * np.array([np.cos(alfa_rad), 0.0, np.sin(alfa_rad)])
    
    @staticmethod
    def _get_ij_indicies(K: int, ny: int):
        return K // ny, K % ny