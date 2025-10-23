import numpy as np
from .Mesh import Mesh
from .Panels import Panels
from .Parameters import Parameters
from .Post import Post  

class Solver:
    def __init__(self, panels: Panels, params: Parameters):
        self._panels = panels
        self._params = params
        self._wing_nx, self._wing_ny = self._panels.get_wing_n_panels()

        self._AIC, self._B, self._RHS = self._compute_aerodynamic_influence()
        
        self._Gammas: np.ndarray | None = None
        self._w_ind: np.ndarray | None = None
        self._post = Post()

    def solve(self):
        Gammas = np.linalg.solve(self._AIC, self._RHS)
        w_ind = self._B @ Gammas

        self._Gammas = Gammas.reshape(self._wing_nx, -1)
        self._w_ind = w_ind.reshape(self._wing_nx, -1)
        
        self._post.compute_coefficients(self._panels._wing_mesh, self._params, self._Gammas, self._w_ind)
        return self._post.export_results()
    
    def print_results(self):
        self._post.print_results()

    def _compute_aerodynamic_influence(self):
        symmetry_CP_factors = np.array([1.0, -1.0, 1.0])

        V_inf_vec = self._params.V_inf_vec()
        n_panels = self._wing_nx * self._wing_ny

        combined_panels = self._panels.get_panels_combined()
        quarter_chords = combined_panels.get_quarter_chords()
        # wake_corners = combined_panels.get_wake_corners()
        collocation_points = combined_panels.get_collocation_points()
        normals = combined_panels.get_normals()

        collocation_points_x = collocation_points.X.ravel()
        collocation_points_y = collocation_points.Y.ravel()
        collocation_points_z = collocation_points.Z.ravel()

        normals_x = normals.X.ravel()
        normals_y = normals.Y.ravel()
        normals_z = normals.Z.ravel()

        AIC = np.zeros((n_panels, n_panels))
        B = np.zeros((n_panels, n_panels))
        RHS = np.zeros((n_panels, 1))

        for k in range(n_panels):
            L = 0

            CP = [collocation_points_x[k], collocation_points_y[k], collocation_points_z[k]]
            normal = [normals_x[k], normals_y[k], normals_z[k]]

            RHS[k] = -np.dot(V_inf_vec, normal)

            for i in range(self._wing_nx):
                for j in range(self._wing_ny):
                    Vi, Vi_star = Solver._VORING(quarter_chords, CP, i, j, 1.0)
                    Vii, Vii_star = Solver._VORING(quarter_chords, CP * symmetry_CP_factors, i, j, 1.0)

                    if i == self._wing_nx - 1:
                        # Vi_wake, Vi_star_wake = Solver._VORING(quarter_chords, CP, i, j, 1.0, wake_corners=wake_corners)
                        # Vii_wake, Vii_star_wake = Solver._VORING(quarter_chords, CP * symmetry_CP_factors, i, j, 1.0, wake_corners=wake_corners)

                        Vi_wake, Vi_star_wake = Solver._VORING(quarter_chords, CP, i + 1, j, 1.0)
                        Vii_wake, Vii_star_wake = Solver._VORING(quarter_chords, CP * symmetry_CP_factors, i + 1, j, 1.0)

                        Vi += Vi_wake
                        Vii += Vii_wake

                        Vi_star += Vi_star_wake
                        Vii_star += Vii_star_wake

                    AIC[k, L] = np.dot(Vi + Vii * [1.0, -1.0, 1.0], normal)
                    B[k, L] = np.dot(Vi_star + Vii_star * [1.0, -1.0, 1.0], normal)
                    L += 1

        return AIC, B, RHS

    @staticmethod
    def _VORING(
            quarter_chords: Mesh.GridVector3, 
            P: np.ndarray, 
            i: int, 
            j: int, 
            Gamma: float, 
            ):#wake_corners: VLM.Mesh.GridVector3 | None = None):
        
        C14_x, C14_y, C14_z = quarter_chords

        # if wake_corners is not None:
        #     wake_x, wake_y, wake_z = wake_corners

        #     P1 = np.array([C14_x[i + 1, j], C14_y[i + 1, j], C14_z[i + 1, j]])  
        #     P2 = np.array([C14_x[i + 1, j + 1], C14_y[i + 1, j + 1], C14_z[i + 1, j + 1]])
        #     P3 = np.array([wake_x[j + 1], wake_y[j + 1], wake_z[j + 1]])
        #     P4 = np.array([wake_x[j], wake_y[j], wake_z[j]])
        
        # else:
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