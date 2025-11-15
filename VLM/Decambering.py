from .Airfoil import Airfoil
from .Flows import Flows
from .Section import Section

import numpy as np

class Decambering:
    def __init__(self, section: Section, nx: int):
        self._nx = nx
        self._chord = 1.0
        self._x2 = 0.8
        airfoil = Airfoil.read(section.airfoil_path, section.xfoil_path)
        self._alfa_visc, self._Cl_visc, self._Cm_visc = airfoil.get_visc_coefs()

        x_vals = np.linspace(0.0, self._chord, nx + 1)  
        self._camber_lineX, self._camber_lineY = airfoil.get_camber_line(x_vals, self._chord, 0.0)
        self._vortices_pos = self._compute_panel_points(self._camber_lineX, self._camber_lineY, 0.25)
        self._control_points = self._compute_panel_points(self._camber_lineX, self._camber_lineY, 0.75)
        self._normals = self._compute_normals(self._camber_lineX, self._camber_lineY)

    def _compute_panel_points(self, pointsX: np.ndarray, pointsY: np.ndarray, frac: float):
        px = pointsX[:-1] + frac * np.diff(pointsX)
        py = pointsY[:-1] + frac * np.diff(pointsY)
        return np.vstack([px, py])
    
    def _compute_normals(self, pointsX: np.ndarray, pointsY: np.ndarray):
        dy_dx = np.diff(pointsY) / np.diff(pointsX)
        ny = 1.0 / np.sqrt(1 + dy_dx**2)
        nx = -dy_dx * ny
        return np.vstack([nx, ny])
    
    def _decamber_airfoil(self, delta_1: float, delta_2: float):
        camber_y = self._camber_lineY + self._camber_lineX * np.tan(-delta_1) 
        camber_y += np.clip((self._camber_lineX - self._x2) * np.tan(-delta_2), a_min=0.0, a_max=None)

        vortices_pos = self._compute_panel_points(self._camber_lineX, camber_y, 0.25)
        control_points = self._compute_panel_points(self._camber_lineX, camber_y, 0.75)
        normals = self._compute_normals(self._camber_lineX, camber_y)
        return vortices_pos, control_points, normals
    
    def _solve_lumped_vortex_2D(self, alfa_rad: float, vortices_pos: np.ndarray, control_points: np.ndarray, normals: np.ndarray):
        V_inf = 12.0
        rho = 1.225

        AIC = np.zeros((self._nx, self._nx)) # Aerodynamic influence coefficients
        RHS = np.zeros((self._nx, 1)) # Right-hand side

        for i in range(self._nx):
            x_i, y_i = control_points[:, i]
            n_i = normals[:, i]

            RHS[i] = -np.dot(V_inf * np.array([np.cos(alfa_rad), np.sin(alfa_rad)]), n_i) # RHS_i = - V_inf * n_i

            for j in range(self._nx):
                x0_j, y0_j = vortices_pos[:, j]
                q_ij = Flows.VOR2D(x0_j, y0_j, x_i, y_i, 1.0)
                AIC[i, j] = np.dot(q_ij, n_i) # a_ij = q_ij(Gamma = 1.0) * n_i

        Gammas = np.linalg.solve(AIC, RHS)

        Cl = (2.0 / (V_inf * self._chord)) * np.sum(Gammas) # Cl = 2 * sum(Gammas_i) / (V_inf * chord)
        Cm_14 = (-2.0 * np.cos(alfa_rad) / (V_inf * self._chord**2)) * np.sum(Gammas.squeeze() * (vortices_pos[0, :] - 0.25 * self._chord)) # Cm_14 = -2 * cos(alfa) * sum(Gammas_i * (x_i - quarter_chord) / (V_inf * chord**2)

        return Cl, Cm_14

    def solve(self, alfa_deg: float):
        Cl_visc = np.interp(alfa_deg, self._alfa_visc, self._Cl_visc)
        Cm_visc = np.interp(alfa_deg, self._alfa_visc, self._Cm_visc)

        Cl_potential, Cm_potential = self._solve_lumped_vortex_2D(np.deg2rad(alfa_deg), self._vortices_pos, self._control_points, self._normals)
        delta_Cl = Cl_visc - Cl_potential
        delta_Cm = Cm_visc - Cm_potential

        theta_2 = np.acos(1.0 - 2.0 * self._x2)

        delta_2 = delta_Cm / (0.25 * np.sin(2.0 * theta_2) - 0.5 * np.sin(theta_2))
        delta_1 = (delta_Cl - (2.0 * (np.pi - theta_2) + 2.0 * np.sin(theta_2)) * delta_2) / (2.0 * np.pi)

        vortices_decamber, control_points_decamber, normals_decamber = self._decamber_airfoil(delta_1, delta_2)
        Cl_decamber, Cm_decamber = self._solve_lumped_vortex_2D(np.deg2rad(alfa_deg), vortices_decamber, control_points_decamber, normals_decamber)
        return Cl_decamber, Cm_decamber
    