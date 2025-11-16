from .Airfoil import Airfoil
from .Flows import Flows
from .Parameters import Parameters
from .Section import Section

import numpy as np

class Decambering:
    def __init__(self, section: Section, nx: int, params: Parameters):
        self._nx = nx
        self._x2 = params.Decamb_x2
        self._Cl_tol = params.Decamb_Cl_tol
        self._Cm_tol = params.Decamb_Cm_tol
        self._max_iter = params.Decamb_max_iter

        airfoil = Airfoil.read(section.airfoil_path, section.xfoil_path)
        self._alfa_visc, self._Cl_visc, self._Cm_visc = airfoil.get_visc_coefs()

        x_vals = np.linspace(0.0, 1.0, nx + 1)  
        self._camber_lineX, self._camber_lineY = airfoil.get_camber_line(x_vals, 1.0, 0.0)
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
        camber_y += np.clip((self._camber_lineX - self._x2), a_min=0.0, a_max=None) * np.tan(-delta_2)

        vortices_pos = self._compute_panel_points(self._camber_lineX, camber_y, 0.25)
        control_points = self._compute_panel_points(self._camber_lineX, camber_y, 0.75)
        normals = self._compute_normals(self._camber_lineX, camber_y)
        return vortices_pos, control_points, normals
    
    def _lumped_vortex_2D(self, alfa_rad: float, vortices_pos: np.ndarray, control_points: np.ndarray, normals: np.ndarray):
        AIC = np.zeros((self._nx, self._nx)) # Aerodynamic influence coefficients
        RHS = np.zeros((self._nx, 1)) # Right-hand side

        for i in range(self._nx):
            x_i, y_i = control_points[:, i]
            n_i = normals[:, i]

            RHS[i] = -np.dot(np.array([np.cos(alfa_rad), np.sin(alfa_rad)]), n_i) # RHS_i = - V_inf * n_i

            for j in range(self._nx):
                x0_j, y0_j = vortices_pos[:, j]
                q_ij = Flows.VOR2D(x0_j, y0_j, x_i, y_i, 1.0)
                AIC[i, j] = np.dot(q_ij, n_i) # a_ij = q_ij(Gamma = 1.0) * n_i

        Gammas = np.linalg.solve(AIC, RHS)

        Cl = 2.0 * np.sum(Gammas) # Cl = 2 * sum(Gammas_i) / (V_inf * chord)
        Cm_14 = -2.0 * np.cos(alfa_rad) * np.sum(Gammas.squeeze() * (vortices_pos[0, :] - 0.25)) # Cm_14 = -2 * cos(alfa) * sum(Gammas_i * (x_i - quarter_chord) / (V_inf * chord**2)

        return Cl, Cm_14

    def solve(self, alfa_deg: float):
        Cl_visc = np.interp(alfa_deg, self._alfa_visc, self._Cl_visc)
        Cm_visc = np.interp(alfa_deg, self._alfa_visc, self._Cm_visc)

        iter = 1
        alfa_rad = np.deg2rad(alfa_deg)
        delta_1, delta_2 = 0.0, 0.0

        Cl, Cm = self._lumped_vortex_2D(alfa_rad, self._vortices_pos, self._control_points, self._normals)
        delta_Cl, delta_Cm = Cl_visc - Cl, Cm_visc - Cm

        while (np.abs(delta_Cl) > self._Cl_tol or np.abs(delta_Cm) > self._Cm_tol) and iter < self._max_iter:
            theta_2 = np.acos(1.0 - 2.0 * self._x2)
            ddelta_2 = delta_Cm / (0.25 * np.sin(2.0 * theta_2) - 0.5 * np.sin(theta_2))
            delta_2 += ddelta_2

            delta_1 += (delta_Cl - (2.0 * (np.pi - theta_2) + 2.0 * np.sin(theta_2)) * ddelta_2) / (2.0 * np.pi)
            vortices_decamber, control_points_decamber, normals_decamber = self._decamber_airfoil(delta_1, delta_2)

            Cl, Cm = self._lumped_vortex_2D(alfa_rad, vortices_decamber, control_points_decamber, normals_decamber)
            delta_Cl, delta_Cm = Cl_visc - Cl, Cm_visc - Cm 
            iter += 1
        
        return Cl, Cm
    