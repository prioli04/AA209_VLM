from .Airfoil import Airfoil
from .Flows import Flows
from .Section import Section

import numpy as np

class Decambering:
    def __init__(self, section: Section, nx: int):
        self._nx = nx
        self._chord = 1.0
        airfoil = Airfoil.read(section.airfoil_path)

        x_vals = np.linspace(0.0, self._chord, nx + 1)  
        self._camber_lineX, self._camber_lineY = airfoil.get_camber_line(x_vals, self._chord, 0.0)
        self._vortexX, self._vortexY = self._compute_panel_points(0.25)
        self._control_pointX, self._control_pointY = self._compute_panel_points(0.75)
        self._normalX, self._normalY = self._compute_normals()

    def _compute_panel_points(self, frac: float):
        px = self._camber_lineX + frac * np.diff(self._camber_lineX)
        py = self._camber_lineY + frac * np.diff(self._camber_lineY)
        return px, py
    
    def _compute_normals(self):
        dy_dx = np.diff(self._camber_lineY) / np.diff(self._camber_lineX)
        ny = 1.0 / np.sqrt(1 + dy_dx**2)
        nx = -dy_dx * ny
        return nx, ny

    def _solve_lumped_vortex_2D(self, alfa_rad: float):
        V_inf = 12.0
        AIC = np.zeros((self._nx, self._nx)) # Aerodynamic influence coefficients
        RHS = np.zeros((self._nx, 1)) # Right-hand side

        for i in range(self._nx):
            x_i, y_i = self._control_pointX[i], self._control_pointY[i]
            nx_i, ny_i = self._normalX[i], self._normalY[i]

            RHS[i] = -np.dot(V_inf * np.array([np.cos(alfa_rad), np.sin(alfa_rad)]), np.array([nx_i, ny_i])) # RHS_i = - V_inf * n_i

            for j in range(self._nx):
                x0_j, y0_j = self._vortexX[j], self._vortexY[j]
                q_ij = Flows.VOR2D(x0_j, y0_j, x_i, y_i, 1.0)
                AIC[i, j] = np.dot(q_ij, np.array([nx_i, ny_i])) # a_ij = q_ij(Gamma = 1.0) * n_i

        Gammas = np.linalg.solve(AIC, RHS)
        self._Cl = (2.0 / (V_inf * self._chord)) * np.sum(Gammas) # Cl = 2 * sum(Gammas_i) / (V_inf * chord)