from .Flows import Flows
from .Parameters import Parameters
from .WingPanels import WingPanels
from scipy.optimize import root_scalar # type: ignore[import-untyped]
from scipy.interpolate import interp1d # type: ignore[import-untyped]
from typing import Callable, List

import numpy as np

class Decambering:
    __p = 0.001

    def __init__(self, wing_mesh: WingPanels, params: Parameters):
        self._x2 = params.decamb_x2
        self._theta2 = params.decamb_theta2
        self._Cl_tol = params.decamb_Cl_tol
        self._Cm_tol = params.decamb_Cm_tol
        self._max_iter = params.decamb_max_iter

        airfoils, self._airfoil_ids = wing_mesh.get_airfoils()
        
        self._alfa_visc: List[np.ndarray] = []
        self._Cl_visc: List[np.ndarray] = []
        self._Cm_visc: List[np.ndarray] = []
        self._alfa0: List[float] = []

        for airfoil in airfoils:
            alfa_visc, Cl_visc, Cm_visc, alfa0 = airfoil.get_visc_coefs()
            self._alfa_visc.append(alfa_visc)
            self._Cl_visc.append(Cl_visc)
            self._Cm_visc.append(Cm_visc)
            self._alfa0.append(alfa0)

        self._camber_lineX: List[np.ndarray] = []
        self._camber_lineY: List[np.ndarray] = []
        self._nx: List[int] = []

        self._vortices_pos: List[np.ndarray] = []
        self._control_points: List[np.ndarray] = []
        self._normals: List[np.ndarray] = []

        for id in self._airfoil_ids:
            camberX, camberY = airfoils[id].get_raw_camber_line()
            self._camber_lineX.append(camberX)
            self._camber_lineY.append(camberY)
            self._nx.append(len(camberX) - 1)

            self._vortices_pos.append(self._compute_panel_points(camberX, camberY, 0.25))
            self._control_points.append(self._compute_panel_points(camberX, camberY, 0.75))
            self._normals.append(self._compute_normals(camberX, camberY))

        self._trajectory_slopes = np.zeros(len(self._airfoil_ids))
        self._delta1, self._delta2 = np.zeros(len(self._airfoil_ids)), np.zeros(len(self._airfoil_ids))
        self._delta1_orig_id, self._delta1_orig_val = 0, 0.0
        self._delta2_orig_id, self._delta2_orig_val = 0, 0.0

    def _compute_panel_points(self, pointsX: np.ndarray, pointsY: np.ndarray, frac: float):
        px = pointsX[:-1] + frac * np.diff(pointsX)
        py = pointsY[:-1] + frac * np.diff(pointsY)
        return np.vstack([px, py])
    
    def _compute_normals(self, pointsX: np.ndarray, pointsY: np.ndarray):
        dy_dx = np.diff(pointsY) / np.diff(pointsX)
        ny = 1.0 / np.sqrt(1 + dy_dx**2)
        nx = -dy_dx * ny
        return np.vstack([nx, ny])
    
    # def _decamber_airfoil(self, delta_1: float, delta_2: float):
    #     camber_y = self._camber_lineY + self._camber_lineX * np.tan(-delta_1) 
    #     camber_y += np.clip((self._camber_lineX - self._x2), a_min=0.0, a_max=None) * np.tan(-delta_2)

    #     vortices_pos = self._compute_panel_points(self._camber_lineX, camber_y, 0.25)
    #     control_points = self._compute_panel_points(self._camber_lineX, camber_y, 0.75)
    #     normals = self._compute_normals(self._camber_lineX, camber_y)
    #     return vortices_pos, control_points, normals
    
    def _lumped_vortex_2D(self, nx: int, alfa_rad: float, vortices_pos: np.ndarray, control_points: np.ndarray, normals: np.ndarray):
        AIC = np.zeros((nx, nx)) # Aerodynamic influence coefficients
        RHS = np.zeros((nx, 1)) # Right-hand side

        for i in range(nx):
            x_i, y_i = control_points[:, i]
            n_i = normals[:, i]

            RHS[i] = -np.dot(np.array([np.cos(alfa_rad), np.sin(alfa_rad)]), n_i) # RHS_i = - V_inf * n_i

            for j in range(nx):
                x0_j, y0_j = vortices_pos[:, j]
                q_ij = Flows.VOR2D(x0_j, y0_j, x_i, y_i, 1.0)
                AIC[i, j] = np.dot(q_ij, n_i) # a_ij = q_ij(Gamma = 1.0) * n_i

        Gammas = np.linalg.solve(AIC, RHS)

        Cl = 2.0 * np.sum(Gammas) # Cl = 2 * sum(Gammas_i) / (V_inf * chord)
        Cm_14 = -2.0 * np.cos(alfa_rad) * np.sum(Gammas.squeeze() * (vortices_pos[0, :] - 0.25)) # Cm_14 = -2 * cos(alfa) * sum(Gammas_i * (x_i - quarter_chord) / (V_inf * chord**2)

        return Cl, Cm_14
    
    def interpolate_visc_Cl(self, alfa_deg: np.ndarray, sec_id: int):
        foil_id = self._airfoil_ids[sec_id]
        return np.interp(alfa_deg, self._alfa_visc[foil_id], self._Cl_visc[foil_id])
    
    def interpolate_visc_Cm(self, alfa_deg: np.ndarray, sec_id: int):
        foil_id = self._airfoil_ids[sec_id]
        return np.interp(alfa_deg, self._alfa_visc[foil_id], self._Cm_visc[foil_id])
    
    def _find_intersection_Cl(self, alfa_s: np.ndarray, Cl_s: np.ndarray, alfa_p: np.ndarray, Cl_p: np.ndarray):
        alfa_s_deg, alfa_p_deg = np.rad2deg(alfa_s), np.rad2deg(alfa_p)
        a = (Cl_s - Cl_p) / (alfa_s_deg - alfa_p_deg)
        b = Cl_s - a * alfa_s_deg

        Cl_intersect, Cm_intersect = np.zeros(len(Cl_s)), np.zeros(len(Cl_s))

        for i in range(len(Cl_s)):
            Cl_trajectory = lambda x: a[i] * x + b[i]
            Cl_visc = lambda x: self.interpolate_visc_Cl(x, i)
            func = lambda x: Cl_trajectory(x) - Cl_visc(x)

            alfa_intersect = root_scalar(func, method="newton", x0=alfa_s_deg[i], xtol=1e-3).root
            Cl_intersect[i] = Cl_trajectory(alfa_intersect)
            Cm_intersect[i] = self.interpolate_visc_Cm(alfa_intersect, i)
            
        return Cl_intersect, Cm_intersect
    
    # def _find_intersection_Cl(self, alfa_s: np.ndarray, Cl_s: np.ndarray, alfa_p: np.ndarray, Cl_p: np.ndarray):
    #     alfa_s_deg = np.rad2deg(alfa_s)
    #     Cl_intersect, Cm_intersect = np.zeros(len(Cl_s)), np.zeros(len(Cl_s))

    #     for i in range(len(Cl_s)):
    #         Cl_intersect[i] = self.interpolate_visc_Cl(alfa_s_deg[i], i)
    #         Cm_intersect[i] = self.interpolate_visc_Cm(alfa_s_deg[i], i)
            
    #     return Cl_intersect, Cm_intersect

    def compute_effective_alfa(self, Cl_sec: float, id_sec: int):
        alfa0 = self._alfa0[self._airfoil_ids[id_sec]]
        return Cl_sec / (2.0 * np.pi) - self._delta1[id_sec] - self._delta2[id_sec] * (1 - (np.sin(self._theta2) - self._theta2) / np.pi) + alfa0

    def compute_residuals_scheme1(self, Cl_sec: np.ndarray, Cm_sec: np.ndarray, alfa_sec: np.ndarray):
        Cl_visc, Cm_visc = np.zeros(len(alfa_sec)), np.zeros(len(alfa_sec))

        for i in range(len(alfa_sec)):
            Cl_visc[i], Cm_visc[i] = self.interpolate_visc_Cl(np.rad2deg(alfa_sec[i]), i)
        
        delta_Cl, delta_Cm = Cl_sec - Cl_visc, Cm_sec - Cm_visc
        return delta_Cl
    
    def compute_residuals_scheme2(self, alfa_s: np.ndarray, Cl_s: np.ndarray, alfa_p: np.ndarray, Cl_p: np.ndarray, Cm_s: np.ndarray):
        Cl_visc, Cm_visc = self._find_intersection_Cl(alfa_s, Cl_s, alfa_p, Cl_p)
        delta_Cl, delta_Cm = Cl_s - Cl_visc, Cm_s - Cm_visc
        # return np.hstack([delta_Cl, delta_Cm])
        return delta_Cl

    def decamber_normals(self, normals: np.ndarray, control_pointX: np.ndarray):
        filter_delta2 = np.clip(np.sign(control_pointX - self._x2), a_min=0.0, a_max=None) * self._delta2

        normals_x = normals[:, 0].reshape(-1, len(self._airfoil_ids))
        normals_y = normals[:, 1].reshape(-1, len(self._airfoil_ids))
        normals_z = normals[:, 2].reshape(-1, len(self._airfoil_ids))

        normals_x_rot1 = np.cos(-self._delta1) * normals_x - np.sin(-self._delta1) * normals_z 
        normals_z_rot1 = np.sin(-self._delta1) * normals_x + np.cos(-self._delta1) * normals_z

        normals_x_rot2 = np.cos(-filter_delta2) * normals_x_rot1 - np.sin(-filter_delta2) * normals_z_rot1
        normals_z_rot2 = np.sin(-filter_delta2) * normals_x_rot1 + np.cos(-filter_delta2) * normals_z_rot1

        return np.hstack((normals_x_rot2.reshape(-1, 1), normals_y.reshape(-1, 1), normals_z_rot2.reshape(-1, 1)))

    # def set_trajectory_slope(self, solver_func: Callable[[bool], np.ndarray], coefs_func: Callable[[np.ndarray], np.ndarray], Gammas_orig: np.ndarray):
    #     Cl_0 = coefs_func(Gammas_orig)
    #     alfa_0 = np.array([self.compute_effective_alfa(Cl_0[i], i) for i in range(len(Cl_0))])

    #     delta_Cl = self.compute_residuals_scheme1(Cl_0, np.zeros_like(Cl_0), alfa_0)
    #     self.update_delta1(delta_Cl)

    #     Gammas_decambering = solver_func(True)
    #     Cl_p = coefs_func(Gammas_decambering)
    #     alfa_p = np.array([self.compute_effective_alfa(Cl_p[i], i) for i in range(len(Cl_p))])
    #     self._trajectory_slopes = (Cl_p - Cl_0) / (alfa_p - alfa_0)
    #     self._delta1 *= 0.0

    #     return alfa_0, Cl_0

    def perturb_delta1_at(self, j: int):
        self._delta1_orig_id, self._delta1_orig_val = j, self._delta1[j]
        self._delta1[j] += Decambering.__p

    def unperturb_delta1(self):
        self._delta1[self._delta1_orig_id] = self._delta1_orig_val
        self._delta1_orig_id, self._delta1_orig_val = 0, 0.0

    def perturb_delta2_at(self, j: int):
        self._delta2_orig_id, self._delta2_orig_val = j, self._delta2[j]
        self._delta2[j] += Decambering.__p

    def unperturb_delta2(self):
        self._delta2[self._delta2_orig_id] = self._delta2_orig_val
        self._delta2_orig_id, self._delta2_orig_val = 0, 0.0

    def update_deltas(self, deltas: np.ndarray):
    #     self._delta1 += deltas[:int(len(deltas) / 2)]
    #     self._delta2 += deltas[int(len(deltas) / 2):]
        self._delta1 += deltas
        self._delta1 = np.clip(self._delta1, a_min=-np.deg2rad(45.0), a_max=np.deg2rad(45.0))
        # self._delta2 = np.clip(self._delta2, a_min=-np.deg2rad(45.0), a_max=np.deg2rad(45.0))

    @classmethod
    def p(cls):
        return cls.__p

    # def solve_2D(self, alfa_deg: float):
    #     Cl_visc, Cm_visc = self.interpolate_visc_coefs(alfa_deg)

    #     iter = 1
    #     alfa_rad = np.deg2rad(alfa_deg)
    #     delta_1, delta_2 = 0.0, 0.0

    #     Cl, Cm = self._lumped_vortex_2D(alfa_rad, self._vortices_pos, self._control_points, self._normals)
    #     delta_Cl, delta_Cm = Cl_visc - Cl, Cm_visc - Cm

    #     while (np.abs(delta_Cl) > self._Cl_tol or np.abs(delta_Cm) > self._Cm_tol) and iter < self._max_iter:
    #         ddelta_2 = delta_Cm / (0.25 * np.sin(2.0 * self._theta2) - 0.5 * np.sin(self._theta2))
    #         delta_2 += ddelta_2

    #         delta_1 += (delta_Cl - (2.0 * (np.pi - self._theta2) + 2.0 * np.sin(self._theta2)) * ddelta_2) / (2.0 * np.pi)
    #         vortices_decamber, control_points_decamber, normals_decamber = self._decamber_airfoil(delta_1, delta_2)

    #         Cl, Cm = self._lumped_vortex_2D(alfa_rad, vortices_decamber, control_points_decamber, normals_decamber)
    #         delta_Cl, delta_Cm = Cl_visc - Cl, Cm_visc - Cm 
    #         iter += 1
        
    #     return Cl, Cm
        

