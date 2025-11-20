from .Parameters import Parameters
from .WingPanels import WingPanels
from typing import List

import numpy as np

class Decambering:
    def __init__(self, wing_mesh: WingPanels, params: Parameters, delta_init_vals: np.ndarray | None = None):
        self._x2 = params.decamb_x2
        self._theta2 = params.decamb_theta2
        self._K = params.decamb_under_relaxation
        self._PI = params.decamb_smoothing

        airfoils, self._airfoil_ids = wing_mesh.get_airfoils()
        
        self._alfa_visc: List[np.ndarray] = []
        self._Cl_visc: List[np.ndarray] = []
        self._Cm_visc: List[np.ndarray] = []
        self._alfa0: List[float] = []

        for airfoil in airfoils:
            if params.decambering and not airfoil.has_viscous_data():
                raise ValueError("Decambering is activated but not all airfoils have viscous data.")

            alfa_visc, Cl_visc, Cm_visc, alfa0 = airfoil.get_visc_coefs()
            self._alfa_visc.append(alfa_visc)
            self._Cl_visc.append(Cl_visc)
            self._Cm_visc.append(Cm_visc)
            self._alfa0.append(alfa0)

        self._delta = np.zeros(len(self._airfoil_ids)) if delta_init_vals is None else delta_init_vals

    def _interpolate_visc_Cl(self, alfa_deg: np.ndarray, sec_id: int):
        foil_id = self._airfoil_ids[sec_id]
        return np.interp(alfa_deg, self._alfa_visc[foil_id], self._Cl_visc[foil_id])
    
    def _interpolate_visc_Cm(self, alfa_deg: np.ndarray, sec_id: int):
        foil_id = self._airfoil_ids[sec_id]
        return np.interp(alfa_deg, self._alfa_visc[foil_id], self._Cm_visc[foil_id])
    
    def compute_effective_alfas(self, Cl_sec: np.ndarray):
        alfa0 = [self._alfa0[self._airfoil_ids[i]] for i in range(len(Cl_sec))]
        return Cl_sec / (2.0 * np.pi) - self._delta + alfa0

    def compute_residuals(self, alfa_sec: np.ndarray, Cl_sec: np.ndarray):
        Cl_visc = np.zeros(len(alfa_sec))

        for i in range(len(alfa_sec)):
            Cl_visc[i] = self._interpolate_visc_Cl(np.rad2deg(alfa_sec[i]), i)
        
        return Cl_visc - Cl_sec 
    
    def decamber_normals(self, normals: np.ndarray):
        normals_x = normals[:, 0].reshape(-1, len(self._airfoil_ids))
        normals_y = normals[:, 1].reshape(-1, len(self._airfoil_ids))
        normals_z = normals[:, 2].reshape(-1, len(self._airfoil_ids))

        normals_x_rot = np.cos(-self._delta) * normals_x - np.sin(-self._delta) * normals_z 
        normals_z_rot = np.sin(-self._delta) * normals_x + np.cos(-self._delta) * normals_z

        return np.hstack((normals_x_rot.reshape(-1, 1), normals_y.reshape(-1, 1), normals_z_rot.reshape(-1, 1)))

    def _apply_smoothing(self):
        new_delta = np.zeros_like(self._delta)

        for i in range(len(self._delta)):
            id1 = i - 1 if i > 0 else 0
            id2 = i + 1 if i < len(self._delta) - 1 else len(self._delta) - 1

            new_delta[i] = (self._delta[i] + 0.5 * self._PI * (self._delta[id1] + self._delta[id2])) / (1.0 + self._PI)

        self._delta = new_delta

    def update_deltas(self, delta_Cl: np.ndarray):
        self._delta += (1.0 / (1.0 + self._K)) * delta_Cl / (2.0 * np.pi)
        self._apply_smoothing()
        self._delta = np.clip(self._delta, a_min=-np.deg2rad(20.0), a_max=np.deg2rad(20.0))