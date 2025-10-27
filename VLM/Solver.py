import numpy as np
from .Flows import Flows
from .Panels import Panels
from .Parameters import Parameters
from .Post import Post

class Solver:
    def __init__(self, panels: Panels, params: Parameters):
        self._params = params        
        self._wake_panels = panels.get_wake_panels()

        self._wing_panels = panels.get_wing_panels()
        self._wing_C14X, self._wing_C14Y, self._wing_C14Z = self._wing_panels.C14_VORING()
        self._wing_nx, self._wing_ny = self._wing_panels.get_dimensions()
        self._n_wing_panels = self._wing_nx * self._wing_ny

        self._AIC: np.ndarray = np.zeros((self._n_wing_panels, self._n_wing_panels))
        self._B: np.ndarray = np.zeros((self._n_wing_panels, self._n_wing_panels))
        self._RHS: np.ndarray = np.zeros((self._n_wing_panels, 1))

        self._compute_aerodynamic_influence()
        self._post = Post()

    def solve(self):
        inv_AIC = np.linalg.inv(self._AIC)
        d_wake = self._params.wake_dt * self._params.V_inf * np.array([1.0, 0.0, 0.0])

        for _ in range(self._params.wake_steps):
            self._update_RHS()

            Gammas = inv_AIC @ self._RHS
            w_ind = self._B @ Gammas

            Gammas = Gammas.reshape(self._wing_nx, -1)
            w_ind = w_ind.reshape(self._wing_nx, -1)

            self._wing_panels.update_Gammas(Gammas)
            self._wing_panels.update_w_ind(w_ind)
            
            self._post.compute_coefficients(self._wing_panels, self._params, Gammas, w_ind)
            self.print_results()
            print()

            self._wake_panels.wake_rollup(self._wing_C14X, self._wing_C14Y, self._wing_C14Z, Gammas, d_wake)

        return self._post.export_results()
    
    def print_results(self):
        self._post.print_results()

    def _compute_aerodynamic_influence(self):
        C14X, C14Y, C14Z = self._wing_panels.C14_VORING()
        control_points = self._wing_panels.control_points_VORING(self._n_wing_panels)
        normals = self._wing_panels.normal_VORING(self._n_wing_panels)

        V, V_star = Flows.VORING(C14X, C14Y, C14Z, control_points, np.ones((self._n_wing_panels, self._n_wing_panels, 1)), True)
        self._AIC[:] = np.sum(V * normals, axis=2)
        self._B[:] = np.sum(V_star * normals, axis=2)

    def _update_RHS(self):
        V_inf_vec = Solver._V_inf_vec(self._params)
        normals = self._wing_panels.normal_RHS()

        wake_influence = self._wake_panels.compute_wake_influence(self._wing_panels, self._wing_ny)
        self._RHS[:] = -np.sum((V_inf_vec + wake_influence) * normals, axis=1).reshape(-1, 1)

    @staticmethod
    def _V_inf_vec(params: Parameters):
        alfa_rad = np.deg2rad(params.alfa_deg)
        return params.V_inf * np.array([np.cos(alfa_rad), 0.0, np.sin(alfa_rad)])