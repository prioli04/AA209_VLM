import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt
from .Flows import Flows
from .PanelGrid import PanelGrid
from .Wing import Wing

class TimeSteppingWake(PanelGrid):
    def __init__(self, nt: int, ny: int, dt: float, wing_TE_points: PanelGrid.GridVector3, plot_ax: matplotlib.axes.Axes):
        super().__init__(nt, ny)
        self._it = 0
        self._dt = dt
        self._wing_TE_points = wing_TE_points
        self._plot_ax = plot_ax
        self._wake_lines = None
        self._add_TE()

    def _C14_VORING(self):
        if self._it < 1:
            raise ValueError("C14_as_controls_points should not be called for 'self._it < 1'.")
        
        C14X_cut = self._C14X[:self._it + 1, :]
        C14Y_cut = self._C14Y[:self._it + 1, :]
        C14Z_cut = self._C14Z[:self._it + 1, :]

        return super()._C14_VORING_base(C14X_cut, C14Y_cut, C14Z_cut)

    def _add_TE(self):
        self._C14X[0, :] = self._wing_TE_points.X
        self._C14Y[0, :] = self._wing_TE_points.Y
        self._C14Z[0, :] = self._wing_TE_points.Z

    def _step_wake(self, d_wake: np.ndarray):
        self._C14X[:self._it + 1, :] += d_wake[0]
        self._C14Y[:self._it + 1, :] += d_wake[1]
        self._C14Z[:self._it + 1, :] += d_wake[2]
        
        self._C14X[1:, :] = self._C14X[:-1, :]
        self._C14Y[1:, :] = self._C14Y[:-1, :]
        self._C14Z[1:, :] = self._C14Z[:-1, :]

        self._add_TE()
        self._compute_control_points(self._C14X, self._C14Y, self._C14Z)
        self._compute_normals(self._C14X, self._C14Y, self._C14Z)

    def _update_Gammas(self, TE_Gammas: np.ndarray):
        self._Gammas[1:, :] = self._Gammas[:-1, :]
        self._Gammas[0, :] = TE_Gammas

    def _C14_as_control_points(self, n_tiles: int):
        if self._it < 1:
            raise ValueError("C14_as_controls_points should not be called for 'self._it < 1'.")

        n_points = (self._it + 1) * (self._ny + 1)

        CPX = np.tile(self._C14X[1:self._it + 2, :].reshape(-1, 1), [1, n_tiles])
        CPY = np.tile(self._C14Y[1:self._it + 2, :].reshape(-1, 1), [1, n_tiles])
        CPZ = np.tile(self._C14Z[1:self._it + 2, :].reshape(-1, 1), [1, n_tiles])

        control_points = np.zeros((n_points, n_tiles, 3))
        control_points[:, :, 0], control_points[:, :, 1], control_points[:, :, 2] = CPX, CPY, CPZ

        return control_points

    def _build_offset_map(self, wing_C14X: np.ndarray, wing_C14Y: np.ndarray, wing_C14Z: np.ndarray, wing_Gammas: np.ndarray):
        wake_C14X, wake_C14Y, wake_C14Z = self._C14_VORING()

        wake_C14_as_CP_wing = self._C14_as_control_points(wing_C14X.shape[0])
        wake_C14_as_CP_wake = self._C14_as_control_points(wake_C14X.shape[0])
        
        wing_Gammas = np.tile(wing_Gammas.reshape(1, -1), [wake_C14_as_CP_wing.shape[0], 1])[:, :, np.newaxis]
        wake_Gammas = np.tile(self._Gammas[:self._it, :].reshape(1, -1), [wake_C14_as_CP_wake.shape[0], 1])[:, :, np.newaxis]

        offset_map_X = np.zeros((self._nx + 1, self._ny + 1))
        offset_map_Y = np.zeros((self._nx + 1, self._ny + 1))
        offset_map_Z = np.zeros((self._nx + 1, self._ny + 1))

        V = np.zeros((wake_C14_as_CP_wake.shape[0], 3))

        dV_wing, _ = Flows.VORING(wing_C14X, wing_C14Y, wing_C14Z, wake_C14_as_CP_wing, wing_Gammas, True)
        V += np.sum(dV_wing, axis=1)

        dV_wake, _ = Flows.VORING(wake_C14X, wake_C14Y, wake_C14Z, wake_C14_as_CP_wake, wake_Gammas, True)
        V += np.sum(dV_wake, axis=1)
            
        offset_point = self._dt * V
        offset_map_X[1:self._it + 2, :] = offset_point[:, 0].reshape(-1, self._ny + 1)
        offset_map_Y[1:self._it + 2, :] = offset_point[:, 1].reshape(-1, self._ny + 1)
        offset_map_Z[1:self._it + 2, :] = offset_point[:, 2].reshape(-1, self._ny + 1)

        return PanelGrid.GridVector3(offset_map_X, offset_map_Y, offset_map_Z)
    
    def _offset_wake(self, offset_map: PanelGrid.GridVector3):
        self._C14X += offset_map.X
        self._C14Y += offset_map.Y
        self._C14Z += offset_map.Z

        self._compute_control_points(self._C14X, self._C14Y, self._C14Z)
        self._compute_normals(self._C14X, self._C14Y, self._C14Z)

    def _update_wake_plot(self):
        if self._wake_lines is not None:
            self._wake_lines.remove()   

        if self._it > 0 and self._plot_ax is not None:
            self._wake_lines = self._plot_ax.plot_wireframe(self._C14X[:self._it + 1, :], self._C14Y[:self._it + 1, :], self._C14Z[:self._it + 1, :])
            self._plot_ax.set_aspect("equal")
            plt.pause(0.01)

    def wake_rollup(self, wing_C14X: np.ndarray, wing_C14Y: np.ndarray, wing_C14Z: np.ndarray, wing_Gammas: np.ndarray, d_wake: float):
        self._step_wake(d_wake)
        self._update_Gammas(wing_Gammas[-1, :])

        if self._it != 0:
            offset_map = self._build_offset_map(wing_C14X, wing_C14Y, wing_C14Z, wing_Gammas)
            self._offset_wake(offset_map)

        self._it += 1
        self._update_wake_plot()

    def compute_wake_influence(self, wing_panels: Wing, wing_ny: int):
        if self._it == 0:
            return 0.0

        wake_C14X, wake_C14Y, wake_C14Z = self._C14_VORING()
        control_points = wing_panels.control_points_VORING(wake_C14X.shape[0])
        wake_Gammas = np.tile(self._Gammas[:self._it, :].reshape(1, -1), [control_points.shape[0], 1])[:, :, np.newaxis]

        dV_w, _ = Flows.VORING(wake_C14X, wake_C14Y, wake_C14Z, control_points, wake_Gammas, True)
        V_w = np.sum(dV_w, axis=1)

        self._w_wake = V_w[:, 2].reshape(-1, wing_ny)
        return V_w