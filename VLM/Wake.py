import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt
from .PanelGrid import PanelGrid

class Wake(PanelGrid):
    def __init__(self, nt: int, ny: int):
        super().__init__(nt, ny)
        self._it = 0

    def _update_wake(self):
        self._compute_control_points(self._C14X, self._C14Y, self._C14Z)
        self._compute_normals(self._C14X, self._C14Y, self._C14Z)

    def _plot_mesh(self, ax: matplotlib.axes.Axes):
       return ax.plot_wireframe(self._C14X[:self._it + 1, :], self._C14Y[:self._it + 1, :], self._C14Z[:self._it + 1, :])

    def update_wake_plot(self, ax: matplotlib.axes.Axes):
        wake_lines = None
       
        if self._it > 0:
            wake_lines = self._plot_mesh(ax)
            ax.set_aspect("equal")
            plt.pause(0.01)

        return wake_lines
    
    def add_TE(self, TE_points: PanelGrid.GridVector3):
        self._C14X[0, :] = TE_points.X
        self._C14Y[0, :] = TE_points.Y
        self._C14Z[0, :] = TE_points.Z

    def step_wake(self, TE_points: PanelGrid.GridVector3, d_wake: np.ndarray):
        self._C14X[:self._it + 1, :] += d_wake[0]
        self._C14Y[:self._it + 1, :] += d_wake[1]
        self._C14Z[:self._it + 1, :] += d_wake[2]
        
        self._C14X[1:, :] = self._C14X[:-1, :]
        self._C14Y[1:, :] = self._C14Y[:-1, :]
        self._C14Z[1:, :] = self._C14Z[:-1, :]

        self._C14X[0, :] = TE_points.X
        self._C14Y[0, :] = TE_points.Y
        self._C14Z[0, :] = TE_points.Z

        self._update_wake()

    def update_Gammas(self, TE_Gammas: np.ndarray):
        self._Gammas[1:, :] = self._Gammas[:-1, :]
        self._Gammas[0, :] = TE_Gammas

    def advance_it(self):
        self._it += 1

    def offset_wake(self, offset_map: PanelGrid.GridVector3):
        self._C14X += offset_map.X
        self._C14Y += offset_map.Y
        self._C14Z += offset_map.Z
        self._update_wake()

    def C14_VORING(self):
        if self._it < 1:
            raise ValueError("C14_as_controls_points should not be called for 'self._it < 1'.")
        
        C14X_cut = self._C14X[:self._it + 1, :]
        C14Y_cut = self._C14Y[:self._it + 1, :]
        C14Z_cut = self._C14Z[:self._it + 1, :]

        return super()._C14_VORING_base(C14X_cut, C14Y_cut, C14Z_cut)
    
    def C14_as_control_points(self, n_tiles: int):
        if self._it < 1:
            raise ValueError("C14_as_controls_points should not be called for 'self._it < 1'.")

        n_points = (self._it + 1) * (self._ny + 1)

        CPX = np.tile(self._C14X[1:self._it + 2, :].reshape(-1, 1), [1, n_tiles])
        CPY = np.tile(self._C14Y[1:self._it + 2, :].reshape(-1, 1), [1, n_tiles])
        CPZ = np.tile(self._C14Z[1:self._it + 2, :].reshape(-1, 1), [1, n_tiles])

        control_points = np.zeros((n_points, n_tiles, 3))
        control_points[:, :, 0], control_points[:, :, 1], control_points[:, :, 2] = CPX, CPY, CPZ

        return control_points
    
    def get_Gammas(self):
        return super()._get_Gammas_base(self._Gammas[:self._it, :])