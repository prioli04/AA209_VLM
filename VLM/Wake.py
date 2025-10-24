import numpy as np
import matplotlib.axes
from .PanelGrid import PanelGrid

class Wake(PanelGrid):
    def __init__(self, nt: int, ny: int):
        super().__init__(nt, ny)

    def _update_wake(self):
        self._compute_control_points(self._C14X, self._C14Y, self._C14Z)
        self._compute_normals(self._C14X, self._C14Y, self._C14Z)

    def plot_mesh(self, ax: matplotlib.axes.Axes):
        ax.plot_wireframe(self._C14X, self._C14Y, self._C14Z)
    
    def add_TE(self, TE_points: PanelGrid.GridVector3):
        self._C14X[0, :] = TE_points.X
        self._C14Y[0, :] = TE_points.Y
        self._C14Z[0, :] = TE_points.Z

    def step_wake(self, it: int, TE_points: PanelGrid.GridVector3, d_wake: np.ndarray):
        self._C14X[:it + 1, :] += d_wake[0]
        self._C14Y[:it + 1, :] += d_wake[1]
        self._C14Z[:it + 1, :] += d_wake[2]
        
        self._C14X[1:, :] = self._C14X[:-1, :]
        self._C14Y[1:, :] = self._C14Y[:-1, :]
        self._C14Z[1:, :] = self._C14Z[:-1, :]

        self._C14X[0, :] = TE_points.X
        self._C14Y[0, :] = TE_points.Y
        self._C14Z[0, :] = TE_points.Z

        # if it != 0:
        self._update_wake()

    def update_Gammas(self, TE_Gammas: np.ndarray):
        self._Gammas[1:, :] = self._Gammas[:-1, :]
        self._Gammas[0, :] = TE_Gammas

    def offset_wake(self, offset_map: PanelGrid.GridVector3):
        self._C14X += offset_map.X
        self._C14Y += offset_map.Y
        self._C14Z += offset_map.Z
        self._update_wake()