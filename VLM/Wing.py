import numpy as np
import matplotlib.axes
from .PanelGrid import PanelGrid

class Wing(PanelGrid):
    def __init__(self, b: float, AR: float, nx: int, ny: int, wake_dx: float):
        points = self._compute_points(b, AR, nx, ny)
        super().__init__(nx, ny, points, wake_dx=wake_dx)
        self._w_ind = np.zeros((nx, ny))

    def _compute_points(self, b: float, AR: float, nx: int, ny: int):
        MAC = b / AR
        x = np.linspace(0, MAC, nx + 1)
        y = np.linspace(0, b / 2.0, ny + 1)

        corners_x, corners_y = np.meshgrid(x, y, indexing="ij")
        corners_z = np.zeros_like(corners_x)
        
        return super().GridVector3(corners_x, corners_y, corners_z)
    
    def update_w_ind(self, w_ind: np.ndarray):
        self._w_ind[:] = w_ind

    def update_Gammas(self, Gammas: np.ndarray):
        self._Gammas[:] = Gammas

    def extract_TE_points(self):
        return super().GridVector3(self._C14X[-1, :], self._C14Y[-1, :], self._C14Z[-1, :])
    
    def plot_mesh(self, ax: matplotlib.axes.Axes):
        # points_x, points_y, points_z = self._points
        # ax.plot_surface(points_x, points_y, points_z)
        ax.scatter(self._C14X, self._C14Y, self._C14Z, c="r", marker="o", depthshade=False)
        ax.scatter(self._control_pointX, self._control_pointY, self._control_pointZ, c="k", marker="D", depthshade=False)