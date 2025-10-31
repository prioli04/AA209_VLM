import numpy as np
import matplotlib.axes
from .PanelGrid import PanelGrid

class Wing(PanelGrid):
    def __init__(self, b: float, MAC: float, nx: int, ny: int, wake_dx: float):
        points = self._compute_points(b, MAC, nx, ny)
        super().__init__(nx, ny, points, wake_dx=wake_dx)
        self._w_ind_trefftz = np.zeros(ny)

    def _compute_points(self, b: float, MAC: float, nx: int, ny: int):
        x = np.linspace(0, MAC, nx + 1)
        y = np.linspace(0, b / 2.0, ny + 1)

        corners_x, corners_y = np.meshgrid(x, y, indexing="ij")
        corners_z = np.zeros_like(corners_x)
        
        return super().GridVector3(corners_x, corners_y, corners_z)
    
    def update_w_ind_trefftz(self, w_ind: np.ndarray):
        self._w_ind_trefftz[:] = w_ind

    def update_Gammas(self, Gammas: np.ndarray):
        self._Gammas[:] = Gammas

    def extract_TE_points(self):
        return super().GridVector3(self._C14X[-1, :], self._C14Y[-1, :], self._C14Z[-1, :])

    def C14_VORING(self):
        return super()._C14_VORING_base(self._C14X, self._C14Y, self._C14Z)
    
    def C14_TREFFTZ(self):
        C14X = self._C14X[-1, :].reshape(-1, 1)
        C14Y = self._C14Y[-1, :].reshape(-1, 1)
        C14Z = self._C14Z[-1, :].reshape(-1, 1)
        return np.hstack((C14X, C14Y, C14Z))
    
    def control_points_VORING(self, n_tiles: int):
        n_points = self._nx * self._ny

        CPX = np.tile(self._control_pointX.reshape(-1, 1), [1, n_tiles])
        CPY = np.tile(self._control_pointY.reshape(-1, 1), [1, n_tiles])
        CPZ = np.tile(self._control_pointZ.reshape(-1, 1), [1, n_tiles])

        control_points = np.zeros((n_points, n_tiles, 3))
        control_points[:, :, 0], control_points[:, :, 1], control_points[:, :, 2] = CPX, CPY, CPZ
        return control_points
    
    def control_points_TREFFTZ(self):
        CPX = self._control_pointX[-1, :].reshape(-1, 1)
        CPY = self._control_pointY[-1, :].reshape(-1, 1)
        CPZ = self._control_pointZ[-1, :].reshape(-1, 1)
        return np.hstack((CPX, CPY, CPZ))

    def normal_RHS(self):
        normalX = self._normalX.reshape(-1, 1)
        normalY = self._normalY.reshape(-1, 1)
        normalZ = self._normalZ.reshape(-1, 1)
        return np.hstack((normalX, normalY, normalZ))
    
    def normal_VORING(self, n_tiles: int):
        n_panels = self._nx * self._ny

        NX = np.tile(self._normalX.reshape(-1, 1), [1, n_tiles])
        NY = np.tile(self._normalY.reshape(-1, 1), [1, n_tiles])
        NZ = np.tile(self._normalZ.reshape(-1, 1), [1, n_tiles])

        normals = np.zeros((n_panels, n_tiles, 3))
        normals[:, :, 0], normals[:, :, 1], normals[:, :, 2] = NX, NY, NZ
        return normals
    
    def normal_TREFFTZ(self):
        normalX = self._normalX[-1, :].reshape(-1, 1)
        normalY = self._normalY[-1, :].reshape(-1, 1)
        normalZ = self._normalZ[-1, :].reshape(-1, 1)
        return np.hstack((normalX, normalY, normalZ))

    def plot_mesh(self, ax: matplotlib.axes.Axes):
        ax.scatter(self._C14X, self._C14Y, self._C14Z, c="r", marker="o", depthshade=False)
        ax.scatter(self._control_pointX, self._control_pointY, self._control_pointZ, c="k", marker="D", depthshade=False)