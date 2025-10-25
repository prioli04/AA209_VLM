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

    def C14_for_vectorized2(self):
        C14X1, C14Y1, C14Z1 = self._C14X[:-1,:-1].reshape(-1, 1), self._C14Y[:-1,:-1].reshape(-1, 1), self._C14Z[:-1,:-1].reshape(-1, 1)
        C14X2, C14Y2, C14Z2 = self._C14X[:-1,1:].reshape(-1, 1), self._C14Y[:-1,1:].reshape(-1, 1), self._C14Z[:-1,1:].reshape(-1, 1)
        C14X3, C14Y3, C14Z3 = self._C14X[1:,1:].reshape(-1, 1), self._C14Y[1:,1:].reshape(-1, 1), self._C14Z[1:,1:].reshape(-1, 1)
        C14X4, C14Y4, C14Z4 = self._C14X[1:,:-1].reshape(-1, 1), self._C14Y[1:,:-1].reshape(-1, 1), self._C14Z[1:,:-1].reshape(-1, 1)

        C14X = np.hstack((C14X1, C14X2, C14X3, C14X4))
        C14Y = np.hstack((C14Y1, C14Y2, C14Y3, C14Y4))
        C14Z = np.hstack((C14Z1, C14Z2, C14Z3, C14Z4))
        return C14X, C14Y, C14Z

    def control_points_for_vectorized(self):
        CPX = self._control_pointX.reshape(-1, 1)
        CPY = self._control_pointY.reshape(-1, 1)
        CPZ = self._control_pointZ.reshape(-1, 1)
        return np.hstack((CPX, CPY, CPZ))
    
    def control_points_for_vectorized2(self, n_tiles: int):
        n_panels = self._nx * self._ny

        CPX = np.tile(self._control_pointX.reshape(-1, 1)[np.newaxis].T, [1, n_tiles])
        CPY = np.tile(self._control_pointY.reshape(-1, 1)[np.newaxis].T, [1, n_tiles])
        CPZ = np.tile(self._control_pointZ.reshape(-1, 1)[np.newaxis].T, [1, n_tiles])

        control_points = np.zeros((n_panels, n_tiles, 3))
        control_points[:, :, 0], control_points[:, :, 1], control_points[:, :, 2] = CPX, CPY, CPZ
        return control_points
        
    def normal_for_vectorized(self):
        normalX = self._normalX.reshape(-1, 1)
        normalY = self._normalY.reshape(-1, 1)
        normalZ = self._normalZ.reshape(-1, 1)
        return np.hstack((normalX, normalY, normalZ))
    
    def normal_for_vectorized2(self, n_tiles: int):
        n_panels = self._nx * self._ny

        NX = np.tile(self._normalX.reshape(-1, 1)[np.newaxis].T, [1, n_tiles])
        NY = np.tile(self._normalY.reshape(-1, 1)[np.newaxis].T, [1, n_tiles])
        NZ = np.tile(self._normalZ.reshape(-1, 1)[np.newaxis].T, [1, n_tiles])

        normals = np.zeros((n_panels, n_tiles, 3))
        normals[:, :, 0], normals[:, :, 1], normals[:, :, 2] = NX, NY, NZ
        return normals
    
    def plot_mesh(self, ax: matplotlib.axes.Axes):
        ax.scatter(self._C14X, self._C14Y, self._C14Z, c="r", marker="o", depthshade=False)
        ax.scatter(self._control_pointX, self._control_pointY, self._control_pointZ, c="k", marker="D", depthshade=False)