from .PanelGrid import PanelGrid
from .WingGeometry import WingGeometry
from mpl_toolkits.mplot3d.axes3d import Axes3D # type: ignore[import-untyped]

import numpy as np

class WingPanels(PanelGrid):
    def __init__(self, wing_geometry: WingGeometry, Z: float, wake_dx: float, sym: bool, alfa_deg: float, beta_deg: float):
        self._patches = wing_geometry.get_patches()

        if sym and beta_deg != 0.0:
            raise ValueError("Sideslip angle is different than 0° and the symmetry flag is activated.")

        self._sym = sym
        self._alfa_rad = np.deg2rad(alfa_deg)
        self._beta_rad = np.deg2rad(beta_deg)

        self._b = wing_geometry.b
        self._S = wing_geometry.S
        self._root_chord = wing_geometry.root_chord
        self._MAC = wing_geometry.MAC

        self._points = self._compute_points(Z)
        nx, ny = self._points.X.shape[0] - 1, self._points.X.shape[1] - 1

        super().__init__(nx, ny, self._points, wake_dx=wake_dx)
        self._w_ind_trefftz = np.zeros(ny)

    def _compute_points(self, Z: float):
        corners_x = np.empty(0)
        corners_y = np.empty(0)
        corners_z = np.empty(0)

        for patch in self._patches:
            patch_x, patch_y, patch_z = patch.compute_points(self._root_chord, self._b / 2.0, Z)
            corners_x = np.hstack((corners_x[:, :-1], patch_x)) if corners_x.size != 0 else patch_x
            corners_y = np.hstack((corners_y[:, :-1], patch_y)) if corners_y.size != 0 else patch_y
            corners_z = np.hstack((corners_z[:, :-1], patch_z)) if corners_z.size != 0 else patch_z

        if not self._sym:
            corners_x = np.hstack([np.flip(corners_x, axis=1), corners_x[:,1:]])
            corners_y = np.hstack([-np.flip(corners_y, axis=1), corners_y[:,1:]])
            corners_z = np.hstack([np.flip(corners_z, axis=1), corners_z[:,1:]])

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

    def plot_mesh(self, ax: Axes3D):
        if self._sym:
            x_sym = np.hstack([np.flip(self._points.X, axis=1), self._points.X[:, 1:]])
            y_sym = np.hstack([-np.flip(self._points.Y, axis=1), self._points.Y[:, 1:]])
            z_sym = np.hstack([np.flip(self._points.Z, axis=1), self._points.Z[:, 1:]])
            ax.plot_surface(x_sym, y_sym, z_sym)

        else:
            ax.plot_surface(self._points.X, self._points.Y, self._points.Z)