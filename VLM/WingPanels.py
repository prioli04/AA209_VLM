from .PanelGrid import PanelGrid
from .WingGeometry import WingGeometry
from mpl_toolkits.mplot3d.axes3d import Axes3D # type: ignore[import-untyped]
from typing import List

import numpy as np

class WingPanels(PanelGrid):
    def __init__(self, wing_geometry: WingGeometry, Z: float, wake_dx: float, sym: bool):
        self._sym = sym
        
        self._b = wing_geometry.b
        self._S = wing_geometry.S
        self._root_chord = wing_geometry.root_chord
        self._MAC = wing_geometry.MAC

        self._patches = wing_geometry.get_patches()
        self._airfoils = wing_geometry.get_airfoils()
        self._airfoil_ids = self._set_airfoil_ids()

        self._points = self._compute_points(Z)
        self._chords = self._compute_chords()
        nx, ny = self._points.X.shape[0] - 1, self._points.X.shape[1] - 1

        super().__init__(nx, ny, self._points, wake_dx=wake_dx)

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
    
    def _compute_chords(self):
        chords_wing = np.empty(0)

        for patch in self._patches:
            chords_patch = patch.compute_chords(self._root_chord)
            chords_wing = np.hstack((chords_wing, chords_patch)) if chords_wing.size != 0 else chords_patch

        if not self._sym:
            chords_wing = np.hstack([np.flip(chords_wing), chords_wing])

        return chords_wing
    
    def _set_airfoil_ids(self) -> List[int]:
        foil_ids = []

        for patch in self._patches:
            foil_ids += [patch._root_foil_id] * patch.ny

        if not self._sym:
            foil_ids = foil_ids[::-1] + foil_ids

        return foil_ids

    def get_chords(self):
        return self._chords

    def get_airfoils(self):
        return self._airfoils, self._airfoil_ids

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

    def control_points_bound_vortex(self, n_tiles: int):
        n_points = self._nx * self._ny

        CPX = 0.5 * (self._C14X[:-1, :-1] + self._C14X[:-1, 1:])
        CPY = 0.5 * (self._C14Y[:-1, :-1] + self._C14Y[:-1, 1:])
        CPZ = 0.5 * (self._C14Z[:-1, :-1] + self._C14Z[:-1, 1:])

        CPX = np.tile(CPX.reshape(-1, 1), [1, n_tiles])
        CPY = np.tile(CPY.reshape(-1, 1), [1, n_tiles])
        CPZ = np.tile(CPZ.reshape(-1, 1), [1, n_tiles])

        control_points = np.zeros((n_points, n_tiles, 3))
        control_points[:, :, 0], control_points[:, :, 1], control_points[:, :, 2] = CPX, CPY, CPZ
        return control_points

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
        normalY = self._normalY[-1, :].reshape(-1, 1)
        normalZ = self._normalZ[-1, :].reshape(-1, 1)
        normals = np.hstack((normalY, normalZ))

        return normals / np.linalg.norm(normals, axis=1).reshape(-1, 1)

    def plot_mesh(self, ax: Axes3D):
        if self._sym:
            x_sym = np.hstack([np.flip(self._points.X, axis=1), self._points.X[:, 1:]])
            y_sym = np.hstack([-np.flip(self._points.Y, axis=1), self._points.Y[:, 1:]])
            z_sym = np.hstack([np.flip(self._points.Z, axis=1), self._points.Z[:, 1:]])
            ax.plot_surface(x_sym, y_sym, z_sym)

        else:
            ax.plot_surface(self._points.X, self._points.Y, self._points.Z)