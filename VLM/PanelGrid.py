from typing import NamedTuple
import numpy as np

class PanelGrid:
    class GridVector3(NamedTuple):
        X: np.ndarray
        Y: np.ndarray
        Z: np.ndarray

    def __init__(self, nx: int, ny: int, points: GridVector3 | None = None, wake_dx: float = 0.0):
        self._nx, self._ny = nx, ny

        self._C14X = np.zeros((nx + 1, ny + 1))
        self._C14Y = np.zeros((nx + 1, ny + 1))
        self._C14Z = np.zeros((nx + 1, ny + 1))

        self._control_pointX = np.zeros((nx, ny))
        self._control_pointY = np.zeros((nx, ny))
        self._control_pointZ = np.zeros((nx, ny))

        self._normalX = np.zeros((nx, ny))
        self._normalY = np.zeros((nx, ny))
        self._normalZ = np.zeros((nx, ny))

        self._Gammas = np.zeros((nx, ny))

        if points is not None:
            PX, PY, PZ = points
            self._C14X, self._C14Y, self._C14Z = self._compute_C14(PX, PY, PZ, wake_dx)
            self._control_pointX, self._control_pointY, self._control_pointZ = self._compute_control_points(PX, PY, PZ)
            self._normalX, self._normalY, self._normalZ = self._compute_normals(PX, PY, PZ)
            
    def _compute_C14(self, points_x: np.ndarray, points_y: np.ndarray, points_z: np.ndarray, wake_dx: float):
        diff_C14_x = 0.25 * np.diff(points_x, 1, 0)
        diff_C14_y = 0.25 * np.diff(points_y, 1, 0)
        diff_C14_z = 0.25 * np.diff(points_z, 1, 0)

        if wake_dx == 0.0:
            C14_x = points_x + np.vstack((diff_C14_x, diff_C14_x[-1, :])) 
            C14_y = points_y + np.vstack((diff_C14_y, diff_C14_y[-1, :])) 
            C14_z = points_z + np.vstack((diff_C14_z, diff_C14_z[-1, :])) 

        else:
            C14_x = points_x + np.vstack((diff_C14_x, np.ones_like(diff_C14_x[-1, :]) * wake_dx)) 
            C14_y = points_y + np.vstack((diff_C14_y, diff_C14_y[-1, :])) 
            C14_z = points_z + np.vstack((diff_C14_z, diff_C14_z[-1, :])) 

        return C14_x, C14_y, C14_z

    def _compute_control_points(self, points_x: np.ndarray, points_y: np.ndarray, points_z: np.ndarray):
        chord_diff_C34_x = 0.75 * np.diff(points_x, axis=0)
        chord_diff_C34_y = 0.75 * np.diff(points_y, axis=0)
        chord_diff_C34_z = 0.75 * np.diff(points_z, axis=0)

        control_chord_x = points_x[0:-1, :] + chord_diff_C34_x
        control_chord_y = points_y[0:-1, :] + chord_diff_C34_y
        control_chord_z = points_z[0:-1, :] + chord_diff_C34_z

        span_diff_C34_x = 0.5 * np.diff(control_chord_x, axis=1)
        span_diff_C34_y = 0.5 * np.diff(control_chord_y, axis=1)
        span_diff_C34_z = 0.5 * np.diff(control_chord_z, axis=1)

        CP_x = control_chord_x[:, 0:-1] + span_diff_C34_x
        CP_y = control_chord_y[:, 0:-1] + span_diff_C34_y
        CP_z = control_chord_z[:, 0:-1] + span_diff_C34_z

        return CP_x, CP_y, CP_z
    
    def _compute_normals(self, points_x: np.ndarray, points_y: np.ndarray, points_z: np.ndarray):
        normalX = np.zeros((self._nx, self._ny))
        normalY = np.zeros((self._nx, self._ny))
        normalZ = np.zeros((self._nx, self._ny))

        for i in range(self._nx):
            for j in range(self._ny):
                P1 = np.array([points_x[i, j], points_y[i, j], points_z[i, j]])
                P2 = np.array([points_x[i, j + 1], points_y[i, j + 1], points_z[i, j + 1]])
                P3 = np.array([points_x[i + 1, j + 1], points_y[i + 1, j + 1], points_z[i + 1, j + 1]])
                P4 = np.array([points_x[i + 1, j], points_y[i + 1, j], points_z[i + 1, j]])

                normal = np.cross(P3 - P1, P2 - P4)
                normal_norm = normal / np.linalg.norm(normal)
                normalX[i, j] = normal_norm[0]
                normalY[i, j] = normal_norm[1]
                normalZ[i, j] = normal_norm[2]

        return normalX, normalY, normalZ

    def get_C14(self):
        return PanelGrid.GridVector3(self._C14X, self._C14Y, self._C14Z)
    
    def get_control_points(self):
        return PanelGrid.GridVector3(self._control_pointX, self._control_pointY, self._control_pointZ)

    def get_dimensions(self):
        return self._nx, self._ny
    
    def get_Gammas(self):
        return self._Gammas
    
    @staticmethod
    def _C14_VORING_base(C14X_orig: np.ndarray, C14Y_orig: np.ndarray, C14Z_orig: np.ndarray):
        C14X1, C14Y1, C14Z1 = C14X_orig[:-1,:-1].reshape(-1, 1), C14Y_orig[:-1,:-1].reshape(-1, 1), C14Z_orig[:-1,:-1].reshape(-1, 1)
        C14X2, C14Y2, C14Z2 = C14X_orig[:-1,1:].reshape(-1, 1), C14Y_orig[:-1,1:].reshape(-1, 1), C14Z_orig[:-1,1:].reshape(-1, 1)
        C14X3, C14Y3, C14Z3 = C14X_orig[1:,1:].reshape(-1, 1), C14Y_orig[1:,1:].reshape(-1, 1), C14Z_orig[1:,1:].reshape(-1, 1)
        C14X4, C14Y4, C14Z4 = C14X_orig[1:,:-1].reshape(-1, 1), C14Y_orig[1:,:-1].reshape(-1, 1), C14Z_orig[1:,:-1].reshape(-1, 1)

        C14X = np.hstack((C14X1, C14X2, C14X3, C14X4))
        C14Y = np.hstack((C14Y1, C14Y2, C14Y3, C14Y4))
        C14Z = np.hstack((C14Z1, C14Z2, C14Z3, C14Z4))
        return C14X, C14Y, C14Z