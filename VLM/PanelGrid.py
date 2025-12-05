from typing import NamedTuple
import numpy as np

# Object that holds a mesh and perform transformations on it
class PanelGrid:
    # Object for packing x, y, z values of a mesh
    class GridVector3(NamedTuple):
        X: np.ndarray
        Y: np.ndarray
        Z: np.ndarray

    def __init__(self, nx: int, ny: int, points: GridVector3 | None = None, wake_dx: float = 0.0):
        # nx: number of panels chordwise
        # ny: number of panels spanwise
        # points: grid of points for the mesh 
        # wake_dx: parameter that adjusts where the ring vortices end and the wake starts
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

        # Compute the information needed for the solver from the points provided
        if points is not None:
            PX, PY, PZ = points
            self._C14X, self._C14Y, self._C14Z = self._compute_C14(PX, PY, PZ, wake_dx)
            self._control_pointX, self._control_pointY, self._control_pointZ = self._compute_control_points(PX, PY, PZ)
            self._normalX, self._normalY, self._normalZ = self._compute_normals(PX, PY, PZ)
    
    # Computes the corners of the ring vortices (located at the panels 1/4 chords)
    def _compute_C14(self, points_x: np.ndarray, points_y: np.ndarray, points_z: np.ndarray, wake_dx: float):
        # Get 1/4 of the chordwise change in coordinates
        diff_C14_x = 0.25 * np.diff(points_x, 1, 0)
        diff_C14_y = 0.25 * np.diff(points_y, 1, 0)
        diff_C14_z = 0.25 * np.diff(points_z, 1, 0)

        if wake_dx == 0.0:
            # Simply add this difference to get the 1/4 chord points (the last value is repeated for the last row of panels)
            C14_x = points_x + np.vstack((diff_C14_x, diff_C14_x[-1, :])) 
            C14_y = points_y + np.vstack((diff_C14_y, diff_C14_y[-1, :])) 
            C14_z = points_z + np.vstack((diff_C14_z, diff_C14_z[-1, :])) 

        else:
            # If wake_dx is provided, adjust the last row x coordinate using this parameter
            C14_x = points_x + np.vstack((diff_C14_x, np.ones_like(diff_C14_x[-1, :]) * wake_dx)) 
            C14_y = points_y + np.vstack((diff_C14_y, diff_C14_y[-1, :])) 
            C14_z = points_z + np.vstack((diff_C14_z, diff_C14_z[-1, :])) 

        return C14_x, C14_y, C14_z

    # Computes the control points of each panel (located at 3/4 chord and midspan of the panel)
    def _compute_control_points(self, points_x: np.ndarray, points_y: np.ndarray, points_z: np.ndarray):
        # Get 3/4 of the chordwise change in coordinates
        chord_diff_C34_x = 0.75 * np.diff(points_x, axis=0)
        chord_diff_C34_y = 0.75 * np.diff(points_y, axis=0)
        chord_diff_C34_z = 0.75 * np.diff(points_z, axis=0)

        # Simply add this difference to get the 3/4 chord points
        # Trailing edge coordinates are not used because there will be 1 less control point than chordwise points
        control_chord_x = points_x[0:-1, :] + chord_diff_C34_x
        control_chord_y = points_y[0:-1, :] + chord_diff_C34_y
        control_chord_z = points_z[0:-1, :] + chord_diff_C34_z

        # Get 1/2 of the spanwise change in coordinates of the 3/4 chord position (not the original mesh points)
        span_diff_C34_x = 0.5 * np.diff(control_chord_x, axis=1)
        span_diff_C34_y = 0.5 * np.diff(control_chord_y, axis=1)
        span_diff_C34_z = 0.5 * np.diff(control_chord_z, axis=1)

        # Simply add this difference to get the control points
        # Wing tip coordinates are not used because there will be 1 less control point than spanwise points
        CP_x = control_chord_x[:, 0:-1] + span_diff_C34_x
        CP_y = control_chord_y[:, 0:-1] + span_diff_C34_y
        CP_z = control_chord_z[:, 0:-1] + span_diff_C34_z

        return CP_x, CP_y, CP_z
    
    # Computes the normal vector of each panel
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

                normal = np.cross(P3 - P1, P2 - P4) # Equation 12.20 from (Katz, Plotkin)
                normal_norm = normal / np.linalg.norm(normal) # Normalize
                normalX[i, j] = normal_norm[0]
                normalY[i, j] = normal_norm[1]
                normalZ[i, j] = normal_norm[2]

        return normalX, normalY, normalZ

    # Getter for the ring vortices' corners
    def get_C14(self):
        return PanelGrid.GridVector3(self._C14X, self._C14Y, self._C14Z)
    
    # Getter for the control points
    def get_control_points(self):
        return PanelGrid.GridVector3(self._control_pointX, self._control_pointY, self._control_pointZ)

    # Getter for the mesh dimensions
    def get_dimensions(self):
        return self._nx, self._ny
    
    # Prepare ring vortices' corners values for the VORING routine
    @staticmethod
    def _C14_VORING_base(C14X_orig: np.ndarray, C14Y_orig: np.ndarray, C14Z_orig: np.ndarray):
        C14X1, C14Y1, C14Z1 = C14X_orig[:-1,:-1].reshape(-1, 1), C14Y_orig[:-1,:-1].reshape(-1, 1), C14Z_orig[:-1,:-1].reshape(-1, 1)
        C14X2, C14Y2, C14Z2 = C14X_orig[:-1,1:].reshape(-1, 1), C14Y_orig[:-1,1:].reshape(-1, 1), C14Z_orig[:-1,1:].reshape(-1, 1)
        C14X3, C14Y3, C14Z3 = C14X_orig[1:,1:].reshape(-1, 1), C14Y_orig[1:,1:].reshape(-1, 1), C14Z_orig[1:,1:].reshape(-1, 1)
        C14X4, C14Y4, C14Z4 = C14X_orig[1:,:-1].reshape(-1, 1), C14Y_orig[1:,:-1].reshape(-1, 1), C14Z_orig[1:,:-1].reshape(-1, 1)

        # C14X, C14Y, C14Z -> (n_vortices, 4) matrix. Each row is a vortex ring and each column is a corner of said ring
        
        # Viewed from above the wing:
        # C14X1, C14Y1, C14Z1 -> Rings' forward-left corner
        # C14X2, C14Y2, C14Z2 -> Rings' forward-right corner
        # C14X3, C14Y3, C14Z3 -> Rings' aft-right corner
        # C14X4, C14Y4, C14Z4 -> Rings' aft-left corner
        C14X = np.hstack((C14X1, C14X2, C14X3, C14X4))
        C14Y = np.hstack((C14Y1, C14Y2, C14Y3, C14Y4))
        C14Z = np.hstack((C14Z1, C14Z2, C14Z3, C14Z4))
        return C14X, C14Y, C14Z