from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt

class Mesh:
    class GridVector3(NamedTuple):
        X: np.ndarray
        Y: np.ndarray
        Z: np.ndarray

    def __init__(self, b: float, AR: float, nx: int, ny: int):
        self._nx = nx
        self._ny = ny

        self._corners = self._compute_corners(b, AR)
        self._quarter_chords = self._compute_quarter_chords()
        self._wake_corners = self._compute_wake_corners()
        self._collocation_points = self._compute_collocation_points()
        self._normals = self._compute_normals()

    def _compute_corners(self, b: float, AR: float):
        MAC = b / AR
        x = np.linspace(0, MAC, self._nx + 1)
        y = np.linspace(0, b / 2.0, self._ny + 1)

        corners_x, corners_y = np.meshgrid(x, y, indexing="ij")
        corners_z = np.zeros_like(corners_x)
        
        return Mesh.GridVector3(corners_x, corners_y, corners_z)

    def _compute_quarter_chords(self):
        corners_x, corners_y, corners_z = self._corners

        diff_C14_x = 0.25 * np.diff(corners_x, 1, 0)
        diff_C14_y = 0.25 * np.diff(corners_y, 1, 0)
        diff_C14_z = 0.25 * np.diff(corners_z, 1, 0)

        quarter_chords_x = corners_x + np.vstack((diff_C14_x, diff_C14_x[-1, :])) 
        quarter_chords_y = corners_y + np.vstack((diff_C14_y, diff_C14_y[-1, :])) 
        quarter_chords_z = corners_z + np.vstack((diff_C14_z, diff_C14_z[-1, :])) 

        return Mesh.GridVector3(quarter_chords_x, quarter_chords_y, quarter_chords_z)

    def _compute_wake_corners(self):
        quarter_chords_x, quarter_chords_y, quarter_chords_z = self._quarter_chords
        wake_corners_x = quarter_chords_x[-1, :] + 1e6
        wake_corners_y = quarter_chords_y[-1, :]
        wake_corners_z = quarter_chords_z[-1, :]

        return wake_corners_x, wake_corners_y, wake_corners_z

    def _compute_collocation_points(self):
        corners_x, corners_y, corners_z = self._corners

        chord_diff_C34_x = 0.75 * np.diff(corners_x, axis=0)
        chord_diff_C34_y = 0.75 * np.diff(corners_y, axis=0)
        chord_diff_C34_z = 0.75 * np.diff(corners_z, axis=0)

        collocation_chord_x = corners_x[0:-1, :] + chord_diff_C34_x
        collocation_chord_y = corners_y[0:-1, :] + chord_diff_C34_y
        collocation_chord_z = corners_z[0:-1, :] + chord_diff_C34_z

        span_diff_C34_x = 0.5 * np.diff(collocation_chord_x, axis=1)
        span_diff_C34_y = 0.5 * np.diff(collocation_chord_y, axis=1)
        span_diff_C34_z = 0.5 * np.diff(collocation_chord_z, axis=1)

        collocation_points_x = collocation_chord_x[:, 0:-1] + span_diff_C34_x
        collocation_points_y = collocation_chord_y[:, 0:-1] + span_diff_C34_y
        collocation_points_z = collocation_chord_z[:, 0:-1] + span_diff_C34_z

        return Mesh.GridVector3(collocation_points_x, collocation_points_y, collocation_points_z)
    
    def _compute_normals(self):
        corners_x, corners_y, corners_z = self._corners
        normals_x = np.zeros((self._nx, self._ny))
        normals_y = np.zeros((self._nx, self._ny))
        normals_z = np.zeros((self._nx, self._ny))

        for i in range(self._nx):
            for j in range(self._ny):
                P1 = np.array([corners_x[i, j], corners_y[i, j], corners_z[i, j]])
                P2 = np.array([corners_x[i, j + 1], corners_y[i, j + 1], corners_z[i, j + 1]])
                P3 = np.array([corners_x[i + 1, j + 1], corners_y[i + 1, j + 1], corners_z[i + 1, j + 1]])
                P4 = np.array([corners_x[i + 1, j], corners_y[i + 1, j], corners_z[i + 1, j]])

                normal = np.cross(P3 - P1, P2 - P4)
                normal_norm = normal / np.linalg.norm(normal)
                normals_x[i, j] = normal_norm[0]
                normals_y[i, j] = normal_norm[1]
                normals_z[i, j] = normal_norm[2]

        return Mesh.GridVector3(normals_x, normals_y, normals_z)
    
    def get_quarter_chords(self):
        return self._quarter_chords
    
    def get_wake_corners(self):
        return self._wake_corners

    def get_collocation_points(self):
        return self._collocation_points
    
    def get_normals(self):
        return self._normals
    
    def get_n_panels(self):
        return self._nx, self._ny
    
    def plot_mesh(self):
        corners_x, corners_y, corners_z = self._corners
        quarter_chords_x, quarter_chords_y, quarter_chords_z = self._quarter_chords
        collocation_points_x, collocation_points_y, collocation_points_z = self._collocation_points

        _, ax = plt.subplots(subplot_kw={"projection": "3d", "computed_zorder": False})
        ax.plot_surface(corners_x, corners_y, corners_z)
        ax.scatter(quarter_chords_x, quarter_chords_y, quarter_chords_z, c="r", marker="o", depthshade=False)
        ax.scatter(collocation_points_x, collocation_points_y, collocation_points_z, c="k", marker="D", depthshade=False)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        plt.show()