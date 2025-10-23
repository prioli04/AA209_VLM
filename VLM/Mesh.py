from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt

class Mesh:
    class GridVector3(NamedTuple):
        X: np.ndarray
        Y: np.ndarray
        Z: np.ndarray

    def __init__(
            self, 
            points: GridVector3, 
            quarter_chords: GridVector3 | None = None, 
            collocation_points: GridVector3 | None = None, 
            normals: GridVector3 | None = None):
        
        self._nx, self._ny = points.X.shape
        self._nx -= 1
        self._ny -= 1

        self._points = points
        self._quarter_chords = self._compute_quarter_chords() if quarter_chords is None else quarter_chords
        self._collocation_points = self._compute_collocation_points() if collocation_points is None else collocation_points
        self._normals = self._compute_normals() if normals is None else normals

    def _compute_quarter_chords(self):
        points_x, points_y, points_z = self._points

        diff_C14_x = 0.25 * np.diff(points_x, 1, 0)
        diff_C14_y = 0.25 * np.diff(points_y, 1, 0)
        diff_C14_z = 0.25 * np.diff(points_z, 1, 0)

        quarter_chords_x = points_x + np.vstack((diff_C14_x, diff_C14_x[-1, :])) 
        quarter_chords_y = points_y + np.vstack((diff_C14_y, diff_C14_y[-1, :])) 
        quarter_chords_z = points_z + np.vstack((diff_C14_z, diff_C14_z[-1, :])) 

        return Mesh.GridVector3(quarter_chords_x, quarter_chords_y, quarter_chords_z)

    def _compute_collocation_points(self):
        points_x, points_y, points_z = self._points

        chord_diff_C34_x = 0.75 * np.diff(points_x, axis=0)
        chord_diff_C34_y = 0.75 * np.diff(points_y, axis=0)
        chord_diff_C34_z = 0.75 * np.diff(points_z, axis=0)

        collocation_chord_x = points_x[0:-1, :] + chord_diff_C34_x
        collocation_chord_y = points_y[0:-1, :] + chord_diff_C34_y
        collocation_chord_z = points_z[0:-1, :] + chord_diff_C34_z

        span_diff_C34_x = 0.5 * np.diff(collocation_chord_x, axis=1)
        span_diff_C34_y = 0.5 * np.diff(collocation_chord_y, axis=1)
        span_diff_C34_z = 0.5 * np.diff(collocation_chord_z, axis=1)

        collocation_points_x = collocation_chord_x[:, 0:-1] + span_diff_C34_x
        collocation_points_y = collocation_chord_y[:, 0:-1] + span_diff_C34_y
        collocation_points_z = collocation_chord_z[:, 0:-1] + span_diff_C34_z

        return Mesh.GridVector3(collocation_points_x, collocation_points_y, collocation_points_z)
    
    def _compute_normals(self):
        points_x, points_y, points_z = self._points
        normals_x = np.zeros((self._nx, self._ny))
        normals_y = np.zeros((self._nx, self._ny))
        normals_z = np.zeros((self._nx, self._ny))

        for i in range(self._nx):
            for j in range(self._ny):
                P1 = np.array([points_x[i, j], points_y[i, j], points_z[i, j]])
                P2 = np.array([points_x[i, j + 1], points_y[i, j + 1], points_z[i, j + 1]])
                P3 = np.array([points_x[i + 1, j + 1], points_y[i + 1, j + 1], points_z[i + 1, j + 1]])
                P4 = np.array([points_x[i + 1, j], points_y[i + 1, j], points_z[i + 1, j]])

                normal = np.cross(P3 - P1, P2 - P4)
                normal_norm = normal / np.linalg.norm(normal)
                normals_x[i, j] = normal_norm[0]
                normals_y[i, j] = normal_norm[1]
                normals_z[i, j] = normal_norm[2]

        return Mesh.GridVector3(normals_x, normals_y, normals_z)
    
    def get_points(self):
        return self._points

    def get_quarter_chords(self):
        return self._quarter_chords
    
    def get_collocation_points(self):
        return self._collocation_points
    
    def get_normals(self):
        return self._normals
    
    def get_n_panels(self):
        return self._nx, self._ny
    
    def plot_mesh(self):
        points_x, points_y, points_z = self._points
        quarter_chords_x, quarter_chords_y, quarter_chords_z = self._quarter_chords
        collocation_points_x, collocation_points_y, collocation_points_z = self._collocation_points

        _, ax = plt.subplots(subplot_kw={"projection": "3d", "computed_zorder": False})
        ax.plot_surface(points_x, points_y, points_z)
        ax.scatter(quarter_chords_x, quarter_chords_y, quarter_chords_z, c="r", marker="o", depthshade=False)
        ax.scatter(collocation_points_x, collocation_points_y, collocation_points_z, c="k", marker="D", depthshade=False)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        plt.show()

    @staticmethod
    def combine_meshes(mesh_1, mesh_2):
        points = Mesh.combine_grid_vector3(mesh_1.get_points(), mesh_2.get_points()) 
        quarter_chords = Mesh.combine_grid_vector3(mesh_1.get_quarter_chords(), mesh_2.get_quarter_chords()) 
        collocation_points = Mesh.combine_grid_vector3(mesh_1.get_collocation_points(), mesh_2.get_collocation_points()) 
        normals = Mesh.combine_grid_vector3(mesh_1.get_normals(), mesh_2.get_normals()) 
        
        return Mesh(points, quarter_chords, collocation_points, normals)

    @staticmethod
    def combine_grid_vector3(grid_1: GridVector3, grid_2: GridVector3, vertical: bool = True):
        x1, y1, z1 = grid_1
        x2, y2, z2 = grid_2

        if vertical:
            return Mesh.GridVector3(np.vstack((x1, x2)), np.vstack((y1, y2)), np.vstack((z1, z2)))
        
        else:
            return Mesh.GridVector3(np.hstack((x1, x2)), np.hstack((y1, y2)), np.hstack((z1, z2)))

