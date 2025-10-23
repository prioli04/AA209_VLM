import numpy as np
from .Mesh import Mesh

class Wing(Mesh):
    def __init__(self, b: float, AR: float, nx: int, ny: int):
        points = self._compute_points(b, AR, nx, ny)
        super().__init__(points)

    def _compute_points(self, b: float, AR: float, nx: int, ny: int):
        MAC = b / AR
        x = np.linspace(0, MAC, nx + 1)
        y = np.linspace(0, b / 2.0, ny + 1)

        corners_x, corners_y = np.meshgrid(x, y, indexing="ij")
        corners_z = np.zeros_like(corners_x)
        
        return super().GridVector3(corners_x, corners_y, corners_z)